import argparse
import gym
import numpy as np
from itertools import count
from os.path import join, exists
from os import mkdir

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from worldmodels.big_model import BigModel, RED_SIZE, RSIZE, RSIZE_simple
from worldmodels.vrnn import reparameterize


parser = argparse.ArgumentParser(description='Actor Critic')
parser.add_argument('--logdir', type=str, default="log_dir",
                    help='Where everything is stored.')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543,
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10,
                    help='interval between training status logs (default: 10)')
parser.add_argument('--simple', action='store_true',
                    help='use the simple model')
parser.add_argument('--use_vrnn', action='store_true',
                    help='use a VRNN for the dynamics model')
parser.add_argument('--bptt_len', type=int, default=2,
                    help='steps into the past that gradients backprop through the rnn (default: 2)')
parser.add_argument('--time_limit', type=int, default=100,
                    help='max number of environment steps per episode (default: 100)')
parser.add_argument('--target_update_period', type=int, default=1000,
                    help='period for updating the weights of the target critic network (default: 1000)')
parser.add_argument('--lamb', type=float, default=0.6,
                    help='lambda for TD(lambda) (default: 0.6)')
parser.add_argument('--kl_coeff', type=float, default=5.0,
                    help='coefficient for the kl portion of the rnn loss (default: 5.0)')
parser.add_argument('--predict_terminals', action='store_true',
                    help='include "done" prediction in rnn loss')
parser.add_argument('--cooperate', action='store_true',
                    help='require both agents to cooperate to move the box')
parser.add_argument('--num_episodes', type=int, default=200,
                    help='episodes to train for each hyperparameter setting (default: 200)')
parser.add_argument('--param_search', action='store_true',
                    help='do a hyperparameter search')
args = parser.parse_args()

img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

# always need these
gpu_num = 1#np.random.randint(0,torch.cuda.device_count())
device = torch.device("cuda:"+str(gpu_num) if torch.cuda.is_available() else "cpu")
eps = np.finfo(np.float32).eps.item() # smallest representable number
batch_size = 20
beta0 = 0.4  # from paper

def hyperparam_search():
    for predict_terminals in [True, False]:
        for use_vrnn in [True, False]:
            for simple in [True]:
                for grabbers_needed in [1,2]:
                    if simple:
                        env = gym.make('BoxCarry-v0', grabbers_needed=grabbers_needed)
                    else:
                        env = gym.make('BoxCarryImg-v0', grabbers_needed=grabbers_needed)

                    env.seed(args.seed)
                    torch.manual_seed(args.seed)

                    if grabbers_needed == 1:
                        logdir = "log_dir_two_easy"
                    elif grabbers_needed == 2:
                        logdir = "log_dir_two"

                    big_model = BigModel(logdir, device, args.time_limit, use_vrnn, simple, args.kl_coeff, predict_terminals, env)
                    optimizer = optim.Adam(big_model.parameters())

                    print("grabbers: {}, vrnn: {}, simple: {}, terminals: {}".format(
                        env.grabbers_needed, big_model.use_vrnn, big_model.simple, big_model.predict_terminals))
                    
                    print("avg_reward:", train(env, big_model, optimizer), "\n")
        

def check_grads(t):
    for name, module in zip(["actor", "critic", "rnn"], [big_model.actor, big_model.critic, big_model.rnn]):
        if all([p.grad is None for p in module.parameters()]):
            print("t = "+str(t)+". No gradients for "+name)
        elif all([(p.grad is None) or all(p.grad.flatten() == 0) for p in module.parameters()]):
            print("t = "+str(t)+". Only zero gradients for "+name)
    if hasattr(big_model, 'vae'):
        if all([p.grad is not None for p in big_model.vae.parameters()]):
            print("t = "+str(t)+". Existing gradients for vae")


# for now, igorning the fact that the target and behavior policies are different
def get_grads(env, big_model):
    # sample from replay buffer
    # weights and indicies are both of length batch_size
    # the others are arrays of size (batch_size, max_rollout_len, *)
    beta = 1 #beta0 + episode*(1-beta0)/episodes   # linear annealing schedule for IS weights. #TODO: turn back on
    states, actions, rewards, probs, weights, indicies = big_model.replay.sample(batch_size, beta)
    states = states.squeeze().transpose((1, 0, 2)) # (max_rollout_len, batch_size, latent_size)

    rewards = rewards.transpose() # (max_rollout_len, batch_size)

    # TODO: calculate importance sampled returns based on action probabilities
    returns = np.cumsum(rewards, axis=0) # returns from each time step in each rollout. shape (max_rollout_len, batch_size)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    states, returns, weights = [torch.Tensor(x).to(device) for x in [states, returns, weights]]

    # initial rnn input
    rnn_size = big_model.rnn_size*2 if big_model.use_vrnn else big_model.rnn_size
    rnn_hidden = [
        torch.zeros(batch_size, rnn_size).to(device)
        for _ in range(2)]

    critic_losses = []
    policy_losses = []
    entropies = []
    avg_advantage = 0

    for t, (state, ret) in enumerate(zip(states, returns)):
        mask = torch.Tensor([not all([x == 0 for x in y]) for y in state]).to(device) # boolean mask of shape (batch_size,)

        # get losses for the critic
        if big_model.use_vrnn:
            state_value = big_model.critic(state, reparameterize(rnn_hidden[0]))
        else:
            state_value = big_model.critic(state, rnn_hidden[0])
        
        # mask state_value and ret
        state_value = state_value * mask
        ret = ret * mask

        mse_criterion = nn.MSELoss()
        critic_losses.append(mse_criterion(state_value, ret))

        # get losses for the actor
        advantage = ret - state_value.detach() # (batch_size,)
        weighted_advantage = advantage * weights # prioritized replay
        _, log_probs, avg_entropy, rnn_hidden, these_probs = big_model.rnn_forward(state, rnn_hidden)
        log_probs = log_probs.transpose() # (num_agents, batch_size)
        policy_losses += [-log_prob * weighted_advantage for log_prob in log_probs] # add all agents' policy losses
        entropies.append(avg_entropy * mask) 

        # periodically detach to limit backprop through time
        if (t+1)%args.bptt_len == 0:
            h = rnn_hidden[0].detach()
            c = rnn_hidden[1].detach()
            rnn_hidden = (h, c)

        # for updating replay priorities later
        avg_advantage += (advantage - avg_advantage)/(t+1)

    # update priority replay priorities
    big_model.replay.update_priorities(indices, np.abs(avg_advantage)+1e-3)   # add small number to prevent never sampling 0 error transitions

    # update gradients
    loss = torch.stack(policy_losses).sum() + torch.stack(critic_losses).sum() - torch.stack(entropies).sum()
    loss.backward()

    # TODO: fix mdrnn loss
    #    mdrnn_loss = big_model.get_mdrnn_loss(rnn_in, reward, done, include_reward=False)["loss"]
    #mdrnn_loss.backward(retain_graph=True)


def train(env, big_model, optimizer):
    best = -np.inf
    avg_ep_reward = 0
    ep_rewards = []

    ep_iter = count(1) if not args.param_search else range(1, args.num_episodes+1)

    # for each episode
    for i_episode in ep_iter:
        # get new experience
        ep_reward = big_model.add_rollout_to_buffer()

        # learn from batch of past experience
        optimizer.zero_grad()
        get_grads(env, big_model)
        optimizer.step()

        ep_rewards.append(ep_reward)
        avg_ep_reward += (ep_reward - avg_ep_reward)/i_episode

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, avg_ep_reward))
                
        if not args.param_search:
            # save reward history
            if i_episode % args.log_interval == 0:
                with open(join(ctrl_dir, "reward_hist.npy"), "wb+") as f:
                    np.save(f, ep_rewards)

            # save state
            if avg_ep_reward > best:
                best = avg_ep_reward
                torch.save(
                    {'episode': i_episode,
                     'reward': avg_ep_reward,
                     'actor_state_dict': big_model.actor.state_dict(),
                     'critic_state_dict': big_model.critic.state_dict()},
                    join(ctrl_dir, 'best.tar'))

                torch.save({'state_dict': big_model.rnn.state_dict()}, 
                        join(rnn_dir, 'best.tar'))
    
    # only returns if not infinite episodes
    return avg_ep_reward


if __name__ == '__main__':
    # if we don't do a hyperparameter search and just use the passed args
    if args.param_search:
        hyperparam_search()
    else:
        # Loading model
        rnn_dir = join(args.logdir, 'rnn')
        rnn_file = join(rnn_dir, 'best.tar')
        if not exists(rnn_dir):
            mkdir(rnn_dir)

        # create ctrl dir if non exitent
        ctrl_dir = join(args.logdir, 'ctrl')
        if not exists(ctrl_dir):
            mkdir(ctrl_dir)

        if args.cooperate:
            grabbers_needed = 2
        else:
            grabbers_needed = 1

        if args.simple:
            env = gym.make('BoxCarry-v0', grabbers_needed=grabbers_needed)
        else:
            env = gym.make('BoxCarryImg-v0', grabbers_needed=grabbers_needed)

        env.seed(args.seed)
        torch.manual_seed(args.seed)

        env_size = env.grid_size**2

        big_model = BigModel(args.logdir, device, args.time_limit, args.use_vrnn, args.simple, args.kl_coeff, args.predict_terminals, env)
        optimizer = optim.Adam(big_model.parameters())

        train(env, big_model, optimizer)
