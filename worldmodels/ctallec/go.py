import argparse
import gym
import numpy as np
from itertools import count
from os.path import join, exists
from os import mkdir

import torch
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


def hyperparam_search():
    global total_steps

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
                    
                    total_steps = 0
                    print("avg_reward:", train(env, big_model, optimizer), "\n")
        

# takes raw obs from environment
# returns latent from vae
def get_rnn_in(obs, device, big_model):
    if big_model.simple:
        obs = torch.Tensor(obs)/2 # normalize to [0,1]
    else:
        obs = img_transform(obs)
    obs = obs.unsqueeze(0).to(device)

    if big_model.simple:
        return obs

    # forward pass through fixed world model
    with torch.no_grad():
        _, latent_mu, _ = big_model.vae(obs)
    return latent_mu


def check_grads(t):
    for name, module in zip(["actor", "critic", "rnn"], [big_model.actor, big_model.critic, big_model.rnn]):
        if all([p.grad is None for p in module.parameters()]):
            print("t = "+str(t)+". No gradients for "+name)
        elif all([(p.grad is None) or all(p.grad.flatten() == 0) for p in module.parameters()]):
            print("t = "+str(t)+". Only zero gradients for "+name)
    if all([p.grad is not None for p in big_model.target_critic.parameters()]):
        print("t = "+str(t)+". Existing gradients for target critic")
    if hasattr(big_model, 'vae'):
        if all([p.grad is not None for p in big_model.vae.parameters()]):
            print("t = "+str(t)+". Existing gradients for vae")


def td_lambda(env, big_model):
    global total_steps

    obs = env.reset()
    rnn_size = big_model.rnn_size*2 if big_model.use_vrnn else big_model.rnn_size
    rnn_hidden = [
        torch.zeros(1, rnn_size).to(device)
        for _ in range(2)]
    discount = 1

    # initial state
    rnn_in = get_rnn_in(obs, device, big_model)

    # don't infinite loop while learning
    for t in range(args.time_limit):
        # copy parameters to target network
        if total_steps % args.target_update_period == 0:
            big_model.target_critic.load_state_dict(big_model.critic.state_dict())
        total_steps += 1

        # get state value from critic
        rnn_in = get_rnn_in(obs, device, big_model)
        if big_model.use_vrnn:
            state_value = big_model.critic(rnn_in, reparameterize(rnn_hidden[0]))
        else:
            state_value = big_model.critic(rnn_in, rnn_hidden[0])

        # get action from actor and step forward through rnn
        actions, log_probs, avg_entropy, rnn_hidden = big_model.rnn_forward(rnn_in, rnn_hidden)

        # step in environment by performing actions
        obs, reward, done, _ = env.step(actions)

        # get delta
        delta = reward - state_value
        delta *= -1 # sign change to make step in the right direction
        
        # get actor and critic gradients with respect to current state before evaluating next state
        state_value.backward(retain_graph=True)
        for log_prob in log_probs:
            log_prob.backward(retain_graph=True) # to backprop through entropy later

        # update gradients with eligibility traces
        big_model.critic._update_grads_with_eligibility(delta, 1, t, args.gamma)
        big_model.actor._update_grads_with_eligibility(delta, discount, t, args.gamma)

        # update gradients due to changed delta
        if not done:
            with torch.no_grad():
                if big_model.use_vrnn:
                    next_state_value = big_model.target_critic(rnn_in, reparameterize(rnn_hidden[0]))
                else:
                    next_state_value = big_model.target_critic(rnn_in, rnn_hidden[0])
            delta_delta = args.gamma*next_state_value
            delta_delta *= -1 # sign change to make step in the right direction
            big_model.critic._increase_delta(delta_delta)
            big_model.actor._increase_delta(delta_delta)

        # update gradients to include entropy loss and mdrnn loss
        aux_loss = -avg_entropy
        if not done:
            mdrnn_loss = big_model.get_mdrnn_loss(rnn_in, reward, done, include_reward=False)
            aux_loss += mdrnn_loss["loss"]
        aux_loss.backward(retain_graph=True)

        # prepare for next time step
        discount *= args.gamma
            
        # only retain the (recurrent) graph for so many time steps
        if (t+1)%args.bptt_len == 0:
            h = rnn_hidden[0].detach()
            c = rnn_hidden[1].detach()
            rnn_hidden = (h, c)

        yield reward, done
    


def train(env, big_model, optimizer):
    best = -np.inf
    avg_ep_reward = 0
    ep_rewards = []

    ep_iter = count(1) if not args.param_search else range(1, args.num_episodes+1)
    for i_episode in ep_iter:
        ep_reward = 0
        grad_gen = td_lambda(env, big_model) # forward and backward through model, generating parameter gradients
        done = False

        while not done:
            optimizer.zero_grad()
            reward, done = next(grad_gen)
            optimizer.step()

            ep_reward += reward
    
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

        big_model = BigModel(args.logdir, device, args.time_limit, args.use_vrnn, args.simple, env_size, args.lamb, args.kl_coeff, args.predict_terminals)
        optimizer = optim.Adam(big_model.parameters())

        total_steps = 0
        train(env, big_model, optimizer)
