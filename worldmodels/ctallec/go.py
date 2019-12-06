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
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--simple', action='store_true',
                    help='use the simple model')
parser.add_argument('--use_vrnn', action='store_true',
                    help='use a VRNN for the dynamics model')
args = parser.parse_args()


img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

# Loading model
rnn_dir = join(args.logdir, 'rnn')
rnn_file = join(rnn_dir, 'best.tar')
if not exists(rnn_dir):
    mkdir(rnn_dir)

# create ctrl dir if non exitent
ctrl_dir = join(args.logdir, 'ctrl')
if not exists(ctrl_dir):
    mkdir(ctrl_dir)

if args.simple:
    env = gym.make('BoxCarry-v0')
else:
    env = gym.make('BoxCarryImg-v0')

env.seed(args.seed)
torch.manual_seed(args.seed)

gpu_num = np.random.randint(0,torch.cuda.device_count())
device = torch.device("cuda:"+str(gpu_num) if torch.cuda.is_available() else "cpu")
time_limit = 100
bptt_len = 4
target_update_period = 1000

env_size = env.grid_size**2
big_model = BigModel(args.logdir, device, time_limit, args.use_vrnn, args.simple, env_size)
optimizer = optim.Adam(big_model.parameters())

eps = np.finfo(np.float32).eps.item() # smallest representable number
hidden_size = RSIZE_simple if args.simple else RSIZE
if args.use_vrnn:
    hidden_size = hidden_size*2

# takes raw obs from environment
# returns latent from vae
def get_rnn_in(obs, device):
    if args.simple:
        obs = torch.Tensor(obs)/2 # normalize to [0,1]
    else:
        obs = img_transform(obs)
    obs = obs.unsqueeze(0).to(device)

    if args.simple:
        return obs

    # forward pass through fixed world model
    with torch.no_grad():
        _, latent_mu, _ = big_model.vae(obs)
    return latent_mu


def train():
    best = -np.inf
    avg_ep_reward = 0
    ep_rewards = []
    total_steps = 0

    # run inifinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        obs = env.reset()
        rnn_hidden = [
            torch.zeros(1, hidden_size).to(device)
            for _ in range(2)]
        done = False
        ep_reward = 0
        discount = 1

        # initial state
        rnn_in = get_rnn_in(obs, device)

        # don't infinite loop while learning
        for t in range(time_limit):
            # copy parameters to target network
            if total_steps % target_update_period == 0:
                big_model.target_critic.load_state_dict(big_model.critic.state_dict())

            # only retain the (recurrent) graph for so many time steps
            if (t+1)%bptt_len == 0:
                h = rnn_hidden[0].detach()
                c = rnn_hidden[1].detach()
                rnn_hidden = (h, c)

            # get state value from critic
            rnn_in = get_rnn_in(obs, device)
            if args.use_vrnn:
                state_value = big_model.critic(rnn_in, reparameterize(rnn_hidden[0]))
            else:
                state_value = big_model.critic(rnn_in, rnn_hidden[0])

            # get action from actor and step forward through rnn
            actions, log_probs, avg_entropy, rnn_hidden = big_model.rnn_forward(rnn_in, rnn_hidden)

            # step
            obs, reward, done, _ = env.step(actions)

            # get delta
            delta = reward - state_value
            delta *= -1 # sign change to make step in the right direction
            
            # get actor and critic gradients with respect to current state before evaluating next state
            optimizer.zero_grad()
            state_value.backward(retain_graph=True)
            for log_prob in log_probs:
                log_prob.backward(retain_graph=True) # to backprop through entropy later

            # update gradients with eligibility traces
            big_model.critic._update_grads_with_eligibility(delta, 1, t, args.gamma)
            big_model.actor._update_grads_with_eligibility(delta, discount, t, args.gamma)

            # update gradients due to changed delta
            if not done:
                with torch.no_grad():
                    if args.use_vrnn:
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

            if t > 0:
                for name, module in zip(["actor", "critic", "rnn"], [big_model.actor, big_model.critic, big_model.rnn]):
                    for p in module.parameters():
                        assert (p.grad is not None), "No gradient for "+name+" parameter with shape "+str(p.grad.shape)
                        assert not all(p.grad.flatten() == 0),  "Zero gradient for "+name+" parameter with shape "+str(p.grad.shape)
                for name, module in zip(["vae", "target_critic"], [big_model.vae, big_model.target_critic]):
                    for p in module.parameters():
                        assert p.grad is None, "Existing gradient for "+name+" parameter with shape "+str(p.grad.shape)

            # update parameters
            optimizer.step()

            # log reward
            ep_reward += reward
            
            # clear memory
            del big_model.rnn_loss_args

            if done:
                break

            # prepare for next time step
            discount *= args.gamma
            total_steps += 1
    
        ep_rewards.append(ep_reward)
        avg_ep_reward += (ep_reward - avg_ep_reward)/i_episode

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, avg_ep_reward))
            
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



if __name__ == '__main__':
    train()
