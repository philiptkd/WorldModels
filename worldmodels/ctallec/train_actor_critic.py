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

from worldmodels.big_model import BigModel
from utils.misc import RED_SIZE, RSIZE


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
args = parser.parse_args()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

# create ctrl dir if non exitent
ctrl_dir = join(args.logdir, 'ctrl')
if not exists(ctrl_dir):
    mkdir(ctrl_dir)

env = gym.make('BoxCarry-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
time_limit = 10000

big_model = BigModel(args.logdir, device, time_limit)
optimizer = optim.Adam(big_model.controller.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item() # smallest representable number

def finish_episode():
    """
    Training code. Calcultes actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = big_model.saved_actions
    policy_losses = []
    value_losses = []
    entropies = []
    returns = []

    for r in big_model.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_probs, value, entropy), R in zip(saved_actions, returns):
        advantage = R - value.item()
        policy_losses += [-log_prob * advantage for log_prob in log_probs] # add all agents' policy losses
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([[R]], device=device)))
        entropies.append(entropy)

    # backprop
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum() - torch.stack(entropies).sum()
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del big_model.rewards[:]
    del big_model.saved_actions[:]


def train():
    avg_ep_reward = 0

    # run inifinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        obs = env.reset()
        ep_reward = 0
        rnn_hidden = [
            torch.zeros(1, RSIZE).to(device)
            for _ in range(2)]

        # for each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        for t in range(time_limit):
            obs = transform(obs).unsqueeze(0).to(device)
            actions, rnn_hidden = big_model.get_action_and_transition(obs, rnn_hidden)
            obs, reward, done, _ = env.step(actions)

            if args.render:
                env.render()

            big_model.rewards.append(reward)
            ep_reward += reward
            
            if done:
                break

        avg_ep_reward += (ep_reward - avg_ep_reward)/i_episode

        # perform backprop
        finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, avg_ep_reward))

            torch.save(
                {'episode': i_episode,
                 'reward': avg_ep_reward,
                 'state_dict': big_model.controller.state_dict()},
                join(ctrl_dir, 'latest.tar'))


if __name__ == '__main__':
    train()
