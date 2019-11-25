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

# Loading model
rnn_dir = join(args.logdir, 'mdrnn')
rnn_file = join(rnn_dir, 'best.tar')
if not exists(rnn_dir):
    mkdir(rnn_dir)

# create ctrl dir if non exitent
ctrl_dir = join(args.logdir, 'ctrl')
if not exists(ctrl_dir):
    mkdir(ctrl_dir)

env = gym.make('BoxCarry-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

gpu_num = np.random.randint(0,torch.cuda.device_count())
device = torch.device("cuda:"+str(gpu_num) if torch.cuda.is_available() else "cpu")
time_limit = 100
bptt_len = 1

big_model = BigModel(args.logdir, device, time_limit)
optimizer = optim.Adam([
                            {'params': big_model.actor.parameters()},
                            {'params': big_model.critic.parameters()},
                            {'params': big_model.mdrnn.parameters()}
                        ])
eps = np.finfo(np.float32).eps.item() # smallest representable number


# takes raw obs from environment
# returns latent from vae
def to_latent(obs, device):
    obs = transform(obs).unsqueeze(0).to(device)
    # forward pass through fixed world model
    with torch.no_grad():
        _, latent_mu, _ = big_model.vae(obs)
    return latent_mu


def train():
    best = -np.inf
    avg_ep_reward = 0
    ep_rewards = []

    # run inifinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        obs = env.reset()
        rnn_hidden = [
            torch.zeros(1, RSIZE).to(device)
            for _ in range(2)]
        done = False
        ep_reward = 0
        discount = 1

        # initial state
        latent_mu = to_latent(obs, device)
        state_value = big_model.critic(latent_mu, rnn_hidden[0])

        # don't infinite loop while learning
        for t in range(time_limit):
            # get action and step forward through rnn
            actions, log_probs, avg_entropy, rnn_hidden = big_model.rnn_forward(latent_mu, rnn_hidden)

            # only retain the graph for so many time steps
            if (t+1)%bptt_len == 0:
                h = rnn_hidden[0].detach()
                c = rnn_hidden[1].detach()
                rnn_hidden = (h, c)

            # step
            obs, reward, done, _ = env.step(actions)
            latent_mu = to_latent(obs, device)

            # get delta
            delta = reward - state_value
            delta *= -1 # sign change to make step in the right direction
            
            # get actor and critic gradients with respect to current state
            optimizer.zero_grad()
            state_value.backward(retain_graph=True)
            for log_prob in log_probs:
                log_prob.backward(retain_graph=True) # to backprop through entropy later

            # update gradients with eligibility traces
            big_model.critic._update_grads_with_eligibility(delta, 1, t, args.gamma)
            big_model.actor._update_grads_with_eligibility(delta, discount, t, args.gamma)

            # update gradients due to changed delta
            if not done:
                next_state_value = big_model.critic(latent_mu, rnn_hidden[0])
                delta_delta = args.gamma*next_state_value
                delta_delta *= -1 # sign change to make step in the right direction
                big_model.critic._increase_delta(delta_delta)
                big_model.actor._increase_delta(delta_delta)

            # update gradients to include entropy loss and mdrnn loss
            aux_loss = -avg_entropy
            if not done:
                mdrnn_loss = big_model.get_mdrnn_loss(latent_mu, reward, done, include_reward=False)
                aux_loss += mdrnn_loss["loss"]
            aux_loss.backward(retain_graph=True)

            # update parameters
            optimizer.step()

            # log reward
            ep_reward += reward
            
            # clear memory
            del big_model.rnn_loss_args

            if done:
                break

            # prepare for next time step
            state_value = next_state_value
            discount *= args.gamma

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

            torch.save({'state_dict': big_model.mdrnn.state_dict()}, 
                    join(rnn_dir, 'best.tar'))



if __name__ == '__main__':
    train()
