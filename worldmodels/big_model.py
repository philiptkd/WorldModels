from worldmodels.ctallec.models import VAE, Actor, Critic
from torch.distributions import Categorical
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as f
from worldmodels.box_carry_env import BoxCarryEnv, BoxCarryImgEnv
from os.path import join, exists
from worldmodels.experience_replay import ExperienceReplay, ProportionalReplay
import numpy as np

# Hardcoded for now
# action_size, latent_size, rnn_size, vae_in_size, 
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    BoxCarryEnv.num_agents, 32, 256, 64, 64
RSIZE_simple = 64

prioritized_replay = True
prioritized_replay_alpha = 0.6  # from paper
max_buffer_size = 1000
beta0 = 0.4  # from paper

class BigModel(nn.Module):
    def __init__(self, mdir, device, time_limit, simple, env):
        super(BigModel, self).__init__()

        self.simple = simple
        self.env = env

        """ Build vae, rnn, controller and environment. """
        # Loading world model and vae
        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'rnn', 'ctrl']]

        # the simple version of the model doesn't have a vae
        if not self.simple:
            assert exists(vae_file), "Trained vae file does not exist at " + vae_file

            vae_state = torch.load(vae_file, map_location={'cuda:0': str(device)})
            print("Loading VAE at epoch {} with test loss {}".format(
                      vae_state['epoch'], vae_state['precision']))
            
            self.vae = VAE(3, LSIZE).to(device)
            self.vae.load_state_dict(vae_state['state_dict'])

        if self.simple:
            self.obs_size = self.env.grid_size**2
        else:
            self.obs_size = LSIZE

        self.actor = Actor(self.obs_size, ASIZE).to(device)
        self.critic = Critic(self.obs_size).to(device)

        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.critic.load_state_dict(ctrl_state['critic_state_dict'])
            self.actor.load_state_dict(ctrl_state['actor_state_dict'])

        self.device = device
        self.time_limit = time_limit

        if prioritized_replay:
            self.replay = ProportionalReplay(max_buffer_size, prioritized_replay_alpha)
        else:
            self.replay = ExperienceReplay(max_buffer_size)


    # assumes obs and hidden are already tensors on device
    def get_actions(self, latent_mu):
        probs = self.actor(latent_mu)
        dists = [Categorical(p) for p in probs] # distribution over actions for each agent
        actions = [dist.sample() for dist in dists] # [(1,), (1,)] or [(20,), (20,)]
        actions = torch.stack(actions, dim=1).float() # (20, 2)

        # save log probs and average entropy
        # dists is a list of length 2, but actions has length 20 (shape (20, 2))
        log_probs = [dist.log_prob(action) for dist, action in zip(dists, actions.transpose(0,1))]
        avg_entropy = sum([dist.entropy() for dist in dists])/len(dists)

        actions = actions.squeeze().cpu().numpy().astype(int)

        # TODO: generalize this to batch case
        # taken_action_probs = [p[0][a] for (p,a) in zip(probs, actions)] # list of num_agents floats

        return [actions, log_probs, avg_entropy] #, taken_action_probs]

   
    def add_rollout_to_buffer(self):
        obs = self.env.reset()
        done = False
        rollout = []
        total_reward = 0

        while not done: # will happen after env time_limit, at latest
            net_in = self.preprocess_obs(obs)
            actions, _, _ = self.get_actions(net_in)

            obs, reward, done, _ = self.env.step(actions)

            state = net_in.to("cpu")
            state, actions, reward = [np.array(x) for x in [state, actions, reward]]

            rollout.append((state, actions, reward))

            total_reward += reward

        self.replay.add(rollout)
        return total_reward
            

    # takes raw obs from environment
    # returns latent from vae
    def preprocess_obs(self, obs):
        if self.simple:
            obs = torch.Tensor(obs)/2 # normalize to [0,1]
        else:
            obs = img_transform(obs)
        obs = obs.unsqueeze(0).to(self.device)

        if self.simple:
            return obs

        # forward pass through fixed world model
        with torch.no_grad():
            _, latent_mu, _ = self.vae(obs)
        return latent_mu
