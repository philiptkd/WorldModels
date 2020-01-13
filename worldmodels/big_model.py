from worldmodels.ctallec.models import MDRNNCell, VAE, Actor, Critic
from worldmodels.vrnn import VRNNCell, reparameterize, kl_divergence
from worldmodels.ctallec.models.mdrnn import gmm_loss
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
    def __init__(self, mdir, device, time_limit, use_vrnn, simple, kl_coeff, predict_terminals, env):
        super(BigModel, self).__init__()

        self.use_vrnn = use_vrnn
        self.simple = simple
        self.kl_coeff = kl_coeff
        self.predict_terminals = predict_terminals
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
            self.rnn_size = RSIZE_simple
        else:
            self.obs_size = LSIZE
            self.rnn_size = RSIZE

        if self.use_vrnn:
            self.rnn = VRNNCell(self.obs_size, ASIZE, self.rnn_size, 5).to(device)
        else:
            self.rnn = MDRNNCell(self.obs_size, ASIZE, self.rnn_size, 5).to(device)

        self.actor = Actor(self.obs_size, self.rnn_size, ASIZE).to(device)
        self.critic = Critic(self.obs_size, self.rnn_size).to(device)

        # load rnn and controller if they were previously saved
        if exists(rnn_file):
            rnn_state = torch.load(rnn_file, map_location={'cuda:0': str(device)})
            print("Loading MDRNN")
            self.rnn.load_state_dict(rnn_state['state_dict'])

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
    def rnn_forward(self, latent_mu, hidden):
        # get actions
        if self.use_vrnn:
            probs = self.actor(latent_mu, reparameterize(hidden[0]))
        else:
            probs = self.actor(latent_mu, hidden[0])

        dists = [Categorical(p) for p in probs] # distribution over actions for each agent
        actions = [dist.sample() for dist in dists] # [(1,), (1,)] or [(20,), (20,)]
        actions = torch.stack(actions, dim=1).float() # (20, 2)

        # save log probs and average entropy
        # dists is a list of length 2, but actions has length 20 (shape (20, 2))
        log_probs = [dist.log_prob(action) for dist, action in zip(dists, actions.transpose(0,1))]
        avg_entropy = sum([dist.entropy() for dist in dists])/len(dists)

        # forward through rnn
        mu, sigma, logpi, r, d, next_hidden = self.rnn(actions, latent_mu, hidden)
        #self.rnn_loss_args.append((mu, sigma, logpi, r, d, latent_mu, next_hidden)) #TODO: include
       
        actions = actions.squeeze().cpu().numpy().astype(int)

        # TODO: generalize this to batch case
        # taken_action_probs = [p[0][a] for (p,a) in zip(probs, actions)] # list of num_agents floats

        return [actions, log_probs, avg_entropy, next_hidden] #, taken_action_probs]

   
    def add_rollout_to_buffer(self):
        obs = self.env.reset()
        rnn_size = self.rnn_size*2 if self.use_vrnn else self.rnn_size
        rnn_hidden = [
            torch.zeros(1, rnn_size).to(self.device)
            for _ in range(2)]
        done = False
        rollout = []
        total_reward = 0

        while not done: # will happen after env time_limit, at latest
            rnn_in = self.get_rnn_in(obs)
            actions, _, _, rnn_hidden = self.rnn_forward(rnn_in, rnn_hidden)

            obs, reward, done, _ = self.env.step(actions)

            state = rnn_in.to("cpu")
            state, actions, reward = [np.array(x) for x in [state, actions, reward]]

            rollout.append((state, actions, reward))

            total_reward += reward

        self.replay.add(rollout)
        return total_reward
            

    # takes raw obs from environment
    # returns latent from vae
    def get_rnn_in(self, obs):
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



    def get_mdrnn_loss(self, latent_next_obs, reward, done, include_reward: bool):
        """ Compute losses.

        The loss that is computed is:
        (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
             BCE(terminal, logit_terminal)) / (LSIZE + 2)
        The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
        approximately linearily with LSIZE. All losses are averaged both on the
        batch and the sequence dimensions (the two first dimensions).

        :returns: dictionary of losses, containing the gmm, the mse, the bce and
            the averaged loss.
        """

        # assumes SEQ_LEN = 1
        mus, sigmas, logpis, rs, ds, latent_obs, hidden = self.rnn_loss_args
        rewards, terminals = [torch.tensor(x, device=self.device, dtype=torch.float32).unsqueeze(0) for x in [reward, done]]

        # predicting the next latent observation
        gmm = gmm_loss(latent_next_obs, mus, sigmas, logpis)
        scale = LSIZE

        # predicting whether the episode has ended
        if self.predict_terminals:
            bce = f.binary_cross_entropy_with_logits(ds, terminals)
            scale += 1
        else:
            bce = 0

        # predicting the last reward
        if include_reward:
            mse = f.mse_loss(rs, rewards)
            scale += 1
        else:
            mse = 0

        # information bottleneck on rnn hidden state. 
        # (not confident in how this scales with LSIZE)
        if self.use_vrnn:
            kl = self.kl_coeff*kl_divergence(hidden[0])
            scale += 1
        else:
            kl = 0
        

        loss = (gmm + bce + mse + kl) / scale
            
        # clear memory
        del self.rnn_loss_args

        return dict(gmm=gmm, bce=bce, mse=mse, loss=loss, kl=kl)
