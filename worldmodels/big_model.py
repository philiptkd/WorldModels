from worldmodels.ctallec.models import MDRNNCell, VAE, QNetwork
from worldmodels.vrnn import VRNNCell, reparameterize, kl_divergence
from worldmodels.ctallec.models.mdrnn import gmm_loss
from torch.distributions import Categorical
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as f
from worldmodels.box_carry_env import BoxCarryEnv, BoxCarryImgEnv
from os.path import join, exists

# Hardcoded for now
# action_size, latent_size, rnn_size, vae_in_size, 
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    BoxCarryEnv.num_agents, 32, 256, 64, 64
RSIZE_simple = 64
eps = 0.1

predict_terminals = False

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

        self.qnet = QNetwork(self.obs_size, self.rnn_size, ASIZE).to(device)
        self.target_qnet = QNetwork(self.obs_size, self.rnn_size, ASIZE).to(device)
        
        # load rnn and controller if they were previously saved
        if exists(rnn_file):
            rnn_state = torch.load(rnn_file, map_location={'cuda:0': str(device)})
            print("Loading MDRNN")
            self.rnn.load_state_dict(rnn_state['state_dict'])

        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.qnet.load_state_dict(ctrl_state['critic_state_dict'])
            self.target_qnet.load_state_dict(ctrl_state['critic_state_dict'])

        self.device = device
        self.time_limit = time_limit

    # assumes obs and hidden are already tensors on device
    def rnn_forward(self, latent_mu, hidden):
        # get actions
        if self.use_vrnn:
            Q = self.qnet(latent_mu, reparameterize(hidden[0]))
        else:
            Q = self.qnet(latent_mu, hidden[0])
        Q = torch.stack(Q) # from list of 1D tensors to 2D tensor

        rands = torch.empty_like(Q).uniform_(0, 1)
        rand_mask = rands.le(eps)
        rand_actions = torch.multinomial(torch.ones(self.env.action_space.n), self.env.num_agents, replacement=True)
        max_actions = Q.argmax(dim=1) # currently not breaking ties fairly
        actions = torch.where(rand_mask, rand_actions, max_actions)

        # forward through rnn
        mu, sigma, logpi, r, d, next_hidden = self.rnn(actions.float().unsqueeze(0), latent_mu, hidden)
        self.rnn_loss_args = (mu, sigma, logpi, r, d, latent_mu, next_hidden)

        actions = [action.item() for action in actions]
        return actions, next_hidden


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
