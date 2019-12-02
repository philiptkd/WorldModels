from worldmodels.ctallec.models import MDRNNCell, VAE, Actor, Critic
from worldmodels.vrnn import VRNNCell, reparameterize, kl_divergence
from worldmodels.ctallec.models.mdrnn import gmm_loss
from worldmodels.ctallec.utils.misc import RolloutGenerator
from worldmodels.ctallec.utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE, kl_coeff
from torch.distributions import Categorical
from collections import namedtuple
import torch
import torch.nn.functional as f

class BigModel(RolloutGenerator):
    # assumes obs and hidden are already tensors on device
    def rnn_forward(self, latent_mu, hidden):
        # get actions
        if self.vrnn:
            probs = self.actor(latent_mu, reparameterize(hidden[0]))
        else:
            probs = self.actor(latent_mu, hidden[0])
        dists = [Categorical(p) for p in probs] # distribution over actions for each agent
        actions = [dist.sample() for dist in dists]
        
        # save log probs and average entropy
        log_probs = [dist.log_prob(action) for dist, action in zip(dists, actions)]
        avg_entropy = sum([dist.entropy() for dist in dists])/len(dists)

        # forward through rnn
        mu, sigma, logpi, r, d, next_hidden = self.rnn(torch.cat(actions).float().unsqueeze(0), latent_mu, hidden)
        self.rnn_loss_args = (mu, sigma, logpi, r, d, latent_mu, next_hidden)

        actions = [action.item() for action in actions]
        return actions, log_probs, avg_entropy, next_hidden


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

        gmm = gmm_loss(latent_next_obs, mus, sigmas, logpis)

        bce = f.binary_cross_entropy_with_logits(ds, terminals)
        if include_reward:
            mse = f.mse_loss(rs, rewards)
            scale = LSIZE + 2
        else:
            mse = 0
            scale = LSIZE + 1

        if self.vrnn:
            scale += 1
            kl = kl_coeff*kl_divergence(hidden[0])
        else:
            kl = 0

        loss = (gmm + bce + mse + kl) / scale
        return dict(gmm=gmm, bce=bce, mse=mse, loss=loss, kl=kl)
