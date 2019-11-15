from worldmodels.ctallec.models import MDRNNCell, VAE, Controller
from worldmodels.ctallec.models.mdrnn import gmm_loss
from worldmodels.ctallec.utils.misc import RolloutGenerator
from worldmodels.ctallec.utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE
from torch.distributions import Categorical
from collections import namedtuple
import torch
import torch.nn.functional as f

SavedActions = namedtuple('SavedActions', ['log_probs', 'value', 'entropy'])

class BigModel(RolloutGenerator):
    def __init__(self, mdir, device, time_limit):
        super(BigModel, self).__init__(mdir, device, time_limit)

        self.saved_actions = []
        self.rewards = []
        self.terminals = []
        self.rnn_loss_args = []


    # assumes obs and hidden are already tensors on device
    def get_action_and_transition(self, obs, hidden):
        # forward pass through fixed world model
        with torch.no_grad():
            _, latent_mu, _ = self.vae(obs)

        # get actions
        probs, state_value = self.controller(latent_mu, hidden[0])
        dists = [Categorical(p) for p in probs] # distribution over actions for each agent
        actions = [dist.sample() for dist in dists]
        
        # save log probs and average entropy
        log_probs = [dist.log_prob(action) for dist, action in zip(dists, actions)]
        avg_entropy = sum([dist.entropy() for dist in dists])/len(dists)
        self.saved_actions.append(SavedActions(log_probs, state_value, avg_entropy))

        # get next hidden state
        mu, sigma, logpi, r, d, next_hidden = self.mdrnn(torch.cat(actions).float().unsqueeze(0), latent_mu, hidden)
        self.rnn_loss_args.append((mu, sigma, logpi, r, d, latent_mu))

        return [action.item() for action in actions], next_hidden


    def get_mdrnn_loss(self, include_reward: bool):
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

        # get in shape (SEQ_LEN, BSIZE, *)
        mus, sigmas, logpis, rs, ds, latent_obs = zip(*self.rnn_loss_args)
        mus, sigmas, logpis, rs, ds, latent_obs= [torch.stack(x).to(self.device) 
                for x in [mus, sigmas, logpis, rs, ds, latent_obs]]

        rewards, terminals = [torch.tensor(x, device=self.device, dtype=torch.float32).unsqueeze(1) for x in [self.rewards, self.terminals]]

        mus, sigmas, logpis, rs, ds, rewards, terminals = [x[:-1] for x in [mus, sigmas, logpis, rs, ds, rewards, terminals]]
        latent_next_obs = latent_obs[1:]

        gmm = gmm_loss(latent_next_obs, mus, sigmas, logpis)

        bce = f.binary_cross_entropy_with_logits(ds, terminals)
        if include_reward:
            mse = f.mse_loss(rs, rewards)
            scale = LSIZE + 2
        else:
            mse = 0
            scale = LSIZE + 1
        loss = (gmm + bce + mse) / scale
        return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)
