from worldmodels.vrnn import VRNNCell, reparameterize
from worldmodels.ctallec.utils.misc import RolloutGenerator
from torch.distributions import Categorical
from collections import namedtuple
import torch

SavedActions = namedtuple('SavedActions', ['log_probs', 'value', 'entropy'])

class BigModel(RolloutGenerator):
    def __init__(self, mdir, device, time_limit):
        super(BigModel, self).__init__(mdir, device, time_limit)

        self.saved_actions = []
        self.rewards = []


    # assumes obs and hidden are already tensors on device
    def get_action_and_transition(self, obs, hidden):
        # forward pass through the fixed world model
        with torch.no_grad():
            _, latent_mu, _ = self.vae(obs)
       
        # get actions
        probs, state_value = self.controller(latent_mu, reparameterize(hidden[0]))
        dists = [Categorical(p) for p in probs] # distribution over actions for each agent
        actions = [dist.sample() for dist in dists]
        
        # save log probs and average entropy
        log_probs = [dist.log_prob(action) for dist, action in zip(dists, actions)]
        avg_entropy = sum([dist.entropy() for dist in dists])/len(dists)
        self.saved_actions.append(SavedActions(log_probs, state_value, avg_entropy))

        # get next hidden state
        with torch.no_grad():
            _, _, _, _, _, next_hidden = self.vrnn(torch.cat(actions).float().unsqueeze(0), latent_mu, hidden)

        return [action.item() for action in actions], next_hidden

