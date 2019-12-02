""" Various auxiliary utilities """
import math
from os.path import join, exists
import torch
from torchvision import transforms
import numpy as np
from models import MDRNNCell, VAE, Actor, Critic
import gym
import worldmodels
from worldmodels.box_carry_env import BoxCarryEnv
from worldmodels.vrnn import VRNNCell

# Hardcoded for now
# action_size, latent_size, rnn_size, vae_in_size, 
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    BoxCarryEnv.num_agents, 32, 256, 64, 64
lamb = 0.6
kl_coeff = 5

# Same
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])


class RolloutGenerator(object):
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.

    :attr vae: VAE model loaded from mdir/vae
    :attr rnn: MDRNN model loaded from mdir/rnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v0 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, mdir, device, time_limit, vrnn: bool):
        self.vrnn = vrnn

        """ Build vae, rnn, controller and environment. """
        # Loading world model and vae
        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'rnn', 'ctrl']]

        assert exists(vae_file), "VAE is untrained." + vae_file

        vae_state = torch.load(vae_file, map_location={'cuda:0': str(device)})
        print("Loading VAE at epoch {} with test loss {}".format(
                  vae_state['epoch'], vae_state['precision']))

        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])

        if vrnn:
            self.rnn = VRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        else:
            self.rnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)

        self.actor = Actor(LSIZE, RSIZE, ASIZE, lamb).to(device)
        self.critic = Critic(LSIZE, RSIZE, lamb).to(device)
        self.target_critic = Critic(LSIZE, RSIZE, lamb).to(device)
        self.target_critic.eval() # don't backprop through target net

        # load rnn and controller if they were previously saved
        if exists(rnn_file):
            rnn_state = torch.load(rnn_file, map_location={'cuda:0': str(device)})
            print("Loading MDRNN")
            self.rnn.load_state_dict(rnn_state['state_dict'])

        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.actor.load_state_dict(ctrl_state['actor_state_dict'])
            self.critic.load_state_dict(ctrl_state['critic_state_dict'])

        #self.env = gym.make('BoxCarry-v0')
        self.device = device

        self.time_limit = time_limit

    def get_action_and_transition(self, obs, hidden):
        """ Get action and transition.

        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.

        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        _, latent_mu, _ = self.vae(obs)
        action = self.controller(latent_mu, hidden[0])
        _, _, _, _, _, next_hidden = self.rnn(action, latent_mu, hidden)
        return action.squeeze().cpu().numpy().astype(int).tolist(), next_hidden

    def rollout(self, params, render=False):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        obs = self.env.reset()

        # This first render is required !
        self.env.render("rgb_array")

        hidden = [
            torch.zeros(1, RSIZE).to(self.device)
            for _ in range(2)]

        cumulative = 0
        i = 0
        while True:
            obs = transform(obs).unsqueeze(0).to(self.device)
            actions, hidden = self.get_action_and_transition(obs, hidden)
            obs, reward, done, _ = self.env.step(actions)

            if render:
                self.env.render("rgb_array")

            cumulative += reward
            if done or i > self.time_limit:
                return - cumulative
            i += 1


def sample_continuous_policy(action_space, seq_len, dt):
    """ Sample a continuous policy.

    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).

    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization

    :returns: sequence of seq_len actions
    """
    actions = [action_space.sample()]
    for _ in range(seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

