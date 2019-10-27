from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.policies.base import Policy
from rlkit.samplers.data_collector import MdpPathCollector
from worldmodels.dynamics_model.train_dynamics_model import RNN_Trainer
from worldmodels.vae.train_vae import VAE_Trainer
from worldmodels.vae.random_stepper import preprocess
from rlkit.torch.core import eval_np
import torch
import numpy as np

models_path = "/home/philip_raeisghasem/worldmodels/worldmodels/working_models/"
latent_size = 32
hidden_size = 256
c_input_size = latent_size+hidden_size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyPolicy(TanhGaussianPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            obs_dim=c_input_size, # hardcoding is brittle but oh well
            action_dim=action_dim,
            std=std,
            init_w=init_w,
            **kwargs
        )
        
        # get vae
        vae_trainer = VAE_Trainer()
        vae_trainer.load_model(filepath=models_path+"vae_model.pt")
        self.vae = vae_trainer.model.to(device)

        # get rnn
        rnn_trainer = RNN_Trainer()
        rnn_trainer.load_model(filepath=models_path+"rnn_model.pt")
        self.rnn = rnn_trainer.model.to(device)

    # get controller inputs from the outputs of the vae and rnn models (the worldmodel)
    def worldmodel(
            self,
            obs,
            last_action,
        ):
        with torch.no_grad():
            # vae
            state = torch.tensor(preprocess(obs)) # (1, 3, 64, 64)
            state = state.type(torch.FloatTensor).cuda()
            z = self.vae.encode(state) # (1, 32)

            # rnn
            a = torch.tensor(last_action)
            a = a.type(torch.FloatTensor).cuda()
            rnn_input = torch.cat((z, a), dim=1) # (1, 35)
            self.rnn(rnn_input)
            
            # controller
            c_input = torch.cat((z, self.rnn.h), dim=1) # (1, 288)
            c_input = c_input.detach()

        del state, z, a, rnn_input
        return c_input

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            last_action=None
        ):

        # if generating a rollout, obs is raw env observation
        if last_action is not None:
            c_input = self.worldmodel(obs,last_action)

        # if training, obs is already concatenated z and h
        else:
            c_input = torch.tensor(obs, device=device)

        rets = super().forward(
            c_input,
            reparameterize=reparameterize,
            deterministic=deterministic,
            return_log_prob=return_log_prob
        )

        torch.cuda.empty_cache()
        return rets + (c_input,)
    

    def get_action(self, obs_np, last_action_np, deterministic=False):
        actions, c_inputs = self.get_actions(obs_np[None], last_action_np[None], deterministic=deterministic)
        return actions[0, :], c_inputs[0, :]


    def get_actions(self, obs_np, last_action_np, deterministic=False):
        rets = self.__call__(obs_np, last_action=last_action_np, deterministic=deterministic)
        actions = rets[0].to("cpu").detach().numpy()
        c_inputs = rets[-1].to("cpu").detach().numpy()
        return actions, c_inputs


def my_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    a = np.zeros(env.action_space.shape)
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, c_input = agent.get_action(o, a)
        next_o, r, d, env_info = env.step(a)
        observations.append(c_input)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append({})
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    
    # encode last observation
    next_o = agent.worldmodel(next_o, actions[-1][None])
    next_o = next_o.to("cpu").numpy()

    next_observations = np.vstack(
        (
            observations[1:, :],
            next_o
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


class MyPathCollector(MdpPathCollector):
    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = my_rollout( # changed this line
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths


class MyMakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy
        self.worldmodel = self.stochastic_policy.worldmodel

    def get_action(self, observation, last_action):
        return self.stochastic_policy.get_action(observation, last_action, deterministic=True)
