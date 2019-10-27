from gym.envs.box2d.car_racing import CarRacing

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from worldmodels.controller.derived import c_input_size, MyPathCollector, MyPolicy, MyMakeDeterministic

def experiment(variant):
    expl_env = NormalizedBoxEnv(CarRacing(verbose=0))
    eval_env = NormalizedBoxEnv(CarRacing(verbose=0))
    obs_dim = c_input_size
    action_dim = eval_env.action_space.low.size

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[],
    )
    policy = MyPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[],
    )
    eval_policy = MyMakeDeterministic(policy)
    eval_path_collector = MyPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MyPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = SimpleReplayBuffer(
        variant['replay_buffer_size'],
        obs_dim,
        action_dim,
        {}
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    variant = dict(
        algorithm="SAC",
        version="normal",
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=2000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=2000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('car-racing-sac', variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
