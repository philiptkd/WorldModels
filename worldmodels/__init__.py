from gym.envs.registration import register
from worldmodels.box_carry_env import BoxCarryEnv

register(
    id='BoxCarry-v0',
    entry_point='worldmodels:BoxCarryEnv',
    max_episode_steps=100,
)
