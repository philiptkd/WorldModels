import multiworld
import gym

multiworld.register_all_envs()
env = gym.make('GoalGridworld-v0')
env.reset()

for i in range(1000):
    env.render()
    env.step(env.action_space.sample())

env.close()
