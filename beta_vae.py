import gym
env = gym.make("CarRacing-v0")
observation = env.reset()

print(observation.shape)
