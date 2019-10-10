from gym.envs.box2d.car_racing import CarRacing

env = CarRacing(3)
env.reset()
i = 0
done = False

while not done:
    a = env.action_space.sample()
    _, _, done, _ = env.step(a)
    i += 1
    if i%10 == 0:
        print(i)
