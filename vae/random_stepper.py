import gym
from gym.envs.box2d.car_racing import CarRacing
from PIL import Image
import numpy as np
import cProfile
import multiprocessing as mp

def preprocess(img_arr, new_size=(64, 64)):
    img = Image.fromarray(img_arr)
    img = img.resize(new_size)
    new_arr = np.asarray(img)/255.0 # (64, 64, 3)
    new_arr = np.transpose(new_arr, (2, 0, 1)) # (3, 64, 64)
    return new_arr 

def _gather_experience(args, ret):
    seed, num_rollouts = args
    env = CarRacing(verbose=1, seed=seed)

    for i in range(num_rollouts):
        print(i)

        env.reset()
        batch = []
        done = False
        while not done:
            a = env.action_space.sample()
            obs, _, done, _ = env.step(a) # obs is 96 x 96 x 3 array of ints
            x = preprocess(obs) # (3, 64, 64)
            batch.append((a, x))

    ret.append(batch)

def gather_experience(args):
    import pstats

    prof_path = 'profiles/profile'+str(mp.current_process().pid)+'.prof'
    
    ret = []
    cProfile.runctx('_gather_experience(args, ret)', globals(), locals(), prof_path) # run the command and collect data
    p = pstats.Stats(prof_path)

    print('\n')
    p.strip_dirs().sort_stats('cumulative').print_stats(10)
    print('\n')

    return ret[0]
