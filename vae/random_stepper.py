import gym
from gym.envs.box2d.car_racing import CarRacing
import torch
from PIL import Image
import multiprocessing as mp
import numpy as np

def preprocess(img_arr, new_size=(64, 64)):
    img = Image.fromarray(img_arr)
    img = img.resize(new_size)
    new_arr = np.asarray(img)/255.0 # (64, 64, 3)
    new_arr = np.transpose(new_arr, (2, 0, 1)) # (3, 64, 64)
    new_arr = np.expand_dims(new_arr, 0) # (1, 3, 64, 64)
    return torch.tensor(new_arr).float()
    
def gather_experience(args):
    seed, conn, num_rollouts, steps_per_rollout = args
    steps_reset_after = 1000 # after about how many steps to reset
    rollouts_reset_after = steps_reset_after//(steps_per_rollout) + 1 # after how many rollouts to reset.
    env = CarRacing(verbose=1, seed=seed)
    env.reset()

    for i in range(num_rollouts):
        batch = None
        for _ in range(steps_per_rollout):
            a = env.action_space.sample()
            obs, _, done, _ = env.step(a) # obs is 96 x 96 x 3 array of ints
            x = preprocess(obs) # (1, 3, 64, 64)

            if batch is None:
                batch = x
            else:
                batch = torch.cat((batch, x))

            if done:
                env.reset()
        if (i+1)%rollouts_reset_after == 0:
            env.reset()
        
        conn.send(batch) # send back tensor of shape (num_step, 3, 64, 64)
