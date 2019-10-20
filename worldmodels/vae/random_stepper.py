import gym
from gym.envs.box2d.car_racing import CarRacing
from PIL import Image
import numpy as np
import cProfile
import multiprocessing as mp
import pickle, pickletools

# transform image into size and type appropriate for our vae model
def preprocess(img_arr, new_size=(64, 64)):
    img = Image.fromarray(img_arr)
    img = img.resize(new_size)
    new_arr = np.asarray(img)/255.0 # (64, 64, 3)
    new_arr = np.transpose(new_arr, (2, 0, 1)) # (3, 64, 64)
    return new_arr 

# randomly interact with the environment until we receive a 'done' signal. this is one rollout
def gather_experience(rollout_idx):
    env = CarRacing()
    env.reset()
    batch = []
    done = False
    n_steps = 0
    while not done:
        n_steps += 1
        a = env.action_space.sample()
        obs, _, done, _ = env.step(a) # obs is 96 x 96 x 3 array of ints
        x = preprocess(obs) # (3, 64, 64)
        batch.append((a, x))

    # write to disk
    filename = "data/rollouts/rollout%d.pkl"%rollout_idx

    # print troubleshooting info
    print("rollout: "+str(rollout_idx)+", total steps: "+str(n_steps))

    batch_bytes = pickletools.optimize(pickle.dumps(batch))
    with open(filename, "wb") as f:
        f.write(batch_bytes)

    # None of this matters, probably
    env.close()
    del env, batch, batch_bytes, a, obs, x
    
