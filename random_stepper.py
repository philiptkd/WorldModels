import gym
from gym.envs.box2d.car_racing import CarRacing
import torch
from PIL import Image

num_steps = 100 # number of experience steps to send back at a time
num_batches = 10 # number of times to send back experience

def preprocess(img_arr, new_size=(64, 64)):
    img = Image.fromarray(img_arr)
    img = img.resize(new_size)
    new_arr = np.asarray(img)/255.0 # (64, 64, 3)
    new_arr = np.transpose(new_arr, (2, 0, 1)) # (3, 64, 64)
    new_arr = np.expand_dims(new_arr, 0) # (1, 3, 64, 64)
    return torch.tensor(new_arr).float()
    
def gather_experience(args):
    seed, conn = args
    env = CarRacing(seed)
    env.reset()

    for _ in range(num_batches):
        batch = None
        for _ in range(num_steps):
            a = env.action_space.sample()
            obs, _, done, _ = env.step(a) # obs is 96 x 96 x 3 array of ints
            x = preprocess(obs) # (1, 3, 64, 64)

            if batch is None:
                batch = x
            else:
                batch = torch.cat((batch, x))

            if done:
                env.reset()
        
        conn.send(batch) # send back tensor of shape (num_steps, 3, 64, 64)
