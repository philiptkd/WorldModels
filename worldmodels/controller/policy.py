# to be the objective function to minimize within cma call
# takes linear layer parameters as input
# returns negative of cumulative reward received throughout episode
# returns when environment returns 'done'

# load both vae and rnn models
# we're not calling backward() so no need to detach() model results
# transform [z,h] with linear parameters into action
# use tanh activation on actions to bound them to their appropriate range [(-1,1), (0,1), (0,1)]

from worldmodels.vae.train_vae import VAE_Trainer
from gym.envs.box2d.car_racing import CarRacing
from worldmodels.dynamics_model.train_dynamics_model import RNN_Trainer
from worldmodels.vae.random_stepper import preprocess
import torch
import cProfile, pstats

latent_size = 32
hidden_size = 256
c_input_size = latent_size+hidden_size
action_size = 3
num_params = c_input_size*action_size + action_size

# exists for the purposes of profiling
def get_fitness(args):
    params, task_idx = args
    ret = []
    cProfile.runctx("_get_fitness(params, ret)", globals(), locals(), "profiles/profile%d.prof"%task_idx)
    return -ret[0]    

def _get_fitness(params, ret):
    assert len(params) == num_params

    models_path = "/home/phil/worldmodels/worldmodels/working_models/"

    # get vae
    vae_trainer = VAE_Trainer()
    vae_trainer.load_model(filepath=models_path+"vae_model.pt")
    vae = vae_trainer.model

    # get rnn
    rnn_trainer = RNN_Trainer()
    rnn_trainer.load_model(filepath=models_path+"rnn_model.pt")
    rnn = rnn_trainer.model

    # initialize environment
    env = CarRacing(verbose=0)
    state = env.reset()
    ret.append(0)
    a = torch.Tensor([[0, 0, 0]]).float()
    done = False

    # shape params
    W_c = params[:-action_size] # the last 3 params are the bias terms
    W_c = W_c.reshape((c_input_size, action_size)) # (288, 3)
    W_c = torch.tensor(W_c).float()
    b_c = params[-action_size:]
    b_c = b_c.reshape((1, action_size)) # (1, 3)
    b_c = torch.tensor(b_c).float()

    while not done:
        state = torch.tensor(preprocess(state)) # (3, 64, 64)
        state = state.unsqueeze(dim=0).float() # (1, 3, 64, 64)
        z = vae.encode(state) # (1, 32)
        rnn_input = torch.cat((z, a), dim=1) # (1, 35)
        z_pred = rnn(rnn_input) # (1, 32)
        c_input = torch.cat((z, rnn.h), dim=1) # (1, 288)
        a = torch.mm(c_input, W_c) + b_c # (1,3)
        a = clip_action(a) # now numpy array
        state, reward, done, _ = env.step(a.squeeze().detach().numpy())
        ret[0] += reward


# ensures the action is within its bounding box
def clip_action(a):
    a = a.sigmoid() # maps every element to (0,1)
    a[0] = (a[0] - 0.5)*2 # first element now in (-1, 1)
    return a
