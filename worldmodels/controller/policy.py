from worldmodels.dynamics_model.train_dynamics_model import RNN_Trainer
from gym.envs.box2d.car_racing import CarRacing
from worldmodels.vae.random_stepper import preprocess
import torch

models_path = "/home/philip_raeisghasem/worldmodels/worldmodels/working_models/"
max_steps = 4000

# controller parameters
latent_size = 32
hidden_size = 256
c_input_size = latent_size+hidden_size
action_size = 3
num_weights = c_input_size*action_size + action_size

# torch devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")


# returns cumulative reward of rollout as fitness for the cma algorithm
def get_fitness(args):
    with torch.no_grad():
        weights, vae = args
        assert len(weights) == num_weights
        
        # get rnn
        rnn_trainer = RNN_Trainer()
        rnn_trainer.load_model(filepath=models_path+"rnn_model.pt")
        rnn = rnn_trainer.model

        # initialize environment
        env = CarRacing(verbose=0)
        state = env.reset()
        ret = 0 # sum of rewards
        a = torch.Tensor([[0, 0, 0]]).float()
        done = False

        # shape weights
        W_c = weights[:-action_size] # the last 3 weights are the bias terms
        W_c = W_c.reshape((c_input_size, action_size)) # (288, 3)
        W_c = torch.tensor(W_c).float()
        b_c = weights[-action_size:]
        b_c = b_c.reshape((1, action_size)) # (1, 3)
        b_c = torch.tensor(b_c).float()

        step = 0
        while not done:
            step += 1
            #if step%100 == 0:
            #    print(step)
            if step >= max_steps:
                break
            
            # vae
            state = torch.tensor(preprocess(state)) # (3, 64, 64)
            state = state.unsqueeze(dim=0).float() # (1, 3, 64, 64)
            state = state.to(device)
            z_gpu = vae.encode(state) # (1, 32)
            z = z_gpu.to(cpu)
            del state, z_gpu # necessary? idk. running out of space on the gpu

            # rnn
            rnn_input = torch.cat((z, a), dim=1) # (1, 35)
            z_pred = rnn(rnn_input) # (1, 32)
            
            # controller
            c_input = torch.cat((z, rnn.h), dim=1) # (1, 288)
            a = torch.mm(c_input, W_c) + b_c # (1,3)
            
            # env step
            a = clip_action(a) # now numpy array
            state, reward, done, _ = env.step(a.squeeze().detach().numpy())
            ret += reward

        torch.cuda.empty_cache()
        return -ret


# ensures the action is within its bounding box
def clip_action(a):
    with torch.no_grad():
        a = a.sigmoid() # maps every element to (0,1)
        a[0] = (a[0] - 0.5)*2 # first element now in (-1, 1)
        return a
