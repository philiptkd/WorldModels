# returns cumulative reward of rollout as fitness for the cma algorithm
def get_fitness(params):
    assert len(params) == num_params

    # get rnn
    rnn_trainer = RNN_Trainer()
    rnn_trainer.load_model(filepath=models_path+"rnn_model.pt")

    # initialize environment
    env = CarRacing(verbose=0)
    state = env.reset()
    ret = 0 # sum of rewards
    a = torch.Tensor([[0, 0, 0]]).float()
    done = False

    # shape params
    W_c = params[:-action_size] # the last 3 params are the bias terms
    W_c = W_c.reshape((c_input_size, action_size)) # (288, 3)
    W_c = torch.tensor(W_c).float()
    b_c = params[-action_size:]
    b_c = b_c.reshape((1, action_size)) # (1, 3)
    b_c = torch.tensor(b_c).float()

    step = 0
    while not done:
        step += 1
        if step%100 == 0:
            print(step)
        
        # vae
        state = torch.tensor(preprocess(state)) # (3, 64, 64)
        state = state.unsqueeze(dim=0).float() # (1, 3, 64, 64)
        state = state.to(device)
        z = vae.encode(state).to(cpu) # (1, 32)

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

    return -ret


# ensures the action is within its bounding box
def clip_action(a):
    a = a.sigmoid() # maps every element to (0,1)
    a[0] = (a[0] - 0.5)*2 # first element now in (-1, 1)
    return a


# distributed wrapper fur objective function
def func_dist(solution_candidates):
    assert len(solution_candidates) == popsize

    with Pool(processes = num_workers) as pool: # create pool of workers
        return pool.map(get_fitness, solution_candidates, chunksize=chunksize)


if __name__ == '__main__':
    from worldmodels.vae.train_vae import VAE_Trainer
    from worldmodels.dynamics_model.train_dynamics_model import RNN_Trainer
    from gym.envs.box2d.car_racing import CarRacing
    from worldmodels.vae.random_stepper import preprocess
    import cma
    import torch
    import numpy as np
    from multiprocessing import Pool
    import pickle
    import os

    # distributed parameters
    num_workers = 24
    chunksize = 1

    # controller parameters
    latent_size = 32
    hidden_size = 256
    c_input_size = latent_size+hidden_size
    action_size = 3
    num_params = c_input_size*action_size + action_size

    # cma parameters
    initial_guess = np.random.rand(num_params)
    step_size = 1
    popsize = num_workers*chunksize

    # torch devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")

    # get vae, to be shared by all processes
    vae_trainer = VAE_Trainer()
    models_path = "/home/philip_raeisghasem/worldmodels/worldmodels/working_models/"
    vae_trainer.load_model(filepath=models_path+"vae_model.pt")
    vae = vae_trainer.model

    # set up logging
    datalog = []
    data_dir = "/home/philip_raeisghasem/worldmodels/worldmodels/controller/data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir) 

    # run evolution strategy
    es = cma.CMAEvolutionStrategy(initial_guess, step_size, {'popsize': popsize})
    while not es.stop():
        X = es.ask() # get potential solutions
        fitnesses = func_dist(X) # get rollout returns
        
        # save progress
        data_log += [X, fitnesses]
        with open(data_dir+"progress.pt", "wb") as f:
            pickle.dump(data_log, f)

        es.tell(X, fitnesses) # update distribution according to solution fitnesses
        es.disp() # print one-line summary

    es.result_pretty() # print results
