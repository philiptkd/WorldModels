from worldmodels.vae.train_vae import VAE_Trainer
import cma, pickle, os, torch
import numpy as np
import multiprocessing as mp
from worldmodels.controller.policy import get_fitness, num_weights, models_path
#import worldmodels.cudaprofile as cudaprofile

# distributed parameters
num_workers = 16
chunksize = 1

# cma parameters
initial_guess = np.random.rand(num_weights)
step_size = 1
popsize = num_workers*chunksize


# distributed wrapper fur objective function
def func_dist(args):
    with mp.Pool(processes = num_workers) as pool: # create pool of workers
        return pool.map(get_fitness, args, chunksize=chunksize)


if __name__ == '__main__':
    with torch.no_grad():
        mp.set_start_method('spawn')
        
        # get vae, to be shared by all processes
        vae_trainer = VAE_Trainer()
        vae_trainer.load_model(filepath=models_path+"vae_model.pt")
        vae = vae_trainer.model

        # set up logging
        data_dir = "/home/teslaadmin/worldmodels/worldmodels/controller/data/"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir) 

        # run evolution strategy
        es = cma.CMAEvolutionStrategy(initial_guess, step_size, {'popsize': popsize, 'maxiter': 1}) # TODO: remove maxiter arg 
        while not es.stop():
            weights_list = es.ask() # get potential solutions
            args = zip(weights_list, [vae]*popsize) # [vae]*popsize should only be list of refs, so much smaller than sizeof(vae)*popsize
            fitnesses = func_dist(args) # get rollout returns
            
            # save progress
            with open(data_dir+"progress.pt", "wb") as f:
                pickle.dump((weights_list, fitnesses), f)

            es.tell(weights_list, fitnesses) # update distribution according to solution fitnesses
            es.disp() # print one-line summary

        es.result_pretty() # print results
        #cudaprofile.stop() # for use with nvprof
