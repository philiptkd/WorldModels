import cma
import numpy as np
from multiprocessing import Pool
from worldmodels.controller.policy import get_fitness, num_params
import pickle
import os

# distributed parameters
num_workers = 24
chunksize = 1

# cma parameters
initial_guess = np.random.rand(num_params)
step_size = 1
popsize = num_workers*chunksize


# distributed wrapper fur objective function
def func_dist(solution_candidates):
    assert len(solution_candidates) == popsize

    with Pool(processes = num_workers) as pool: # create pool of workers
        return pool.map(get_fitness, solution_candidates, chunksize=chunksize)


if __name__ == '__main__':
    es = cma.CMAEvolutionStrategy(initial_guess, step_size, {'popsize': popsize})
    datalog = []
    
    data_dir = "/home/philip_raeisghasem/worldmodels/worldmodels/controller/data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir) 

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
