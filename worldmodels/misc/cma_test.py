import cma
import numpy as np
from multiprocessing import Pool

# distributed parameters
num_workers = 48
chunksize = 1

# cma parameters
initial_guess = np.random.rand(2)
step_size = 1
popsize = num_workers*chunksize


# objective function with minimum at (2,1)
def func(x):
    return (x[0]-2)*(x[0]-2) + (x[1]-1)*(x[1]-1)


# distributed wrapper fur objective function
def func_dist(solution_candidates):
    assert len(solution_candidates) == popsize
    with Pool(processes = num_workers) as pool: # create pool of workers
        return pool.map(func, solution_candidates, chunksize=chunksize)


if __name__ == '__main__':
    es = cma.CMAEvolutionStrategy(initial_guess, step_size, {'popsize': popsize})

    while not es.stop():
        X = es.ask() # get potential solutions
        es.tell(X, func_dist(X)) # update distribution according to solution fitnesses
        es.disp() # print one-line summary

    es.result_pretty() # print results
