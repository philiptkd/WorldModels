#   create N = os.cpu_count() - 1 workers
#   create a pipe for each worker
#   call map_async(), passing experience gathering function and iterable of N seeds
#   map_async() doesn't need to return anything upon completion
#   each worker should periodically place batch of experience on pipe

#   loop:    
#       parent process receives experience tuples from each worker that has a batch ready
#       if replay buffer is large enough:
#           parent process sends minibatch from replay buffer to GPU for VAE training
   

import multiprocessing as mp
from multiprocessing import Pool, Pipe
import os

# dummy function that continually runs until told to stop
def f(args):
    print("entered worker", mp.current_process())
    conn, seed = args
    i = 0
    j = 0
    while True:
        i += 1
        if i % 100000 == 0:
            i = 0
            j += 1
            print(mp.current_process(), "sending")
            conn.send(j) # send message to parent
        if conn.poll() and conn.recv() == "kill":
            break

if __name__ == '__main__':
    
    # setup workers
    num_workers = os.cpu_count() # not sure of the performance implications of using all cores on workers, leaving the main process to share with one of them
    pipes = [Pipe(duplex=True) for worker in range(num_workers)] # one two-way pipe per worker
    child_conns, parent_conns = zip(*pipes)
    seeds = range(1234, 1234+num_workers) # arbitrary positive integers
    process_args = zip(child_conns, seeds) # an iterable that yields one tuple of arguments per task/worker

    with Pool(processes = num_workers) as pool:
        pool.map_async(f, process_args, chunksize=1)

        while True:
            messages = []
            for conn in parent_conns:
                messages.append(conn.recv()) # blocks until there is something to receive or the connection is closed
            print(mp.current_process(), messages)
