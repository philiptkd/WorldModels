# this was a prototype for distributed training back when I was still communicating with Pipes.
# I've since stopped doing that in favor of writing to disk, but I'll leave this here for now just in case.

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
