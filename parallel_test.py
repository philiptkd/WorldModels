import multiprocessing as mp
from multiprocessing import Pool

def f(x):
    print(mp.current_process(), x*x)
    return x*x

if __name__ == '__main__':
    with Pool() as pool:
        print(pool.map(f, range(10), chunksize=1))

