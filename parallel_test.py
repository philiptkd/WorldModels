import multiprocessing as mp
from multiprocessing import Pool

def f(x):
    print(mp.current_process(), x*x)
    return x*x

def g(z):
    x,y = z
    print(mp.current_process(), x*x)
    return x*x

def h(z):
    x,conn = z
    conn.send([x*x])
    print(mp.current_process(), x*x)
    conn.close()
    print("closed connection",mp.current_process)
    return x*x

if __name__ == '__main__':
    with Pool() as pool:
        print(pool.map(g, zip(range(10), range(10)), chunksize=1))


