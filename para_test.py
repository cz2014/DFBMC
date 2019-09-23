from multiprocessing import Pool
import numpy as np

shared_mem = np.arange(10)

def target_fun(x, a=shared_mem):
    su = 0
    for i in range(100000):
        su += i 
        
    return "input x: "+str(x)+"; len(a) = "+str(len(a))+"; id(a) = "+str(id(a))

if __name__ == '__main__':
    # shared_mem = np.arange(10)

    inputs = list(range(10))
    pool = Pool(processes=5)
    res = pool.map(target_fun, inputs)
    pool.close()
    pool.join()

    for i in res:
        print(i)
 
