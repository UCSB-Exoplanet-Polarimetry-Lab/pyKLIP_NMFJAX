__author__ = 'JB'

import numpy as np
import scipy.linalg as la
import time

if __name__ == "__main__":

    Nr = 100
    Npix = 5000

    t0 = time.clock()
    rng_state = np.random.get_state()
    for k in range(10):
        R = np.random.random((Nr, Npix))

        U,s,V = np.linalg.svd(R, full_matrices=0)
    print(time.clock() - t0)


    t0 = time.clock()
    np.random.set_state(rng_state)
    for k in range(10):
        R = np.random.random((Nr, Npix))

        evals, evecs = la.eigh(np.dot(R,np.transpose(R)))
        np.dot(R,np.transpose(R))
        np.dot(R,np.transpose(R))
    print(time.clock() - t0)