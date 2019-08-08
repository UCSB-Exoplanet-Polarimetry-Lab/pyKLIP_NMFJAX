#!/usr/bin/env python

"""
Weighted Principal Component Analysis using Expectation Maximization

Classic PCA is great but it doesn't know how to handle noisy or
missing data properly.  This module provides Weighted Expectation
Maximization PCA, an iterative method for solving PCA while properly
weighting data.  Missing data is simply the limit of weight=0.

Given data[nobs, nvar] and weights[nobs, nvar],

    lowrankbasis, lowrankfit = empca(data, weights, options...)

returns the low rank basis set (nvec vectors each of length nvar) and
the best resulting approximation to the data in a weighted
least-squares sense.
    
Original: Stephen Bailey, Spring 2012
Rewritten by Timothy Brandt, Spring 2016
"""

import numpy as np
import multiprocessing
import time

import matutils

def _random_orthonormal(nvec, nvar, seed=1):
    """
    Return array of random orthonormal vectors A[nvec, nvar] 

    Doesn't protect against rare duplicate vectors leading to 0s
    """

    if seed is not None:
        np.random.seed(seed)
        
    A = np.random.normal(size=(nvec, nvar))
    for i in range(nvec):
        A[i] /= np.linalg.norm(A[i])

    for i in range(1, nvec):
        for j in range(0, i):
            A[i] -= np.dot(A[j], A[i])*A[j]
            A[i] /= np.linalg.norm(A[i])

    return A



def empca(data, weights=None, niter=25, nvec=5, randseed=1, maxcpus=None, silent=False):
    """
    Perform iterative lower rank matrix approximation of data[obs, var]
    using weights[obs, var].
    
    Generated model vectors are not orthonormal and are not
    rotated/ranked by ability to model the data, but as a set
    they are good at describing the data.
    
    Optional:
      - niter : maximum number of iterations to perform
      - nvec  : number of vectors to solve
      - randseed : rand num generator seed; if None, don't re-initialize
    
    Returns [basis, model], where basis is an [nvec, var] sized array
    and represents the basis set for the low-rank approximation, while
    model is the low-rank approximation to data (and is of the same
    shape).
    """
    
    if weights is None:
        weights = np.ones(data.shape, np.float32)

    ##################################################################
    # The following code makes sure that there are two copies each 
    # of data and weights, one in C format (last axis fast) and one 
    # in Fortran format (first axis fast).  This costs a factor of
    # two in memory usage but speeds up access later.
    ##################################################################

    if not (isinstance(data, np.ndarray) and isinstance(weights, np.ndarray)):
        raise TypeError("'data' and 'weights' must be numpy ndarrays.")
    if not (data.shape == weights.shape and len(data.shape) == 2):
        raise ValueError("'data' and 'weights' must be 2D arrays of the same shape.")
   
    if data.flags['C']:
        dataC = data.astype(np.float32)
        dataF = (dataC.T).copy(order='C')
    elif data.flags['F']:
        dataC = data.copy(order='C').astype(np.float32)
        dataF = data.T.astype(np.float32)
    else:
        raise AttributeError("Attribute 'flags' missing from data.")
    
    if weights.flags['C']:
        weightsC = weights.astype(np.float32)
        weightsF = (weightsC.T).copy(order='C')
    elif weights.flags['F']:
        weightsC = weights.copy(order='C').astype(np.float32)
        weightsF = weights.T.astype(np.float32)
    else:
        raise AttributeError("Attribute 'flags' missing from weights.")

    ##################################################################
    # Random initial guess for the low-rank approximation, zero
    # for the initial fit/approximation coefficients.
    ##################################################################

    nobs, nvar = data.shape
    P = _random_orthonormal(nvec, nvar, seed=randseed)
    C = np.zeros((nobs, nvec))

    if not silent:
        print ("iter     dchi2      R2          time (s)")

    ncpus = multiprocessing.cpu_count()
    if maxcpus is not None:
        ncpus = min(ncpus, maxcpus)

    chisq_orig = matutils.calc_chisq(dataC, P*0, weightsC, C, maxproc=ncpus)
    chisq_last = chisq_orig
    datwgt = dataC*weightsC

    for itr in range(1, niter + 1):

        tstart = time.time()

        ##############################################################
        # Solve for best-fit coefficients with the previous/first
        # low-rank approximation.
        ##############################################################

        P3D = np.empty((P.shape[0], P.shape[0], P.shape[1]))
        for i in range(P.shape[0]):
            P3D[i] = P*P[i]
        A = np.tensordot(weights, P3D.T, axes=1)
        b = np.dot(datwgt, P.T)

        C = matutils.lstsq(A, b, maxproc=ncpus).T

        ##############################################################
        # Compute the weighted residual (chi squared) value from the
        # previous fit.
        ##############################################################

        chisq = matutils.calc_chisq(dataC, P, weightsC, C.T, maxproc=ncpus)

        if itr == niter:

            ##########################################################
            # Compute the low-rank approximation to the data.
            ##########################################################

            model = matutils.dot(C.T, P, maxproc=ncpus)

        else:

            ##########################################################
            # Update the low-rank approximation.
            ##########################################################

            C3D = np.empty((C.shape[0], C.shape[0], C.shape[1]))
            for i in range(C.shape[0]):
                C3D[i] = C*C[i]
            A = np.tensordot(weights.T, C3D.T, axes=1)
            b = np.dot(datwgt.T, C.T)
            P = matutils.lstsq(A, b, maxproc=ncpus).T
        
        if not silent:
            print ('%3d  %9.3g  %12.6f %11.3f' % (itr, chisq - chisq_last, 1 - chisq/chisq_orig, time.time() - tstart))
        
        chisq_last = chisq

    ##################################################################
    # Normalize the low-rank approximation.
    ##################################################################

    for k in range(nvec):
        P[k] /= np.linalg.norm(P[k])

    return P, model

