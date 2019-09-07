#!/usr/bin/env python

#this is the version MC implemented in pyklip previously, to be compared to the original empca and merged

import numpy as np
import numpy.fft as fft
import scipy.linalg as la
import scipy.ndimage as ndimage
import scipy.interpolate as sinterp
import scipy.signal as signal
from scipy.stats import t
import sys
import multiprocessing
import time
import matutils

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

def np_calc_chisq(data, b, w, coef):
    """
    Calculate chi squared

    Args:
        im: nim x npix, single-precision numpy.ndarray. Data to be fit by the basis images
        b: nvec x npts, double precision numpy.ndarray. The nvec basis images.
        w: nim x npts, single-precision numpy.ndarray. Weights (inverse variances) of the data.
        coef: nvec x npts, double precision numpy.ndarray. The coefficients of the basis image fits.

    Returns:
        chisq, the total chi squared summed over all points and all images
    """

    chisq = 0
    nim = data.shape[0]
    for i in range(nim):
        chisq += np.sum((data[i] - np.sum(coef[i] * b.T, axis=1)) ** 2 * w[i])

    return chisq

def set_pixel_weights(imflat, rflat, mode='standard', inner_sup=17, outer_sup=66, normalize_weights=False):
    '''
    MC edited function

    :param imflat: array of flattend images, shape (N, number of section indices)
    :param rflat: radial component of the polar coordinates flattened to 1D, length = number of section indices
    :param inner_sup: radius within which to supress weights
    :param outer_sup: radius beyond which to supress weights
    :param mode:
        'standard': assume poission statistics to calculate variance as sqrt(photon count)
                    use inverse sqrt(variance) as pixel weights and multiply by a radial weighting
    :return: pixel weights for empca
    '''

    #default weights are ones
    weights = np.ones(imflat.shape)

    if mode.lower() == 'standard':
        weights = 1. / (np.sqrt(np.abs(imflat)) + 10)
        weights *= imflat != 0
        weights *= 1 / (1 + np.exp((inner_sup - rflat) / 1.))
        weights *= 1 / (1 + np.exp((rflat - outer_sup) / 1.))

    if normalize_weights:
        #TODO: implement correct axis for np.nanmean
        weights /= np.nanmean(weights)

    return weights

def _random_orthonormal(nvec, nvar, seed=1):
    '''
    Return array of random orthonormal vectors A[nvec, nvar] 
    Doesn't protect against rare duplicate vectors leading to 0s
    '''

    if seed is not None:
        np.random.seed(seed)
        
    A = np.random.normal(size=(nvec, nvar))
    for i in range(nvec):
        A[i] /= np.linalg.norm(A[i])

    for i in range(1, nvec):
        for j in range(0, i):
            A[i] -= np.dot(A[j], A[i])*A[j]
            A[i] /= np.linalg.norm(A[i])

    if np.any(np.isnan(A)):
        raise ValueError("random orthonormal is nan")
    return A

def nan_smooth(im, ivar, sig=1, spline_filter=False):
    '''
    Private function _smooth smooths an image accounting for the
    inverse variance.
    Parameters
    ----------
    im : ndarray
        2D image to smooth
    ivar : ndarray
        2D inverse variance, shape should match im
    sig : float (optional)
        standard deviation of Gaussian smoothing kernel
        Default 1
    spline_filter: boolean (optional)
        Spline filter the result?  Default False.
    Returns
    -------
    imsmooth : ndarray
        smoothed image of the same size as im
    '''

    if not isinstance(im, np.ndarray) or not isinstance(ivar, np.ndarray):
        raise TypeError("image and ivar passed to _smooth must be ndarrays")
    if im.shape != ivar.shape or len(im.shape) != 2:
        raise ValueError("image and ivar must be 2D ndarrays of the same shape")

    masked = np.copy(im)
    nan_locs = np.where(np.isnan(im))
    masked[nan_locs] = 0

    nx = int(sig * 4 + 1) * 2 + 1
    x = np.arange(nx) - nx / 2
    x, y = np.meshgrid(x, x)

    window = np.exp(-(x ** 2 + y ** 2) / (2 * sig ** 2))
    # think about whether this is properly normalized (accounting for nan values in the image that should not contribute)
    imsmooth = signal.convolve2d(masked*ivar, window, mode='same')
    imsmooth /= signal.convolve2d(ivar, window, mode='same') + 1e-35
    imsmooth *= (masked != 0)
    imsmooth[nan_locs] = np.nan

    if spline_filter:
        imsmooth = ndimage.spline_filter(imsmooth)

    return imsmooth

def weighted_empca(data, weights=None, niter=25, nvec=5, randseed=1, maxcpus=1, silent=True):
    '''
    replaes the linear algebra in clip_math with weighted low-rank approximation as an alternative model psf construction to test the performance on S/N. Code is adapted from Tim's ADI algorithm
    Perform iterative lower rank matrix approximation of data[obs, var]
    using weights[obs, var].
    
    Generated model vectors are not orthonormal and are not
    rotated/ranked by ability to model the data, but as a set
    they are good at describing the data.
    
    Optional:
      - niter : maximum number of iterations to perform
      - nvec  : number of vectors to solve
      - randseed : rand num generator seed; if None, don't re-initialize
    Returns:
    psfs: the weighted low-rank approximation model for the psf, dot(C.T, P), where P is the basis matrix of shape k*i, this model psf will replace klip_psf in klip_math
    '''

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
        print('iter     dchi2      R2          time (s)')

    ncpus = multiprocessing.cpu_count()
    if maxcpus is not None:
        ncpus = min(ncpus, maxcpus)

    #chisq_orig = matutils.calc_chisq(dataC, P*0, weightsC, C, maxproc=ncpus)
    chisq_orig = np_calc_chisq(dataC, P*0, weightsC, C)
    chisq_last = chisq_orig
    datwgt = dataC*weightsC

    # for np.linalg.lstsq instead of matutils, C shape needs to be inverted now
    C = C.T

    debug_time = time.time()
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
        C = matutils.lstsq(A.copy(), b.copy(), maxproc=ncpus).T

        C3 = np.zeros(C.shape)
        Ainv = np.linalg.pinv(A)
        import pdb; pdb.set_trace()
        C3 = np.diagonal(np.tensordot(Ainv, b.T, axes = 1), axis1=0, axis2=2)
        # for k in range(Ainv.shape[0]):
        #     C3.T[k] = np.dot(Ainv[k], b[k])

        # for k in range(A.shape[0]):
        #     C2.T[k] = np.linalg.lstsq(A[k], b[k], rcond=None)[0]
        #     Ainv = np.linalg.pinv(A[k])
        #     C3.T[k] = np.dot(Ainv, b[k])

        #b = np.dot(P, ( weightsC[i_obs]*dataC[i_obs] )) # shape nvec
        #A = np.dot(P, (P*weightsC[i_obs]).T) # shape (nvec, nvec)
        #C = np.linalg.solve(A, b).T # shape nvec

        ##############################################################
        # Compute the weighted residual (chi squared) value from the
        # previous fit.
        ##############################################################

        if not silent:
            #chisq = matutils.calc_chisq(dataC, P, weightsC, C.T, maxproc=ncpus)
            chisq = np_calc_chisq(dataC, P, weightsC, C.T)
            print('%3d  %9.3g  %12.6f %11.3f' % (itr, chisq - chisq_last, 1 - chisq / chisq_orig, time.time() - tstart))
            chisq_last = chisq

        if itr == niter:

            ##########################################################
            # Compute the low-rank approximation to the data.
            ##########################################################

            # model = matutils.dot(C.T, P, maxproc=ncpus)
            model = np.dot(C.T, P)

        else:

            ##########################################################
            # Update the low-rank approximation.
            ##########################################################
            C3D = np.empty((C.shape[0], C.shape[0], C.shape[1]))
            for i in range(C.shape[0]):
                C3D[i] = C*C[i]
            A = np.tensordot(weights.T, C3D.T, axes=1)
            b = np.dot(datwgt.T, C.T)
            P = matutils.lstsq(A.copy(), b.copy(), maxproc=ncpus)

            P3 = np.zeros(P.shape)
            Ainv = np.linalg.pinv(A)
            P3 = np.diagonal(np.tensordot(Ainv, b.T, axes=1), axis1=0, axis2=2).T
            # for k in range(Ainv.shape[0]):
            #     P3.T[k] = np.dot(Ainv[k], b[k])

            # for k in range(A.shape[0]):
            #     P2.T[k] = np.linalg.lstsq(A[k], b[k], rcond=None)[0]
            #     Ainv = np.linalg.pinv(A[k])
            #     P3.T[k] = np.dot(Ainv, b[k])

            #singular_matrix = 0
            #b = np.dot(C, weightsC.T[i_var] * dataC.T[i_var]) # shape nvec
            #A = np.dot(C, (C * weightsC.T[i_var]).T) # shape (nvec,nvec)
            #try:
            #    P2 = np.linalg.solve(A, b).T # shape nvec
            #except:
            #    cond_num = np.linalg.cond(A)
            
            # np.linalg.svd should fix the problem without having to resort to the following comments:            
            # grab the good indices for which cond_num < some large number
            # linalg.solve the good indices
            # loop solve the bad indices and check if they are just 0s
    debug_time = time.time() - debug_time
    import pdb; pdb.set_trace()

    ##################################################################
    # Normalize the low-rank approximation.
    ##################################################################

    #TODO: think more carefully about whether normalization is needed
    for k in range(nvec):
        P[k] /= np.linalg.norm(P[k])
    
    return model
