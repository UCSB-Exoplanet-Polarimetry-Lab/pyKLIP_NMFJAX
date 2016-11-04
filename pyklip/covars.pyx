from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def matern32(np.ndarray[DTYPE_t, ndim=1] x_coords, np.ndarray[DTYPE_t, ndim=1] y_coords,
                np.ndarray[DTYPE_t, ndim=1] sigmas, float corr_len):
    """
    Generates a Matern (\nu=3/2) covariance matrix that assumes x/y has the same correlation length

    C_ij = \sigma_i \sigma_j (1 + sqrt(3) r_ij / l) exp(-sqrt(3) r_ij / l)

    Args:
        x_coords: 1-D array of x coordiantes
        y_coords: 1-D array of y coordinates
        sigmas: 1-D array with the error in each pixel.
    """
    cdef int cov_size = x_coords.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] cov = np.zeros([cov_size, cov_size], dtype=DTYPE)
    cdef int i, j
    cdef float dist

    for j in range(cov_size):
        for i in range(j, cov_size):

            dist = (x_coords[i]-x_coords[j])**2 + (y_coords[i] - y_coords[j])**2
            dist = np.sqrt(3*dist)/corr_len
            cov_val = sigmas[i] * sigmas[j] * (1+ dist) * exp(-dist)
            cov[j,i] = cov_val
            cov[i,j] = cov_val

    return cov


def sq_exp(np.ndarray[DTYPE_t, ndim=1] x_coords, np.ndarray[DTYPE_t, ndim=1] y_coords,
                np.ndarray[DTYPE_t, ndim=1] sigmas, float corr_len):
    """
    Generates square exponential covariance matrix that assumes x/y has the same correlation length

    C_ij = \sigma_i \sigma_j exp(-r_ij^2/[2 l^2])

    Args:
        x_coords: 1-D array of x coordiantes
        y_coords: 1-D array of y coordinates
        sigmas: 1-D array with the error in each pixel.
    """
    cdef int cov_size = x_coords.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] cov = np.zeros([cov_size, cov_size], dtype=DTYPE)
    cdef int i, j
    cdef float dist

    for j in range(cov_size):
        for i in range(j, cov_size):

            dist_sq = (x_coords[i]-x_coords[j])**2 + (y_coords[i] - y_coords[j])**2
            cov_val = sigmas[i] * sigmas[j] * np.exp(-dist_sq/2/(corr_len*corr_len))
            cov[j,i] = cov_val
            cov[i,j] = cov_val

    return cov