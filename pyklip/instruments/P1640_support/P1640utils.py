#!/usr/bin/env

import numpy as np

"""
Various useful functions specific to the P1640 data
"""

def set_zeros_to_nan(data):
    """
    PyKLIP expects values outside the detector to be set to nan.
    P1640 sets these (and also saturated pixels) to identically 0.
    Find all the zeros and convert them to nans
    Input:
        data: N x Npix x Npix datacube or appended set of datacubes
    Returns:
        nandata: data with nans instead of zeros
    """

    zeros = np.where(data==0)
    data[zeros]=np.nan
    return data

def median_filter(img, box_side=2, thresh=4):
    """
    For 2-D image, get the median of a box of pixels, and look for 
    any pixels that are above some threshold number of standard deviations.
    Input:
        img: 2-D image
        box_side: half the length of box side to use (default: 2)
        thresh: number of standard deviations to use for threshold (default: 4)
    """
    coords = np.indices(img.shape)
    img_median = None
    