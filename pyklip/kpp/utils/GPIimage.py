__author__ = 'JB'

import numpy as np
from copy import copy

def get_occ(image, centroid = None):
    '''
    Get the IWA (inner working angle) of the central disk of nans and return the mask corresponding to the inner disk.

    :param image: A GPI image with a disk full of nans at the center.
    :param centroid: center of the nan disk
    :return:
    '''
    ny,nx = image.shape

    if centroid is None :
        x_cen = np.ceil((nx-1)/2) ; y_cen = np.ceil((ny-1)/2)
    else:
        x_cen, y_cen = centroid

    
    # IWA = 0
    # while np.isnan(image[x_cen,y_cen+IWA]):
    #     IWA += 1
        
    # Build the x and y coordinates grids
    x, y = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
    # Calculate the radial distance of each pixel
    r = abs(x +y*1j)

    mask = np.ones((ny,nx))
    if np.sum(np.isnan(image)) == 0:
        IWA = -1
        N_nans, N_nans_new = -1, 0
        while N_nans_new != N_nans:
            IWA = IWA+1
            N_nans = N_nans_new
            N_nans_new = np.size(np.where((image==0)*(r<IWA+1))[0])

        mask[np.where(image==0)] = np.nan
    else:
        IWA = -1
        N_nans, N_nans_new = -1, 0
        while N_nans_new != N_nans:
            IWA = IWA+1
            N_nans = N_nans_new
            N_nans_new = np.size(np.where(np.isnan(image)*(r<IWA+1))[0])

        mask[np.where(np.isnan(image))] = np.nan



    inner_mask = copy(mask)
    inner_mask[np.where(r > IWA+3.)] = 1

    outer_mask = copy(mask)
    outer_mask[np.where(np.isnan(inner_mask))] = 1
    OWA = np.min(r[np.where(np.isnan(outer_mask))])

    return IWA,OWA,inner_mask,outer_mask