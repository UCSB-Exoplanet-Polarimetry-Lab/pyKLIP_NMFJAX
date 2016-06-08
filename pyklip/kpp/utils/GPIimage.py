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

    
    IWA,OWA = get_IOWA(image, centroid = centroid)
        
    # Build the x and y coordinates grids
    x, y = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
    # Calculate the radial distance of each pixel
    r = abs(x +y*1j)

    mask = np.ones((ny,nx))
    if np.sum(np.isnan(image)) == 0:
        mask[np.where(image==0)] = np.nan
    else:
        mask[np.where(np.isnan(image))] = np.nan


    inner_mask = copy(mask)
    inner_mask[np.where(r > IWA+1.)] = 1

    outer_mask = copy(mask)
    outer_mask[np.where(r > OWA-1.)] = 1

    return IWA,OWA,inner_mask,outer_mask


def get_IOWA(image, centroid = None):
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



    if np.sum(np.isnan(image)) == 0:
        image_tmp = copy(image)
        image_tmp[np.where(image==0)]
    else:
        image_tmp = image

    # Build the x and y coordinates grids
    x, y = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
    # Calculate the radial distance of each pixel
    r = abs(x +y*1j)

    dr = 0.5
    Dr = 0.5
    r_samp = np.arange(0,np.max(r)+dr,dr)
    radial_val = np.zeros(np.size(r_samp))

    for r_id, r_it in enumerate(r_samp):
        selec_pix = np.where( ((r_it-Dr/2.0) < r) * (r < (r_it+Dr/2.0)) )
        selec_y, selec_x = selec_pix
        radial_val[r_id] = np.sum(image_tmp[selec_y, selec_x])

    IWA = r_samp[np.where(np.isfinite(radial_val))[0][0]]
    OWA = r_samp[np.where(np.isfinite(radial_val))[0][-1]]

    return IWA,OWA