__author__ = 'JB'

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from scipy.signal import convolve
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import leastsq
import astropy.io.fits as pyfits
from astropy.modeling import models, fitting
from copy import copy
import warnings
from scipy.stats import nanmedian
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import multiprocessing as mp
import itertools
import glob, os
from sys import stdout
import platform


def get_spatial_cova_func(image,IOWA,N,centroid = None,n_neigh=11, corr = False):
    IWA,OWA = IOWA
    ny,nx = image.shape

    image_mask = np.ones((ny,nx))
    image_mask[np.where(np.isnan(image))] = 0
    image_mask[0:n_neigh/2,:] = 0
    image_mask[:,0:n_neigh/2] = 0
    image_mask[(ny-n_neigh/2):ny,:] = 0
    image_mask[:,(nx-n_neigh/2):nx] = 0

    if centroid is None :
        x_cen = np.ceil((nx-1)/2) ; y_cen = np.ceil((ny-1)/2)
    else:
        x_cen, y_cen = centroid

    # Build the x and y coordinates grids
    x, y = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
    # Calculate the radial distance of each pixel
    r_grid = abs(x +y*1j)
    th_grid = np.arctan2(x,y)

    # Define the radii intervals for each annulus
    r0 = IWA
    annuli_radii = []
    while np.sqrt(N/np.pi+r0**2) < OWA:
        annuli_radii.append((r0,np.sqrt(N/np.pi+r0**2)))
        r0 = np.sqrt(N/np.pi+r0**2)

    annuli_radii.append((r0,np.max([ny,nx])))
    N_annuli = len(annuli_radii)
    #print(annuli_radii)

    xneigh0, yneigh0 = np.meshgrid(np.arange(n_neigh)-n_neigh/2, np.arange(n_neigh)-n_neigh/2)
    rneigh = abs(xneigh0 +yneigh0*1j)

    correlation_list_of_values = []
    correlation_stamps= np.zeros((n_neigh,n_neigh,N_annuli))
    if corr:
        variance_ring_values = np.zeros((N_annuli,))
    for k in range(n_neigh*n_neigh*N_annuli):
        correlation_list_of_values.append([])

    for it, rminmax in enumerate(annuli_radii):
        r_min,r_max = rminmax

        where_ring = np.where((r_min< r_grid) * (r_grid < r_max) * image_mask)

        #im = copy(image)
        #im[where_ring] = 1.0
        #plt.imshow(im,interpolation="nearest")
        #plt.show()

        if corr:
            variance_ring_values[it] = np.nanvar(image[where_ring])

        for k,l in zip(where_ring[0],where_ring[1]):
            yneigh = k+yneigh0
            xneigh = l+xneigh0
            for i in range(n_neigh):
                for j in range(n_neigh):
                    correlation_list_of_values[it*n_neigh**2+ i*n_neigh + j].append(image[k,l]*image[yneigh[i,j],xneigh[i,j]])


    for it in range(N_annuli):
        for i in range(n_neigh):
            for j in range(n_neigh):
                correlation_stamps[i,j,it] = np.nanmean(correlation_list_of_values[it*n_neigh**2+ i*n_neigh + j])

    if corr:
        for it in range(N_annuli):
            correlation_stamps[:,:,it] /= variance_ring_values[it]

    #plt.plot(np.reshape(rneigh,n_neigh*n_neigh),np.reshape(correlation_stamps[:,:,0],n_neigh*n_neigh),".")
    #plt.show()

    return np.reshape(rneigh,n_neigh*n_neigh),np.reshape(correlation_stamps,(n_neigh*n_neigh,N_annuli))

def get_spectral_cova_func(cube,IOWA,N,centroid = None,n_neigh=3, corr = False):
    IWA,OWA = IOWA

    nl,ny,nx = cube.shape

    flat_cube = np.mean(cube,axis=0)

    image_mask = np.ones((ny,nx))
    image_mask[np.where(np.isnan(flat_cube))] = 0
    image_mask[0:n_neigh/2,:] = 0
    image_mask[:,0:n_neigh/2] = 0
    image_mask[(ny-n_neigh/2):ny,:] = 0
    image_mask[:,(nx-n_neigh/2):nx] = 0

    if centroid is None :
        x_cen = np.ceil((nx-1)/2) ; y_cen = np.ceil((ny-1)/2)
    else:
        x_cen, y_cen = centroid

    # Build the x and y coordinates grids
    x, y = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
    # Calculate the radial distance of each pixel
    r_grid = abs(x +y*1j)
    th_grid = np.arctan2(x,y)

    # Define the radii intervals for each annulus
    r0 = IWA
    annuli_radii = []
    while np.sqrt(N/np.pi+r0**2) < OWA:
        annuli_radii.append((r0,np.sqrt(N/np.pi+r0**2)))
        r0 = np.sqrt(N/np.pi+r0**2)

    annuli_radii.append((r0,np.max([ny,nx])))
    N_annuli = len(annuli_radii)

    correlation_list_of_values = []
    correlation= np.zeros((n_neigh,N_annuli))
    for k in range(n_neigh*N_annuli):
        correlation_list_of_values.append([])



    for it, rminmax in enumerate(annuli_radii):
        r_min,r_max = rminmax

        where_ring = np.where((r_min< r_grid) * (r_grid < r_max) * image_mask)

        #cube[:,where_ring[0],where_ring[1]] = 1.0
        spectral_stddev = np.nanstd(cube[:,where_ring[0],where_ring[1]],axis=1)
        for l in range(nl):
            cube[l,where_ring[0],where_ring[1]] /= spectral_stddev[l]

            if l == 2:
                print(cube[:,where_ring[0],where_ring[1]])

        for l in range(nl):
            for i in range(l,n_neigh):
                #print(l,i,i-l)
                correlation_list_of_values[(i-l)+it*n_neigh].append(np.mean(cube[i,where_ring[0],where_ring[1]]*cube[l,where_ring[0],where_ring[1]]))
                #print(correlation_list_of_values)


    for it in range(N_annuli):
        for i in range(n_neigh):
            correlation[i,it] = np.nanmean(correlation_list_of_values[i+it*n_neigh])



    return correlation,annuli_radii
