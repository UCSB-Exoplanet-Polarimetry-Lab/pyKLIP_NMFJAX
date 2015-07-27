import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from scipy.signal import convolve
from scipy.optimize import curve_fit
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
import xml.etree.cElementTree as ET

import spectra_management as spec
from kpp_utils import *
from kpp_pdf import *
from kpp_std import *
import instruments.GPI as GPI


def calculate_shape_metric_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call calculate_shape_metric() with a tuple of parameters.
    """
    return calculate_shape_metric(*params)

def calculate_shape_metric(row_indices,col_indices,cube,PSF_cube_cpy,stamp_PSF_mask):

    nl,ny_PSF,nx_PSF = PSF_cube_cpy.shape

    row_m = np.floor(ny_PSF/2.0)
    row_p = np.ceil(ny_PSF/2.0)
    col_m = np.floor(nx_PSF/2.0)
    col_p = np.ceil(nx_PSF/2.0)

    N_it = row_indices.size
    shape_map = np.zeros((N_it)) + np.nan
    #stdout.write("\r%d" % 0)
    for id,k,l in zip(range(N_it),row_indices,col_indices):
        #stdout.flush()
        #stdout.write("\r%d" % k)

        stamp_cube = copy(cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)])
        for slice_id in range(nl):
            stamp_cube[slice_id,:,:] -= np.nanmean(stamp_cube[slice_id,:,:]*stamp_PSF_mask)
        ampl = np.nansum(PSF_cube_cpy*stamp_cube)
        try:
            shape_map[id] = np.sign(ampl)*ampl**2/np.nansum(stamp_cube**2)
        except:
            shape_map[id] =  np.nan

    # criterion is here a cosine squared so we take the square root to get something similar to a cosine.
    return np.sign(shape_map)*np.sqrt(abs(shape_map))


def calculate_MF_metric_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call calculate_MF_metric() with a tuple of parameters.
    """
    return calculate_MF_metric(*params)

def calculate_MF_metric(row_indices,col_indices,cube,PSF_cube_cpy,stamp_PSF_mask):

    nl,ny_PSF,nx_PSF = PSF_cube_cpy.shape

    row_m = np.floor(ny_PSF/2.0)
    row_p = np.ceil(ny_PSF/2.0)
    col_m = np.floor(nx_PSF/2.0)
    col_p = np.ceil(nx_PSF/2.0)

    N_it = row_indices.size
    matchedFilter_map = np.zeros((N_it)) + np.nan
    #stdout.write("\r%d" % 0)
    for id,k,l in zip(range(N_it),row_indices,col_indices):
        #stdout.flush()
        #stdout.write("\r%d" % k)

        stamp_cube = copy(cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)])
        for slice_id in range(nl):
            stamp_cube[slice_id,:,:] -= np.nanmean(stamp_cube[slice_id,:,:]*stamp_PSF_mask)
        ampl = np.nansum(PSF_cube_cpy*stamp_cube)
        matchedFilter_map[id] = np.sign(ampl)*ampl**2

    # criterion is here a cosine squared so we take the square root to get something similar to a cosine.
    return matchedFilter_map

def calculate_shapeAndMF_metric_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call calculate_MF_metric() with a tuple of parameters.
    """
    return calculate_shapeAndMF_metric(*params)

def calculate_shapeAndMF_metric(row_indices,col_indices,cube,PSF_cube_cpy,stamp_PSF_mask):

    nl,ny_PSF,nx_PSF = PSF_cube_cpy.shape

    row_m = np.floor(ny_PSF/2.0)
    row_p = np.ceil(ny_PSF/2.0)
    col_m = np.floor(nx_PSF/2.0)
    col_p = np.ceil(nx_PSF/2.0)

    N_it = row_indices.size
    matchedFilter_map = np.zeros((N_it)) + np.nan
    shape_map = np.zeros((N_it)) + np.nan

    #stdout.write("\r%d" % 0)
    for id,k,l in zip(range(N_it),row_indices,col_indices):
        stdout.flush()
        stdout.write("\r%d" % k)

        stamp_cube = copy(cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)])
        for slice_id in range(nl):
            stamp_cube[slice_id,:,:] -= np.nanmean(stamp_cube[slice_id,:,:]*stamp_PSF_mask)
        ampl = np.nansum(PSF_cube_cpy*stamp_cube)
        matchedFilter_map[id] = np.sign(ampl)*ampl**2
        try:
            shape_map[id] = np.sign(ampl)*ampl**2/np.nansum(stamp_cube**2)
        except:
            shape_map[id] =  np.nan

    shape_map = np.sign(shape_map)*np.sqrt(abs(shape_map))

    # criterion is here a cosine squared so we take the square root to get something similar to a cosine.
    return shape_map,matchedFilter_map

def calculate_metrics(filename,
                        metrics = None,
                        PSF_cube = None,
                        outputDir = None,
                        folderName = None,
                        spectrum = None,
                        mute = False,
                        filename_no_planets = None,
                        SNR = True,
                        probability = True,
                        GOI_list = None,
                        overwrite_metric = False,
                        overwrite_stat = False,
                        proba_using_mask_per_pixel = False,
                        N_threads = None):
    '''
    Calculate the metrics for future planet detection. The SNR map is a metric for example but JB thinks it's not the best one.

    Inputs:
        filename: Path and name of the fits file to be analyzed.
        metrics: flatCube is calculated by default
            - "weightedFlatCube"
            - "matchedFilter"
            - "shape"
        PSF_cube: User-defined cube PSF. PSF_cube should not have any spectral


        outputDir: Directory where to save the outputs

    Outputs:

    '''
    hdulist = pyfits.open(filename)

    #grab the data and headers
    try:
        cube = hdulist[1].data
        exthdr = hdulist[1].header
        prihdr = hdulist[0].header
    except:
        print("Couldn't read the fits file normally. Try another way.")
        cube = hdulist[0].data
        prihdr = hdulist[0].header


    # Normalization to have reasonable values of the pixel.
    # Indeed some image are in contrast units and the code below doesn't like slices with values around 10**-7.
    cube /= np.nanstd(cube[10,:,:])
    #cube *= 10.0**7


    # Get input cube dimensions
    # Transform a 2D image into a cube with one slice because the code below works for cubes
    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
        #print(cube.shape)
        if nl != 37:
            if not mute:
                print("Spectral dimension of "+filename+" is not correct... quitting")
            return
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        cube_cpy = cube[None,:]
        nl = 1

    if nl != 1:
        # If data cube we can use the spectrum of the planet to more accurately detect planets.
        flat_cube = np.mean(cube,0)

        # Build the PSF.
        PSF_cube_cpy = copy(PSF_cube)
        if PSF_cube_cpy is not None:
            if np.size(np.shape(PSF_cube_cpy)) != 3:
                if not mute:
                    print("Wrong PSF dimensions. Image is 3D.")
                return 0
            # The PSF is user-defined.
            nl, ny_PSF, nx_PSF = PSF_cube_cpy.shape
            tmp_spectrum = np.nanmax(PSF_cube_cpy,axis=(1,2))
        else:
            if not mute:
                print("No specified PSF cube so one is built with simple gaussian PSF. (no spectral widening)")
            # Gaussian PSF with 1.5pixel sigma as nothing was specified by the user.
            # Build the grid for PSF stamp.
            ny_PSF = 8 # should be even
            nx_PSF = 8
            x_PSF_grid, y_PSF_grid = np.meshgrid(np.arange(0,ny_PSF,1)-ny_PSF/2,np.arange(0,nx_PSF,1)-nx_PSF/2)
            # Use a simple 2d gaussian PSF for now. The width is probably not even the right one.
            # I just set it so that "it looks" right.
            PSF = gauss2d(x_PSF_grid, y_PSF_grid,1.0,0.0,0.0,1.5,1.5)
            # Normalize the PSF with a norm 2
            PSF /= np.sqrt(np.sum(PSF**2))

            # Duplicate the PSF to get a PSF cube.
            # Besides apply the spectrum of the planet on that cube before renormalization.
            PSF_cube_cpy = np.tile(PSF,(nl,1,1))

        spectrum_cpy = spectrum
        if spectrum_cpy is not None:
            for k in range(nl):
                PSF_cube_cpy[k,:,:] *= spectrum_cpy[k]
        else:
            if PSF_cube_cpy is not None:
                spectrum_cpy = tmp_spectrum
            else:
                spectrum_cpy = np.ones(nl)

        spectrum_cpy /= np.sqrt(np.nansum(spectrum_cpy**2))
        # normalize PSF with norm 2.
        PSF_cube_cpy /= np.sqrt(np.sum(PSF_cube_cpy**2))

    else: # Assuming 2D image
        flat_cube = cube
        print("Metric calculation Not ready for 2D images to be corrected first")
        return

    try:
        # Retrieve the center of the image from the fits keyword.
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    except:
        # If the keywords could not be found.
        if not mute:
            print("Couldn't find PSFCENTX and PSFCENTY keywords.")
        center = [(nx-1)/2,(ny-1)/2]


    try:
        filter = prihdr['IFSFILT'].split('_')[1]
    except:
        # If the keywords could not be found.
        if not mute:
            print("Couldn't find IFSFILT keyword. Assuming H.")
        filter = "H"

    ##
    # Preliminaries and some sanity checks before saving the metrics maps fits file.
    if outputDir is None:
        outputDir = "."+os.path.sep
    else:
        outputDir = outputDir+os.path.sep

    if folderName is None:
        folderName = os.path.sep+"default_out" +os.path.sep
    else:
        folderName = folderName+os.path.sep

    if not os.path.exists(outputDir+folderName): os.makedirs(outputDir+folderName)

    try:
        # OBJECT: keyword in the primary header with the name of the star.
        prefix = prihdr['OBJECT'].strip().replace (" ", "_")
    except:
        prefix = "UNKNOWN_OBJECT"

    # Smoothing of the image. Remove the median of an arc centered on each pixel.
    # Actually don't do pixel per pixel but consider small boxes.
    # This function has to be cleaned.
    #flat_cube = subtract_radialMed(flat_cube,2,20,center)
    flat_cube_nans = np.where(np.isnan(flat_cube))

    # Calculate metrics only if the pixel is 5 pixel away from a nan
    flat_cube_mask = np.ones((ny,nx))
    flat_cube_mask[flat_cube_nans] = np.nan
    #widen the nans region (disabled)
    #conv_kernel = np.ones((5,5))
    #flat_cube_wider_mask = convolve2d(flat_cube_mask,conv_kernel,mode="same")
    flat_cube_wider_mask = copy(flat_cube_mask)
    # remove the edges if not already nans
    flat_cube_wider_mask[0:ny_PSF/2,:] = np.nan
    flat_cube_wider_mask[:,0:nx_PSF/2] = np.nan
    flat_cube_wider_mask[(ny-ny_PSF/2):ny,:] = np.nan
    flat_cube_wider_mask[:,(nx-nx_PSF/2):nx] = np.nan
    # Exctract the finite pixels from the mask
    flat_cube_wider_notNans = np.where(np.isnan(flat_cube_wider_mask) == 0)

    std_function = radialStdMap
    #std_function = ringSection_based_StdMap

    activate_metric_calc_WFC = True
    activate_metric_calc_MF = True
    activate_metric_calc_shape = True

    activate_proba_calc_FC = True
    activate_proba_calc_WFC = True
    activate_proba_calc_MF = True
    activate_proba_calc_shape = True

    activate_SNR_calc_FC = True
    activate_SNR_calc_WFC = True
    activate_SNR_calc_MF = True
    activate_SNR_calc_shape = True

    if metrics is not None:
        if len(metrics) == 1 and not isinstance(metrics,list):
            metrics = [metrics]

        if "flatCube" in metrics:
            if not overwrite_stat:
                if SNR:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-flatCube_SNR.fits')
                    if len(doesFileExist) != 0:
                        activate_SNR_calc_FC = False
                if probability:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-flatCube_proba.fits')
                    if len(doesFileExist) != 0:
                        activate_proba_calc_FC = False

            if SNR and activate_SNR_calc_FC:
                if not mute:
                    print("Calculating SNR of flatCube for "+filename)
                # Calculate the standard deviation map.
                # the standard deviation is calculated on annuli of width Dr. There is an annulus centered every dr.
                #flat_cube_std = radialStdMap(flat_cube, centroid=center)
                flat_cube_std = std_function(flat_cube, centroid=center)

                # Divide the convolved flat cube by the standard deviation map to get the SNR.
                flat_cube_SNR = flat_cube/flat_cube_std

            if probability and activate_proba_calc_FC:
                if not mute:
                    print("Calculating proba of flatCube for "+filename)
                image = flat_cube
                image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                #image_without_planet = copy(image)
                center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
                flat_cube_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)


        if "weightedFlatCube" in metrics:
            if not overwrite_metric:
                activate_metric_calc_WFC = False
                try:
                    hdulist = pyfits.open(outputDir+folderName+prefix+'-weightedFlatCube.fits')
                    weightedFlatCube = hdulist[1].data
                    hdulist.close()
                except:
                    activate_metric_calc_WFC = True

            if activate_metric_calc_WFC:
                if not mute:
                    print("Calculating weightedFlatCube for "+filename)
                weightedFlatCube = np.average(cube,axis=0,weights=spectrum_cpy)
                #weightedFlatCube_SNR = weightedFlatCube/radialStdMap(weightedFlatCube,dr,Dr, centroid=center)

            if not overwrite_stat:
                if SNR:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-weightedFlatCube_SNR.fits')
                    if len(doesFileExist) != 0:
                        activate_SNR_calc_WFC = False
                if probability:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-weightedFlatCube_proba.fits')
                    if len(doesFileExist) != 0:
                        activate_proba_calc_WFC = False

            if SNR and activate_SNR_calc_WFC:
                if not mute:
                    print("Calculating SNR of weightedFlatCube for "+filename)
                weightedFlatCube_SNR = weightedFlatCube/std_function(weightedFlatCube,centroid=center)
            if probability and activate_proba_calc_WFC:
                if not mute:
                    print("Calculating proba of weightedFlatCube for "+filename)
                image = weightedFlatCube
                image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                #image_without_planet = copy(image)
                center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
                weightedFlatCube_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)

        if "matchedFilter" in metrics and "shape" not in metrics:
            if not overwrite_metric:
                activate_metric_calc_MF = False
                try:
                    hdulist = pyfits.open(outputDir+folderName+prefix+'-matchedFilter.fits')
                    matchedFilter_map = hdulist[1].data
                    hdulist.close()
                except:
                    activate_metric_calc_MF = True

            if activate_metric_calc_MF:
                if not mute:
                    print("Calculating matchedFilter (no shape) for "+filename)
                matchedFilter_map = np.ones((ny,nx)) + np.nan

                # Calculate the criterion map.
                # For each pixel calculate the dot product of a stamp around it with the PSF.
                # We use the PSF cube to consider also the spectrum of the planet we are looking for.
                if not mute:
                    print("Calculate the criterion map. It is done pixel per pixel so it might take a while...")
                stamp_PSF_x_grid, stamp_PSF_y_grid = np.meshgrid(np.arange(0,nx_PSF,1)-nx_PSF/2,np.arange(0,ny_PSF,1)-ny_PSF/2)
                stamp_PSF_mask = np.ones((nl,ny_PSF,nx_PSF))
                r_PSF_stamp = abs((stamp_PSF_x_grid) +(stamp_PSF_y_grid)*1j)
                #r_PSF_stamp = np.tile(r_PSF_stamp,(nl,1,1))
                stamp_PSF_mask[np.where(r_PSF_stamp < 2.5)] = np.nan

                if N_threads is not None:
                    pool = NoDaemonPool(processes=N_threads)
                    #pool = mp.Pool(processes=N_threads)

                    ## cut images in N_threads part
                    # get the first and last index of each chunck
                    N_pix = flat_cube_wider_notNans[0].size
                    chunk_size = N_pix/N_threads
                    N_chunks = N_pix/chunk_size

                    # Get the chunks
                    chunks_row_indices = []
                    chunks_col_indices = []
                    for k in range(N_chunks-1):
                        chunks_row_indices.append(flat_cube_wider_notNans[0][(k*chunk_size):((k+1)*chunk_size)])
                        chunks_col_indices.append(flat_cube_wider_notNans[1][(k*chunk_size):((k+1)*chunk_size)])
                    chunks_row_indices.append(flat_cube_wider_notNans[0][((N_chunks-1)*chunk_size):N_pix])
                    chunks_col_indices.append(flat_cube_wider_notNans[1][((N_chunks-1)*chunk_size):N_pix])

                    outputs_list = pool.map(calculate_MF_metric_star, itertools.izip(chunks_row_indices,
                                                                                       chunks_col_indices,
                                                                                       itertools.repeat(cube),
                                                                                       itertools.repeat(PSF_cube_cpy),
                                                                                       itertools.repeat(stamp_PSF_mask)))

                    for row_indices,col_indices,out in zip(chunks_row_indices,chunks_col_indices,outputs_list):
                        matchedFilter_map[(row_indices,col_indices)] = out
                else:
                    matchedFilter_map[flat_cube_wider_notNans] = calculate_MF_metric(flat_cube_wider_notNans[0],flat_cube_wider_notNans[1],cube,PSF_cube_cpy,stamp_PSF_mask)


                IWA,OWA,inner_mask,outer_mask = get_occ(matchedFilter_map, centroid = center)
                conv_kernel = np.ones((10,10))
                wider_mask = convolve2d(outer_mask,conv_kernel,mode="same")
                matchedFilter_map[np.where(np.isnan(wider_mask))] = np.nan

            if not overwrite_stat:
                if SNR:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-matchedFilter_SNR.fits')
                    if len(doesFileExist) != 0:
                        activate_SNR_calc_MF = False
                if probability:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-matchedFilter_proba.fits')
                    if len(doesFileExist) != 0:
                        activate_proba_calc_MF = False

            if SNR and activate_SNR_calc_MF:
                if not mute:
                    print("Calculating SNR of matchedFilter (no shape) for "+filename)
                matchedFilter_SNR_map = matchedFilter_map/std_function(matchedFilter_map, centroid=center)
            if probability and activate_proba_calc_MF:
                if not mute:
                    print("Calculating proba of matchedFilter (no shape) for "+filename)
                image = matchedFilter_map
                image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                #image_without_planet = copy(image)
                center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
                matchedFilter_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)

        if "shape" in metrics and "matchedFilter" not in metrics:
            if not overwrite_metric:
                activate_metric_calc_shape = False
                try:
                    hdulist = pyfits.open(outputDir+folderName+prefix+'-shape.fits')
                    shape_map = hdulist[1].data
                    hdulist.close()
                except:
                    activate_metric_calc_shape = True

            if activate_metric_calc_shape:
                if not mute:
                    print("Calculating shape (no matchedFilter) for "+filename)
                shape_map = -np.ones((ny,nx)) + np.nan

                # Calculate the criterion map.
                # For each pixel calculate the dot product of a stamp around it with the PSF.
                # We use the PSF cube to consider also the spectrum of the planet we are looking for.
                if not mute:
                    print("Calculate the criterion map. It is done pixel per pixel so it might take a while...")
                stamp_PSF_x_grid, stamp_PSF_y_grid = np.meshgrid(np.arange(0,nx_PSF,1)-nx_PSF/2,np.arange(0,ny_PSF,1)-ny_PSF/2)
                stamp_PSF_mask = np.ones((nl,ny_PSF,nx_PSF))
                r_PSF_stamp = abs((stamp_PSF_x_grid) +(stamp_PSF_y_grid)*1j)
                #r_PSF_stamp = np.tile(r_PSF_stamp,(nl,1,1))
                stamp_PSF_mask[np.where(r_PSF_stamp < 2.5)] = np.nan

                if N_threads is not None:
                    pool = NoDaemonPool(processes=N_threads)
                    #pool = mp.Pool(processes=N_threads)

                    ## cut images in N_threads part
                    # get the first and last index of each chunck
                    N_pix = flat_cube_wider_notNans[0].size
                    chunk_size = N_pix/N_threads
                    N_chunks = N_pix/chunk_size

                    # Get the chunks
                    chunks_row_indices = []
                    chunks_col_indices = []
                    for k in range(N_chunks-1):
                        chunks_row_indices.append(flat_cube_wider_notNans[0][(k*chunk_size):((k+1)*chunk_size)])
                        chunks_col_indices.append(flat_cube_wider_notNans[1][(k*chunk_size):((k+1)*chunk_size)])
                    chunks_row_indices.append(flat_cube_wider_notNans[0][((N_chunks-1)*chunk_size):N_pix])
                    chunks_col_indices.append(flat_cube_wider_notNans[1][((N_chunks-1)*chunk_size):N_pix])

                    outputs_list = pool.map(calculate_shape_metric_star, itertools.izip(chunks_row_indices,
                                                                                       chunks_col_indices,
                                                                                       itertools.repeat(cube),
                                                                                       itertools.repeat(PSF_cube_cpy),
                                                                                       itertools.repeat(stamp_PSF_mask)))

                    for row_indices,col_indices,out in zip(chunks_row_indices,chunks_col_indices,outputs_list):
                        shape_map[(row_indices,col_indices)] = out
                else:
                    shape_map[flat_cube_wider_notNans] = calculate_shape_metric(flat_cube_wider_notNans[0],flat_cube_wider_notNans[1],cube,PSF_cube_cpy,stamp_PSF_mask)


                IWA,OWA,inner_mask,outer_mask = get_occ(shape_map, centroid = center)
                conv_kernel = np.ones((10,10))
                wider_mask = convolve2d(outer_mask,conv_kernel,mode="same")
                shape_map[np.where(np.isnan(wider_mask))] = np.nan

            if not overwrite_stat:
                if SNR:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-shape_SNR.fits')
                    if len(doesFileExist) != 0:
                        activate_SNR_calc_shape = False
                if probability:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-shape_proba.fits')
                    if len(doesFileExist) != 0:
                        activate_proba_calc_shape = False

            if SNR and activate_SNR_calc_shape:
                if not mute:
                    print("Calculating SNR of shape (no matchedFilter) for "+filename)
                shape_SNR_map = shape_map/std_function(shape_map, centroid=center)
            if probability and activate_proba_calc_shape:
                if not mute:
                    print("Calculating proba of shape (no matchedFilter) for "+filename)
                image = shape_map
                image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                #image_without_planet = copy(image)
                center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
                shape_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)


        if "matchedFilter" in metrics and "shape" in metrics:
            if not overwrite_metric:
                activate_metric_calc_MF = False
                activate_metric_calc_shape = False
                try:
                    hdulist = pyfits.open(outputDir+folderName+prefix+'-matchedFilter.fits')
                    matchedFilter_map = hdulist[1].data
                    hdulist.close()
                    hdulist = pyfits.open(outputDir+folderName+prefix+'-shape.fits')
                    shape_map = hdulist[1].data
                    hdulist.close()
                except:
                    activate_metric_calc_MF = True
                    activate_metric_calc_shape = True

            if activate_metric_calc_MF and activate_metric_calc_shape:
                if not mute:
                    print("Calculating shape and matchedFilter for "+filename)
                matchedFilter_map = np.ones((ny,nx)) + np.nan
                shape_map = -np.ones((ny,nx)) + np.nan

                # Calculate the criterion map.
                # For each pixel calculate the dot product of a stamp around it with the PSF.
                # We use the PSF cube to consider also the spectrum of the planet we are looking for.
                if not mute:
                    print("Calculate the criterion map. It is done pixel per pixel so it might take a while...")
                stamp_PSF_x_grid, stamp_PSF_y_grid = np.meshgrid(np.arange(0,nx_PSF,1)-nx_PSF/2,np.arange(0,ny_PSF,1)-ny_PSF/2)
                stamp_PSF_mask = np.ones((nl,ny_PSF,nx_PSF))
                r_PSF_stamp = abs((stamp_PSF_x_grid) +(stamp_PSF_y_grid)*1j)
                #r_PSF_stamp = np.tile(r_PSF_stamp,(nl,1,1))
                stamp_PSF_mask[np.where(r_PSF_stamp < 2.5)] = np.nan

                if N_threads is not None:
                    pool = NoDaemonPool(processes=N_threads)
                    #pool = mp.Pool(processes=N_threads)

                    ## cut images in N_threads part
                    # get the first and last index of each chunck
                    N_pix = flat_cube_wider_notNans[0].size
                    chunk_size = N_pix/N_threads
                    N_chunks = N_pix/chunk_size

                    # Get the chunks
                    chunks_row_indices = []
                    chunks_col_indices = []
                    for k in range(N_chunks-1):
                        chunks_row_indices.append(flat_cube_wider_notNans[0][(k*chunk_size):((k+1)*chunk_size)])
                        chunks_col_indices.append(flat_cube_wider_notNans[1][(k*chunk_size):((k+1)*chunk_size)])
                    chunks_row_indices.append(flat_cube_wider_notNans[0][((N_chunks-1)*chunk_size):N_pix])
                    chunks_col_indices.append(flat_cube_wider_notNans[1][((N_chunks-1)*chunk_size):N_pix])

                    outputs_list = pool.map(calculate_shapeAndMF_metric_star, itertools.izip(chunks_row_indices,
                                                                                       chunks_col_indices,
                                                                                       itertools.repeat(cube),
                                                                                       itertools.repeat(PSF_cube_cpy),
                                                                                       itertools.repeat(stamp_PSF_mask)))

                    for row_indices,col_indices,out in zip(chunks_row_indices,chunks_col_indices,outputs_list):
                        shape_map[(row_indices,col_indices)] = out[0]
                        matchedFilter_map[(row_indices,col_indices)] = out[1]
                else:
                    shape_map[flat_cube_wider_notNans],matchedFilter_map[flat_cube_wider_notNans] = \
                        calculate_shapeAndMF_metric(flat_cube_wider_notNans[0],flat_cube_wider_notNans[1],cube,PSF_cube_cpy,stamp_PSF_mask)


                IWA,OWA,inner_mask,outer_mask = get_occ(shape_map, centroid = center)
                conv_kernel = np.ones((10,10))
                wider_mask = convolve2d(outer_mask,conv_kernel,mode="same")
                shape_map[np.where(np.isnan(wider_mask))] = np.nan
                matchedFilter_map[np.where(np.isnan(wider_mask))] = np.nan

            if not overwrite_stat:
                if SNR:
                    doesFileExist1 = glob.glob(outputDir+folderName+prefix+'-shape_SNR.fits')
                    doesFileExist2 = glob.glob(outputDir+folderName+prefix+'-matchedFilter_SNR.fits')
                    if len(doesFileExist1) != 0 and len(doesFileExist2) != 0:
                        activate_SNR_calc_MF = False
                        activate_SNR_calc_shape = False
                if probability:
                    doesFileExist1 = glob.glob(outputDir+folderName+prefix+'-shape_proba.fits')
                    doesFileExist2 = glob.glob(outputDir+folderName+prefix+'-matchedFilter_proba.fits')
                    if len(doesFileExist1) != 0 and len(doesFileExist2) != 0:
                        activate_proba_calc_MF = False
                        activate_proba_calc_shape = False

            if SNR and activate_SNR_calc_MF and activate_SNR_calc_shape:
                if not mute:
                    print("Calculating SNR of shape and matchedFilter for "+filename)
                #shape_SNR_map = shape_map/radialStdMap(shape_map,dr,Dr, centroid=center)
                shape_SNR_map = shape_map/std_function(shape_map,centroid=center)
                #matchedFilter_SNR_map = matchedFilter_map/radialStdMap(matchedFilter_map,dr,Dr, centroid=center)
                matchedFilter_SNR_map = matchedFilter_map/std_function(matchedFilter_map,centroid=center)
            if probability and activate_proba_calc_MF and activate_proba_calc_shape:
                if not mute:
                    print("Calculating proba of shape and matchedFilter for "+filename)
                center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
                image = shape_map
                image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                #image_without_planet = copy(image)
                shape_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)
                image = matchedFilter_map
                image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                #image_without_planet = copy(image)
                matchedFilter_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)



    ## ortho_criterion is actually the sine squared between the two vectors
    ## ortho_criterion_map = 1 - criterion_map
    ## ratio_shape_SNR = 10
    ## criterion_map = np.minimum(ratio_shape_SNR*shape_map,flat_cube_SNR)



    if not mute:
        print("Saving metrics maps as: "+outputDir+folderName+prefix+'-<myMetric>.fits')
        print("Saving stat maps as: "+outputDir+folderName+prefix+'-<myMetric>_<stat>.fits')
    hdulist2 = pyfits.HDUList()
    #try:
    hdulist2.append(pyfits.PrimaryHDU(header=prihdr))
    hdulist2.append(pyfits.ImageHDU(header=exthdr, data=flat_cube, name="Sci"))
    #hdulist2[1].data = flat_cube
    hdulist2.writeto(outputDir+folderName+prefix+'-flatCube.fits', clobber=True)
    if metrics is not None:
        if "flatCube" in metrics:
            if SNR and activate_SNR_calc_FC:
                hdulist2[1].data = flat_cube_SNR
                hdulist2.writeto(outputDir+folderName+prefix+'-flatCube_SNR.fits', clobber=True)
            if probability and activate_proba_calc_FC:
                hdulist2[1].data = flat_cube_proba_map
                hdulist2.writeto(outputDir+folderName+prefix+'-flatCube_proba.fits', clobber=True)
        if "weightedFlatCube" in metrics:
            if activate_metric_calc_WFC:
                hdulist2[1].data = weightedFlatCube
                hdulist2.writeto(outputDir+folderName+prefix+'-weightedFlatCube.fits', clobber=True)
            if SNR and activate_SNR_calc_WFC:
                hdulist2[1].data = weightedFlatCube_SNR
                hdulist2.writeto(outputDir+folderName+prefix+'-weightedFlatCube_SNR.fits', clobber=True)
            if probability and activate_proba_calc_WFC:
                hdulist2[1].data = weightedFlatCube_proba_map
                hdulist2.writeto(outputDir+folderName+prefix+'-weightedFlatCube_proba.fits', clobber=True)
        if "matchedFilter" in metrics:
            if activate_metric_calc_MF:
                hdulist2[1].data = matchedFilter_map
                hdulist2.writeto(outputDir+folderName+prefix+'-matchedFilter.fits', clobber=True)
            if SNR and activate_SNR_calc_MF:
                hdulist2[1].data = matchedFilter_SNR_map
                hdulist2.writeto(outputDir+folderName+prefix+'-matchedFilter_SNR.fits', clobber=True)
            if probability and activate_proba_calc_MF:
                hdulist2[1].data = matchedFilter_proba_map
                hdulist2.writeto(outputDir+folderName+prefix+'-matchedFilter_proba.fits', clobber=True)
        if "shape" in metrics:
            if activate_metric_calc_shape:
                hdulist2[1].data = shape_map
                hdulist2.writeto(outputDir+folderName+prefix+'-shape.fits', clobber=True)
            if SNR and activate_SNR_calc_shape:
                hdulist2[1].data = shape_SNR_map
                hdulist2.writeto(outputDir+folderName+prefix+'-shape_SNR.fits', clobber=True)
            if probability and activate_proba_calc_shape:
                hdulist2[1].data = shape_proba_map
                hdulist2.writeto(outputDir+folderName+prefix+'-shape_proba.fits', clobber=True)
    """
    except:
        print("No exthdr so only using primary to save data...")
        hdulist2.append(pyfits.PrimaryHDU(header=prihdr))
        hdulist2[0].data = flat_cube
        hdulist2.writeto(outputDir+folderName+prefix+'-flatCube.fits', clobber=True)
        hdulist2[0].data = flat_cube_SNR
        hdulist2.writeto(outputDir+folderName+prefix+'-flatCubeSNR.fits', clobber=True)
        if metrics is not None:
            if "weightedFlatCube" in metrics:
                hdulist2[0].data = weightedFlatCube
                hdulist2.writeto(outputDir+folderName+prefix+'-weightedFlatCube.fits', clobber=True)
                hdulist2[0].data = weightedFlatCube_SNR
                hdulist2.writeto(outputDir+folderName+prefix+'-weightedFlatCube_SNR.fits', clobber=True)
            if "matchedFilter" in metrics:
                hdulist2[0].data = matchedFilter_map
                hdulist2.writeto(outputDir+folderName+prefix+'-matchedFilter.fits', clobber=True)
                hdulist2[0].data = matchedFilter_SNR_map
                hdulist2.writeto(outputDir+folderName+prefix+'-matchedFilter_SNR.fits', clobber=True)
            if "shape" in metrics:
                hdulist2[0].data = shape_map
                hdulist2.writeto(outputDir+folderName+prefix+'-shape.fits', clobber=True)
                hdulist2[0].data = shape_SNR_map
                hdulist2.writeto(outputDir+folderName+prefix+'-shape_SNR.fits', clobber=True)
    """
    hdulist2.close()

    # Save a plot of the spectrum used
    spec_sampling = spec.get_gpi_wavelength_sampling(filter)
    plt.close(1)
    plt.figure(1)
    plt.plot(spec_sampling,spectrum_cpy,"rx-",markersize = 7, linewidth = 2)
    plt.title("Template spectrum use in this folder")
    plt.xlabel("Wavelength (mum)")
    plt.ylabel("Norm-2 Normalized")
    plt.savefig(outputDir+folderName+'template_spectrum.png', bbox_inches='tight')
    plt.close(1)

    hdulist2[1].data = [spec_sampling,spectrum_cpy]
    hdulist2.writeto(outputDir+folderName+'template_spectrum.fits', clobber=True)

    return 1
# END calculate_metrics() DEFINITION

def candidate_detection(metrics_foldername,
                        mute = False,
                        confirm_candidates = None):
    '''

    Inputs:


    Outputs:

    '''
    shape_filename_list = glob.glob(metrics_foldername+os.path.sep+"*-shape_proba.fits")
    matchedFilter_filename_list = glob.glob(metrics_foldername+os.path.sep+"*-matchedFilter_proba.fits")
    if len(shape_filename_list) == 0:
        if not mute:
            print("Couldn't find shape_SNR map in "+metrics_foldername)
            return 0
    else:
        hdulist_shape = pyfits.open(shape_filename_list[0])
    if len(matchedFilter_filename_list) == 0:
        if not mute:
            print("Couldn't find matchedFilter_SNR map in "+metrics_foldername)
            return 0
    else:
        hdulist_matchedFilter = pyfits.open(matchedFilter_filename_list[0])

    #grab the data and headers
    try:
        shape_map = hdulist_shape[1].data
        matchedFilter_map = hdulist_matchedFilter[1].data
        exthdr = hdulist_shape[1].header
        prihdr = hdulist_shape[0].header
    except:
        print("Couldn't read the fits file normally. Try another way.")
        shape_map = hdulist_shape[0].data
        prihdr = hdulist_shape[0].header

    if np.size(shape_map.shape) == 2:
        ny,nx = shape_map.shape

    try:
        # Retrieve the center of the image from the fits keyword.
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    except:
        # If the keywords could not be found.
        if not mute:
            print("Couldn't find PSFCENTX and PSFCENTY keywords.")
        center = [(nx-1)/2,(ny-1)/2]


    try:
        # OBJECT: keyword in the primary header with the name of the star.
        star_name = prihdr['OBJECT'].strip().replace (" ", "_")
    except:
        star_name = "UNKNOWN_OBJECT"


    #criterion_map = np.max([shape_map,matchedFilter_map],axis=0)
    criterion_map = shape_map
    criterion_map_cpy = copy(criterion_map)
    if 0:
        IWA,OWA,inner_mask,outer_mask = get_occ(criterion_map, centroid = center)

        # Ignore all the pixel too close from an edge with nans
        #flat_cube_nans = np.where(np.isnan(criterion_map))
        #flat_cube_mask = np.ones((ny,nx))
        #flat_cube_mask[flat_cube_nans] = np.nan
        #widen the nans region
        conv_kernel = np.ones((10,10))
        flat_cube_wider_mask = convolve2d(outer_mask,conv_kernel,mode="same")
        criterion_map[np.where(np.isnan(flat_cube_wider_mask))] = np.nan


    # Build as grids of x,y coordinates.
    # The center is in the middle of the array and the unit is the pixel.
    # If the size of the array is even 2n x 2n the center coordinates is [n,n].
    x_grid, y_grid = np.meshgrid(np.arange(0,nx,1)-center[0],np.arange(0,ny,1)-center[1])


    # Definition of the different masks used in the following.
    stamp_nrow = 13
    stamp_ncol = 13
    # Mask to remove the spots already checked in criterion_map.
    stamp_x_grid, stamp_y_grid = np.meshgrid(np.arange(0,stamp_nrow,1)-6,np.arange(0,stamp_ncol,1)-6)
    stamp_mask = np.ones((stamp_nrow,stamp_ncol))
    r_stamp = abs((stamp_x_grid) +(stamp_y_grid)*1j)
    stamp_mask[np.where(r_stamp < 4.0)] = np.nan

    row_m = np.floor(stamp_nrow/2.0)
    row_p = np.ceil(stamp_nrow/2.0)
    col_m = np.floor(stamp_ncol/2.0)
    col_p = np.ceil(stamp_ncol/2.0)

    root = ET.Element("root")
    star_elt = ET.SubElement(root, star_name)
    candidates_elt = ET.SubElement(star_elt, "candidates")
    all_elt = ET.SubElement(star_elt, "all")

    # Count the number of valid detected candidates.
    N_candidates = 0.0

    # Maximum number of iterations on local maxima.
    max_attempts = 60
                ## START FOR LOOP.
                ## Each iteration looks at one local maximum in the criterion map.
    k = 0
    max_val_criter = np.nanmax(criterion_map)
    while max_val_criter >= 2.5 and k <= max_attempts:
        k += 1
        # Find the maximum value in the current SNR map. At each iteration the previous maximum is masked out.
        max_val_criter = np.nanmax(criterion_map)
        # Locate the maximum by retrieving its coordinates
        max_ind = np.where( criterion_map == max_val_criter )
        row_id,col_id = max_ind[0][0],max_ind[1][0]
        x_max_pos, y_max_pos = x_grid[row_id,col_id],y_grid[row_id,col_id]

        # Mask the spot around the maximum we just found.
        criterion_map[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] *= stamp_mask

        potential_planet = max_val_criter > 4.0


        ET.SubElement(all_elt,"localMax",
                      id = str(k),
                      max_val_criter = str(max_val_criter),
                      x_max_pos= str(x_max_pos),
                      y_max_pos= str(y_max_pos),
                      row_id= str(row_id),
                      col_id= str(col_id))

        if potential_planet:
            ET.SubElement(candidates_elt, "localMax",
                          id = str(k),
                          max_val_criter = str(max_val_criter),
                          x_max_pos= str(x_max_pos),
                          y_max_pos= str(y_max_pos),
                          row_id= str(row_id),
                          col_id= str(col_id))

            # Increment the number of detected candidates
            N_candidates += 1


    tree = ET.ElementTree(root)
    tree.write(metrics_foldername+os.path.sep+star_name+'-detections.xml')

    # Highlight the detected candidates in the criterion map
    if not mute:
        print("Number of candidates = " + str(N_candidates))
    plt.close(3)
    plt.figure(3,figsize=(16,16))
    plt.imshow(criterion_map_cpy[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
    ax = plt.gca()
    for candidate in candidates_elt:
        candidate_id = int(candidate.attrib["id"])
        max_val_criter = float(candidate.attrib["max_val_criter"])
        x_max_pos = float(candidate.attrib["x_max_pos"])
        y_max_pos = float(candidate.attrib["y_max_pos"])

        ax.annotate(str(candidate_id)+","+"{0:02.1f}".format(max_val_criter), fontsize=20, color = "black", xy=(x_max_pos+0.0, y_max_pos+0.0),
                xycoords='data', xytext=(x_max_pos+10, y_max_pos-10),
                textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                linewidth = 1.,
                                color = 'black')
                )
    plt.clim(0.,10.0)
    plt.savefig(metrics_foldername+os.path.sep+star_name+'-detectionIm_candidates.png', bbox_inches='tight')
    plt.close(3)

    plt.close(3)
    # Show the local maxima in the criterion map
    plt.figure(3,figsize=(16,16))
    plt.imshow(criterion_map_cpy[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
    ax = plt.gca()
    for spot in all_elt:
        candidate_id = int(spot.attrib["id"])
        max_val_criter = float(spot.attrib["max_val_criter"])
        x_max_pos = float(spot.attrib["x_max_pos"])
        y_max_pos = float(spot.attrib["y_max_pos"])
        ax.annotate(str(candidate_id)+","+"{0:02.1f}".format(max_val_criter), fontsize=20, color = 'black', xy=(x_max_pos+0.0, y_max_pos+0.0),
                xycoords='data', xytext=(x_max_pos+10, y_max_pos-10),
                textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                linewidth = 1.,
                                color = 'black')
                )
    plt.clim(0.,10.0)
    plt.savefig(metrics_foldername+os.path.sep+star_name+'-detectionIm_all.png', bbox_inches='tight')
    plt.close(3)

def gather_detections(planet_detec_dir, PSF_cube_filename, mute = True):

    spectrum_folders_list = glob.glob(planet_detec_dir+os.path.sep+"*"+os.path.sep)
    N_spectra_folders = len(spectrum_folders_list)
    plt.figure(1,figsize=(8*N_spectra_folders,16))


    # First let's recover the name of the file that was reduced for this folder.
    # The name of the file is in the folder name
    planet_detec_dir_glob = glob.glob(planet_detec_dir)
    if np.size(planet_detec_dir_glob) == 0:
        if not mute:
            print("/!\ Quitting! Couldn't find the following directory " + planet_detec_dir)
        return 0
    else:
        planet_detec_dir = planet_detec_dir_glob[0]

    planet_detec_dir_splitted = planet_detec_dir.split(os.path.sep)
    # Filename of the original klipped cube
    original_cube_filename = planet_detec_dir_splitted[len(planet_detec_dir_splitted)-2].split("planet_detec_")[1]
    original_cube_filename += "-speccube.fits"

    if isinstance(PSF_cube_filename, basestring):
        hdulist = pyfits.open(PSF_cube_filename)
        PSF_cube = hdulist[1].data
        hdulist.close()
    else:
        PSF_cube = PSF_cube_filename

    hdulist = pyfits.open(planet_detec_dir+os.path.sep+".."+os.path.sep+original_cube_filename)
    cube = hdulist[1].data
    exthdr = hdulist[1].header
    prihdr = hdulist[0].header
    hdulist.close()

    nl,ny,nx = np.shape(cube)

    try:
        # Retrieve the center of the image from the fits keyword.
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    except:
        # If the keywords could not be found.
        center = [(nx-1)/2,(ny-1)/2]


    try:
        prihdr = hdulist[0].header
        date = prihdr["DATE"]
        hdulist.close()
    except:
        date = "no_date"
        hdulist.close()

    try:
        filter = prihdr['IFSFILT'].split('_')[1]
    except:
        # If the keywords could not be found.
        filter = "no_filter"


    if not mute:
        print("Looking for folders in " + planet_detec_dir + " ...")
        print("... They should contain spectrum template based detection algorithm outputs.")
    all_templates_detections = []
    for spec_id,spectrum_folder in enumerate(spectrum_folders_list):
        spectrum_folder_splitted = spectrum_folder.split(os.path.sep)
        spectrum_name = spectrum_folder_splitted[len(spectrum_folder_splitted)-2]
        if not mute:
            print("Found the folder " + spectrum_name)


        # Gather the detection png in a single png
        candidates_log_file_list = glob.glob(spectrum_folder+os.path.sep+"*-detections.xml")
        #weightedFlatCube_file_list = glob.glob(spectrum_folder+os.path.sep+"*-weightedFlatCube_proba.fits")
        shape_proba_file_list = glob.glob(spectrum_folder+os.path.sep+"*-shape_proba.fits")
        shape_file_list = glob.glob(spectrum_folder+os.path.sep+"*-shape.fits")
        template_spectrum_file_list = glob.glob(spectrum_folder+os.path.sep+"template_spectrum.fits")
        if len(candidates_log_file_list) == 1 and len(shape_proba_file_list) == 1 and len(shape_file_list) == 1 and len(template_spectrum_file_list) == 1:
            candidates_log_file = candidates_log_file_list[0]
            shape_proba_file = shape_proba_file_list[0]
            shape_file = shape_file_list[0]
            template_spectrum_file = template_spectrum_file_list[0]

            splitted_str =  candidates_log_file.split(os.path.sep)
            star_name = splitted_str[len(splitted_str)-1].split("-")[0]

            # Read flatCube_file
            hdulist = pyfits.open(shape_proba_file)
            shape_proba = hdulist[1].data
            exthdr = hdulist[1].header
            prihdr = hdulist[0].header
            hdulist.close()
            x_grid, y_grid = np.meshgrid(np.arange(0,nx,1)-center[0],np.arange(0,ny,1)-center[1])

            plt.subplot(2,N_spectra_folders,spec_id+1)
            plt.imshow(shape_proba[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
            ax = plt.gca()


            tree = ET.parse(candidates_log_file)
            root = tree.getroot()
            for candidate in root[0].find("candidates"):
                candidate_id = int(candidate.attrib["id"])
                max_val_criter = float(candidate.attrib["max_val_criter"])
                x_max_pos = float(candidate.attrib["x_max_pos"])
                y_max_pos = float(candidate.attrib["y_max_pos"])
                row_id = float(candidate.attrib["row_id"])
                col_id = float(candidate.attrib["col_id"])

                all_templates_detections.append((spectrum_name,int(candidate_id),float(max_val_criter),float(x_max_pos),float(y_max_pos), int(row_id),int(col_id)))

                ax.annotate(str(int(candidate_id))+","+"{0:02.1f}".format(float(max_val_criter)), fontsize=30, color = "red", xy=(float(x_max_pos), float(y_max_pos)),
                        xycoords='data', xytext=(float(x_max_pos)+20, float(y_max_pos)-20),
                        textcoords='data',
                        arrowprops=dict(arrowstyle="->",
                                        linewidth = 2.,
                                        color = 'red')
                        )
            plt.title(star_name +" "+ spectrum_name)
            plt.clim(0.,5.0)

            # Read flatCube_file
            hdulist = pyfits.open(shape_file)
            shape = hdulist[1].data
            hdulist.close()

            plt.subplot(2,N_spectra_folders,N_spectra_folders+spec_id+1)
            plt.imshow(shape[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
            plt.colorbar()


            if 0:
                # Read template_spectrum_file
                hdulist = pyfits.open(template_spectrum_file)
                spec_sampling = hdulist[1].data[0]
                spectrum = hdulist[1].data[1]
                hdulist.close()

                plt.subplot(3,N_spectra_folders,2*N_spectra_folders+spec_id+1)
                plt.plot(spec_sampling,spectrum,"rx-",markersize = 7, linewidth = 2)
                plt.title("Template spectrum use in this folder")
                plt.xlabel("Wavelength (mum)")
                plt.ylabel("Norm-2 Normalized")

    #plt.show()

    ##
    # Extract centroid for all detections
    root = ET.Element("root")
    star_elt = ET.SubElement(root, star_name)
    #position_no_duplicated_detec = []
    no_duplicates_id = 0
    for detection in all_templates_detections:
        spectrum_name,candidate_it,max_val_criter,x_max_pos,y_max_pos, row_id,col_id = detection

        row_cen,col_cen = spec.extract_planet_centroid(cube, (row_id,col_id), PSF_cube)

        candidate_already_seen = False
        candidate_seen_with_template = False


        #Check that the detection hasn't already been taken care of.
        for candidate in star_elt:
            if abs(col_cen-float(candidate.attrib['col_centroid'])) < 1. \
                and abs(row_cen-float(candidate.attrib['row_centroid'])) < 1.:
                candidate_already_seen = True
                candidate_elt = candidate

        if not candidate_already_seen:
            no_duplicates_id+=1
            candidate_elt = ET.SubElement(star_elt,"candidate",
                                          id = str(no_duplicates_id),
                                          x_max_pos= str(x_max_pos),
                                          y_max_pos= str(y_max_pos),
                                          col_centroid= str(col_cen),
                                          row_centroid= str(row_cen),
                                          row_id= str(row_id),
                                          col_id= str(col_id))
            ET.SubElement(candidate_elt,"spectrumTemplate",
                          candidate_it = str(candidate_it),
                          name = spectrum_name,
                          max_val_criter = str(max_val_criter))

            for spec_id in range(N_spectra_folders):
                plt.subplot(2,N_spectra_folders,N_spectra_folders+spec_id+1)
                ax = plt.gca()
                ax.annotate(str(int(no_duplicates_id)), fontsize=30, color = "black", xy=(float(x_max_pos), float(y_max_pos)),
                        xycoords='data', xytext=(float(x_max_pos)+20, float(y_max_pos)-20),
                        textcoords='data',
                        arrowprops=dict(arrowstyle="->",
                                        linewidth = 2.,
                                        color = 'black')
                        )
        else:
            for template in candidate_elt:
                if template.attrib["name"] == spectrum_name:
                    candidate_seen_with_template = True

            if not candidate_seen_with_template:
                ET.SubElement(candidate_elt,"spectrumTemplate",
                              candidate_it = str(candidate_it),
                              name = spectrum_name,
                              max_val_criter = str(max_val_criter))

    for candidate in star_elt:
        wave_samp,spectrum = spec.extract_planet_spectrum(planet_detec_dir+os.path.sep+".."+os.path.sep+original_cube_filename,
                                                          (float(candidate.attrib["row_centroid"]), float(candidate.attrib['col_centroid'])),
                                                          PSF_cube, method="aperture")

        plt.close(2)
        plt.figure(2)
        plt.plot(wave_samp,spectrum/np.nanmean(spectrum),"rx-",markersize = 7, linewidth = 2)
        legend_str = ["candidate spectrum"]
        for spectrum_folder in spectrum_folders_list:
            spectrum_folder_splitted = spectrum_folder.split(os.path.sep)
            spectrum_name = spectrum_folder_splitted[len(spectrum_folder_splitted)-2]

            hdulist = pyfits.open(spectrum_folder+os.path.sep+'template_spectrum.fits')
            spec_data = hdulist[1].data
            hdulist.close()

            wave_samp = spec_data[0]
            spectrum_template = spec_data[1]

            plt.plot(wave_samp,spectrum_template/np.nanmean(spectrum_template),"--")
            legend_str.append(spectrum_name)
        plt.title("Candidate spectrum and templates used")
        plt.xlabel("Wavelength (mum)")
        plt.ylabel("Mean Normalized")
        ax = plt.gca()
        ax.legend(legend_str, loc = 'upper right', fontsize=12)
        #print(planet_detec_dir)
        #plt.show()
        plt.savefig(planet_detec_dir+os.path.sep+star_name+'-'+filter+'-'+date+'-candidate'+"_"+ \
                    str(candidate.attrib["id"]) +'.png', bbox_inches='tight')
        plt.close(2)

        #myAttributes = {"proba":max_val_criter}
        #ET.SubElement(object_elt, "detec "+str(candidate_it), name="coucou").text = "some value1"


    tree = ET.ElementTree(root)
    print(planet_detec_dir+os.path.sep+star_name+'-'+filter+'-'+date+'-candidates.xml')
    tree.write(planet_detec_dir+os.path.sep+star_name+'-'+filter+'-'+date+'-candidates.xml')


    plt.savefig(planet_detec_dir+os.path.sep+star_name+'-'+filter+'-'+date+'-candidates.png', bbox_inches='tight')
    plt.close(1)


def planet_detection_in_dir_per_file_per_spectrum_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call planet_detection_in_dir_per_file_per_spectrum() with a tuple of parameters.
    """
    return planet_detection_in_dir_per_file_per_spectrum(*params)

def planet_detection_in_dir_per_file_per_spectrum(spectrum_name_it,
                                                  filename,
                                                  filter,
                                                  sat_spot_spec = None,
                                                  metrics = None,
                                                  outputDir = '',
                                                  star_type = "",
                                                  star_temperature = None,
                                                  PSF_cube = None,
                                                  metrics_only = False,
                                                  planet_detection_only = False,
                                                  mute = False,
                                                  GOI_list = None,
                                                  overwrite_metric = False,
                                                  overwrite_stat = False,
                                                  proba_using_mask_per_pixel = False,
                                                  SNR = True,
                                                  probability = True,
                                                  N_threads_metric = None):

        # Define the output Foldername
        if spectrum_name_it != "":
            spectrum_name = spectrum_name_it.split(os.path.sep)
            spectrum_name = spectrum_name[len(spectrum_name)-1].split(".")[0]
        else:
            spectrum_name = "satSpotSpec"

        folderName = spectrum_name+os.path.sep



        if spectrum_name_it != "":
            if not mute:
                print("spectrum model: "+spectrum_name_it)
            # Interpolate the spectrum of the planet based on the given filename
            wv,planet_sp = spec.get_planet_spectrum(spectrum_name_it,filter)

            if sat_spot_spec is not None and (star_type !=  "" or star_temperature is not None):
                # Interpolate a spectrum of the star based on its spectral type/temperature
                wv,star_sp = spec.get_star_spectrum(filter,star_type,star_temperature)
                spectrum = (sat_spot_spec/star_sp)*planet_sp
            else:
                if not mute:
                    print("No star spec or sat spot spec so using sole planet spectrum.")
                spectrum = copy(planet_sp)
        else:
            if sat_spot_spec is not None:
                if not mute:
                    print("Default sat spot spectrum will be used.")
                spectrum = copy(sat_spot_spec)
            else:
                if not mute:
                    print("Using gpi filter "+filter+" spectrum. Could find neither sat spot spectrum nor planet spectrum.")
                wv,spectrum = spec.get_gpi_filter(filter)

        if 0:
            plt.plot(sat_spot_spec)
            plt.plot(spectrum)
            plt.show()


        if not planet_detection_only:
            if not mute:
                print("Calling calculate_metrics() on "+filename)
            calculate_metrics(filename,
                              metrics,
                                PSF_cube = PSF_cube,
                                outputDir = outputDir,
                                folderName = folderName,
                                spectrum=spectrum,
                                mute = mute,
                                SNR = SNR,
                                probability = probability,
                                GOI_list = GOI_list,
                                overwrite_metric = overwrite_metric,
                                overwrite_stat = overwrite_stat,
                                proba_using_mask_per_pixel = proba_using_mask_per_pixel,
                                N_threads = N_threads_metric )


        if not metrics_only:
            if not mute:
                print("Calling candidate_detection() on "+outputDir+folderName)
            candidate_detection(outputDir+folderName,
                                mute = mute)



def planet_detection_in_dir_per_file_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call planet_detection_in_dir_per_file() with a tuple of parameters.
    """
    return planet_detection_in_dir_per_file(*params)

def planet_detection_in_dir_per_file(filename,
                                      metrics = None,
                                      directory = "."+os.path.sep,
                                      outputDir = '',
                                      spectrum_model = "",
                                      star_type = "",
                                      star_temperature = None,
                                      user_defined_PSF_cube = None,
                                      metrics_only = False,
                                      planet_detection_only = False,
                                      mute = False,
                                      threads = False,
                                      GOI_list = None,
                                      overwrite_metric = False,
                                      overwrite_stat = False,
                                      proba_using_mask_per_pixel = False,
                                      SNR = True,
                                      probability = True):
    # Get the number of KL_modes for this file based on the filename *-KL#-speccube*
    splitted_name = filename.split("-KL")
    splitted_after_KL = splitted_name[1].split("-speccube.")
    N_KL_modes = int(splitted_after_KL[0])

    # Get the prefix of the filename
    splitted_before_KL = splitted_name[0].split(os.path.sep)
    prefix = splitted_before_KL[np.size(splitted_before_KL)-1]

    compact_date = prefix.split("-")[1]



    #grab the headers
    hdulist = pyfits.open(filename)
    prihdr = hdulist[0].header
    try:
        filter = prihdr['IFSFILT'].split('_')[1]
    except:
        # If the keywords could not be found.
        if not mute:
            print("Couldn't find IFSFILT keyword. Assuming H.")
        filter = "H"

    if user_defined_PSF_cube is not None:
        if not mute:
            print("User defined PSF cube: "+user_defined_PSF_cube)
        filelist_ori_PSFs_cube = glob.glob(user_defined_PSF_cube)
    else:
        filelist_ori_PSFs_cube = glob.glob(directory+os.path.sep+"*"+compact_date+"*-original_radial_PSF_cube.fits")

    if np.size(filelist_ori_PSFs_cube) == 1:
        if not mute:
            print("I found a radially averaged PSF. I'll take it.")
        hdulist = pyfits.open(filelist_ori_PSFs_cube[0])
        #print(hdulist[1].data.shape)
        PSF_cube = hdulist[1].data[:,::-1,:]
        #PSF_cube = np.transpose(hdulist[1].data,(1,2,0))[:,::-1,:] #np.swapaxes(np.rollaxis(hdulist[1].data,2,1),0,2)
        sat_spot_spec = np.nanmax(PSF_cube,axis=(1,2))
        # Remove the spectral shape from the psf cube because it is dealt with independently
        for l_id in range(PSF_cube.shape[0]):
            PSF_cube[l_id,:,:] /= sat_spot_spec[l_id]
    elif np.size(filelist_ori_PSFs_cube) == 0:
        if not mute:
            print("I didn't find any PSFs file so trying to generate one.")

        print(directory+os.path.sep+compact_date+"*_spdc_distorcorr.fits")
        cube_filename_list = glob.glob(directory+os.path.sep+compact_date+"*_spdc_distorcorr.fits")
        if len(cube_filename_list) == 0:
            cube_filename_list = glob.glob(directory+os.path.sep+compact_date+"*_spdc.fits")
            if len(cube_filename_list) == 0:
                if not mute:
                    print("I can't find the cubes to calculate the PSF from so I will use a default gaussian.")
                sat_spot_spec = None
                PSF_cube = None

        dataset = GPI.GPIData(cube_filename_list)
        if not mute:
            print("Calculating the planet PSF from the satellite spots...")
        dataset.generate_psf_cube(20)
        # Save the original PSF calculated from combining the sat spots
        dataset.savedata(directory + os.path.sep + prefix+"-original_PSF_cube.fits", dataset.psfs,
                                  astr_hdr=dataset.wcs[0], filetype="PSF Spec Cube")
        # Calculate and save the rotationally invariant psf (ie smeared out/averaged).
        PSF_cube = dataset.get_radial_psf(save = directory + os.path.sep + prefix)
        sat_spot_spec = np.nanmax(PSF_cube,axis=(1,2))
        # Remove the spectral shape from the psf cube because it is dealt with independently
        for l_id in range(PSF_cube.shape[0]):
            PSF_cube[l_id,:,:] /= sat_spot_spec[l_id]
    else:
        if not mute:
            print("I found several files for the PSF cube matching the given filter so I don't know what to do and I quit")
        return 0


    prefix = prefix+"-KL"+str(N_KL_modes)

    if len(spectrum_model) == 1 and not isinstance(spectrum_model,list):
        spectrum_model =[spectrum_model]
    else:
        if not mute:
            print("Iterating over several model spectra")

    if outputDir == '':
        outputDir = directory
    outputDir += os.path.sep+"planet_detec_"+prefix+os.path.sep

    if threads:
        N_threads = np.size(spectrum_model)
        pool = NoDaemonPool(processes=N_threads)
        #pool = mp.Pool(processes=N_threads)

        N_threads_metric = mp.cpu_count()/N_threads
        print(mp.cpu_count(),N_threads,N_threads_metric)
        if N_threads_metric <= 1:
            N_threads_metric = None

        pool.map(planet_detection_in_dir_per_file_per_spectrum_star, itertools.izip(spectrum_model,
                                                                           itertools.repeat(filename),
                                                                           itertools.repeat(filter),
                                                                           itertools.repeat(sat_spot_spec),
                                                                           itertools.repeat(metrics),
                                                                           itertools.repeat(outputDir),
                                                                           itertools.repeat(star_type),
                                                                           itertools.repeat(star_temperature),
                                                                           itertools.repeat(PSF_cube),
                                                                           itertools.repeat(metrics_only),
                                                                           itertools.repeat(planet_detection_only),
                                                                           itertools.repeat(mute),
                                                                           itertools.repeat(GOI_list),
                                                                           itertools.repeat(overwrite_metric),
                                                                           itertools.repeat(overwrite_stat),
                                                                           itertools.repeat(proba_using_mask_per_pixel),
                                                                           itertools.repeat(SNR),
                                                                           itertools.repeat(probability),
                                                                           itertools.repeat(N_threads_metric)))
    else:
        for spectrum_name_it in spectrum_model:
            planet_detection_in_dir_per_file_per_spectrum(spectrum_name_it,
                                                          filename,
                                                          filter,
                                                          sat_spot_spec = sat_spot_spec,
                                                          metrics = metrics,
                                                          outputDir = outputDir,
                                                          star_type = star_type,
                                                          star_temperature = star_temperature,
                                                          PSF_cube = PSF_cube,
                                                          metrics_only = metrics_only,
                                                          planet_detection_only = planet_detection_only,
                                                          mute = mute,
                                                          GOI_list = GOI_list,
                                                          overwrite_metric = overwrite_metric,
                                                          overwrite_stat = overwrite_stat,
                                                          proba_using_mask_per_pixel = proba_using_mask_per_pixel,
                                                          SNR = SNR,
                                                          probability = probability,
                                                          N_threads_metric = None)


    if not metrics_only:
        if not mute:
            print("Calling gather_detections() on "+outputDir)
        gather_detections(outputDir,PSF_cube, mute = mute)

def planet_detection_in_dir_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call planet_detection_in_dir() with a tuple of parameters.
    """
    return planet_detection_in_dir(*params)

def planet_detection_in_dir(directory = "."+os.path.sep,
                            filename_prefix_is = '',
                            numbasis = None,
                            outputDir = '',
                            spectrum_model = "",
                            star_type = "",
                            star_temperature = None,
                            user_defined_PSF_cube = None,
                            metrics = None,
                            threads = False,
                            metrics_only = False,
                            planet_detection_only = False,
                            mute = True,
                            GOI_list = None,
                            overwrite_metric = False,
                            overwrite_stat = False,
                            proba_using_mask_per_pixel = False,
                            SNR = True,
                            probability = True):
    '''
    Apply the planet detection algorithm for all pyklip reduced cube respecting a filter in a given folder.
    By default the filename filter used is pyklip-*-KL*-speccube.fits.
    It will look for a PSF file like pyklip-*-KL*-speccube-solePSFs.fits to extract a klip reduced PSF.
        Note: If you want to include a spectrum it can be already included in the PSF when reducing it with klip.
            The other option if there is no psf available for example is to give a spectrum as an input.
    If no pyklip-*-KL*-speccube-solePSFs.fits is found it will for a pyklip-*-original_radial_PSF_cube.fits which is a PSF
    built from the sat spots but not klipped.

    /!\ The klipped PSF was not well understood by JB so anything related to it in pyklip will not work. It needs to be
    entirely "rethought" and implemented from scratch.

    Inputs:
        directory: directory in which the function will look for suitable fits files and run the planet detection algorithm.
        filename_prefix_is: Look for file containing of the form "/"+filename_filter+"-KL*-speccube.fits"
        numbasis: Integer. Apply algorithm only to klipped images that used numbasis KL modes.
        outputDir: Directory to where to save the output folder. By default it is saved in directory.
        spectrum_model: Mark Marley's spectrum filename. E.g. "/Users/jruffio/gpi/pyklip/t800g100nc.flx"
        star_type: Spectral type of the star (works only for type V). E.g. "G5" for G5V type.
        star_temperature: Temperature of the star. (replace star_type)
        threads: If true, parallel computation of several files (no prints in the console).
                Otherwise sequential with bunch of prints (for debugging).
        metrics_only: If True, compute only the metrics (Matched filter SNR, shape SNR, ...) but the planet detection
                algorithm is not applied.
        planet_detection_only: ????????
        mute: ???????????????

    Outputs:
        For each file an output folder is created:
            outputDir = directory + "/planet_detec_"+prefix+"_KL"+str(N_KL_modes)+"/"
        The outputs of the detection can be found there.

    '''

    if numbasis is not None:
        numbasis = str(numbasis)
    else:
        numbasis = '*'

    if filename_prefix_is == '':
        filelist_klipped_cube = glob.glob(directory+os.path.sep+"pyklip-*-KL"+numbasis+"-speccube.fits")
    else:
        filelist_klipped_cube = glob.glob(directory+os.path.sep+filename_prefix_is+"-KL"+numbasis+"-speccube.fits")
        #print(directory+"/"+filename_prefix_is+"-KL"+numbasis+"-speccube.fits")

    if len(filelist_klipped_cube) == 0:
        if not mute:
            print("No suitable files found in: "+directory)
    else:
        if not mute:
            print(directory+"contains suitable file for planet detection:")
            for f_name in filelist_klipped_cube:
                print(f_name)

        if threads:
            N_threads = np.size(filelist_klipped_cube)
            pool = NoDaemonPool(processes=N_threads)
            #pool = mp.Pool(processes=N_threads)
            pool.map(planet_detection_in_dir_per_file_star, itertools.izip(filelist_klipped_cube,
                                                                           itertools.repeat(metrics),
                                                                           itertools.repeat(directory),
                                                                           itertools.repeat(outputDir),
                                                                           itertools.repeat(spectrum_model),
                                                                           itertools.repeat(star_type),
                                                                           itertools.repeat(star_temperature),
                                                                           itertools.repeat(user_defined_PSF_cube),
                                                                           itertools.repeat(metrics_only),
                                                                           itertools.repeat(planet_detection_only),
                                                                           itertools.repeat(mute),
                                                                           itertools.repeat(threads),
                                                                           itertools.repeat(GOI_list),
                                                                           itertools.repeat(overwrite_metric),
                                                                           itertools.repeat(overwrite_stat),
                                                                           itertools.repeat(proba_using_mask_per_pixel),
                                                                           itertools.repeat(SNR),
                                                                           itertools.repeat(probability)))
        else:
            for filename in filelist_klipped_cube:
                planet_detection_in_dir_per_file(filename,
                                                 metrics = metrics,
                                                 directory = directory,
                                                 outputDir = outputDir,
                                                 spectrum_model = spectrum_model,
                                                 star_type = star_type,
                                                 star_temperature = star_temperature,
                                                 user_defined_PSF_cube = user_defined_PSF_cube,
                                                 metrics_only = metrics_only,
                                                 planet_detection_only = planet_detection_only,
                                                 mute = mute,
                                                 threads = False,
                                                 GOI_list = GOI_list,
                                                 overwrite_metric = overwrite_metric,
                                                 overwrite_stat = overwrite_stat,
                                                 proba_using_mask_per_pixel = proba_using_mask_per_pixel,
                                                 SNR = SNR,
                                                 probability = probability)





def planet_detection_campaign(campaign_dir = "."+os.path.sep):
    outputDir = ''
    star_type = ''
    metrics = None

    filename_filter = "pyklip-*-k100a7s4m3"
    numbasis = 20
    #spectrum_model = ["."+os.path.sep+"spectra"+os.path.sep+"t800g100nc.flx",""]
    if 1:
        spectrum_model = ["."+os.path.sep+"spectra"+os.path.sep+"g100ncflx"+os.path.sep+"t1500g100nc.flx",
                              "."+os.path.sep+"spectra"+os.path.sep+"g32ncflx"+os.path.sep+"t950g32nc.flx",
                              "."+os.path.sep+"spectra"+os.path.sep+"g32ncflx"+os.path.sep+"t600g32nc.flx",
                              ""]
    star_type = "G4"
    metrics = ["weightedFlatCube","matchedFilter","shape"]

    if 0:
        if platform.system() == "Windows":
            user_defined_PSF_cube = "C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\code\\pyklipH-S20141218-k100a7s4m3-original_radial_PSF_cube.fits"
        else:
            user_defined_PSF_cube = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB/code/pyklipH-S20141218-k100a7s4m3-original_radial_PSF_cube.fits"
    else:
        user_defined_PSF_cube = None

    if platform.system() == "Windows":
        GOI_list = "C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\GOI_list.txt"
    elif platform.system() == "Linux":
        GOI_list = "/home/sda/Dropbox (GPI)/SCRATCH/Scratch/JB/GOI_list.txt"
    else:
        GOI_list = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB/GOI_list.txt"

    inputDirs = []
    #for inputDir in os.listdir(campaign_dir):
    for inputDir in ["c_Eri"]:
        if not inputDir.startswith('.'):

            inputDirs.append(campaign_dir+inputDir+os.path.sep+"autoreduced"+os.path.sep)

            inputDir = campaign_dir+inputDir+os.path.sep+"autoreduced"+os.path.sep
            print(inputDir)
            if 1:
                planet_detection_in_dir(inputDir,
                                        filename_prefix_is=filename_filter,
                                        spectrum_model=spectrum_model,
                                        star_type=star_type,
                                        metrics = metrics,
                                        numbasis=numbasis,
                                        user_defined_PSF_cube=user_defined_PSF_cube,
                                        metrics_only = False,
                                        planet_detection_only = False,
                                        threads = True,
                                        mute = False,
                                        GOI_list = GOI_list,
                                        overwrite_metric=False,
                                        overwrite_stat=False,
                                        proba_using_mask_per_pixel = True,
                                        SNR = True,
                                        probability = True)

    if 0 and 0:
        N_threads = len(inputDirs)
        print(N_threads)
        pool = mp.Pool(processes=N_threads)
        #Check parameters
        #pool.map(planet_detection_in_dir_star, itertools.izip(inputDirs,
        #                                                               itertools.repeat(filename_filter),
        #                                                               itertools.repeat(numbasis),
        #                                                               itertools.repeat(outputDir),
        #                                                               itertools.repeat(spectrum_model),
        #                                                               itertools.repeat(star_type),
        #                                                               itertools.repeat(None),
        #                                                               itertools.repeat(user_defined_PSF_cube),
        #                                                               itertools.repeat(metrics),
        #                                                               itertools.repeat(False),
        #                                                               itertools.repeat(True),
        #                                                               itertools.repeat(False),
        #                                                               itertools.repeat(False)))

