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

import spectra_management as spec
from kpp_utils import *
from kpp_pdf import *
from kpp_std import *

def calculate_metrics(filename,
                        metrics = None,
                        PSF_cube = None,
                        outputDir = None,
                        folderName = None,
                        spectrum = None,
                        mute = False,
                        filename_no_planets = None,
                        SNR = True,
                        probability = True):
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
        if PSF_cube is not None:
            if np.size(np.shape(PSF_cube)) != 3:
                if not mute:
                    print("Wrong PSF dimensions. Image is 3D.")
                return 0
            # The PSF is user-defined.
            nl, ny_PSF, nx_PSF = PSF_cube.shape
            tmp_spectrum = np.nanmax(PSF_cube,axis=(1,2))
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
            PSF_cube = np.tile(PSF,(nl,1,1))

        if spectrum is not None:
            for k in range(nl):
                PSF_cube[k,:,:] *= spectrum[k]
        else:
            if PSF_cube is not None:
                spectrum = tmp_spectrum
            else:
                spectrum = np.ones(nl)

        spectrum /= np.sqrt(np.nansum(spectrum**2))
        # normalize PSF with norm 2.
        PSF_cube /= np.sqrt(np.sum(PSF_cube**2))

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

    # Calculate the standard deviation map.
    # the standard deviation is calculated on annuli of width Dr. There is an annulus centered every dr.
    #flat_cube_std = radialStdMap(flat_cube, centroid=center)
    flat_cube_std = std_function(flat_cube, centroid=center)

    # Divide the convolved flat cube by the standard deviation map to get the SNR.
    flat_cube_SNR = flat_cube/flat_cube_std

    if probability:
        if platform.system() == "Windows":
            GOI_list = "C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\GOI_list.txt"
        else:
            GOI_list = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB/GOI_list.txt"
        image = flat_cube
        image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
        IWA,OWA,inner_mask,outer_mask = get_occ(image, centroid = center)
        flat_cube_proba_map = get_image_probability_map(image,image_without_planet,(IWA,OWA),2000,centroid = center)

    if metrics is not None:
        if len(metrics) == 1 and not isinstance(metrics,list):
            metrics = [metrics]

        if "weightedFlatCube" in metrics:
            if not mute:
                print("Calculating weightedFlatCube for "+filename)
            weightedFlatCube = np.average(cube,axis=0,weights=spectrum)
            #weightedFlatCube_SNR = weightedFlatCube/radialStdMap(weightedFlatCube,dr,Dr, centroid=center)
            if SNR:
                weightedFlatCube_SNR = weightedFlatCube/std_function(weightedFlatCube,centroid=center)
            if probability:
                if platform.system() == "Windows":
                    GOI_list = "C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\GOI_list.txt"
                else:
                    GOI_list = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB/GOI_list.txt"
                image = weightedFlatCube
                image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
                IWA,OWA,inner_mask,outer_mask = get_occ(image, centroid = center)
                weightedFlatCube_proba_map = get_image_probability_map(image,image_without_planet,(IWA,OWA),2000,centroid = center)

        if "matchedFilter" in metrics and "shape" not in metrics:
            if not mute:
                print("Calculating matchedFilter (no shape) for "+filename)
            matchedFilter_map = np.ones((ny,nx)) + np.nan
                #ortho_criterion_map = np.zeros((ny,nx))
            row_m = np.floor(ny_PSF/2.0)
            row_p = np.ceil(ny_PSF/2.0)
            col_m = np.floor(nx_PSF/2.0)
            col_p = np.ceil(nx_PSF/2.0)

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

            #stdout.write("\r%d" % 0)
            for k,l in zip(flat_cube_wider_notNans[0],flat_cube_wider_notNans[1]):
                #stdout.flush()
                #stdout.write("\r%d" % k)

                stamp_cube = copy(cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)])
                for slice_id in range(nl):
                    stamp_cube[slice_id,:,:] -= np.nanmean(stamp_cube[slice_id,:,:]*stamp_PSF_mask)
                ampl = np.nansum(PSF_cube*stamp_cube)
                matchedFilter_map[k,l] = np.sign(ampl)*ampl**2

            if SNR:
                matchedFilter_SNR_map = matchedFilter_map/std_function(matchedFilter_map, centroid=center)
            if probability:
                if platform.system() == "Windows":
                    GOI_list = "C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\GOI_list.txt"
                else:
                    GOI_list = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB/GOI_list.txt"
                image = matchedFilter_SNR_map
                image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
                IWA,OWA,inner_mask,outer_mask = get_occ(image, centroid = center)
                matchedFilter_proba_map = get_image_probability_map(image,image_without_planet,(IWA,OWA),2000,centroid = center)

        if "shape" in metrics and "matchedFilter" not in metrics:
            if not mute:
                print("Calculating shape (no matchedFilter) for "+filename)
            shape_map = -np.ones((ny,nx)) + np.nan
                #ortho_criterion_map = np.zeros((ny,nx))
            row_m = np.floor(ny_PSF/2.0)
            row_p = np.ceil(ny_PSF/2.0)
            col_m = np.floor(nx_PSF/2.0)
            col_p = np.ceil(nx_PSF/2.0)

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

            #stdout.write("\r%d" % 0)
            for k,l in zip(flat_cube_wider_notNans[0],flat_cube_wider_notNans[1]):
                #stdout.flush()
                #stdout.write("\r%d" % k)

                stamp_cube = copy(cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)])
                for slice_id in range(nl):
                    stamp_cube[slice_id,:,:] -= np.nanmean(stamp_cube[slice_id,:,:]*stamp_PSF_mask)
                ampl = np.nansum(PSF_cube*stamp_cube)
                try:
                    shape_map[k,l] = np.sign(ampl)*ampl**2/np.nansum(stamp_cube**2)
                except:
                    shape_map[k,l] =  np.nan

            # criterion is here a cosine squared so we take the square root to get something similar to a cosine.
            shape_map = np.sign(shape_map)*np.sqrt(abs(shape_map))
            if SNR:
                shape_SNR_map = shape_map/std_function(shape_map, centroid=center)
            if probability:
                if platform.system() == "Windows":
                    GOI_list = "C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\GOI_list.txt"
                else:
                    GOI_list = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB/GOI_list.txt"
                image = matchedFilter_SNR_map
                image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
                IWA,OWA,inner_mask,outer_mask = get_occ(image, centroid = center)
                shape_proba_map = get_image_probability_map(image,image_without_planet,(IWA,OWA),2000,centroid = center)


        if "matchedFilter" in metrics and "shape" in metrics:
            if not mute:
                print("Calculating shape and matchedFilter for "+filename)
            matchedFilter_map = np.ones((ny,nx)) + np.nan
            shape_map = -np.ones((ny,nx)) + np.nan
                #ortho_criterion_map = np.zeros((ny,nx))
            row_m = np.floor(ny_PSF/2.0)
            row_p = np.ceil(ny_PSF/2.0)
            col_m = np.floor(nx_PSF/2.0)
            col_p = np.ceil(nx_PSF/2.0)

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

            #stdout.write("\r%d" % 0)
            for k,l in zip(flat_cube_wider_notNans[0],flat_cube_wider_notNans[1]):
                #stdout.flush()
                #stdout.write("\r%d" % k)

                stamp_cube = copy(cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)])
                for slice_id in range(nl):
                    stamp_cube[slice_id,:,:] -= np.nanmean(stamp_cube[slice_id,:,:]*stamp_PSF_mask)
                ampl = np.nansum(PSF_cube*stamp_cube)
                matchedFilter_map[k,l] = np.sign(ampl)*ampl**2
                try:
                    shape_map[k,l] = np.sign(ampl)*ampl**2/np.nansum(stamp_cube**2)
                except:
                    shape_map[k,l] =  np.nan

            shape_map = np.sign(shape_map)*np.sqrt(abs(shape_map))
            if SNR:
                #shape_SNR_map = shape_map/radialStdMap(shape_map,dr,Dr, centroid=center)
                shape_SNR_map = shape_map/std_function(shape_map,centroid=center)
                #matchedFilter_SNR_map = matchedFilter_map/radialStdMap(matchedFilter_map,dr,Dr, centroid=center)
                matchedFilter_SNR_map = matchedFilter_map/std_function(matchedFilter_map,centroid=center)
            if probability:
                if platform.system() == "Windows":
                    GOI_list = "C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\GOI_list.txt"
                else:
                    GOI_list = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB/GOI_list.txt"
                image = shape_map
                image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
                IWA,OWA,inner_mask,outer_mask = get_occ(image, centroid = center)
                shape_proba_map = get_image_probability_map(image,image_without_planet,(IWA,OWA),2000,centroid = center)
                image = matchedFilter_map
                image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
                IWA,OWA,inner_mask,outer_mask = get_occ(image, centroid = center)
                matchedFilter_proba_map = get_image_probability_map(image,image_without_planet,(IWA,OWA),2000,centroid = center)



    ## ortho_criterion is actually the sine squared between the two vectors
    ## ortho_criterion_map = 1 - criterion_map
    ## ratio_shape_SNR = 10
    ## criterion_map = np.minimum(ratio_shape_SNR*shape_map,flat_cube_SNR)

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
        if not mute:
            print("Saving metrics maps as: "+outputDir+folderName+prefix+'-#myMetric#.fits')
    except:
        prefix = "UNKNOWN_OBJECT"

    hdulist2 = pyfits.HDUList()
    try:
        hdulist2.append(pyfits.PrimaryHDU(header=prihdr))
        hdulist2.append(pyfits.ImageHDU(header=exthdr, data=flat_cube, name="Sci"))
        #hdulist2[1].data = flat_cube
        hdulist2.writeto(outputDir+folderName+prefix+'-flatCube.fits', clobber=True)
        hdulist2[1].data = flat_cube_SNR
        hdulist2.writeto(outputDir+folderName+prefix+'-flatCube_SNR.fits', clobber=True)
        if probability:
            hdulist2[1].data = flat_cube_proba_map
            hdulist2.writeto(outputDir+folderName+prefix+'-flatCube_proba.fits', clobber=True)

        if metrics is not None:
            if "weightedFlatCube" in metrics:
                hdulist2[1].data = weightedFlatCube
                hdulist2.writeto(outputDir+folderName+prefix+'-weightedFlatCube.fits', clobber=True)
                if SNR:
                    hdulist2[1].data = weightedFlatCube_SNR
                    hdulist2.writeto(outputDir+folderName+prefix+'-weightedFlatCube_SNR.fits', clobber=True)
                if probability:
                    hdulist2[1].data = weightedFlatCube_proba_map
                    hdulist2.writeto(outputDir+folderName+prefix+'-weightedFlatCube_proba.fits', clobber=True)
            if "matchedFilter" in metrics:
                hdulist2[1].data = matchedFilter_map
                hdulist2.writeto(outputDir+folderName+prefix+'-matchedFilter.fits', clobber=True)
                if SNR:
                    hdulist2[1].data = matchedFilter_SNR_map
                    hdulist2.writeto(outputDir+folderName+prefix+'-matchedFilter_SNR.fits', clobber=True)
                if probability:
                    hdulist2[1].data = matchedFilter_proba_map
                    hdulist2.writeto(outputDir+folderName+prefix+'-matchedFilter_proba.fits', clobber=True)
            if "shape" in metrics:
                hdulist2[1].data = shape_map
                hdulist2.writeto(outputDir+folderName+prefix+'-shape.fits', clobber=True)
                if SNR:
                    hdulist2[1].data = shape_SNR_map
                    hdulist2.writeto(outputDir+folderName+prefix+'-shape_SNR.fits', clobber=True)
                if probability:
                    hdulist2[1].data = shape_proba_map
                    hdulist2.writeto(outputDir+folderName+prefix+'-shape_proba.fits', clobber=True)
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
    hdulist2.close()

    # Save a plot of the spectrum used
    # todo /!\ Works only for H-Band. To be generalized.
    wave_step = 0.00841081142426 # in mum
    lambdaH0 = 1.49460536242  # in mum
    spec_sampling = np.arange(nl)*wave_step + lambdaH0
    plt.close(1)
    plt.figure(1)
    plt.plot(spec_sampling,spectrum,"rx-",markersize = 7, linewidth = 2)
    plt.title("Template spectrum use in this folder")
    plt.xlabel("Wavelength (mum)")
    plt.ylabel("Norm-2 Normalized")
    plt.savefig(outputDir+folderName+'template_spectrum.png', bbox_inches='tight')
    plt.close(1)

    return 1
# END calculate_metrics() DEFINITION

def candidate_detection(metrics_foldername,
                        mute = False,
                        confirm_candidates = None):
    '''

    Inputs:


    Outputs:

    '''
    shape_filename_list = glob.glob(metrics_foldername+os.path.sep+"*-shape_SNR.fits")
    if len(shape_filename_list) == 0:
        if not mute:
            print("Couldn't find shape_SNR map in "+metrics_foldername)
    else:
        hdulist = pyfits.open(shape_filename_list[0])

    #grab the data and headers
    try:
        criterion_map = hdulist[1].data
        criterion_map_cpy = copy(criterion_map)
        exthdr = hdulist[1].header
        prihdr = hdulist[0].header
    except:
        print("Couldn't read the fits file normally. Try another way.")
        criterion_map = hdulist[0].data
        prihdr = hdulist[0].header

    if np.size(criterion_map.shape) == 2:
        ny,nx = criterion_map.shape

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
        prefix = prihdr['OBJECT'].strip().replace (" ", "_")
    except:
        prefix = "UNKNOWN_OBJECT"

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

    # List of local maxima
    checked_spots_list = []
    # List of local maxima that are valid candidates
    candidates_list = []

    logFile_all = open(metrics_foldername+os.path.sep+prefix+'-detectionLog_all.txt', 'w')
    logFile_candidates = open(metrics_foldername+os.path.sep+prefix+'-detectionLog_candidates.txt', 'w')

    myStr = "# Log some values for each local maxima \n" +\
            "# Meaning of the columns from left to right. \n" +\
            "# 1/ Index \n" +\
            "# 2/ Boolean. True if the local maximum is a valid candidate. \n" +\
            "# 3/ Value of the criterion at this local maximum. \n"+\
            "# 4/ y coordinate of the maximum. \n"+\
            "# 5/ x coordinate of the maximum. \n"+\
            "# 6/ Row index of the maximum. (y-coord in DS9) \n"+\
            "# 7/ Column index of the maximum. (x-coord in DS9) \n"
    logFile_all.write(myStr)

    myStr2 = "# Log some values for each local maxima \n" +\
            "# Meaning of the columns from left to right. \n" +\
            "# 1/ Index \n" +\
            "# 2/ Value of the criterion at this local maximum. \n"+\
            "# 3/ y coordinate of the maximum. \n"+\
            "# 4/ x coordinate of the maximum. \n"+\
            "# 5/ Row index of the maximum. (y-coord in DS9) \n"+\
            "# 6/ Column index of the maximum. (x-coord in DS9) \n"
    logFile_candidates.write(myStr2)

    # Count the number of valid detected candidates.
    N_candidates = 0.0

    # Maximum number of iterations on local maxima.
    max_attempts = 60
                ## START FOR LOOP.
                ## Each iteration looks at one local maximum in the criterion map.
    k = 0
    max_val_criter = np.nanmax(criterion_map)
    while max_val_criter >= 2.0 and k <= max_attempts:
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

        myStr = str(k)+', '+\
                str(potential_planet)+', '+\
                str(max_val_criter)+', '+\
                str(x_max_pos)+', '+\
                str(y_max_pos)+', '+\
                str(row_id)+', '+\
                str(col_id)+'\n'
        logFile_all.write(myStr)

        checked_spots_list.append((k,potential_planet,max_val_criter,x_max_pos,y_max_pos, row_id,col_id))

        if potential_planet:
            myStr = str(k)+', '+\
                    str(max_val_criter)+', '+\
                    str(x_max_pos)+', '+\
                    str(y_max_pos)+', '+\
                    str(row_id)+', '+\
                    str(col_id)+'\n'
            logFile_candidates.write(myStr)

            # Increment the number of detected candidates
            N_candidates += 1
            # Append the useful things about the candidate in the list.
            candidates_list.append((k,max_val_criter,x_max_pos,y_max_pos, row_id,col_id))

    logFile_all.close()
    logFile_candidates.close()

    # Highlight the detected candidates in the criterion map
    if not mute:
        print("Number of candidates = " + str(N_candidates))
    plt.close(3)
    plt.figure(3,figsize=(16,16))
    plt.imshow(criterion_map_cpy[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
    ax = plt.gca()
    for candidate in candidates_list:
        candidate_it,max_val_criter,x_max_pos,y_max_pos, row_id,col_id = candidate

        ax.annotate(str(candidate_it)+","+"{0:02.1f}".format(max_val_criter), fontsize=20, color = "black", xy=(x_max_pos+0.0, y_max_pos+0.0),
                xycoords='data', xytext=(x_max_pos+10, y_max_pos-10),
                textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                linewidth = 1.,
                                color = 'black')
                )
        plt.clim(-4.,4.0)
        plt.savefig(metrics_foldername+os.path.sep+prefix+'-detectionIm_candidates.png', bbox_inches='tight')
    plt.close(3)

    plt.close(3)
    # Show the local maxima in the criterion map
    plt.figure(3,figsize=(16,16))
    plt.imshow(criterion_map_cpy[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
    ax = plt.gca()
    for spot in checked_spots_list:
        k,potential_planet,max_val_criter,x_max_pos,y_max_pos, row_id,col_id = spot
        ax.annotate(str(k)+","+"{0:02.1f}".format(max_val_criter), fontsize=20, color = 'black', xy=(x_max_pos+0.0, y_max_pos+0.0),
                xycoords='data', xytext=(x_max_pos+10, y_max_pos-10),
                textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                linewidth = 1.,
                                color = 'black')
                )
        plt.clim(-4.,4.0)
        plt.savefig(metrics_foldername+os.path.sep+prefix+'-detectionIm_all.png', bbox_inches='tight')
    plt.close(3)


def confirm_candidates(GOI_list_filename, logFilename_all, candidate_indices,candidate_status, object_name = None):
    with open(GOI_list_filename, 'a') as GOI_list:

        with open(logFilename_all, 'r') as logFile_all:
            for myline in logFile_all:
                if not myline.startswith("#"):
                    k,potential_planet,max_val_criter,x_max_pos,y_max_pos, row_id,col_id = myline.rstrip().split(",")
                    if int(k) in candidate_indices:
                        GOI_list.write(object_name+", "+candidate_status[np.where(np.array(candidate_indices)==int(k))[0]]+", "+myline)

    #"/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB/planet_detec_pyklip-S20141218-k100a7s4m3_KL20/t800g100nc/c_Eri-detectionLog_candidates.txt"

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
                                      mute = False):
    # Get the number of KL_modes for this file based on the filename *-KL#-speccube*
    splitted_name = filename.split("-KL")
    splitted_after_KL = splitted_name[1].split("-speccube.")
    N_KL_modes = int(splitted_after_KL[0])

    # Get the prefix of the filename
    splitted_before_KL = splitted_name[0].split(os.path.sep)
    prefix = splitted_before_KL[np.size(splitted_before_KL)-1]



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
        filelist_ori_PSFs_cube = glob.glob(directory+os.path.sep+prefix+"-original_radial_PSF_cube.fits")

    if np.size(filelist_ori_PSFs_cube) == 1:
        if not mute:
            print("I found a radially averaged PSF. I'll take it.")
        hdulist = pyfits.open(filelist_ori_PSFs_cube[0])
        #print(hdulist[1].data.shape)
        PSF_cube = hdulist[1].data[:,::-1,:]
        #PSF_cube = np.transpose(hdulist[1].data,(1,2,0))[:,::-1,:] #np.swapaxes(np.rollaxis(hdulist[1].data,2,1),0,2)
        sat_spot_spec = np.nanmax(PSF_cube,axis=(1,2))
        sat_spot_spec_exist = True
        # Remove the spectral shape from the psf cube because it is dealt with independently
        for l_id in range(PSF_cube.shape[0]):
            PSF_cube[l_id,:,:] /= sat_spot_spec[l_id]
    elif np.size(filelist_ori_PSFs_cube) == 0:
        if not mute:
            print("I didn't find any PSFs file so I will use a default gaussian. Default sat spot spectrum = filter spectrum.")
        wv,sat_spot_spec = spec.get_gpi_filter(filter)
        sat_spot_spec_exist = False
        PSF_cube = None
    else:
        if not mute:
            print("I found several *-original_radial_PSF_cube.fits so I don't know what to do and I quit")
        return 0

    prefix = prefix+"_KL"+str(N_KL_modes)


    if len(spectrum_model) == 1 and not isinstance(spectrum_model,list):
        spectrum_model =[spectrum_model]
    else:
        if not mute:
            print("Iterating over several model spectra")

    for spectrum_name_it in spectrum_model:

        # Define the output Foldername
        if spectrum_name_it != "":
            spectrum_name = spectrum_name_it.split(os.path.sep)
            spectrum_name = spectrum_name[len(spectrum_name)-1].split(".")[0]
        else:
            spectrum_name = "satSpotSpec"

        if outputDir == '':
            outputDir = directory
        folderName = os.path.sep+"planet_detec_"+prefix+os.path.sep+spectrum_name+os.path.sep



        if spectrum_name_it != "":
            if not mute:
                print("spectrum model: "+spectrum_name_it)
            # Interpolate the spectrum of the planet based on the given filename
            wv,planet_sp = spec.get_planet_spectrum(spectrum_name_it,filter)

            if sat_spot_spec_exist and (star_type !=  "" or star_temperature is not None):
                # Interpolate a spectrum of the star based on its spectral type/temperature
                wv,star_sp = spec.get_star_spectrum(filter,star_type,star_temperature)
                spectrum = (sat_spot_spec/star_sp)*planet_sp
            else:
                if not mute:
                    print("No star spec or sat spot spec so using sole planet spectrum.")
                spectrum = planet_sp
        else:
            spectrum = sat_spot_spec

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
                                SNR = True,
                                probability = True)


        if not metrics_only:
            if not mute:
                print("Calling candidate_detection() on "+outputDir+folderName)
            if 1:
                candidate_detection(outputDir+folderName,
                                    mute = mute)

def planet_detection_in_dir_per_file_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call planet_detection_in_dir_per_file() with a tuple of parameters.
    """
    return planet_detection_in_dir_per_file(*params)

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
                            mute = True):
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

        if 1:
            if threads:
                N_threads = np.size(filelist_klipped_cube)
                pool = mp.Pool(processes=N_threads)
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
                                                                               itertools.repeat(mute)))
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
                                                     mute = mute)





def planet_detection_campaign(campaign_dir = "."+os.path.sep):
    outputDir = ''
    star_type = ''
    metrics = None

    filename_filter = "pyklip-*-k100a7s4m3"
    numbasis = 20
    spectrum_model = ["."+os.path.sep+"spectra"+os.path.sep+"t800g100nc.flx",""]
    if 0:
        spectrum_model = ["."+os.path.sep+"spectra"+os.path.sep+"t800g100nc.flx",
                          "."+os.path.sep+"spectra"+os.path.sep+"t700g178nc.flx",
                          "."+os.path.sep+"spectra"+os.path.sep+"t650g18nc.flx",
                          "."+os.path.sep+"spectra"+os.path.sep+"t650g32nc.flx",
                          "."+os.path.sep+"spectra"+os.path.sep+"t650g56nc.flx",
                          ""]
        spectrum_model = ["/Users/jruffio/gpi/pyklip/spectra/t650g18nc.flx",
                          "/Users/jruffio/gpi/pyklip/spectra/t650g32nc.flx",
                          "/Users/jruffio/gpi/pyklip/spectra/t650g56nc.flx"]
    star_type = "G4"
    metrics = ["weightedFlatCube","matchedFilter","shape"]

    if platform.system() == "Windows":
        user_defined_PSF_cube = "C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\code\\pyklipH-S20141218-k100a7s4m3-original_radial_PSF_cube.fits"
    else:
        user_defined_PSF_cube = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB/code/pyklipH-S20141218-k100a7s4m3-original_radial_PSF_cube.fits"


    inputDirs = []
    for inputDir in os.listdir(campaign_dir):
        if not inputDir.startswith('.'):
            inputDirs.append(campaign_dir+inputDir+os.path.sep+"autoreduced"+os.path.sep)

            if 1:
                inputDir = campaign_dir+inputDir+os.path.sep+"autoreduced"+os.path.sep
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
                                        mute = True)

    if 0:
        N_threads = len(inputDirs)
        print(N_threads)
        pool = mp.Pool(processes=N_threads)
        pool.map(planet_detection_in_dir_star, itertools.izip(inputDirs,
                                                                       itertools.repeat(filename_filter),
                                                                       itertools.repeat(numbasis),
                                                                       itertools.repeat(outputDir),
                                                                       itertools.repeat(spectrum_model),
                                                                       itertools.repeat(star_type),
                                                                       itertools.repeat(None),
                                                                       itertools.repeat(user_defined_PSF_cube),
                                                                       itertools.repeat(metrics),
                                                                       itertools.repeat(False),
                                                                       itertools.repeat(False),
                                                                       itertools.repeat(False),
                                                                       itertools.repeat(True)))

