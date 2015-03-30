import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from scipy.signal import convolve
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
from astropy import wcs
import glob
import os
import sys
import cmath
import time
from scipy.interpolate import griddata
from scipy.interpolate import bisplrep
from scipy.interpolate import bisplev
from sys import stdout
import numexpr as ne


import spectra_management as spec


def extract_PSFs(filename, stamp_width = 10, mute = False):
    '''
    Extract the PSFs in a pyklip reduced cube in which fake planets have been injected.
    The position of the fake planets is stored in the headers when pyklip is used.
    A cube stamp is extracted for each radius and angle and they are all sorted in the out_stamp_PSFs array.

    /!\ The cube is normalized following cube /= np.nanstd(cube[10,:,:])
    This is because I don't want the very small numbers in Jason's output as he uses contrast units

    :param filename: Name of the file from which the fake planets cube stamps should be extracted.
    :param stamp_width: Spatial width of the stamps to be extracted around each fake planets.
    :param mute: If true print some text in the console.
    :return out_stamp_PSFs: A (nl,stamp_width,stamp_width,nth,nr) array with nl the number of wavelength of the cube,
            nth the number of section in the klip reduction and nr the number of annuli. Therefore the cube defined by
            out_stamp_PSFs[:,:,:,0,2] is a cube stamp of the planet in the first section of the third annulus. In order
            to know what position it exactly corresponds to please look at the FAKPLPAR keyword in the primary headers.
    '''
    hdulist = pyfits.open(filename)
    cube = hdulist[1].data
    cube /= np.nanstd(cube[10,:,:])
    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        cube = cube[None,:]
        nl = 1
    #slice = hdulist[1].data[2,:,:]
    #ny,nx = slice.shape
    prihdr = hdulist[0].header
    exthdr = hdulist[1].header

    try:
        # Retrieve the center of the image from the fits keyword.
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    except:
        # If the keywords could not be found.
        if not mute:
            print("Couldn't find PSFCENTX and PSFCENTY keywords.")
        center = [(nx-1)/2,(ny-1)/2]

    try:
        # Retrieve the position of the fake planets from the fits keyword.
        fakePlparams_str = prihdr['FAKPLPAR']
    except:
        # If the keywords could not be found.
        if not mute:
            print("ERROR. Couldn't find FAKPLPAR (Fake planets parameters) keyword. Has to quit extract_PSFs().")
        return 0

    fakePlparams_splitted_str = fakePlparams_str.split(";")
    planet_angles_str = fakePlparams_splitted_str[6]
    planet_radii_str = fakePlparams_splitted_str[7]
    planet_angles =eval(planet_angles_str.split("=")[1])
    planet_radii =eval(planet_radii_str.split("=")[1])

    nth = np.size(planet_angles)
    nr = np.size(planet_radii)

    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    #x.shape = (x.shape[0] * x.shape[1])
    #y.shape = (y.shape[0] * y.shape[1])
    x -= center[0]
    y -= center[1]
    x_planets = np.dot(np.array([planet_radii]).transpose(),np.array([np.cos(np.radians(planet_angles))]))
    y_planets = np.dot(np.array([planet_radii]).transpose(),np.array([np.sin(np.radians(planet_angles))]))
    #print(x_planets+center[0])
    #print(y_planets+center[1])

    out_stamp_PSFs = np.zeros((nl,stamp_width,stamp_width,nth,nr))

    for l_id in range(nl):
        for r_id,r_it in enumerate(planet_radii):
            for th_id, th_it in enumerate(planet_angles):
                x_plnt = np.round(x_planets[r_id,th_id]+center[0])
                y_plnt = np.round(y_planets[r_id,th_id]+center[1])

                out_stamp_PSFs[l_id,:,:,th_id,r_id] = cube[l_id,
                                                            (y_plnt-np.floor(stamp_width/2.)):(y_plnt+np.ceil(stamp_width/2.)),
                                                            (x_plnt-np.floor(stamp_width/2.)):(x_plnt+np.ceil(stamp_width/2.))]

        #print(l_id,r_id,r_it,th_id, th_it,x_plnt,y_plnt)
        #plt.imshow(out_stamp_PSFs[:,:,l_id,th_id,r_id],interpolation = 'nearest')
        #plt.show()

    return out_stamp_PSFs

def extract_merge_PSFs(filename, radii, thetas, stamp_width = 10):
    hdulist = pyfits.open(filename)
    cube = hdulist[1].data
    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        cube = cube[None,:]
        nl = 1
    nth = np.size(thetas)
    nr = np.size(radii)
    #slice = hdulist[1].data[2,:,:]
    #ny,nx = slice.shape
    prihdr = hdulist[0].header
    exthdr = hdulist[1].header
    try:
        # Retrieve the center of the image from the fits keyword.
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    except:
        # If the keywords could not be found.
        center = [(nx-1)/2,(ny-1)/2]

    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    #x.shape = (x.shape[0] * x.shape[1])
    #y.shape = (y.shape[0] * y.shape[1])
    x -= center[0]
    y -= center[1]
    x_planets = np.dot(np.array([radii]).transpose(),np.array([np.cos(np.radians(thetas))]))
    y_planets = np.dot(np.array([radii]).transpose(),np.array([np.sin(np.radians(thetas))]))

    out_stamp_PSFs = np.zeros((ny,nx,nth,nr,nl))

    dn = stamp_width
    for l_id in range(nl):
        for r_id,r_it in enumerate(radii):
            for th_id, th_it in enumerate(thetas):
                x_plnt = np.ceil(x_planets[r_id,th_id]+center[0])
                y_plnt = np.ceil(y_planets[r_id,th_id]+center[1])

                stamp = cube[l_id,(y_plnt-dn/2):(y_plnt+dn/2),(x_plnt-dn/2):(x_plnt+dn/2)]
                stamp_x, stamp_y = np.meshgrid(np.arange(dn, dtype=np.float32), np.arange(dn, dtype=np.float32))
                stamp_x += (x_planets[r_id,th_id]+center[0]-x_plnt)
                stamp_y += y_planets[r_id,th_id]+center[1]-y_plnt
                stamp = ndimage.map_coordinates(stamp, [stamp_y, stamp_x])

                #plt.imshow(stamp,interpolation = 'nearest')
                #plt.show()
                #return

                if k == 0 and l == 0:
                    PSFs_stamps = [stamp]
                else:
                    PSFs_stamps = np.concatenate((PSFs_stamps,[stamp]),axis=0)

    return np.mean(PSFs_stamps,axis=0)

def subtract_radialMed(image,w,l,center):
    ny,nx = image.shape

    n_area_x = np.floor(nx/w)
    n_area_y = np.floor(ny/w)

    x, y = np.meshgrid(np.arange(nx)-center[0], np.arange(ny)-center[1])
    r_grid = abs(x +y*1j)
    th_grid = np.arctan2(x,y)

    for p in np.arange(n_area_x):
        for q in np.arange(n_area_y):
            #stamp = image[(q*w):((q+1)*w),(p*w):((p+1)*w)]
            #image[(q*w):((q+1)*w),(p*w):((p+1)*w)] -= nanmedian(stamp)

            r = r_grid[((q+0.5)*w),((p+0.5)*w)]
            th = th_grid[((q+0.5)*w),((p+0.5)*w)]

            arc_id = np.where(((r-w/2.0) < r_grid) * (r_grid < (r+w/2.0)) * ((th-l/r) < th_grid) * (th_grid < (th+l/r)))
            image[(q*w):((q+1)*w),(p*w):((p+1)*w)] -= nanmedian(image[arc_id])

            if 0 and p == 50 and q == 50:
                image[arc_id] = 100
                print(image[arc_id].size)
                plt.figure(2)
                plt.imshow(image, interpolation="nearest")
                plt.show()


    return image


def radialStd(cube,dr,Dr,centroid = None, rejection = False):
    '''
    Return the standard deviation with respect to the radius on an image.
    It cuts annuli of radial width Dr separated by dr and calculate the standard deviation in them.

    Note: This function should work on cubes however JB has never really used it... So no guarantee

    Inputs:
        cube: 2D (ny,nx) or 3D (nl,ny,nx) array in which to calculate
        dr: Sampling of the radial axis for calculating the standard deviation
        Dr: width of the annuli used to calculate the standard deviation
        centroid: [col,row] with the coordinates of the center of the image.
        rejection: reject 3 sigma values

    Outputs:
        radial_std: Vector containing the standard deviation for each radii
        r: Map with the distance of each pixel to the center.
        r_samp: Radial sampling used for radial_std based on dr.

    :return:
    '''
    # Get dimensions of the image
    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        cube = cube[None,:]
        nl = 1

    # Get the center of the image
    #TODO centroid should be different for each slice?
    if centroid is None :
        x_cen = np.ceil((nx-1)/2) ; y_cen = np.ceil((ny-1)/2)
    else:
        x_cen, y_cen = centroid

    # Build the x and y coordinates grids
    x, y = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)

    # Calculate the radial distance of each pixel
    r = abs(x +y*1j)
    r_samp = np.arange(0,max(r.reshape(np.size(r))),dr)

    # Declare the output array that will contain the std values
    radial_std = np.zeros((nl,np.size(r_samp)))


    for r_id, r_it in enumerate(r_samp):
        # Get the coordinates of the pixels inside the current annulus
        selec_pix = np.where( ((r_it-Dr/2.0) < r) * (r < (r_it+Dr/2.0)) )
        selec_y, selec_x = selec_pix
        # Extract the pixels inside the current annulus
        data = cube[:,selec_y, selec_x]
        for l_it in np.arange(nl):
            data_slice = data[l_it,:]
            # Check that everything is not Nan
            if np.size(np.where(np.isnan(data_slice))) != data_slice.size:
                # 3 sigma rejection if required
                if rejection:
                    mn_data = np.nanmean(data_slice)
                    std_data = np.nanstd(data_slice)
                    # Remove nans from the array but ensure they be put back afterwards
                    data_slice[np.where(np.isnan(data_slice))] = mn_data + 10 * std_data
                    out_3sig = np.where((data_slice-mn_data)>(3.*std_data))
                    if out_3sig is not ():
                        data[l_it,out_3sig] = np.nan

                # Calculate the standard deviation on the annulus.
                radial_std[l_it,r_id] = np.nanstd(data[l_it,:])
            else:
                radial_std[l_it,r_id] = np.nan

    # Remove singleton dimension if the input is 2D.
    radial_std = np.squeeze(radial_std)

    return radial_std,r_samp,r

def radialStdMap(cube,dr,Dr,centroid = None, rejection = False,treshold=10**(-6)):
    '''
    Return a cube of the same size as input cube with the radial standard deviation value. The pixels falling at same
    radius will have the same value.

    Inputs:
        cube: 2D (ny,nx) or 3D (nl,ny,nx) array in which to calculate
        dr: Sampling of the radial axis for calculating the standard deviation
        Dr: width of the annuli used to calculate the standard deviation
        centroid: [col,row] with the coordinates of the center of the image.
        rejection: reject 3 sigma values
        treshold: Set to nans the values of cube below the treshold.

    Output:
        cube_std: The standard deviation map.
    '''

    # Get the standard deviation function of radius
    radial_std,r_samp,r = radialStd(cube,dr,Dr,centroid,rejection = rejection)

    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        radial_std = radial_std[None,:]
        nl = 1

    # Interpolate the standard deviation function on each point of the image.
    cube_std = np.zeros((nl,ny,nx))
    radial_std_nans = np.isnan(radial_std)
    radial_std[radial_std_nans] = 0.0
    for l_id in np.arange(nl):
        #f = interp1d(r_samp, radial_std[l_id,:], kind='cubic',bounds_error=False, fill_value=np.nan)
        #a = f(r[0].reshape(nx*ny))
        f = interp1d(r_samp, radial_std[l_id,:], kind='cubic',bounds_error=False, fill_value=np.nan)
        a = f(r.reshape(nx*ny))
        cube_std[l_id,:,:] = a.reshape(ny,nx)

    # Remove nans from the array but ensure they be put back afterwards
    cube_std[np.where(np.isnan(cube_std))] = 0.0
    cube_std[np.where(cube_std < treshold)] = np.nan

    # Remove singleton dimension if the input is 2D.
    cube_std = np.squeeze(cube_std)

    return cube_std

def gauss2d(x, y, amplitude = 1.0, xo = 0.0, yo = 0.0, sigma_x = 1.0, sigma_y = 1.0, theta = 0, offset = 0):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g


def candidate_detection(filename,
                        PSF = None,
                        outputDir = None,
                        folderName = None,
                        toPNG='',
                        toFits='',
                        toDraw=False,
                        logFile='',
                        spectrum = None,
                        mute = False ):
    '''
    Should take into account PSF wavelength dependence.
    3d convolution to take into account spectral shift if useful
    but 3d conv takes too long

    Inputs:
        filename: Path and name of the fits file to be analyzed.
        PSF: User-defined 2D PSF. If None, gaussian PSF is assumed.
        allmodes:

        outputDir: Directory where to save the outputs
        toPNG: Save some plots as PNGs. toPNG must be a string being a prefix of the filename of the images.
        toFits: Save some fits files in memory. toFits must be a string being a prefix of the filename of the images.
        toDraw: Plot some figures using matplotlib.pyplot. First a SNR map with the candidate list and a criterion map
                with all the checked spots. toDraw is a string ie the prefix of the filename.
        logFile: Log the result of the detection in text files. logFile is a string ie the prefix of the filename.

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
        if PSF is not None:
            PSF_cube = PSF
            if np.size(np.shape(PSF)) != 3:
                if not mute:
                    print("Wrong PSF dimensions. Image is 3D.")
                return 0
            # The PSF is user-defined.
            nl, ny_PSF, nx_PSF = PSF_cube.shape
            tmp_spectrum = np.nanmax(PSF_cube,axis=(1,2))
        else:
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
            if PSF is not None:
                spectrum = tmp_spectrum
            else:
                spectrum = np.ones(nl)

        spectrum /= np.sqrt(np.nansum(spectrum**2))
        # normalize PSF with norm 2.
        PSF_cube /= np.sqrt(np.sum(PSF_cube**2))

    else: # Assuming 2D image
        flat_cube = cube

        # Build the PSF.
        if PSF is not None:
            if np.size(np.shape(PSF)) != 2:
                if not mute:
                    print("Wrong PSF dimensions. Image is 2D.")
                return 0
            # The PSF is user-defined.
            # normalize PSF with norm 2.
            PSF /= np.sqrt(np.sum(PSF**2))
            ny_PSF, nx_PSF = PSF.shape
        else:
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


    try:
        # Retrieve the center of the image from the fits keyword.
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    except:
        # If the keywords could not be found.
        if not mute:
            print("Couldn't find PSFCENTX and PSFCENTY keywords.")
        center = [(nx-1)/2,(ny-1)/2]

    candidates_KLs_list = []


    # Smoothing of the image. Remove the median of an arc centered on each pixel.
    # Actually don't do pixel per pixel but consider small boxes.
    # This function has to be cleaned.
    #flat_cube = subtract_radialMed(flat_cube,2,20,center)
    flat_cube_cpy = copy(flat_cube)
    flat_cube_nans = np.where(np.isnan(flat_cube))


    # Build as grids of x,y coordinates.
    # The center is in the middle of the array and the unit is the pixel.
    # If the size of the array is even 2n x 2n the center coordinates is [n,n].
    x_grid, y_grid = np.meshgrid(np.arange(0,nx,1)-center[0],np.arange(0,ny,1)-center[1])

            # Replace nans by zeros.
            # Otherwise we loose the border of the image because of the convolution which will give NaN if there is any NaNs in
            # the area.
            # /!\ Desactivated because there is no hope in real life to get anything there anyway. Only for Baade's window...
            #flat_cube[np.where(np.isnan(flat_cube))] = 0.0
            #flat_cube = copy(flat_cube_cpy)

            # Perform a "match-filtering". Simply the convolution of the transposed PSF with the image.
            # It should still be zero if there is no signal. Assuming a zero mean noise after KLIP.
            # The value at a point centered on a planet should be the L2 norm of the planet.
            # /!\ Desactivated because matched filtering doesn't work on correlated images.
            #flat_cube_convo = convolve2d(flat_cube,PSF,mode="same")
            # The 3d convolution takes a while so the idea is to detect the interesting spot in the 2d flat cube and then
            # perform the 3d convolution on the cube stamp around it.

    # Calculate the standard deviation map.
    # the standard deviation is calculated on annuli of width Dr. There is an annulus centered every dr.
    dr = 2 ; Dr = 5 ;
    flat_cube_std = radialStdMap(flat_cube,dr,Dr, centroid=center)

    # Divide the convolved flat cube by the standard deviation map to get the SNR.
    flat_cube_SNR = flat_cube/flat_cube_std


    # Definition of the different masks used in the following.
    stamp_nrow = 13
    stamp_ncol = 13
    # Mask to remove the spots already checked in criterion_map.
    stamp_x_grid, stamp_y_grid = np.meshgrid(np.arange(0,stamp_nrow,1)-6,np.arange(0,stamp_ncol,1)-6)
    stamp_mask = np.ones((stamp_nrow,stamp_ncol))
    r_stamp = abs((stamp_x_grid) +(stamp_y_grid)*1j)
    stamp_mask[np.where(r_stamp < 4.0)] = np.nan
    stamp_mask_small = np.ones((stamp_nrow,stamp_ncol))
    stamp_mask_small[np.where(r_stamp < 2.0)] = 0.0
    stamp_cube_small_mask = np.tile(stamp_mask_small[None,:,:],(nl,1,1))




    shape_crit_map = -np.ones((ny,nx))
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
    if nl !=1:
        stamp_PSF_x_grid, stamp_PSF_y_grid = np.meshgrid(np.arange(0,nx_PSF,1)-nx_PSF/2,np.arange(0,ny_PSF,1)-ny_PSF/2)
        stamp_PSF_mask = np.ones((nl,ny_PSF,nx_PSF))
        r_PSF_stamp = abs((stamp_PSF_x_grid) +(stamp_PSF_y_grid)*1j)
        #r_PSF_stamp = np.tile(r_PSF_stamp,(nl,1,1))
        stamp_PSF_mask[np.where(r_PSF_stamp < 2.5)] = np.nan

        #plt.figure(1)
        #plt.imshow(stamp_PSF_mask[5,:,:], interpolation="nearest")
        #plt.show()

        stdout.write("\r%d" % 0)
        for k in np.arange(10,ny-10):
            stdout.flush()
            stdout.write("\r%d" % k)

            for l in np.arange(10,nx-10):
                stamp_cube = copy(cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)])
                for slice_id in range(nl):
                    stamp_cube[slice_id,:,:] -= np.nanmean(stamp_cube[slice_id,:,:]*stamp_PSF_mask)
                ampl = np.nansum(PSF_cube*stamp_cube)
                if ampl != 0.0:
                    square_norm_stamp = np.nansum(stamp_cube**2)
                    shape_crit_map[k,l] = np.sign(ampl)*ampl**2/square_norm_stamp
                    #shape_crit_map[k,l] = ampl
                else:
                    shape_crit_map[k,l] = -1.0
                #ortho_criterion_map[k,l] = np.nansum((stamp_cube-ampl*PSF_cube)**2)/square_norm_stamp
    else:
        print("planet detection Not ready for 2D images to be corrected first")
        return
        for k in np.arange(10,ny-10):
            for l in np.arange(10,nx-10):
                stamp = cube[(k-row_m):(k+row_p), (l-col_m):(l+col_p)]
                ampl = np.nansum(PSF*stamp)
                if ampl != 0.0:
                    square_norm_stamp = np.nansum(stamp**2)
                    shape_crit_map[k,l] = np.sign(ampl)*ampl**2/square_norm_stamp
                else:
                    shape_crit_map[k,l] = -1.0
                #ortho_criterion_map[k,l] = np.nansum((stamp_cube-ampl*PSF_cube)**2)/square_norm_stamp


        ## ortho_criterion is actually the sine between the two vectors
        # ortho_criterion_map = 1 - criterion_map
    # criterion is here a cosine squared so we take the square root to get something similar to a cosine.
    shape_crit_map = np.sign(shape_crit_map)*np.sqrt(abs(shape_crit_map))
    #Save a copy of the flat cube because we will mask the detected spots as the algorithm goes.

    ratio_shape_SNR = 10

    #criterion_map = np.minimum(ratio_shape_SNR*shape_crit_map,flat_cube_SNR)
    criterion_map = shape_crit_map/radialStdMap(shape_crit_map,dr,Dr, centroid=center)
    #criterion_map = flat_cube_SNR
    criterion_map[flat_cube_nans] = np.nan
    #criterion_map /= radialStdMap(flat_cube,dr,Dr, centroid=center)
    criterion_map_cpy = copy(criterion_map)

    # List of local maxima
    checked_spots_list = []
    # List of local maxima that are valid candidates
    candidates_list = []

    if outputDir is None:
        outputDir = "./"
    else:
        outputDir = outputDir+"/"


    if folderName is None:
        folderName = "/default_out/"
    else:
        folderName = folderName+"/"

    if not os.path.exists(outputDir+folderName): os.makedirs(outputDir+folderName)

    if logFile:
        logFile_all = open(outputDir+folderName+logFile+'-log_allLocalMax.txt', 'w')
        logFile_candidates = open(outputDir+folderName+logFile+'-log_candidates.txt', 'w')

        myStr = "# Log some values for each local maxima \n" +\
                "# Meaning of the columns from left to right. \n" +\
                "# 1/ Index \n" +\
                "# 2/ Boolean. True if the local maximum is a valid candidate. \n" +\
                "# 3/ Value of the criterion at this local maximum. \n"+\
                "# 3/ Value of the shape criterion at this local maximum. \n"+\
                "# 3/ Value of the SNR at this local maximum. \n"+\
                "# 4/ Error check of the gaussian fit. \n"+\
                "# 5/ Distance between the fitted centroid and the original max position. \n"+\
                "# 6/ x-axis width of the gaussian. \n"+\
                "# 7/ y-axis width of the gaussian. \n"+\
                "# 8/ Amplitude of the gaussian. \n"+\
                "# 9/ Row index of the maximum. (y-coord in DS9) \n"+\
                "# 10/ Column index of the maximum. (x-coord in DS9) \n"
        logFile_all.write(myStr)

        myStr2 = "# Log some values for each valid candidates. \n" +\
                "# Meaning of the columns from left to right. \n" +\
                "# 1/ Index \n" +\
                "# 2/ Boolean. True if the centroid of the maximum is stable-ish. \n" +\
                "# 3/ Value of the criterion at this local maximum. \n"+\
                "# 3/ Value of the shape criterion at this local maximum. \n"+\
                "# 3/ Value of the SNR at this local maximum. \n"+\
                "# 4/ Error check of the gaussian fit. \n"+\
                "# 5/ Distance between the fitted centroid and the original max position. \n"+\
                "# 6/ x-axis width of the gaussian. \n"+\
                "# 7/ y-axis width of the gaussian. \n"+\
                "# 8/ Amplitude of the gaussian. \n"+\
                "# 9/ Row index of the maximum. (y-coord in DS9) \n"+\
                "# 10/ Column index of the maximum. (x-coord in DS9) \n"
        logFile_candidates.write(myStr2)


    flat_cube[flat_cube_nans] = 0.0

    # Count the number of valid detected candidates.
    N_candidates = 0.0

    # Maximum number of iterations on local maxima.
    max_attempts = 60
                ## START FOR LOOP.
                ## Each iteration looks at one local maximum in the criterion map.
                ## Then it verifies some other criteria to check if it is worse looking at.
                #for k in np.arange(max_attempts):
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

        # Check if the maximum is next to a nan. If so it should not be considered.
        if not np.isnan(np.sum(criterion_map[(row_id-1):(row_id+2), (col_id-1):(col_id+2)])):
            valid_potential_planet = max_val_criter > 3.0
        else:
            valid_potential_planet = False

        #Extract a stamp around the maximum in the flat cube (without the convolution)
        row_m = np.floor(stamp_nrow/2.0)
        row_p = np.ceil(stamp_nrow/2.0)
        col_m = np.floor(stamp_ncol/2.0)
        col_p = np.ceil(stamp_ncol/2.0)
        stamp = copy(flat_cube[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)])
        stamp_SNR = copy(flat_cube_SNR[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)])
        stamp_x_grid = x_grid[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
        stamp_y_grid = y_grid[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
        max_val_SNR = np.nanmax(flat_cube_SNR[(row_id-2):(row_id+3), (col_id-2):(col_id+3)])

        stamp[np.where(np.isnan(stamp))] = 0.0

        # Definition of a 2D gaussian fitting to be used on the stamp.
        g_init = models.Gaussian2D(max_val_SNR,x_max_pos,y_max_pos,1.5,1.5)
        fit_g = fitting.LevMarLSQFitter()
        # Fit the 2d Gaussian to the stamp
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        g = fit_g(g_init, stamp_x_grid, stamp_y_grid, stamp,np.abs(stamp)**0.5)

            # Calculate the fitting residual using the square root of the summed squared error.
            #ampl = np.nansum(stamp*g(stamp_x_grid, stamp_y_grid))
            #khi = np.sign(ampl)*ampl**2/(np.nansum(stamp**2)*np.nansum(g(stamp_x_grid, stamp_y_grid)**2))

        # JB Todo: Should explore the meaning of 'ierr' but I can't find a good clear documentation of astropy.fitting
        sig_min = 1.0 ; sig_max = 3.0 ;
        # The condition for a local maximum to be considered as a candidate are:
        #       - Positive criterion. Always verified because we are looking at maxima...
        #       - Reasonable SNR. ie greater than one.
        #         I prefer to be conservative on the SNR because we never know and I think it isn't the best criterion.
        #       - The gaussian fit had to succeed.
        #       - Reasonable width of the gaussian. Not wider than 3.5pix and not smaller than 0.5pix in both axes.
        #       - Centroid of the Gaussian fit not too far from the center of the stamp.
        #       - Amplitude of the Gaussian fit should be positive.
        valid_gaussian = (flat_cube_SNR[row_id,col_id] > 1.0 and
                                 #fit_g.fit_info['ierr'] <= 3 and
                                 sig_min < g.x_stddev < sig_max and
                                 sig_min < g.y_stddev < sig_max and
                                 np.sqrt((g.x_mean-x_max_pos)**2+(g.y_mean-y_max_pos)**2)<1.5 and
                                 g.amplitude > 0.0)
                                        #khi/khi0 < 1.0 and # Check that the fit was good enough. Not a weird looking speckle.

        #fit_g.fit_info['ierr'] == 1 and # Check that the fitting actually occured. Actually I have no idea what the number mean but it looks like when it succeeds it is 1.

        # If the spot verifies the conditions above it is considered as a valid candidate.
        checked_spots_list.append((k,row_id,col_id,max_val_criter,max_val_SNR,x_max_pos,y_max_pos,g))

        # Mask the spot around the maximum we just found.
        criterion_map[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] *= stamp_mask
        flat_cube[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] *= stamp_mask

        # Todo: Remove prints, for debugging purpose only
        if not mute:
            print(k,row_id,col_id,max_val_criter,ratio_shape_SNR*shape_crit_map[row_id,col_id],max_val_SNR, g.x_stddev+0.0, g.y_stddev+0.0,g.x_mean-x_max_pos,g.y_mean-y_max_pos,flat_cube_SNR[row_id,col_id])
        if k == 79 or 0:
            plt.figure(1)
            plt.imshow(stamp, interpolation="nearest")
            plt.figure(2)
            plt.imshow(g(stamp_x_grid, stamp_y_grid), interpolation="nearest")
            plt.figure(3)
            plt.imshow(stamp_SNR, interpolation="nearest")
            plt.show()

        if logFile:
            myStr = str(k)+', '+\
                    str(valid_potential_planet)+', '+\
                    str(max_val_criter)+', '+\
                    str(ratio_shape_SNR*shape_crit_map[row_id,col_id])+', '+\
                    str(max_val_SNR)+', '+\
                    str(fit_g.fit_info['ierr'])+', '+\
                    str(np.sqrt((g.x_mean-x_max_pos)**2+(g.y_mean-y_max_pos)**2))+', '+\
                    str(g.x_stddev+0.0)+', '+\
                    str(g.y_stddev+0.0)+', '+\
                    str(g.amplitude+0.0)+', '+\
                    str(row_id)+', '+\
                    str(col_id)+'\n'
            logFile_all.write(myStr)

        # If the spot is a valid candidate we add it to the candidates list
        stable_cent = True
        if valid_potential_planet:
            # Check that the centroid is stable over all the slices of the cube.
            # It basically fit a gaussian on each slice and verifies that the resulting centroid is not chaotic
            # and that it doesn't move with wavelength.
            if nl != 1:
                stamp_cube = cube[:,(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
                stamp_x_grid = x_grid[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
                stamp_y_grid = y_grid[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]

                h_init = models.Gaussian2D(g.amplitude+0.0,g.x_mean+0.0,g.y_mean+0.0,g.x_stddev+0.0,g.y_stddev+0.0)
                fit_h = fitting.LevMarLSQFitter()
                warnings.simplefilter('ignore')

                stamp_cube_cent_x = np.zeros(nl)
                stamp_cube_cent_y = np.zeros(nl)
                wave_step = 0.00841081142426 # in mum
                lambdaH0 = 1.49460536242  # in mum
                spec_sampling = np.arange(nl)*wave_step + lambdaH0

                fit_weights = np.sqrt(np.abs(spectrum))

                for k_slice in np.arange(nl):
                    h = fit_h(h_init, stamp_x_grid, stamp_y_grid, stamp_cube[k_slice,:,:],np.abs(stamp)**0.5)
                    if fit_h.fit_info['ierr'] <= 3:
                        stamp_cube_cent_x[k_slice] = h.x_mean+0.0
                        stamp_cube_cent_y[k_slice] = h.y_mean+0.0
                    else:
                        fit_weights[k_slice] = 0.0

                    if 0 and k == 7:
                        print(fit_h.fit_info['ierr'])
                    if 0 and k == 7:
                        plt.imshow(stamp_cube[k_slice,:,:], interpolation="nearest")
                        plt.show()
                stamp_cube_cent_r = np.sqrt(stamp_cube_cent_x**2 + stamp_cube_cent_y**2)
                fit_weights[abs(stamp_cube_cent_r - np.nanmean(stamp_cube_cent_r)) > 3 * np.nanstd(stamp_cube_cent_r)] = 0.0
                #print(stamp_cube_cent_r - np.nanmean(stamp_cube_cent_r),3 * np.nanstd(stamp_cube_cent_r))
                #print(stamp_cube_cent_r,fit_weights)
                fit_coefs_r = np.polynomial.polynomial.polyfit(spec_sampling, stamp_cube_cent_r, 1, w=fit_weights)
                #print(fit_coefs_r)
                r_poly_fit = np.poly1d(fit_coefs_r[::-1])
                r_fit = r_poly_fit(spec_sampling)

                candidate_radius = np.sqrt(x_grid[row_id,col_id]**2 + y_grid[row_id,col_id]**2)
                fit_sig = np.sqrt(np.nansum((r_fit-stamp_cube_cent_r)**2*fit_weights**2)/r_fit.size)
                # Todo: Remove prints, for debugging purpose only
                if not mute:
                    print(fit_coefs_r[1]*37*wave_step,0.5*candidate_radius/lambdaH0*37*wave_step,0.5*30/lambdaH0*37*wave_step,candidate_radius, fit_sig)
                #if fit_coefs_r[1] > 0.5*min(30/lambdaH0,candidate_radius/lambdaH0):
                if abs(fit_coefs_r[1])*37*wave_step > 1.5 or fit_sig>0.4:
                    stable_cent = False

                if 0 and k ==13:
                    stamp_cube_cent_r[np.where(abs(fit_weights) == 0.0)] = np.nan
                    plt.figure(1)
                    plt.plot(spec_sampling,r_fit, 'g-',spec_sampling,stamp_cube_cent_r, 'go')
                    plt.show()

            if logFile:
                myStr = str(k)+', '+\
                        str(stable_cent)+', '+\
                        str(max_val_criter)+', '+\
                        str(ratio_shape_SNR*shape_crit_map[row_id,col_id])+', '+\
                        str(max_val_SNR)+', '+\
                        str(fit_g.fit_info['ierr'])+', '+\
                        str(np.sqrt((g.x_mean-x_max_pos)**2+(g.y_mean-y_max_pos)**2))+', '+\
                        str(g.x_stddev+0.0)+', '+\
                        str(g.y_stddev+0.0)+', '+\
                        str(g.amplitude+0.0)+', '+\
                        str(row_id)+', '+\
                        str(col_id)+'\n'
                logFile_candidates.write(myStr)


        if 0 and k==12:
            stamp_cube = cube[:,(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
            stamp_cube[np.where(stamp_cube_small_mask != 0)] = 0.0
            spectrum = np.nansum(np.nansum(stamp_cube,axis=1),axis=1)
            print(spectrum)
            #plt.figure(2)
            #plt.imshow(stamp_cube[5,:,:], interpolation="nearest")
            plt.figure(3)
            plt.plot(spectrum)
            plt.show()


        # Todo: Remove prints, for debugging purpose only
        if not mute:
            print(valid_potential_planet,stable_cent,fit_g.fit_info['ierr'])


        if valid_potential_planet:
            # Increment the number of detected candidates
            N_candidates += 1
            # Append the useful things about the candidate in the list.
            candidates_list.append((k,valid_gaussian,stable_cent,x_max_pos, y_max_pos,max_val_criter,max_val_SNR,g))

    # END FOR LOOP

    candidates_KLs_list.append(candidates_list)

    if logFile:
        logFile_all.close()
        logFile_candidates.close()

    # START IF STATEMENT
    if toDraw or toPNG:
        # Highlight the detected candidates in the criterion map
        if not mute:
            print(N_candidates)
        criterion_map_checkedArea = criterion_map_cpy
        plt.figure(3,figsize=(16,16))
        #*flat_cube_mask[::-1,:]
        plt.imshow(criterion_map_checkedArea[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
        ax = plt.gca()
        for candidate in candidates_list:
            candidate_it,valid_gaussian,stable_cent,x_max_pos, y_max_pos,max_val_criter,max_val_SNR,g = candidate
            if valid_gaussian:
                color = 'green'
                if not stable_cent:
                    color = 'red'
            else:
                color = 'black'

            ax.annotate(str(candidate_it)+","+"{0:02.1f}".format(max_val_criter)+","+"{0:02.1f}".format(max_val_SNR), fontsize=20, color = color, xy=(x_max_pos+0.0, y_max_pos+0.0),
                    xycoords='data', xytext=(x_max_pos+10, y_max_pos-10),
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    linewidth = 1.,
                                    color = 'black')
                    )

        # Show the local maxima in the criterion map
        plt.figure(4,figsize=(16,16))
        plt.imshow(criterion_map_checkedArea[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
        ax = plt.gca()
        for spot in checked_spots_list:
            spot_it,row_id,col_id,max_val_criter,max_val_SNR,x_max_pos,y_max_pos,g = spot
            ax.annotate(str(spot_it)+","+"{0:02.1f}".format(max_val_criter), fontsize=20, color = 'black', xy=(x_max_pos+0.0, y_max_pos+0.0),
                    xycoords='data', xytext=(x_max_pos+10, y_max_pos-10),
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    linewidth = 1.,
                                    color = 'black')
                    )

    # END IF STATEMENT

    if toPNG:
        plt.figure(3,figsize=(16,16))
        plt.clim(-5.,10.0)
        plt.savefig(outputDir+folderName+toPNG+'_candidates_SNR.png', bbox_inches='tight')
        plt.clf()
        plt.close(3)
        plt.figure(4,figsize=(16,16))
        plt.clim(-5.,10.0)
        plt.savefig(outputDir+folderName+toPNG+'_allSpots_criterion.png', bbox_inches='tight')
        plt.clf()
        plt.close(4)
        plt.figure(5,figsize=(16,16))
        where_shape_bigger_than_SNR = np.zeros((ny,nx))
        where_shape_bigger_than_SNR[np.where(ratio_shape_SNR*shape_crit_map > flat_cube_SNR)] = 1
        plt.imshow(where_shape_bigger_than_SNR[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
        plt.savefig(outputDir+folderName+toPNG+'_shape_biggerThan_SNR.png', bbox_inches='tight')
        plt.clf()
        plt.close(5)


    if toFits:
        hdulist2 = pyfits.HDUList()
        try:
            hdulist2.append(pyfits.PrimaryHDU(header=prihdr))
            hdulist2.append(pyfits.ImageHDU(header=exthdr, data=criterion_map_cpy, name="Sci"))
            hdulist2.writeto(outputDir+folderName+toFits+'-criterion.fits', clobber=True)
            hdulist2[1].data = flat_cube_cpy
            hdulist2.writeto(outputDir+folderName+toFits+'-flatCube.fits', clobber=True)
            hdulist2[1].data = shape_crit_map
            hdulist2.writeto(outputDir+folderName+toFits+'-shape.fits', clobber=True)
            hdulist2[1].data = flat_cube_SNR
            hdulist2.writeto(outputDir+folderName+toFits+'-SNR.fits', clobber=True)
        except:
            print("Couldn't save using the normal way so trying something else.")
            hdulist2.append(pyfits.PrimaryHDU(header=prihdr))
            hdulist2[0].data = criterion_map_cpy
            hdulist2.writeto(outputDir+folderName+toFits+'-criterion.fits', clobber=True)
            hdulist2[0].data = flat_cube_cpy
            hdulist2.writeto(outputDir+folderName+toFits+'-flatCube.fits', clobber=True)
            hdulist2[0].data = shape_crit_map
            hdulist2.writeto(outputDir+folderName+toFits+'-shape.fits', clobber=True)
            hdulist2[0].data = flat_cube_SNR
            hdulist2.writeto(outputDir+folderName+toFits+'-SNR.fits', clobber=True)
        hdulist2.close()


    if toDraw:
        plt.figure(3,figsize=(16,16))
        plt.clim(-5.,10.0)
        plt.figure(4,figsize=(16,16))
        plt.clim(-5.,10.0)
        plt.show()

    return 1
# END candidate_detection() DEFINITION


def planet_detection_in_dir_per_file(filename,pipeline_dir,directory = "./", outputDir = '',
                            spectrum_model = "", star_type = "", star_temperature = None, mute = False):
    # Get the number of KL_modes for this file based on the filename *-KL#-speccube*
    splitted_name = filename.split("-KL")
    splitted_after_KL = splitted_name[1].split("-speccube.")
    N_KL_modes = int(splitted_after_KL[0])

    # Get the prefix of the filename
    splitted_before_KL = splitted_name[0].split("/")
    prefix = splitted_before_KL[np.size(splitted_before_KL)-1]

    # Define the output Filer
    if outputDir == '':
        outputDir = directory
    folderName = "/planet_detec_"+prefix+"_KL"+str(N_KL_modes)+"/"
    tofits = prefix+"_KL"+str(N_KL_modes)


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


    filelist_klipped_PSFs_cube = glob.glob(directory+"/"+prefix+"-KL"+str(N_KL_modes)+"-speccube-solePSFs.fits")
    #print(filelist_klipped_PSFs_cube)
    if np.size(filelist_klipped_PSFs_cube) == 1:
        # We found a PSF klip reduction
        # Hard coded position of the planets... Not great but I don't want to really think about it right now.
        #thetas = 90+np.array([54.7152978, -35.2847022, -125.2847022, -215.2847022])
        #radii = np.array([16.57733255996081, 32.33454364876502, 48.09175473756924, 63.84896582637346, 79.60617691517767])
        all_PSFs = extract_PSFs(filelist_klipped_PSFs_cube[0], stamp_width = 20)

        # merge the PSFs that have different radii and angles but same wavelength
        PSF_cubes = np.nanmean(all_PSFs, axis = (3,4))
        sat_spot_spec = np.nanmax(PSF_cubes,axis=(1,2))
        sat_spot_spec_exist = True
        # Remove the spectral shape from the psf cube because it is dealt with independently
        for l_id in range(PSF_cubes.shape[0]):
            PSF_cubes[l_id,:,:] /= sat_spot_spec[l_id]
    elif np.size(filelist_klipped_PSFs_cube) == 0:
        filelist_ori_PSFs_cube = glob.glob(directory+"/"+prefix+"-original_radial_PSF_cube.fits")
        if np.size(filelist_ori_PSFs_cube) == 1:
            if not mute:
                print("I didn't find any klipped PSFs file but I found a non klipped PSF. I'll take it.")
            hdulist = pyfits.open(filelist_ori_PSFs_cube[0])
            #print(hdulist[1].data.shape)
            PSF_cubes = hdulist[1].data[:,::-1,:]
            #PSF_cubes = np.transpose(hdulist[1].data,(1,2,0))[:,::-1,:] #np.swapaxes(np.rollaxis(hdulist[1].data,2,1),0,2)
            sat_spot_spec = np.nanmax(PSF_cubes,axis=(1,2))
            sat_spot_spec_exist = True
            # Remove the spectral shape from the psf cube because it is dealt with independently
            for l_id in range(PSF_cubes.shape[0]):
                PSF_cubes[l_id,:,:] /= sat_spot_spec[l_id]
        else:
            if not mute:
                print("I didn't find any PSFs file so I will use a default gaussian. Default sat spot spectrum = filter spectrum.")
            wv,sat_spot_spec = spec.get_gpi_filter(pipeline_dir,filter)
            sat_spot_spec_exist = False
            PSF_cubes = None
    else:
        if not mute:
            print("I found several *-solePSFs.fits so I don't know what to do and I quit")
        return 0

    if spectrum_model != "":
        if not mute:
            print("spectrum model: "+spectrum_model)
        # Interpolate the spectrum of the planet based on the given filename
        wv,planet_sp = spec.get_planet_spectrum(spectrum_model,filter)

        if sat_spot_spec_exist and (star_type !=  "" or star_temperature is not None):
            # Interpolate a spectrum of the star based on its spectral type/temperature
            wv,star_sp = spec.get_star_spectrum(pipeline_dir,filter,star_type,star_temperature)
            spectrum = (sat_spot_spec/star_sp)*planet_sp
        else:
            if not mute:
                print("No star spec or sat spot spec so using sole planet spectrum.")
            spectrum = planet_sp
    else:
        spectrum = sat_spot_spec


    candidate_detection(filename,
                        PSF = PSF_cubes,
                        outputDir = outputDir,
                        folderName = folderName,
                        toDraw=False,
                        toFits=tofits,
                        toPNG=tofits,
                        logFile=tofits,
                        spectrum=spectrum,
                        mute = mute)#toPNG="Baade", logFile='Baade')

def planet_detection_in_dir_per_file_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call planet_detection_in_dir_per_file() with a tuple of parameters.
    """
    return planet_detection_in_dir_per_file(*params)

def planet_detection_in_dir(pipeline_dir,directory = "./",filename_prefix_is = '', outputDir = '',
                            spectrum_model = "", star_type = "", star_temperature = None,threads = False):
    '''
    Apply the planet detection algorithm for all pyklip reduced cube respecting a filter in a given folder.
    By default the filename filter used is pyklip-*-KL*-speccube.fits.
    It will look for a PSF file like pyklip-*-KL*-speccube-solePSFs.fits to extract a klip reduced PSF.
        Note: If you want to include a spectrum it can be already included in the PSF when reducing it with klip.
            The other option if there is no psf available for example is to give a spectrum to
    If no pyklip-*-KL*-speccube-solePSFs.fits is found it will for a pyklip-*-original_radial_PSF_cube.fits which is a PSF
    built from the sat spots but not klipped.

    Inputs:
        pipeline_dir: GPI pipeline directory. E.g. "/Users/jruffio/gpi/pipeline/".
        directory: directory in which the function will look for suitable fits files and run the planet detection algorithm.
        filename_prefix_is: Look for file containing of the form "/"+filename_filter+"-KL*-speccube.fits"
        outputDir: Directory to where to save the output folder. By default it is saved in directory.
        spectrum_model: Mark Marley's spectrum filename. E.g. "/Users/jruffio/gpi/pyklip/t800g100nc.flx"
        star_type: Spectral type of the star (works only for type V). E.g. "G5" for G5V type.
        star_temperature: Temperature of the star. (replace star_type)
        threads: If true, parallel computation of several files (no prints in the console).
                Otherwise sequential with bunch of prints (for debugging).

    Outputs:
        For each file an output folder is created:
            outputDir = directory + "/planet_detec_"+prefix+"_KL"+str(N_KL_modes)+"/"
        The outputs of the detection can be found there.

    '''
    if filename_prefix_is == '':
        filelist_klipped_cube = glob.glob(directory+"/pyklip-*-KL*-speccube.fits")
    else:
        filelist_klipped_cube = glob.glob(directory+"/"+filename_prefix_is+"-KL*-speccube.fits")

    if threads:
        N_threads = np.size(filelist_klipped_cube)
        pool = mp.Pool(processes=N_threads)
        pool.map(planet_detection_in_dir_per_file_star, itertools.izip(filelist_klipped_cube,
                                                                       itertools.repeat(pipeline_dir),
                                                                       itertools.repeat(directory),
                                                                       itertools.repeat(outputDir),
                                                                       itertools.repeat(spectrum_model),
                                                                       itertools.repeat(star_type),
                                                                       itertools.repeat(star_temperature),
                                                                       itertools.repeat(True)))
    else:
        for filename in filelist_klipped_cube:
            print(filename)
            planet_detection_in_dir_per_file(filename,
                                             pipeline_dir,
                                             directory = directory,
                                             outputDir = outputDir,
                                             spectrum_model = spectrum_model,
                                             star_type = star_type,
                                             star_temperature = star_temperature,
                                             mute = False)




