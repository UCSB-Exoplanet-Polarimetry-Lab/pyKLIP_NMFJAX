import numpy as np
#import cmath
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

import spectra_management as spec


def extract_PSFs(filename, radii, thetas, stamp_width = 10):
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
    center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]

    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    #x.shape = (x.shape[0] * x.shape[1])
    #y.shape = (y.shape[0] * y.shape[1])
    x -= center[0]
    y -= center[1]
    x_planets = np.dot(np.array([radii]).transpose(),np.array([np.cos(np.radians(thetas))]))
    y_planets = np.dot(np.array([radii]).transpose(),np.array([np.sin(np.radians(thetas))]))

    dn = stamp_width
    out_stamp_PSFs = np.zeros((nl,dn,dn,nth,nr))

    for l_id in range(nl):
        for r_id,r_it in enumerate(radii):
            for th_id, th_it in enumerate(thetas):
                x_plnt = np.round(x_planets[r_id,th_id]+center[0])
                y_plnt = np.round(y_planets[r_id,th_id]+center[1])

                out_stamp_PSFs[l_id,:,:,th_id,r_id] = cube[l_id,(y_plnt-np.floor(dn/2.)):(y_plnt+np.ceil(dn/2.)),(x_plnt-np.floor(dn/2.)):(x_plnt+np.ceil(dn/2.))]

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


def radialStd(cube,dr,Dr,centroid = None, r = None, r_samp = None):
    '''
    Return the standard deviation with respect to the radius computed in annuli of radial width Dr separated by dr.
    :return:
    '''
    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        cube = cube[None,:]
        nl = 1

    if r is None:
        r = [np.nan]
    if r_samp is None:
        r_samp = [np.nan]

    #TODO centroid should be different for each slice?
    if centroid is None :
        x_cen = np.ceil((nx-1)/2) ; y_cen = np.ceil((ny-1)/2)
    else:
        x_cen, y_cen = centroid

    x, y = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
    r[0] = abs(x +y*1j)
    #r_samp[0] = np.arange(dr,max(r[0].reshape(np.size(r[0]))),dr)
    r_samp[0] = np.arange(0,max(r[0].reshape(np.size(r[0]))),dr)

    radial_std = np.zeros((nl,np.size(r_samp[0])))

    bins = np.arange(-3,3,0.1)
    #print(r_samp[0])
    #print(r_samp[0].size)
    histo = np.zeros([bins.size-1,r_samp[0].size])

    for r_id, r_it in enumerate(r_samp[0]):
        selec_pix = np.where( ((r_it-Dr/2.0) < r[0]) * (r[0] < (r_it+Dr/2.0)) )
        selec_y, selec_x = selec_pix
        data = cube[:,selec_y, selec_x]
        '''
            print(data)
            plt.figure(1)
            plt.show()
            mn_data = np.nanmean(data)
            std_data = np.nanstd(data)
            data[abs(data - mn_data) > 3 * std_data] = np.nan
            #print(data.shape)
            radial_std[:,r_id] = np.nanstd(data,1)
        '''
        for l_it in np.arange(nl):
            data_slice = data[l_it,:]
            # Check that everything is not Nan
            if np.size(np.where(np.isnan(data_slice))) != data_slice.size:
                mn_data = np.nanmean(data_slice)
                std_data = np.nanstd(data_slice)
                # Remove nans from the array but ensure they be put back afterwards
                data_slice[np.where(np.isnan(data_slice))] = mn_data + 10 * std_data
                out_3sig = np.where((data_slice-mn_data)>(3.*std_data))
                if out_3sig is not ():
                    data[l_it,out_3sig] = np.nan
                radial_std[l_it,r_id] = np.nanstd(data[l_it,:])
            else:
                radial_std[l_it,r_id] = np.nan


        # Study histogram of the pixels
        if 0 and r_it>30 and r_it<80:
            #print(data.shape)
            data_histo = np.histogram(data[np.where(np.isfinite(data))], bins=bins)
            #print(a.shape)
            histo[:,r_id] = data_histo[0]

            # Definition of a 2D gaussian fitting to be used on the stamp.
            g_init = models.Gaussian1D(amplitude=100., mean=10, stddev=5.)
            # JB Todo: look at the different fitting methods.
            fit_g = fitting.LevMarLSQFitter()

            # Fit the 2d Gaussian to the stamp
            # Ignore model linearity warning from the fitter
            warnings.simplefilter('ignore')
            g = fit_g(g_init, bins[0:bins.size-1], histo[:,r_id])

            plt.figure(10)
            #print(bins[0:bins.size-1].shape,histo[:,50].shape,g.shape)
            plt.plot(bins[0:bins.size-1],histo[:,r_id],'rx',bins[0:bins.size-1],g(bins[0:bins.size-1]),'bo')
            plt.show()

    radial_std = np.squeeze(radial_std)

    if 0:
        # Definition of a 2D gaussian fitting to be used on the stamp.
        g_init = models.Gaussian1D(amplitude=100., mean=10, stddev=5.)
        # JB Todo: look at the different fitting methods.
        fit_g = fitting.LevMarLSQFitter()

        # Fit the 2d Gaussian to the stamp
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        g = fit_g(g_init, bins[0:bins.size-1], histo[:,50])

        plt.figure(10)
        #print(bins[0:bins.size-1].shape,histo[:,50].shape,g.shape)
        plt.plot(bins[0:bins.size-1],histo[:,50],'rx',bins[0:bins.size-1],g(bins[0:bins.size-1]),'bo')
        plt.show()


    return radial_std

def radialStdMap(cube,dr,Dr,centroid = None,treshold=10**(-6)):
    '''
    Return an cube of the same size as cube on which the value for each pixel is the standard deviation of the original
    cube inside an annulus of width Dr.
    :return:
    '''

    r = [np.nan]
    r_samp = [np.nan]
    radial_std = radialStd(cube,dr,Dr,centroid, r = r, r_samp = r_samp)

    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        radial_std = radial_std[None,:]
        nl = 1

    cube_std = np.zeros((nl,ny,nx))
    radial_std_nans = np.isnan(radial_std)
    radial_std[radial_std_nans] = 0.0
    for l_id in np.arange(nl):
        f = interp1d(r_samp[0], radial_std[l_id,:], kind='cubic',bounds_error=False, fill_value=np.nan)
        a = f(r[0].reshape(nx*ny))
        cube_std[l_id,:,:] = a.reshape(ny,nx)

    # Remove nans from the array but ensure they be put back afterwards
    cube_std[np.where(np.isnan(cube_std))] = 0.0
    cube_std[np.where(cube_std < treshold)] = np.nan

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
    cube = hdulist[1].data
    scaled_cube = copy(cube)
    exthdr = hdulist[1].header
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
    flat_cube = subtract_radialMed(flat_cube,2,20,center)
    flat_cube_cpy = copy(flat_cube)


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
    flat_cube_std = radialStdMap(flat_cube,dr,Dr, centroid=center, treshold=10**(-3))


    # Divide the convolved flat cube by the standard deviation map to get the SNR.
    flat_cube_SNR = flat_cube/flat_cube_std



    criterion_map = -np.ones((ny,nx))
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
        for k in np.arange(10,ny-10):
            for l in np.arange(10,nx-10):
                stamp_cube = cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)]
                ampl = np.nansum(PSF_cube*stamp_cube)
                if ampl != 0.0:
                    square_norm_stamp = np.nansum(stamp_cube**2)
                    criterion_map[k,l] = np.sign(ampl)*ampl**2/square_norm_stamp
                else:
                    criterion_map[k,l] = -1.0
                #ortho_criterion_map[k,l] = np.nansum((stamp_cube-ampl*PSF_cube)**2)/square_norm_stamp
    else:
        for k in np.arange(10,ny-10):
            for l in np.arange(10,nx-10):
                stamp = cube[(k-row_m):(k+row_p), (l-col_m):(l+col_p)]
                ampl = np.nansum(PSF*stamp)
                if ampl != 0.0:
                    square_norm_stamp = np.nansum(stamp**2)
                    criterion_map[k,l] = np.sign(ampl)*ampl**2/square_norm_stamp
                else:
                    criterion_map[k,l] = -1.0
                #ortho_criterion_map[k,l] = np.nansum((stamp_cube-ampl*PSF_cube)**2)/square_norm_stamp


        ## ortho_criterion is actually the sine between the two vectors
        # ortho_criterion_map = 1 - criterion_map
    # criterion is here a cosine squared so we take the square root to get something similar to a cosine.
    criterion_map = np.sign(criterion_map)*np.sqrt(abs(criterion_map))
    #Save a copy of the flat cube because we will mask the detected spots as the algorithm goes.
    criterion_map_cpy = copy(criterion_map)


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
                "# 4/ Error check of the gaussian fit. \n"+\
                "# 5/ Distance between the fitted centroid and the original max position. \n"+\
                "# 6/ x-axis width of the gaussian. \n"+\
                "# 7/ y-axis width of the gaussian. \n"+\
                "# 8/ Amplitude of the gaussian. \n"+\
                "# 9/ Row index of the maximum. (y-coord in DS9) \n"+\
                "# 10/ Column index of the maximum. (x-coord in DS9) \n"+\
                "# 11/ SNR at the position of the max. \n"
        logFile_all.write(myStr)

        myStr2 = "# Log some values for each valid candidates. \n" +\
                "# Meaning of the columns from left to right. \n" +\
                "# 1/ Index \n" +\
                "# 2/ Boolean. True if the centroid of the maximum is stable-ish. \n" +\
                "# 3/ Value of the criterion at this local maximum. \n"+\
                "# 4/ Error check of the gaussian fit. \n"+\
                "# 5/ Distance between the fitted centroid and the original max position. \n"+\
                "# 6/ x-axis width of the gaussian. \n"+\
                "# 7/ y-axis width of the gaussian. \n"+\
                "# 8/ Amplitude of the gaussian. \n"+\
                "# 9/ Row index of the maximum. (y-coord in DS9) \n"+\
                "# 10/ Column index of the maximum. (x-coord in DS9) \n"+\
                "# 11/ SNR at the position of the max. \n"
        logFile_candidates.write(myStr2)


    flat_cube[np.where(np.isnan(flat_cube))] = 0.0

    # Count the number of valid detected candidates.
    N_candidates = 0.0
    # Maximum number of iterations on local maxima.
    max_attempts = 100
    # START FOR LOOP.
    # Each iteration looks at one local maximum in the criterion map.
    # Then it verifies some other criteria to check if it is worse looking at.
    for k in np.arange(max_attempts):
        # Find the maximum value in the current SNR map. At each iteration the previous maximum is masked out.
        max_val_criter = np.nanmax(criterion_map)
        # Locate the maximum by retrieving its coordinates
        max_ind = np.where( criterion_map == max_val_criter )
        row_id,col_id = max_ind[0][0],max_ind[1][0]
        x_max_pos, y_max_pos = x_grid[row_id,col_id],y_grid[row_id,col_id]
        max_val = flat_cube_SNR[row_id,col_id]

        #Extract a stamp around the maximum in the flat cube (without the convolution)
        row_m = np.floor(stamp_nrow/2.0)
        row_p = np.ceil(stamp_nrow/2.0)
        col_m = np.floor(stamp_ncol/2.0)
        col_p = np.ceil(stamp_ncol/2.0)
        stamp = copy(flat_cube[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)])
        stamp_SNR = copy(criterion_map[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)])
        stamp_x_grid = x_grid[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
        stamp_y_grid = y_grid[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]


        # Definition of a 2D gaussian fitting to be used on the stamp.
        g_init = models.Gaussian2D(max_val,x_max_pos,y_max_pos,1.5,1.5)
        fit_g = fitting.LevMarLSQFitter()
        # Fit the 2d Gaussian to the stamp
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        g = fit_g(g_init, stamp_x_grid, stamp_y_grid, stamp,np.abs(stamp)**0.5)

            # Calculate the fitting residual using the square root of the summed squared error.
            #ampl = np.nansum(stamp*g(stamp_x_grid, stamp_y_grid))
            #khi = np.sign(ampl)*ampl**2/(np.nansum(stamp**2)*np.nansum(g(stamp_x_grid, stamp_y_grid)**2))

        # JB Todo: Should explore the meaning of 'ierr' but I can't find a good clear documentation of astropy.fitting
        sig_min = 0.5 ; sig_max = 3.5 ;
        # The condition for a local maximum to be considered as a candidate are:
        #       - Positive criterion. Always verified because we are looking at maxima...
        #       - Reasonable SNR. ie greater than one.
        #         I prefer to be conservative on the SNR because we never know and I think it isn't the best criterion.
        #       - The gaussian fit had to succeed.
        #       - Reasonable width of the gaussian. Not wider than 3.5pix and not smaller than 0.5pix in both axes.
        #       - Centroid of the Gaussian fit not too far from the center of the stamp.
        #       - Amplitude of the Gaussian fit should be positive.
        valid_potential_planet = (max_val_criter > 0.0 and
                                 flat_cube_SNR[row_id,col_id] > 1.0 and
                                 fit_g.fit_info['ierr'] <= 3 and
                                 sig_min < g.x_stddev < sig_max and
                                 sig_min < g.y_stddev < sig_max and
                                 np.sqrt((g.x_mean-x_max_pos)**2+(g.y_mean-y_max_pos)**2)<1.5 and
                                 g.amplitude > 0.0)
                                    #khi/khi0 < 1.0 and # Check that the fit was good enough. Not a weird looking speckle.

        #fit_g.fit_info['ierr'] == 1 and # Check that the fitting actually occured. Actually I have no idea what the number mean but it looks like when it succeeds it is 1.

        # If the spot verifies the conditions above it is considered as a valid candidate.
        checked_spots_list.append((k,row_id,col_id,max_val,x_max_pos,y_max_pos,g))

        # Mask the spot around the maximum we just found.
        criterion_map[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] *= stamp_mask
        flat_cube[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] *= stamp_mask

        # Todo: Remove prints, for debugging purpose only
        if not mute:
            print(k,row_id,col_id,x_max_pos,y_max_pos,max_val_criter,max_val, g.x_stddev+0.0, g.y_stddev+0.0,g.x_mean-x_max_pos,g.y_mean-y_max_pos,flat_cube_SNR[row_id,col_id])
        if k == 79 and 0:
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
                    str(fit_g.fit_info['ierr'])+', '+\
                    str(np.sqrt((g.x_mean-x_max_pos)**2+(g.y_mean-y_max_pos)**2))+', '+\
                    str(g.x_stddev+0.0)+', '+\
                    str(g.y_stddev+0.0)+', '+\
                    str(g.amplitude+0.0)+', '+\
                    str(row_id)+', '+\
                    str(col_id)+', '+\
                    str(max_val)+'\n'
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
                        str(fit_g.fit_info['ierr'])+', '+\
                        str(np.sqrt((g.x_mean-x_max_pos)**2+(g.y_mean-y_max_pos)**2))+', '+\
                        str(g.x_stddev+0.0)+', '+\
                        str(g.y_stddev+0.0)+', '+\
                        str(g.amplitude+0.0)+', '+\
                        str(row_id)+', '+\
                        str(col_id)+', '+\
                        str(max_val)+'\n'
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
            candidates_list.append((k,stable_cent,row_id,col_id,max_val,g))

    # END FOR LOOP

    candidates_KLs_list.append(candidates_list)

    if logFile:
        logFile_all.close()
        logFile_candidates.close()

    # START IF STATEMENT
    if toDraw or toPNG:
        # Highlight the detected candidates in the 2d flat cube.
        if not mute:
            print(N_candidates)
        z = criterion_map
        plt.figure(3,figsize=(16,16))
        #*flat_cube_mask[::-1,:]
        plt.imshow(flat_cube_cpy[::-1,:]/flat_cube_std[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
        ax = plt.gca()
        for candidate in candidates_list:
            candidate_it,stable_cent,row_id,col_id,max_val,g = candidate
            if not stable_cent:
                color = 'red'
            else:
                color = 'black'

            ax.annotate(str(candidate_it), fontsize=20, color = color, xy=(g.x_mean+0.0, g.y_mean+0.0),
                    xycoords='data', xytext=(g.x_mean+15, g.y_mean-15),
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    linewidth = 1.,
                                    color = 'black')
                    )

        # Shouw the local maxima in the SNR map
        criterion_map_checkedArea = criterion_map_cpy #np.log10(abs(flat_cube))
        plt.figure(4,figsize=(16,16))
        plt.imshow(criterion_map_checkedArea[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
        ax = plt.gca()
        for spot in checked_spots_list:
            spot_it,row_id,col_id,max_val,x_max_pos,y_max_pos,g = spot
            ax.annotate(str(spot_it), fontsize=20, color = 'black', xy=(x_max_pos+0.0, y_max_pos+0.0),
                    xycoords='data', xytext=(x_max_pos+15, y_max_pos-15),
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    linewidth = 1.,
                                    color = 'black')
                    )
    # END IF STATEMENT

    if toPNG:
        plt.figure(3,figsize=(16,16))
        plt.savefig(outputDir+folderName+toPNG+'_candidates_SNR.png', bbox_inches='tight')
        plt.figure(4,figsize=(16,16))
        plt.savefig(outputDir+folderName+toPNG+'_allSpots_criterion.png', bbox_inches='tight')


    if toFits:
        hdulist2 = pyfits.HDUList()
        hdulist2.append(pyfits.PrimaryHDU(header=prihdr))
        hdulist2.append(pyfits.ImageHDU(header=exthdr, data=criterion_map_cpy, name="Sci"))
        hdulist2.writeto(outputDir+folderName+toFits+'-criterion.fits', clobber=True)
        hdulist2[1].data = flat_cube_cpy
        hdulist2.writeto(outputDir+folderName+toFits+'-flatCube.fits', clobber=True)
        hdulist2[1].data = flat_cube_SNR
        hdulist2.writeto(outputDir+folderName+toFits+'-SNR.fits', clobber=True)
        hdulist2.close()


    if toDraw:
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
        thetas = 90+np.array([54.7152978, -35.2847022, -125.2847022, -215.2847022])
        radii = np.array([16.57733255996081, 32.33454364876502, 48.09175473756924, 63.84896582637346, 79.60617691517767])
        all_PSFs = extract_PSFs(filelist_klipped_PSFs_cube[0], radii, thetas, stamp_width = 20)

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
    Apply the planet detection algorithm for all pyklip reduced cube in a given folder.
    It will look for a PSF file like pyklip-*-KL*-speccube-solePSFs.fits to extract a klip reduced PSF.
        Note: If you want to include a spectrum it can be already included in the PSF when reducing it with klip.
            The other option if there is no psf available for example is to give a spectrum to
    If no pyklip-*-KL*-speccube-solePSFs.fits is found it will for a pyklip-*-original_PSF_cube.fits which is a PSF
    built from the sat spots but not klipped.

    Inputs:
        directory: directory in which the function will look for suitable fits files and run the planet detection algorithm.
        filename_filter: Look for file containing of the form "/*"+filename_filter+"-KL*-speccube.fits"

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





if __name__ == "__main__":
    import astropy.io.fits as pyfits
    import glob
    import numpy as np
    from astropy import wcs
    import os
    #different importants depending on if python2.7 or python3
    import sys
    import cmath
    import time
    from scipy.interpolate import interp1d
    import astropy.io.fits as pyfits

    from scipy.interpolate import griddata
    from scipy.interpolate import bisplrep
    from scipy.interpolate import bisplev


    outputDir = ''
    star_type = ''

    #inputDir = "/Users/jruffio/gpi/pyklip/outputs/baade"
    #inputDir = "/Users/jruffio/gpi/pyklip/outputs/test_detec_folder"
    inputDir = "/Users/jruffio/Dropbox (GPI)/GPIDATA/c_Eri/autoreduced/"
    #outputDir = "/Users/jruffio/gpi/pyklip/outputs/dropbox_prior_test/"
    outputDir = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB"
    #inputDir = "/Users/jruffio/gpi/pyklip/outputs/Baade_5im"
    filename_filter = "pyklip-S20141218-k100a7s4m3"
    #filename_filter = "*baade*"
    spectrum_model = "/Users/jruffio/gpi/pyklip/t800g100nc.flx"
    #star_type = "G4"
    pipeline_dir = "/Users/jruffio/gpi/pipeline/"
    planet_detection_in_dir(pipeline_dir,
                            inputDir,
                            outputDir=outputDir,
                            filename_prefix_is=filename_filter,
                            spectrum_model=spectrum_model,
                            star_type=star_type,
                            threads = True)













####################################################################################
####################### CODE GARBAGE ###############################################
####################################################################################


'''
    thetas = 90+np.array([54.7152978, -35.2847022, -125.2847022, -215.2847022])
    radii = np.array([16.57733255996081, 32.33454364876502, 48.09175473756924, 63.84896582637346, 79.60617691517767])
    filename_PSFs = "/Users/jruffio/gpi/pyklip/outputs/baade-PSFs-KLmodes-all-PSFs.fits"
    #filename_PSFs = "/Users/jruffio/gpi/pyklip/outputs/baade-PSFs2-KLmodes-all-PSFs.fits"
    #filename_PSFs = "/Users/jruffio/gpi/pyklip/outputs/baade-PSFs15-KLmodes-all-PSFs.fits"
    #filename_PSFs = "/Users/jruffio/gpi/pyklip/outputs/baade-PSFs20-KLmodes-all-PSFs.fits"
    if 0:
        PSF = extract_merge_PSFs(filename_PSFs, radii, thetas)
    #print(PSF.shape)
    #plt.figure(20)
    #plt.imshow(PSF,interpolation = 'nearest')
    #plt.show()


    #filelist = glob.glob("/Users/jruffio/gpi/pyklip/outputs/HD114174-KL20-speccube.fits")

    #dataset = GPI.GPIData(filelist)


    filename = "/Users/jruffio/gpi/pyklip/outputs/baade/baade-h-k300a7s4m3-KL20-speccube.fits"
    filename = "/Users/jruffio/Dropbox/GPIDATA/c_Eri/autoreduced/pyklip-S20141218-k100a5s4m3-KL20-speccube.fits"

    tofits = ''
    outputDir = "/Users/jruffio/gpi/pyklip/outputs/"
    folderName = "/baade_out"
    folderName = "/c_Eri"
    tofits = folderName
    spectrumHEri = np.array([141.9296769,   198.73423838 , 242.22243818,  299.85427039,  335.37585698, # spec c_eri H
  370.88441648 , 398.06479267 , 411.32024875 , 424.47542857,  429.61704089,
  421.74684239 , 377.85252056 , 305.35767397 , 247.08267071, 196.72520657,
  138.06660696 , 104.14917438 , 104.38252497 ,  95.81786937 ,  73.66560944,
   37.55078903 ,  37.32665492  , 46.10064065  , 35.90442611  , 18.09452635,
   25.52058438  , 34.30831346  , 13.11016915  ,  0.95286082  ,  7.54171253,
   10.2595994 ,   10.93531723  , 15.62122062 ,   7.88143819   , 4.68372895])
    spectrumH = np.array([0.0,141.9296769,   198.73423838 , 242.22243818,  299.85427039,  335.37585698, # spec c_eri H
  370.88441648 , 398.06479267 , 411.32024875 , 424.47542857,  429.61704089,
  421.74684239 , 377.85252056 , 305.35767397 , 247.08267071, 196.72520657,
  138.06660696 , 104.14917438 , 104.38252497 ,  95.81786937 ,  73.66560944,
   37.55078903 ,  37.32665492  , 46.10064065  , 35.90442611  , 18.09452635,
   25.52058438  , 34.30831346  , 13.11016915  ,  0.95286082  ,  7.54171253,
   10.2595994 ,   10.93531723  , 15.62122062 ,   7.88143819   , 4.68372895,0.0])
    spectrumJ = np.array([ -0.11320329 , -0.07596916 , -2.7990458 ,  -3.79993844 , -4.38923454,  # spec c_eri J
  -5.78503418 , -6.53705502 , -4.93471336 , -4.2245512 ,  -2.4959271,
   0.46197677 ,  0.99514222 ,  4.06949472 , 10.77979755 , 11.6944828,
  20.47476196 , 20.85378075 , 19.65199089 , 24.29648018 , 27.36954117,
  34.65512085 , 40.13297653 , 35.40809631 , 31.18817902 , 32.92845154,
  36.96004486 , 39.82872009 , 41.06404877 , 35.63868332 , 30.73090744,
  27.26691818 , 23.48490906 , 21.04096222 , 14.74596024 ,  8.45730782,
   3.58565426 ,  0.78945696])
    #spectrum_flat = np.ones(37)
    #spectrum = spectrum[1:36]
    #spectrum = np.ones(5)
    #spectrum[0] = 0.0

    #candidate_detection(filename, outputDir = outputDir, folderName = folderName, PSF = None, toDraw=True, toFits=tofits,toPNG=tofits,logFile=tofits, spectrum=spectrumHEri)#toPNG="Baade", logFile='Baade')
        #candidate_detection(filename,PSF = None, toDraw=True, toFits="HD114174")#toPNG="HD114174", logFile='HD114174')
'''


def flatten_annulus(im,rad_bounds,center):
    ny,nx = im.shape

    x, y = np.meshgrid(np.arange(ny), np.arange(nx))
    #x.shape = (x.shape[0] * x.shape[1])
    #y.shape = (y.shape[0] * y.shape[1])
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    phi = np.arctan2(y - center[1], x - center[0])

    r_min, r_max = rad_bounds
    annulus_indices = np.where((r >= r_min) & (r < r_max))
    ann_y_id = annulus_indices[0]
    ann_x_id = annulus_indices[1]

    annulus_pix = slice_PSFs[ann_y_id,ann_x_id]
    annulus_x = x[ann_y_id,ann_x_id] - center[0]
    annulus_y = y[ann_y_id,ann_x_id]- center[1]
    annulus_r = r[ann_y_id,ann_x_id]
    annulus_phi = phi[ann_y_id,ann_x_id]

    annulus_pix = annulus_pix.reshape(np.size(annulus_pix))
    annulus_x = annulus_x.reshape(np.size(annulus_pix))
    annulus_y = annulus_y.reshape(np.size(annulus_pix))
    annulus_r = annulus_r.reshape(np.size(annulus_pix))
    annulus_phi = annulus_phi.reshape(np.size(annulus_pix))

    #slice_PSFs[ann_y_id,ann_x_id] = 100
    #plt.imshow(slice_PSFs,interpolation = 'nearest')
    #plt.show()

    n_phi = np.floor(2*np.pi*r_max)
    dphi = 2*np.pi/n_phi
    #r_arr, phi_arr = np.meshgrid(np.arange(np.ceil(r_min),np.ceil(r_min)+n_r,0.01),np.arange(-np.pi,np.pi,dphi))
    r_arr, phi_arr = np.meshgrid(np.arange(np.min(annulus_r),np.max(annulus_r),1),np.arange(-np.pi,np.pi,dphi))

    points = np.array([annulus_r,(r_max+r_min)/2.*annulus_phi])
    points = points.transpose()
    grid_z2 = griddata(points, annulus_pix, (r_arr, (r_max+r_min)/2.*phi_arr), method='cubic')

    '''
    tck = bisplrep(annulus_x, annulus_y, annulus_pix)
    nrow, ncol = r_arr.shape
    #znew = bisplev(35, 35, tck)
    #print(znew)
    #print(np.min(annulus_x))
    #print(np.max(annulus_x))
    znew = np.zeros([nrow, ncol])
    for i_it in range(nrow):
        for j_it in range(ncol):
            xp = r_arr[i_it,j_it]*np.cos(phi_arr[i_it,j_it])
            yp = r_arr[i_it,j_it]*np.sin(phi_arr[i_it,j_it])
            znew[i_it,j_it] = bisplev(xp, yp, tck)
    #xnew, ynew = np.meshgrid(np.arange(np.min(annulus_x),np.max(annulus_x),1), np.arange(np.min(annulus_x),np.max(annulus_x),1))
    #znew = bisplev(xnew[:,0], ynew[0,:], tck)
    '''

    plt.figure(2)
    plt.imshow(grid_z2.transpose(),interpolation = 'nearest',extent=[-np.pi*((r_max+r_min)/2.),np.pi*((r_max+r_min)/2.),np.min(annulus_r),np.max(annulus_r)], origin='lower')
    plt.plot(points[:,1], points[:,0],'b.')
    #plt.imshow(znew,interpolation = 'nearest')
    plt.show()

    return grid_z2

def badPixelFilter(cube,scale = 2,maxDeviation = 10):

    cube_cpy = cube[:]

    if np.size(cube_cpy.shape) == 3:
        nl,ny,nx = cube_cpy.shape
    elif np.size(cube_cpy.shape) == 2:
        ny,nx = cube_cpy.shape
        cube_cpy = cube_cpy[None,:]
        nl = 1

    smooth_cube = np.zeros((nl,ny,nx))

    ker = np.ones((3,3))/8.0
    n_bad = np.zeros((nl,))
    ker[1,1] = 0.0
    for l_id in np.arange(nl):
        smooth_cube[l_id,:,:] = convolve2d(cube_cpy[l_id,:,:], ker, mode='same')
        slice_diff = cube_cpy[l_id,:,:] - smooth_cube[l_id,:,:]
        stdmap = radialStdMap(slice_diff,2,10)
        bad_pixs = np.where(abs(slice_diff) > maxDeviation*stdmap)
        n_bad[l_id] = np.size(bad_pixs)
        bad_pixs_x, bad_pixs_y = bad_pixs
        cube_cpy[l_id,bad_pixs_x, bad_pixs_y] = np.nan ;

    return cube_cpy,bad_pixs

def radialMed(cube,dr,Dr,centroid = None, r = None, r_samp = None):
    '''
    Return the mean with respect to the radius computed in annuli of radial width Dr separated by dr.
    :return:
    '''
    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        cube = cube[None,:]
        nl = 1

    if centroid is None :
        x_cen = np.ceil((nx-1)/2) ; y_cen = np.ceil((ny-1)/2)
    else:
        x_cen, y_cen = centroid

    if r is None:
        r = [np.nan]
    if r_samp is None:
        r_samp = [np.nan]

    x, y = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
    r[0] = abs(x +y*1j)
    r_samp[0] = np.arange(dr,max(r[0].reshape(np.size(r[0]))),dr)

    radial_std = np.zeros((nl,np.size(r_samp[0])))

    for r_id, r_it in enumerate(r_samp[0]):
        selec_pix = np.where( ((r_it-Dr/2.0) < r[0]) * (r[0] < (r_it+Dr/2.0)) )
        selec_y, selec_x = selec_pix
        radial_std[:,r_id] = nanmedian(cube[:,selec_y, selec_x],1)

    radial_std = np.squeeze(radial_std)

    return radial_std


def radialMean(cube,dr,Dr,centroid = None, r = None, r_samp = None):
    '''
    Return the mean with respect to the radius computed in annuli of radial width Dr separated by dr.
    :return:
    '''
    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        cube = cube[None,:]
        nl = 1

    #TODO centroid should be different for each slice?
    if centroid is None :
        x_cen = np.ceil((nx-1)/2) ; y_cen = np.ceil((ny-1)/2)
    else:
        x_cen, y_cen = centroid

    if r is None:
        r = [np.nan]
    if r_samp is None:
        r_samp = [np.nan]

    x, y = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
    r[0] = abs(x +y*1j)
    r_samp[0] = np.arange(dr,max(r[0].reshape(np.size(r[0]))),dr)

    radial_std = np.zeros((nl,np.size(r_samp[0])))

    for r_id, r_it in enumerate(r_samp[0]):
        selec_pix = np.where( ((r_it-Dr/2.0) < r[0]) * (r[0] < (r_it+Dr/2.0)) )
        selec_y, selec_x = selec_pix
        radial_std[:,r_id] = np.nanmean(cube[:,selec_y, selec_x],1)

    radial_std = np.squeeze(radial_std)

    return radial_std
