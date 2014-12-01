import numpy as np
#import cmath
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from scipy.signal import convolve
import astropy.io.fits as pyfits
from astropy.modeling import models, fitting
from copy import copy
import warnings

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

    #TODO centroid should be different for each slice?
    if centroid is None :
        x_cen = np.ceil((nx-1)/2) ; y_cen = np.ceil((ny-1)/2)
    else:
        x_cen, y_cen = centroid

    x, y = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
    r[0] = abs(x +y*1j)
    r_samp[0] = np.arange(dr,max(r[0].reshape(np.size(r[0]))),dr)

    radial_std = np.zeros((nl,np.size(r_samp[0])))

    for r_id, r_it in enumerate(r_samp[0]):
        selec_pix = np.where( ((r_it-Dr/2.0) < r[0]) * (r[0] < (r_it+Dr/2.0)) )
        selec_y, selec_x = selec_pix
        radial_std[:,r_id] = np.nanstd(cube[:,selec_y, selec_x],1)

    radial_std = np.squeeze(radial_std)

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

    cube_std[np.where(cube_std < treshold)] = np.nan

    cube_std = np.squeeze(cube_std)

    return cube_std

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

def gauss2d(x, y, amplitude = 1.0, xo = 0.0, yo = 0.0, sigma_x = 1.0, sigma_y = 1.0, theta = 0, offset = 0):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g

def candidate_detection(filename,toPNG='', toDraw=False, logFile='' ):
    '''
    Should take into account PSF wavelength dependence.
    3d convolution to take into account spectral shift if useful
    but 3d conv takes too long
    '''
    hdulist = pyfits.open(filename)

    #grab the data and headers
    cube = hdulist[1].data
    exthdr = hdulist[1].header
    prihdr = hdulist[0].header

    # Get input cube dimensions
    # Transform a 2D image into a cube with one slice because the code below works for cubes
    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        cube_cpy = cube[None,:]
        nl = 1

    # Build the PSF.
    # It should be the real PSF derived from the satellite spots and reduced through KLIP.
    # Build the grid for PSF stamp.
    x_PSF_grid, y_PSF_grid = np.meshgrid(np.arange(0,12,1)-5,np.arange(0,12,1)-5)
    # Use a simple 2d gaussian PSF for now. The width is probably not even the right one.
    # I just set it so that "it looks" right.
    PSF = gauss2d(x_PSF_grid, y_PSF_grid,1.0,0.0,0.0,1.5,1.5)
    # Normalize the PSF with a norm 2
    PSF /= np.sqrt(np.sum(PSF**2))
    # In case we want to do 3d conv we should have a PSF cube. It should also include a priori on the spectrum.
    #PSF_cube = [PSF for _ in np.arange(nl)]
    #PSF_cube = np.squeeze(PSF_cube)

    # The detection is performed on the flat cube.
    # It could be a first step and then the spectra could be looked at to check if they are planets.
    # IT could also be done with a 3d conv if we take into account the wavelength dependence of the PSF.
    flat_cube = np.nanmean(cube,0)

    try:
        # Retrieve the center of the image from the fits keyword.
        center = [exthdr['PPSFCENTX'], exthdr['PSFCENTY']]
    except:
        # If the keywords could not be found.
        center = [(nx-1)/2,(ny-1)/2]

    # Build as grids of x,y coordinates.
    # The center is in the middle of the array and the unit is the pixel.
    # If the size of the array is even 2n x 2n the center coordinates is [n,n].
    x_grid, y_grid = np.meshgrid(np.arange(0,nx,1)-center[0],np.arange(0,ny,1)-center[1])

    # Calculate the standard deviation map.
    # the standard deviation is calculated on annuli of width Dr. There is an annulus centered every dr.
    dr = 2 ; Dr = 10 ;
    flat_cube_std = radialStdMap(flat_cube,dr,Dr, centroid=center, treshold=10**(-3))

    # Replace nans by zeros.
    # Otherwise we loose the border of the image because of the convolution which will give NaN if there is any NaNs in
    # the area.
    flat_cube[np.where(np.isnan(flat_cube))] = 0.0

    # Perform a "match-filtering". Simply the convolution of the transposed PSF with the image.
    # It should still be zero if there is no signal. Assuming a zero mean noise after KLIP.
    # The value at a point centered on a planet should be the L2 norm of the planet.
    flat_cube_convo = convolve2d(flat_cube,PSF,mode="same")
    # The 3d convolution takes a while so the idea is to detect the interesting spot in the 2d flat cube and then
    # perform the 3d convolution on the cube stamp around it.

    # Divide the convolved flat cube by the standard deviation map to get something similar to a SNR ratio.
    # It is not the classic SNR because it is not the simple flat cube.
    flat_cube_SNR = flat_cube_convo/flat_cube_std
    #Save a copy of the flat cube because we will mask the detected spots as the algorithm goes.
    flat_cube_SNR_cpy = copy(flat_cube_SNR)
    # use convolve for 3d convolution

    # Definition of the different masks used in the following.
    stamp_nrow = 15
    stamp_ncol = 15
    # Mask to remove the spots already checked in flat_cube_SNR.
    stamp_mask_nans = np.zeros((stamp_nrow,stamp_ncol))
    # Mask used to act on flat_cube_mask
    stamp_mask_ones = np.zeros((stamp_nrow,stamp_ncol))
    # Mask that will divide by 10 the flux of the image everywhere but where there is a valid candidate.
    flat_cube_mask = 0.1 * np.ones((ny,nx))

    # List of local maxima
    checked_spots_list = []
    # List of local maxima that are valid candidates
    candidates_list = []

    if logFile:
        f_out = open(logFile+'-logFile.txt', 'w')

    # Count the number of valid detected candidates.
    N_candidates = 0.0
    # Maximum number of iterations on local maxima.
    max_attempts = 30
    # START FOR LOOP
    for k in np.arange(max_attempts):
        # Find the maximum value in the current SNR map. At each iteration the previous maximum is masked out.
        max_val = np.nanmax(flat_cube_SNR)
        # Locate the maximum by retrieving its coordinates
        max_ind = np.where( flat_cube_SNR == max_val )
        row_id,col_id = max_ind[0][0],max_ind[1][0]
        x_max_pos, y_max_pos = x_grid[max_ind],y_grid[max_ind]

        #Extract a stamp around the maximum in the flat cube (without the convolution)
        row_m = np.floor(stamp_nrow/2.0)
        row_p = np.ceil(stamp_nrow/2.0)
        col_m = np.floor(stamp_ncol/2.0)
        col_p = np.ceil(stamp_ncol/2.0)
        stamp = flat_cube[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
        stamp_x_grid = x_grid[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
        stamp_y_grid = y_grid[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]

        # Definition of a 2D gaussian fitting to be used on the stamp.
        g_init = models.Gaussian2D(max_val,x_max_pos,y_max_pos,1.0,1.0)
        # JB Todo: look at the different fitting methods.
        fit_g = fitting.LevMarLSQFitter()

        # Fit the 2d Gaussian to the stamp
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        g = fit_g(g_init, stamp_x_grid, stamp_y_grid, stamp)

        # Calculate the fitting residual using the square root of the summed squared error.
        khi = np.sqrt(np.sum((stamp-g(stamp_x_grid, stamp_y_grid))**2))

        # JB Todo: Should explore the meaning of 'ierr' but I can't find a good clear documentation of astropy.fitting
        sig_min = 1.0 ; sig_max = 2.0 ;
        valid_potential_planet = (khi/max_val < 3 and # Check that the fit was good enough. Not a weird looking speckle.
                                 fit_g.fit_info['ierr'] == 1 and # Check that the fitting actually occured. Actually I have no idea what the number mean but it looks like when it succeeds it is 1.
                                 sig_min < g.x_stddev < sig_max and # Check the shape of the gaussian. It should be more or less circular and not too wide to be a planet.
                                 sig_min < g.y_stddev < sig_max)
        # Todo: Remove prints, for debugging purpose only
        print(k,row_id,col_id,khi,max_val,g.x_mean+0.0, g.y_mean+0.0, g.x_stddev+0.0, g.y_stddev+0.0)
        print(valid_potential_planet)

        if logFile:
            myStr = str(k)+', '+\
                    str(row_id)+', '+\
                    str(col_id)+', '+\
                    str(khi)+', '+\
                    str(max_val)+', '+\
                    str(g.x_mean+0.0)+', '+\
                    str(g.y_mean+0.0)+', '+\
                    str(g.x_stddev+0.0)+', '+\
                    str(g.y_stddev+0.0)+'\n'
            f_out.write(myStr)
            f_out.write(str(valid_potential_planet)+'\n')

        # If the spot verifies the conditions above it is considered as a valid candidate.
        checked_spots_list.append((k,row_id,col_id,max_val,x_max_pos,y_max_pos))

        # Mask the spot around the maximum we just found.
        r_stamp = abs((stamp_x_grid-g.x_mean) +(stamp_y_grid-g.y_mean)*1j)
        stamp_mask_nans[np.where(r_stamp < 0.33*max([stamp_nrow, stamp_ncol]))] = np.NaN
        flat_cube_SNR[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] += stamp_mask_nans

        #JB Todo: Analyze the stamp cube to look for spectrum or others. It could help distinguishing bad pixels.
        '''
        #plt.figure(3)
        #f,ax = plt.subplots(6,6)
        if valid_potential_planet:
            stamp_cube = cube[:,(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
            for l_it in np.arange(nl):
                #Sky subtraction
                stamp_slice = stamp_cube[l_it]
                surroundings_indices = np.where(r_stamp > 0.5*max([stamp_nrow, stamp_ncol]))
                #print(surroundings_indices[0].size)
                sky_func_init = models.Polynomial2D(degree=1)
                fit_sky = fitting.LevMarLSQFitter()
                sky_func = fit_sky(sky_func_init, stamp_x_grid[surroundings_indices], stamp_y_grid[surroundings_indices], stamp_slice[surroundings_indices])
                #print(np.nanmean(stamp_slice[surroundings_indices]))
                #plt.figure(2)
                #coucou = np.zeros(stamp_slice.shape)
                #coucou[surroundings_indices] = stamp_slice[surroundings_indices]
                #plt.imshow(coucou)
                stamp_slice -= sky_func(stamp_x_grid, stamp_y_grid)
                #print(np.nanmean(stamp_slice[surroundings_indices]))
                #bonjour = np.zeros(stamp_slice.shape)
                #bonjour[surroundings_indices] = stamp_slice[surroundings_indices]
                #if l_it != nl-1:
                #    ax[l_it/6,l_it-6*(l_it/6)].imshow(stamp_slice,interpolation='nearest')
                stamp_cube[l_it] = stamp_slice
            #convolve(stamp_cube) = PSF_cube
        '''

        # If the spot is a valid candidate we add it to the candidates list
        if valid_potential_planet:
            # Build the mask to highlight the candidates.
            stamp_mask_ones[np.where(r_stamp < 3*max([g.x_stddev, g.y_stddev]))] = 0.9
            flat_cube_mask[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] += stamp_mask_ones
            # Increment the number of detected candidates
            N_candidates += 1
            # Append the useful things about the candidate in the list.
            candidates_list.append((N_candidates,row_id,col_id,max_val,g))
    # END FOR LOOP

    if logFile:
        f_out.close()

    # START IF STATEMENT
    if toDraw or toPNG:
        # Highlight the detected candidates in the 2d flat cube.
        print(N_candidates)
        z = flat_cube_SNR
        plt.figure(3)
        plt.imshow(flat_cube[::-1,:]*flat_cube_mask[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
        ax = plt.gca()
        for candidate in candidates_list:
            candidate_it,row_id,col_id,max_val,g = candidate
            ax.annotate(str(candidate_it), fontsize=20, color = 'white', xy=(g.x_mean+0.0, g.y_mean+0.0),
                    xycoords='data', xytext=(g.x_mean+15, g.y_mean-15),
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    linewidth = 1.,
                                    color = 'white')
                    )

        # Shouw the local maxima in the SNR map
        flat_cube_checkedArea = flat_cube_SNR_cpy #np.log10(abs(flat_cube))
        plt.figure(4)
        plt.imshow(flat_cube_checkedArea[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
        ax = plt.gca()
        for spot in checked_spots_list:
            spot_it,row_id,col_id,max_val,x_max_pos,y_max_pos = spot
            ax.annotate(str(spot_it), fontsize=20, color = 'white', xy=(x_max_pos+0.0, y_max_pos+0.0),
                    xycoords='data', xytext=(x_max_pos+15, y_max_pos-15),
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    linewidth = 1.,
                                    color = 'white')
                    )
    # END IF STATEMENT

    if toPNG:
        plt.figure(3)
        plt.savefig(toPNG+'-candidates.png', bbox_inches='tight')
        plt.figure(4)
        plt.savefig(toPNG+'-allSpots.png', bbox_inches='tight')
    #'''

    if toDraw:
        plt.show()


    #pyfits.writeto("/Users/jruffio/gpi/pyklip/outputs/SNR.fits", flat_cube_SNR, clobber=True)
    return 1
# END candidate_detection() DEFINITION

def test(c, d=0.0):
    print(d)
    c = 2

def jbgradient(slice):
    ny,nx = slice.shape
    #print(ny,nx)
    xder = np.zeros((ny,nx))
    yder = np.zeros((ny,nx))
    xder[:,1:nx] = slice[:,1:nx] - slice[:,0:(nx-1)]
    yder[1:ny,:] = slice[1:ny,:] - slice[0:(ny-1),:]
    #print(slice[:,1:(nx)].shape, slice[:,0:(nx-1)].shape)
    return [xder,yder]

if __name__ == "__main__":
    import astropy.io.fits as pyfits
    import glob
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    from astropy import wcs
    import os
    #different importants depending on if python2.7 or python3
    import sys
    import cmath
    import time
    from scipy.interpolate import interp1d
    import astropy.io.fits as pyfits

    print('coucou')
    #

    #filelist = glob.glob("D:/gpi/pyklip/fits/baade/*.fits")

    #filelist = glob.glob("/Users/jruffio/gpi/pyklip/outputs/baade-KL20-speccube.fits")
    filelist = glob.glob("/Users/jruffio/gpi/pyklip/outputs/HD114174-KL20-speccube.fits")

    #dataset = GPI.GPIData(filelist)

    filename = filelist[0]
    #candidate_detection(filename, toPNG="Baade", logFile='Baade')
    candidate_detection(filename, toPNG="HD114174", logFile='HD114174')



    #filt_cube, n_badpixs = badPixelFilter(cube)
    #print(n_badpixs)
    #imgplot =
    '''
    plt.figure(1)
    plt.imshow(r)
    '''
    '''
    sig = 1
    slice = cube[15,:,:]
    g = np.array(jbgradient(slice))
    #g = np.gradient(slice)
    print(np.array(g).shape)
    gradNorm = np.sqrt(np.sum(g**2,0))
    ker = np.ones((2*sig+1,2*sig+1))
    #ker[sig,sig] = 0.0
    flux = convolve2d(abs(slice), ker, mode='same')
    #print((1/(10.*(sig**3))), np.median(gradNorm/flux))
    #gradNorm[np.where(gradNorm/flux < (1/(100.*(sig**3))))] = np.nan
    #print(gradNorm.shape)
    plt.figure(1)
    plt.imshow(gradNorm, interpolation="nearest")
    #plt.figure(2)
    #plt.imshow(np.log10(abs(filt_cube[15,:,:])), interpolation="nearest")
    '''

    '''
    import pyklip.parallelized as parallelized

    parallelized.klip_dataset(dataset, outputdir="D:/gpi/pyklip/outputs/", fileprefix="test", numbasis = [10,20])
    '''
    '''
    #print(exthdr['XTENSION'])
    w = wcs.WCS(header=exthdr, naxis=[1,2])
    hdulist = w.to_fits()
    #print(hdulist[0])
    hdulist[0].header.update({'coucou':42,'bonjou':53})
    hdulist.append(hdulist[0])
    hdulist[1].data = cube
    hdulist.writeto("/Users/jruffio/gpi/pyklip/outputs/coucou.fits", clobber=True)
    cards = [pyfits.Card(keyword='coucou',value=43),pyfits.Card(keyword='bonjour',value=52)]
    pyfits.writeto("/Users/jruffio/gpi/pyklip/outputs/coucou.fits", np.array([[1,1],[0,0]]), header = pyfits.Header(cards), clobber=True)
    hdulist.close()
    '''