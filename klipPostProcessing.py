import numpy as np
#import cmath
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from scipy.signal import convolve
import astropy.io.fits as pyfits
from astropy.modeling import models, fitting

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

    #TODO centroid should be different for each slice
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

def get_spots(cube):
    '''
    Should take into account PSF wavelength dependence.
    3d convolution to take into account spectral shift if useful
    but 3d conv takes too long
    '''
    print('bonjour')
    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        cube_cpy = cube[None,:]
        nl = 1

    x_grid, y_grid = np.meshgrid(np.arange(0,nx,1)-(nx-1)/2,np.arange(0,ny,1)-(ny-1)/2)

    x_PSF_grid, y_PSF_grid = np.meshgrid(np.arange(0,11,1)-5,np.arange(0,11,1)-5)
    PSF = gauss2d(x_PSF_grid, y_PSF_grid,1.0,0.0,0.0,1.5,1.5)
    # In case we want to do 3d conv
    #PSF_cube = [PSF for _ in np.arange(nl)]
    #PSF_cube = np.squeeze(PSF_cube)

    flat_cube = np.nanmean(cube,0)

    dr = 2 ; Dr = 10 ;
    flat_cube_std = radialStdMap(flat_cube,dr,Dr,treshold=10**(-3))


    # use convolve for 3d convolution
    flat_cube_convo = convolve2d(flat_cube,PSF,mode="same")
    flat_cube_SNR = flat_cube_convo/flat_cube_std

    max_val = np.nanmax(flat_cube_SNR)
    max_ind = np.where( flat_cube_SNR == max_val )
    x_max_pos, y_max_pos = x_grid[max_ind],y_grid[max_ind]

    return flat_cube_SNR, x_max_pos, y_max_pos

def test(c):
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
    from astropy.io.fits import open
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

    print('coucou')

    #filelist = glob.glob("D:/gpi/pyklip/fits/baade/*.fits")

    filelist = glob.glob("/Users/jruffio/gpi/pyklip/outputs/baade4Jason-KL20-speccube.fits")

    #dataset = GPI.GPIData(filelist)


    hdulist = open(filelist[0])

    #grab the data and headers
    cube = hdulist[1].data
    exthdr = hdulist[1].header
    prihdr = hdulist[0].header

    myim = cube[0,:,:] ;
    nl,ny,nx = cube.shape
    print(nl,ny,nx)

    #z, x, y = get_spots(cube)

    print('bonjour')
    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        cube_cpy = cube[None,:]
        nl = 1

    x_grid, y_grid = np.meshgrid(np.arange(0,nx,1)-(nx-1)/2,np.arange(0,ny,1)-(ny-1)/2)

    x_PSF_grid, y_PSF_grid = np.meshgrid(np.arange(0,11,1)-5,np.arange(0,11,1)-5)
    PSF = gauss2d(x_PSF_grid, y_PSF_grid,1.0,0.0,0.0,1.5,1.5)
    # In case we want to do 3d conv
    #PSF_cube = [PSF for _ in np.arange(nl)]
    #PSF_cube = np.squeeze(PSF_cube)

    flat_cube = np.nanmean(cube,0)
    flat_cube_cpy = flat_cube[:]

    dr = 2 ; Dr = 10 ;
    flat_cube_std = radialStdMap(flat_cube,dr,Dr,treshold=10**(-3))

    #replace nans by zeros
    flat_cube[np.where(np.isnan(flat_cube))] = 0.0

    # use convolve for 3d convolution
    flat_cube_convo = convolve2d(flat_cube,PSF,mode="same")
    flat_cube_SNR = flat_cube_convo/flat_cube_std


    stamp_nrow = 15
    stamp_ncol = 15
    stamp_mask_nans = np.zeros((stamp_nrow,stamp_ncol))
    stamp_mask_ones = np.zeros((stamp_nrow,stamp_ncol))
    flat_cube_mask = 0.1 * np.ones((ny,nx))

    N_candidates = 0.0
    max_attempts = 10
    for k in np.arange(max_attempts):
        max_val = np.nanmax(flat_cube_SNR)

        #if max_val < 3.0:
        #    break

        max_ind = np.where( flat_cube_SNR == max_val )
        row_id,col_id = max_ind[0][0],max_ind[1][0]
        x_max_pos, y_max_pos = x_grid[max_ind],y_grid[max_ind]

        #Extract stamp
        row_m = np.floor(stamp_nrow/2.0)
        row_p = np.ceil(stamp_nrow/2.0)
        col_m = np.floor(stamp_ncol/2.0)
        col_p = np.ceil(stamp_ncol/2.0)
        stamp = flat_cube_SNR[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
        stamp_cpy = stamp[:]
        stamp_x_grid = x_grid[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
        stamp_y_grid = y_grid[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]

        g_init = models.Gaussian2D(max_val,x_max_pos,y_max_pos,1.0,1.0)
        fit = fitting.LevMarLSQFitter()

        # Ignore model linearity warning from the fitter
        #warnings.simplefilter('ignore')
        g = fit(g_init, stamp_x_grid, stamp_y_grid, stamp)


        print(row_id,col_id,max_val,g.x_mean+0.0, g.y_mean+0.0, g.x_stddev+0.0, g.y_stddev+0.0)
        valid_potential_planet = 1.0 < g.x_stddev < 3.0 and 1.0 < g.y_stddev < 3.0
        print(valid_potential_planet)

        '''
        if k == 2:
            plt.figure(2)
            plt.imshow(stamp_cpy, interpolation="nearest")
            plt.figure(3)
            plt.imshow(g(stamp_x_grid, stamp_y_grid), interpolation="nearest")
            break
        '''

        r_stamp = abs((stamp_x_grid-g.x_mean) +(stamp_y_grid-g.y_mean)*1j)
        if valid_potential_planet:
            stamp_mask_nans[np.where(r_stamp < 3*max([g.x_stddev, g.y_stddev]))] = np.NaN
        else:
            stamp_mask_nans[np.where(r_stamp < 0.5*max([stamp_nrow, stamp_ncol]))] = np.NaN

        flat_cube_SNR[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] += stamp_mask_nans

        if valid_potential_planet:
            stamp_mask_ones[np.where(r_stamp < 3*max([g.x_stddev, g.y_stddev]))] = 0.9
            flat_cube_mask[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] += stamp_mask_ones
            N_candidates += 1


    print(N_candidates)
    z = flat_cube_SNR
    plt.figure(3)
    plt.imshow(flat_cube*flat_cube_mask, interpolation="nearest")
    plt.figure(4)
    plt.imshow(flat_cube_SNR, interpolation="nearest")
    #plt.figure(5)
    #plt.imshow(g(x_grid, y_grid), interpolation="nearest")
    plt.show()

    pyfits.writeto("/Users/jruffio/gpi/pyklip/outputs/SNR.fits", flat_cube_SNR, clobber=True)


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