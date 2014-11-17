import numpy as np
#import cmath
from scipy.interpolate import interp1d

def radialStd(cube,dr,Dr,centroid = None, r = None, r_samp = None):
    '''
    Return the standard deviation with respect to the radius computed in annuli of radial width Dr separated by dr.
    :return:
    '''
    nl,ny,nx = cube.shape

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

    return radial_std


def radialStdMap(cube,dr,Dr,centroid = None,treshold=10**(-6)):
    '''
    Return an cube of the same size as cube on which the value for each pixel is the standard deviation of the original
    cube inside an annulus of width Dr.
    :return:
    '''

    nl,ny,nx = cube.shape
    r = [np.nan]
    r_samp = [np.nan]
    radial_std = radialStd(cube,dr,Dr,centroid, r = r, r_samp = r_samp)

    cube_std = np.zeros((nl,ny,nx))
    radial_std_nans = np.isnan(radial_std)
    radial_std[radial_std_nans] = 0.0
    for l_id in np.arange(nl):
        f = interp1d(r_samp[0], radial_std[l_id,:], kind='cubic',bounds_error=False, fill_value=np.nan)
        a = f(r[0].reshape(nx*ny))
        cube_std[l_id,:,:] = a.reshape(ny,nx)

    cube_std[np.where(cube_std < treshold)] = np.nan

    return cube_std

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

    filelist = glob.glob("D:/gpi/pyklip/fits/baade/*.fits")

    #dataset = GPI.GPIData(filelist)


    hdulist = open(filelist[0])

    #grab the data and headers
    cube = hdulist[1].data
    exthdr = hdulist[1].header
    prihdr = hdulist[0].header

    myim = cube[0,:,:] ;
    nl,ny,nx = cube.shape
    print(nl,ny,nx)

    dr = 2 ; Dr = 10 ;
    r = [np.nan]
    r_samp = [np.nan]
    radial_std = radialStd(cube,dr,Dr, r = r, r_samp = r_samp)

    cube_std = radialStdMap(cube,dr,Dr,treshold=10**(-3))

    cube_SNR = cube/cube_std
    #imgplot =
    '''
    plt.figure(1)
    plt.imshow(r)
    '''
    plt.figure(1)
    plt.imshow(cube_std[0,:,:], interpolation="nearest")
    plt.figure(2)
    plt.imshow(cube_SNR[0,:,:], interpolation="nearest")

    '''
    plt.figure(2)
    plt.plot(radial_std.T) #[1,:]
    plt.ylabel('some numbers')
    '''
    plt.show()

    '''
    import pyklip.parallelized as parallelized

    parallelized.klip_dataset(dataset, outputdir="D:/gpi/pyklip/outputs/", fileprefix="test", numbasis = [10,20])
    '''