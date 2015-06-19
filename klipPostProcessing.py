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
import glob, os
from sys import stdout

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

    IWA = 0
    while np.isnan(image[x_cen,y_cen+IWA]):
        IWA += 1

    # Build the x and y coordinates grids
    x, y = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
    # Calculate the radial distance of each pixel
    r = abs(x +y*1j)

    mask = np.ones((ny,nx))
    mask[np.where(np.isnan(image))] = np.nan

    inner_mask = copy(mask)
    inner_mask[np.where(r > IWA+2.)] = 1

    outer_mask = copy(mask)
    outer_mask[np.where(np.isnan(inner_mask))] = 1
    OWA = np.min(r[np.where(np.isnan(outer_mask))])

    return IWA,OWA,inner_mask,outer_mask

def mask_known_objects(cube,prihdr,GOI_list_filename, mask_radius = 7):

    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        cube = cube[None,:]
        nl = 1

    width = 2*mask_radius+1
    stamp_x_grid, stamp_y_grid = np.meshgrid(np.arange(0,width,1)-width/2,np.arange(0,width,1)-width/2)
    stamp_mask = np.ones((width,width))
    r_stamp = abs((stamp_x_grid) +(stamp_y_grid)*1j)
    stamp_mask[np.where(r_stamp < mask_radius)] = np.nan

    try:
        # OBJECT: keyword in the primary header with the name of the star.
        object_name = prihdr['OBJECT'].strip().replace (" ", "_")
    except:
        object_name = "UNKNOWN_OBJECT"

    print(object_name)

    candidates_list = []

    with open(GOI_list_filename, 'r') as GOI_list:
        for myline in GOI_list:
            if not myline.startswith("#"):
                GOI_name, status, k,potential_planet,max_val_criter,x_max_pos,y_max_pos, row_id,col_id = myline.rstrip().split(",")
                if GOI_name == object_name:
                    candidates_list.append((int(k),bool(potential_planet),float(max_val_criter),float(x_max_pos),float(y_max_pos), int(row_id),int(col_id)))


    row_m = np.floor(width/2.0)
    row_p = np.ceil(width/2.0)
    col_m = np.floor(width/2.0)
    col_p = np.ceil(width/2.0)

    for candidate in candidates_list:
        k,potential_planet,max_val_criter,x_max_pos,y_max_pos, k,l = candidate
        cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = np.tile(stamp_mask,(nl,1,1)) * cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)]

    return np.squeeze(cube)

def get_image_PDF(image,(IWA,OWA),N,centroid = None):
    ny,nx = image.shape

    image_mask = np.ones((ny,nx))
    image_mask[np.where(np.isnan(image))] = 0

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


    for it, rminmax in enumerate(annuli_radii):
        r_min,r_max = rminmax

        where_ring = np.where((r_min< r_grid) * (r_grid < r_max) * image_mask)

        #im = copy(image)
        #im[where_ring] = 1.0
        #plt.imshow(im,interpolation="nearest")
        #plt.show()

        im_std = np.std(image[where_ring])
        bins = np.arange(-10.*im_std,10.*im_std,im_std/10.)
        data = image[where_ring]
        im_histo = np.histogram(data, bins=bins)[0]
        N_inHisto = np.sum(im_histo)
        im_histo = im_histo/float(N_inHisto)
        print(im_histo)
        print(N_inHisto,np.size(where_ring[0]))


        if 1:
            im_histo_max = np.max(im_histo)
    
            g_init = models.Gaussian1D(amplitude=np.max(im_histo), mean=0.0, stddev=im_std)
            fit_g = fitting.LevMarLSQFitter()
            warnings.simplefilter('ignore')
            g = fit_g(g_init, bins[0:bins.size-1], im_histo)

            fig = 1
            plt.figure(fig,figsize=(12,12))
            plt.plot(bins[0:bins.size-1],im_histo,'bx-', markersize=5,linewidth=3)
            plt.plot(bins[0:bins.size-1],g(bins[0:bins.size-1]),'c--',linewidth=1)
    
            plt.xlabel('criterion value', fontsize=20)
            plt.ylabel('Probability of the value', fontsize=20)
            plt.xlim((-10.* im_std,10.*im_std))
            plt.grid(True)
            ax = plt.gca()
            #ax.text(10.*im_std, 2.0*im_histo_max/5., str(N_high_SNR_planets),
            #        verticalalignment='bottom', horizontalalignment='right',
            #        color='red', fontsize=50)
            #ax.text(3.*im_std, 2.0*im_histo_max/5., str(N_low_SNR_planets),
            #        verticalalignment='bottom', horizontalalignment='right',
            #        color='red', fontsize=50)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            ax.legend(['flat cube histogram','flat cube histogram (Gaussian fit)','planets'], loc = 'upper right', fontsize=12)
            #plt.savefig(outputDir+"histo_"+filename+".png", bbox_inches='tight')
            #plt.clf()
            #plt.close(fig)
            ax.set_yscale('log')
            plt.ylim((10**-7,1))
            plt.show()
    return

def get_spatial_cova_func(image,(IWA,OWA),N,centroid = None,n_neigh=11, corr = False):
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

def get_spectral_cova_func(cube,(IWA,OWA),N,centroid = None,n_neigh=3, corr = False):

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



def stamp_based_StdMap(image,width = 20,inner_mask_radius = 2.5):
    '''
    Return a map of the same size as input image with its standard deviation for each location in the image.
    The standard deviation is calculated as follow:
    For each pixel a 2D stamp is extracted around it. The width of the stamp is equal to the input width.
    The center disk of radius inner_mask_radius is masked out from the stamp.
    The standard deviation value for the center pixel is the standard deviation in the non masked surroundings.

    Inputs:
        image: 2D (ny,nx) array from which to calculate the standard deviation.
        width: Width of the stamp to be extracted at each pixel.
        inner_mask_radius: Radius of the inner disk to be masked out form the stamp.

    Output:
        im_std: The standard deviation map.
    '''
    ny,nx = image.shape

    stamp_PSF_x_grid, stamp_PSF_y_grid = np.meshgrid(np.arange(0,width,1)-width/2,np.arange(0,width,1)-width/2)
    stamp_PSF_mask = np.ones((width,width))
    r_PSF_stamp = abs((stamp_PSF_x_grid) +(stamp_PSF_y_grid)*1j)
    stamp_PSF_mask[np.where(r_PSF_stamp < inner_mask_radius)] = np.nan

    im_std = np.zeros((ny,nx)) + np.nan
    row_m = np.floor(width/2.0)
    row_p = np.ceil(width/2.0)
    col_m = np.floor(width/2.0)
    col_p = np.ceil(width/2.0)

    for k in np.arange(10,ny-10):
        stdout.write("\r{0}/{1}".format(k,ny))
        stdout.flush()
        for l in np.arange(10,nx-10):
            stamp_cube = image[(k-row_m):(k+row_p), (l-col_m):(l+col_p)]
            im_std[k,l] = np.nanstd(stamp_cube*stamp_PSF_mask)


    return im_std


def ringSection_based_StdMap(image,Dr = 8,Dth = 45,Dpix_mask = 2.5,centroid = None):
    '''
    Return a map of the same size as input image with its standard deviation for each location in the image.
    The standard deviation is calculated as follow:
    For each pixel a 2D stamp is extracted around it.
    The stamp has the shape of a piece of ring centered on the current pixel.
    The piece of rings are constructed such that all the stamp have the same number of pixel roughly no matter the
    separation.

    Inputs:
        image: 2D (ny,nx) array from which to calculate the standard deviation.
        Dr: Width of the ring.
        Dth: Angle of the section for a separation of 100 pixels.
        Dpix_mask: radius of the PSF that should be masked in pixel
        centroid: [col,row] with the coordinates of the center of the image.

    Output:
        im_std: The standard deviation map.
    '''
    ny,nx = image.shape

    x, y = np.meshgrid(np.arange(nx)-centroid[0], np.arange(ny)-centroid[1])
    #x-axis points right
    #y-axis points down
    #theta measured from y to x.
    #print(x)
    #print(y)
    r_grid = abs(x +y*1j)
    th_grid = np.arctan2(x,y)
    #print(th_grid)
    #print(np.arctan2(10,10))
    #print(np.arctan2(-10,10))
    #print(np.arctan2(10,-10))
    #print(np.arctan2(-10,-10))

    Dth_rad = Dth/180.*np.pi
    im_std = np.zeros((ny,nx)) + np.nan

    for k in np.arange(10,ny-10):
        #stdout.write("\r{0}/{1}".format(k,ny))
        #stdout.flush()
        for l in np.arange(10,nx-10):
            if not np.isnan(image[k,l]):
                r = r_grid[(k,l)]
                th = th_grid[(k,l)]

                delta_th_grid = np.mod(th_grid - th +np.pi,2.*np.pi)-np.pi

                ring_section = ((r-Dr/2.0) < r_grid) * (r_grid < (r+Dr/2.0)) * \
                                (abs(delta_th_grid)<(+Dth_rad*50./r)) * \
                                (abs(delta_th_grid)>(Dpix_mask/r))
                                #((+Dpix_mask/r) < delta_th_grid) * (delta_th_grid < (-Dpix_mask/r))
                ring_section_id = np.where(ring_section)
                im_std[k,l] = np.nanstd(image[ring_section_id])


                if 0 and ((k == 75 and l == 150) or ((k == 173 and l == 165))):
                    print("coucou")
                    print(r,th,Dth*50./r,Dpix_mask/r,+Dpix_mask/r,-Dpix_mask/r)
                    image[ring_section_id] = 100
                    print(image[ring_section_id].size)
                    print("coucou")
                    plt.figure(2)
                    plt.imshow(image, interpolation="nearest")
                    plt.show()


    return im_std

def gauss2d(x, y, amplitude = 1.0, xo = 0.0, yo = 0.0, sigma_x = 1.0, sigma_y = 1.0, theta = 0, offset = 0):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g

def calculate_metrics(filename,
                        metrics = None,
                        PSF_cube = None,
                        outputDir = None,
                        folderName = None,
                        spectrum = None,
                        mute = False ):
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

    # Calculate the standard deviation map.
    # the standard deviation is calculated on annuli of width Dr. There is an annulus centered every dr.
    dr = 2 ; Dr = 5 ;
    #flat_cube_std = radialStdMap(flat_cube,dr,Dr, centroid=center)
    flat_cube_std = ringSection_based_StdMap(flat_cube, centroid=center)

    # Divide the convolved flat cube by the standard deviation map to get the SNR.
    flat_cube_SNR = flat_cube/flat_cube_std


    if metrics is not None:
        if len(metrics) == 1 and not isinstance(metrics,list):
            metrics = [metrics]

        if "weightedFlatCube" in metrics:
            weightedFlatCube = np.average(cube,axis=0,weights=spectrum)
            #weightedFlatCube_SNR = weightedFlatCube/radialStdMap(weightedFlatCube,dr,Dr, centroid=center)
            weightedFlatCube_SNR = weightedFlatCube/ringSection_based_StdMap(weightedFlatCube,centroid=center)

        if "matchedFilter" in metrics and "shape" not in metrics:
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

            matchedFilter_SNR_map = matchedFilter_map/radialStdMap(matchedFilter_map,dr,Dr, centroid=center)

        if "shape" in metrics and "matchedFilter" not in metrics:
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
            shape_SNR_map = shape_map/radialStdMap(shape_map,dr,Dr, centroid=center)

        if "matchedFilter" in metrics and "shape" in metrics:
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
            #shape_SNR_map = shape_map/radialStdMap(shape_map,dr,Dr, centroid=center)
            shape_SNR_map = shape_map/ringSection_based_StdMap(shape_map,centroid=center)
            #matchedFilter_SNR_map = matchedFilter_map/radialStdMap(matchedFilter_map,dr,Dr, centroid=center)
            matchedFilter_SNR_map = matchedFilter_map/ringSection_based_StdMap(matchedFilter_map,centroid=center)


    ## ortho_criterion is actually the sine squared between the two vectors
    ## ortho_criterion_map = 1 - criterion_map
    ## ratio_shape_SNR = 10
    ## criterion_map = np.minimum(ratio_shape_SNR*shape_map,flat_cube_SNR)

    ##
    # Preliminaries and some sanity checks before saving the metrics maps fits file.
    if outputDir is None:
        outputDir = "./"
    else:
        outputDir = outputDir+"/"

    if folderName is None:
        folderName = "/default_out/"
    else:
        folderName = folderName+"/"

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
        if metrics is not None:
            if "weightedFlatCube" in metrics:
                hdulist2[1].data = weightedFlatCube
                hdulist2.writeto(outputDir+folderName+prefix+'-weightedFlatCube.fits', clobber=True)
                hdulist2[1].data = weightedFlatCube_SNR
                hdulist2.writeto(outputDir+folderName+prefix+'-weightedFlatCube_SNR.fits', clobber=True)
            if "matchedFilter" in metrics:
                hdulist2[1].data = matchedFilter_map
                hdulist2.writeto(outputDir+folderName+prefix+'-matchedFilter.fits', clobber=True)
                hdulist2[1].data = matchedFilter_SNR_map
                hdulist2.writeto(outputDir+folderName+prefix+'-matchedFilter_SNR.fits', clobber=True)
            if "shape" in metrics:
                hdulist2[1].data = shape_map
                hdulist2.writeto(outputDir+folderName+prefix+'-shape.fits', clobber=True)
                hdulist2[1].data = shape_SNR_map
                hdulist2.writeto(outputDir+folderName+prefix+'-shape_SNR.fits', clobber=True)
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
    shape_filename_list = glob.glob(metrics_foldername+"/*-shape_SNR.fits")
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


    # Ignore all the pixel too close from an edge with nans
    flat_cube_nans = np.where(np.isnan(criterion_map))
    flat_cube_mask = np.ones((ny,nx))
    flat_cube_mask[flat_cube_nans] = np.nan
    #widen the nans region
    conv_kernel = np.ones((5,5))
    flat_cube_wider_mask = convolve2d(flat_cube_mask,conv_kernel,mode="same")
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

    logFile_all = open(metrics_foldername+"/"+prefix+'-detectionLog_all.txt', 'w')
    logFile_candidates = open(metrics_foldername+"/"+prefix+'-detectionLog_candidates.txt', 'w')

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
        plt.savefig(metrics_foldername+"/"+prefix+'-detectionIm_candidates.png', bbox_inches='tight')
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
        plt.savefig(metrics_foldername+"/"+prefix+'-detectionIm_all.png', bbox_inches='tight')
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

def calculate_metrics_in_dir_per_file(filename,pipeline_dir,
                                      metrics = None,
                                      directory = "./",
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
    splitted_before_KL = splitted_name[0].split("/")
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
        filelist_ori_PSFs_cube = glob.glob(directory+"/"+prefix+"-original_radial_PSF_cube.fits")

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
        wv,sat_spot_spec = spec.get_gpi_filter(pipeline_dir,filter)
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
            spectrum_name = spectrum_name_it.split("/")
            spectrum_name = spectrum_name[len(spectrum_name)-1].split(".")[0]
        else:
            spectrum_name = "satSpotSpec"

        if outputDir == '':
            outputDir = directory
        folderName = "/planet_detec_"+prefix+"/"+spectrum_name+"/"



        if spectrum_name_it != "":
            if not mute:
                print("spectrum model: "+spectrum_name_it)
            # Interpolate the spectrum of the planet based on the given filename
            wv,planet_sp = spec.get_planet_spectrum(spectrum_name_it,filter)

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

        if not planet_detection_only:
            if not mute:
                print("Calling calculate_metrics() on "+filename)
            calculate_metrics(filename,
                              metrics,
                                PSF_cube = PSF_cube,
                                outputDir = outputDir,
                                folderName = folderName,
                                spectrum=spectrum,
                                mute = mute)


        if not metrics_only:
            if not mute:
                print("Calling candidate_detection() on "+outputDir+folderName)
            if 1:
                candidate_detection(outputDir+folderName,
                                    mute = mute)

def calculate_metrics_in_dir_per_file_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call calculate_metrics_in_dir_per_file() with a tuple of parameters.
    """
    return calculate_metrics_in_dir_per_file(*params)

def planet_detection_in_dir_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call planet_detection_in_dir() with a tuple of parameters.
    """
    return planet_detection_in_dir(*params)

def planet_detection_in_dir(pipeline_dir,
                            directory = "./",
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
        pipeline_dir: GPI pipeline directory. E.g. "/Users/jruffio/gpi/pipeline/".
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
        filelist_klipped_cube = glob.glob(directory+"/pyklip-*-KL"+numbasis+"-speccube.fits")
    else:
        filelist_klipped_cube = glob.glob(directory+"/"+filename_prefix_is+"-KL"+numbasis+"-speccube.fits")
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
                pool.map(calculate_metrics_in_dir_per_file_star, itertools.izip(filelist_klipped_cube,
                                                                               itertools.repeat(pipeline_dir),
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
                    calculate_metrics_in_dir_per_file(filename,
                                                     pipeline_dir,
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





def planet_detection_campaign(pipeline_dir,
                              campaign_dir = "./"):
    outputDir = ''
    star_type = ''
    metrics = None

    filename_filter = "pyklip-*-k100a7s4m3"
    numbasis = 20
    spectrum_model = ["/Users/jruffio/gpi/pyklip/spectra/t800g100nc.flx",
                      "/Users/jruffio/gpi/pyklip/spectra/t700g178nc.flx",
                      "/Users/jruffio/gpi/pyklip/spectra/t650g18nc.flx",
                      "/Users/jruffio/gpi/pyklip/spectra/t650g32nc.flx",
                      "/Users/jruffio/gpi/pyklip/spectra/t650g56nc.flx",
                      ""]
    #spectrum_model = ["/Users/jruffio/gpi/pyklip/spectra/t650g18nc.flx",
    #                  "/Users/jruffio/gpi/pyklip/spectra/t650g32nc.flx",
    #                  "/Users/jruffio/gpi/pyklip/spectra/t650g56nc.flx"]
    star_type = "G4"
    metrics = ["weightedFlatCube","matchedFilter","shape"]
    user_defined_PSF_cube = "/Users/jruffio/gpi/pyklip/outputs/dropbox_prior_test/pyklipH-S20141218-k100a7s4m3-original_radial_PSF_cube.fits"

    inputDirs = []
    for inputDir in os.listdir(campaign_dir):
        if not inputDir.startswith('.'):
            inputDirs.append(campaign_dir+inputDir+"/autoreduced/")

            if 1:
                inputDir = campaign_dir+inputDir+"/autoreduced/"
                planet_detection_in_dir(pipeline_dir,
                                        inputDir,
                                        filename_prefix_is=filename_filter,
                                        spectrum_model=spectrum_model,
                                        star_type=star_type,
                                        metrics = metrics,
                                        numbasis=numbasis,
                                        user_defined_PSF_cube=user_defined_PSF_cube,
                                        threads = True,
                                        mute = False)

    if 0:
        N_threads = len(inputDirs)
        print(N_threads)
        pool = mp.Pool(processes=N_threads)
        pool.map(planet_detection_in_dir_star, itertools.izip(itertools.repeat(pipeline_dir),
                                                                       inputDirs,
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

