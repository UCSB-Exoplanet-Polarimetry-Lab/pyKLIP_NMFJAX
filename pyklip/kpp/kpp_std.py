__author__ = 'JB'

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sys import stdout

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

def radialStdMap(cube,dr = 2,Dr = 5,centroid = None, rejection = False,treshold=10**(-6)):
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