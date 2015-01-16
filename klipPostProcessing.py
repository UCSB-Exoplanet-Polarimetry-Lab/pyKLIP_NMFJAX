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


def extract_PSF(filename, radii, thetas, stamp_width = 10):
    hdulist = pyfits.open(filename)
    slice = hdulist[1].data[2,:,:]
    ny,nx = slice.shape
    prihdr = hdulist[1].header
    center = [prihdr['PSFCENTX'], prihdr['PSFCENTY']]

    x, y = np.meshgrid(np.arange(ny), np.arange(nx))
    #x.shape = (x.shape[0] * x.shape[1])
    #y.shape = (y.shape[0] * y.shape[1])
    x -= center[0]
    y -= center[1]
    x_planets = np.dot(np.array([radii]).transpose(),np.array([np.cos(np.radians(thetas))]))
    y_planets = np.dot(np.array([radii]).transpose(),np.array([np.sin(np.radians(thetas))]))

    dn = stamp_width
    for k in np.arange(radii.size):
        for l in np.arange(thetas.size):
            x_plnt = np.ceil(x_planets[k,l]+center[0])
            y_plnt = np.ceil(y_planets[k,l]+center[1])

            stamp = slice[(y_plnt-dn/2):(y_plnt+dn/2),(x_plnt-dn/2):(x_plnt+dn/2)]
            stamp_x = x[(y_plnt-dn/2):(y_plnt+dn/2),(x_plnt-dn/2):(x_plnt+dn/2)]-x_planets[k,l]
            stamp_y = y[(y_plnt-dn/2):(y_plnt+dn/2),(x_plnt-dn/2):(x_plnt+dn/2)]-y_planets[k,l]

            #plt.imshow(stamp,interpolation = 'nearest')
            #plt.show()
            #break

            if k == 0 and l == 0:
                PSFs_stamps = [stamp]
                PSFs_x = [stamp_x]
                PSFs_y = [stamp_y]
            else:
                PSFs_stamps = np.concatenate((PSFs_stamps,[stamp]),axis=0)
                PSFs_x = np.concatenate((PSFs_x,[stamp_x]),axis=0)
                PSFs_y = np.concatenate((PSFs_y,[stamp_y]),axis=0)

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
            '''
            if p == 10 and q == 10:
                image[arc_id] = 100
                print(image[arc_id].size)
                plt.figure(2)
                plt.imshow(image, interpolation="nearest")
                plt.show()
            '''

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
    r_samp[0] = np.arange(dr,max(r[0].reshape(np.size(r[0]))),dr)

    radial_std = np.zeros((nl,np.size(r_samp[0])))

    for r_id, r_it in enumerate(r_samp[0]):
        selec_pix = np.where( ((r_it-Dr/2.0) < r[0]) * (r[0] < (r_it+Dr/2.0)) )
        selec_y, selec_x = selec_pix
        data = cube[:,selec_y, selec_x]
        #print(data.shape)
        radial_std[:,r_id] = np.nanstd(data,1)
        for l_it in np.arange(nl):
            out_3sig = np.where(data[l_it,:]>(3.*radial_std[l_it,r_id]))
            if out_3sig is not ():
                data[l_it,out_3sig] = np.nan
        radial_std[:,r_id] = np.nanstd(data,1)

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

def gauss2d(x, y, amplitude = 1.0, xo = 0.0, yo = 0.0, sigma_x = 1.0, sigma_y = 1.0, theta = 0, offset = 0):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g


def candidate_detection(filename,PSF = None, toPNG='', toFits='', toDraw=False, logFile='' ):
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
    if PSF is None:
        PSF /= np.sqrt(np.sum(PSF**2))
    else:
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
    #flat_cube = np.nanmean(cube,0)
    flat_cube = nanmedian(cube,0)


    #a = np.concatenate((flat_cube[None,1:ny,1:nx],flat_cube[None,0:(ny-1),1:nx],flat_cube[None,1:ny,0:(nx-1)],flat_cube[None,0:(ny-1),0:(nx-1)]),axis=0)
    #flat_cube[0:(ny-1),0:(nx-1)] = nanmedian(a,axis=0)

    try:
        # Retrieve the center of the image from the fits keyword.
        center = [prihdr['PSFCENTX'], prihdr['PSFCENTY']]
    except:
        # If the keywords could not be found.
        center = [(nx-1)/2,(ny-1)/2]

    flat_cube = subtract_radialMed(flat_cube,5,20,center)
    flat_cube_cpy = copy(flat_cube)

    # Build as grids of x,y coordinates.
    # The center is in the middle of the array and the unit is the pixel.
    # If the size of the array is even 2n x 2n the center coordinates is [n,n].
    x_grid, y_grid = np.meshgrid(np.arange(0,nx,1)-center[0],np.arange(0,ny,1)-center[1])

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

    # Calculate the standard deviation map.
    # the standard deviation is calculated on annuli of width Dr. There is an annulus centered every dr.
    dr = 2 ; Dr = 5 ;
    flat_cube_std = radialStdMap(flat_cube_convo,dr,Dr, centroid=center, treshold=10**(-3))
    flat_cube = copy(flat_cube_cpy)

    # Divide the convolved flat cube by the standard deviation map to get something similar to a SNR ratio.
    # It is not the classic SNR because it is not the simple flat cube.
    flat_cube_SNR = flat_cube_convo/flat_cube_std
    #Save a copy of the flat cube because we will mask the detected spots as the algorithm goes.
    flat_cube_SNR_cpy = copy(flat_cube_SNR)
    # use convolve for 3d convolution

    # Definition of the different masks used in the following.
    stamp_nrow = 13
    stamp_ncol = 13
    # Mask to remove the spots already checked in flat_cube_SNR.
    stamp_mask = np.ones((stamp_nrow,stamp_ncol))
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
    max_attempts = 75
    # START FOR LOOP
    for k in np.arange(max_attempts):
        # Find the maximum value in the current SNR map. At each iteration the previous maximum is masked out.
        max_val_SNR = np.nanmax(flat_cube_SNR)
        # Locate the maximum by retrieving its coordinates
        max_ind = np.where( flat_cube_SNR == max_val_SNR )
        row_id,col_id = max_ind[0][0],max_ind[1][0]
        x_max_pos, y_max_pos = x_grid[max_ind],y_grid[max_ind]
        max_val = flat_cube[max_ind]

        #Extract a stamp around the maximum in the flat cube (without the convolution)
        row_m = np.floor(stamp_nrow/2.0)
        row_p = np.ceil(stamp_nrow/2.0)
        col_m = np.floor(stamp_ncol/2.0)
        col_p = np.ceil(stamp_ncol/2.0)
        stamp = flat_cube[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
        stamp_x_grid = x_grid[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
        stamp_y_grid = y_grid[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]

        # Definition of a 2D gaussian fitting to be used on the stamp.
        g_init = models.Gaussian2D(max_val,x_max_pos,y_max_pos,2.0,2.0)
        # JB Todo: look at the different fitting methods.
        fit_g = fitting.LevMarLSQFitter()

        # Fit the 2d Gaussian to the stamp
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        g = fit_g(g_init, stamp_x_grid, stamp_y_grid, stamp,np.abs(stamp)**0.5)




        # Calculate the fitting residual using the square root of the summed squared error.
        khi = np.sqrt(np.nansum((stamp-g(stamp_x_grid, stamp_y_grid))**2*np.abs(stamp)**0.5))
        khi0 = np.sqrt(np.nansum((stamp)**2*np.abs(stamp)**0.5))

        # JB Todo: Should explore the meaning of 'ierr' but I can't find a good clear documentation of astropy.fitting
        sig_min = 1.0 ; sig_max = 3.0 ;
        valid_potential_planet = (max_val_SNR > 3.0 and
                                 fit_g.fit_info['ierr'] <= 2 and
                                 khi/khi0 < 1.0 and # Check that the fit was good enough. Not a weird looking speckle.
                                 sig_min < g.x_stddev < sig_max and # Check the shape of the gaussian. It should be more or less circular and not too wide to be a planet.
                                 sig_min < g.y_stddev < sig_max)
        #fit_g.fit_info['ierr'] == 1 and # Check that the fitting actually occured. Actually I have no idea what the number mean but it looks like when it succeeds it is 1.

        # Todo: Remove prints, for debugging purpose only
        print(k,max_val_SNR,khi0,khi/khi0,max_val, g.x_stddev+0.0, g.y_stddev+0.0)
        print(valid_potential_planet,fit_g.fit_info['ierr'])

        if k == 26 and 0:
            #print(fit_g.objective_function([g.amplitude,g.x_mean,g.y_mean,g.x_stddev,g.y_stddev,g.theta],g,stamp**2))
            print(np.max(stamp))
            print(np.max(g(stamp_x_grid, stamp_y_grid)))
            print(stamp_x_grid[1,:])
            print(stamp_y_grid[:,1])
            print(max_val)
            print(x_max_pos)
            print(y_max_pos)
            #print(stamp)
            #print(stamp-g(stamp_x_grid, stamp_y_grid))
            plt.figure(1)
            plt.imshow(stamp, interpolation="nearest")
            plt.figure(2)
            plt.imshow(g(stamp_x_grid, stamp_y_grid), interpolation="nearest")
            plt.show()

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
        stamp_mask[np.where(r_stamp < 0.66*0.5*max([stamp_nrow, stamp_ncol]))] = 0.0
        flat_cube_SNR[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] *= stamp_mask

        # If the spot is a valid candidate we add it to the candidates list
        if valid_potential_planet:
            if 0:
                stamp_cube = cube[:,(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
                stamp_cube_mask = np.tile(stamp_mask[None,:,:],(nl,1,1))
                stamp_cube[np.where(stamp_cube_mask != 0)] = 0.0
                spectrum = np.nansum(np.nansum(stamp_cube,axis=1),axis=1)
                #plt.figure(2)
                #plt.imshow(stamp_cube[5,:,:], interpolation="nearest")
                plt.figure(3)
                plt.plot(spectrum)
                plt.show()


            flat_cube[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] *= stamp_mask

            # Build the mask to highlight the candidates.
            flat_cube_mask[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] += (1-stamp_mask)*0.9
            # Increment the number of detected candidates
            N_candidates += 1
            # Append the useful things about the candidate in the list.
            candidates_list.append((k,row_id,col_id,max_val,g))
    # END FOR LOOP

    if logFile:
        f_out.close()

    # START IF STATEMENT
    if toDraw or toPNG:
        # Highlight the detected candidates in the 2d flat cube.
        print(N_candidates)
        z = flat_cube_SNR
        plt.figure(3)
        plt.imshow(flat_cube_cpy[::-1,:]*flat_cube_mask[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
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

    if toFits:
        hdulist2 = pyfits.HDUList()
        hdulist2.append(pyfits.PrimaryHDU(header=prihdr))
        hdulist2.append(pyfits.ImageHDU(header=exthdr, data=flat_cube_SNR_cpy, name="Sci"))
        hdulist2.writeto(toFits+'-flatCubeSNR.fits', clobber=True)
        hdulist2[1].data = flat_cube_cpy
        hdulist2.writeto(toFits+'-flatCube.fits', clobber=True)
        hdulist2.close()


    if toDraw:
        plt.show()


    #pyfits.writeto("/Users/jruffio/gpi/pyklip/outputs/SNR.fits", flat_cube_SNR, clobber=True)
    return 1
# END candidate_detection() DEFINITION


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

    from scipy.interpolate import griddata
    from scipy.interpolate import bisplrep
    from scipy.interpolate import bisplev

    thetas = 90+np.array([54.7152978, -35.2847022, -125.2847022, -215.2847022])
    radii = np.array([16.57733255996081, 32.33454364876502, 48.09175473756924, 63.84896582637346, 79.60617691517767])
    filename_PSFs = "/Users/jruffio/gpi/pyklip/outputs/baade-PSFs-KLmodes-all-PSFs.fits"
    #filename_PSFs = "/Users/jruffio/gpi/pyklip/outputs/baade-PSFs2-KLmodes-all-PSFs.fits"
    #filename_PSFs = "/Users/jruffio/gpi/pyklip/outputs/baade-PSFs15-KLmodes-all-PSFs.fits"
    #filename_PSFs = "/Users/jruffio/gpi/pyklip/outputs/baade-PSFs20-KLmodes-all-PSFs.fits"
    PSF = extract_PSF(filename_PSFs, radii, thetas)
    #print(PSF.shape)
    #plt.figure(20)
    #plt.imshow(PSF,interpolation = 'nearest')
    #plt.show()

    filelist = glob.glob("/Users/jruffio/gpi/pyklip/outputs/baade/baade-h-k300a7s4m3-KL20-speccube.fits")
    #filelist = glob.glob("/Users/jruffio/gpi/pyklip/outputs/baade-PSFs2-KL20-speccube.fits")
    #filelist = glob.glob("/Users/jruffio/gpi/pyklip/outputs/baade-PSFs15-KL20-speccube.fits")
    #filelist = glob.glob("/Users/jruffio/gpi/pyklip/outputs/baade-PSFs20-KL20-speccube.fits")

    #dataset = GPI.GPIData(filelist)

    filename = filelist[0]
    candidate_detection(filename,PSF = PSF, toDraw=True, toFits="Baade")#toPNG="Baade", logFile='Baade')
    #candidate_detection(filename, toDraw=True)#toPNG="HD114174", logFile='HD114174')






















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
