import numpy as np
from scipy import optimize
import scipy.ndimage as ndimage
import scipy.interpolate as interp

def covert_pa_to_image_polar(pa, astr_hdr):
    """
    Given a parallactic angle (angle from N to Zenith rotating in the Eastward direction), calculate what
    polar angle theta (angle from +X CCW towards +Y) it corresponds to

    Input:
        pa: parallactic angle in degrees
        astr_hdr: wcs astrometry header (astropy.wcs)

    Output:
        theta: polar angle in degrees
    """
    rot_det = astr_hdr.wcs.cd[0,0] * astr_hdr.wcs.cd[1,1] - astr_hdr.wcs.cd[0,1] * astr_hdr.wcs.cd[1,0]
    if rot_det < 0:
        rot_sgn = -1.
    else:
        rot_sgn = 1.
    #calculate CCW rotation from +Y to North in radians
    rot_YN = np.arctan2(rot_sgn * astr_hdr.wcs.cd[0,1],rot_sgn * astr_hdr.wcs.cd[0,0])
    #now that we know where north it, find the CCW rotation from +Y to find location of planet
    rot_YPA = rot_YN - rot_sgn*pa*np.pi/180. #radians
    #rot_YPA = rot_YN + pa*np.pi/180. #radians

    theta = rot_YPA * 180./np.pi + 90.0 #degrees
    return theta

def _inject_gaussian_planet(frame, xpos, ypos, amplitude, fwhm=3.5):
    """
    Injects a fake planet with a Gaussian PSF into a dataframe

    Inputs:
        frame: a 2D data frame
        xpos,ypos: x,y location (in pixels) where the planet should be
        amplitude: peak of the Gaussian PSf (in appropriate units not dictacted here)
        fwhm: fwhm of gaussian

    Outputs:
        frame: the frame with the injected planet
    """

    #figure out sigma when given FWHM
    sigma = fwhm/(2.*np.sqrt(2*np.log(2)))

    #create a meshgrid for the psf
    x,y = np.meshgrid(np.arange(1.0*frame.shape[1]), np.arange(1.0*frame.shape[0]))
    x -= xpos
    y -= ypos

    psf = amplitude * np.exp(-(x**2./(2.*fwhm) + y**2./(2.*fwhm)))

    frame += psf
    return frame

def inject_planet(frames, centers, inputflux, astr_hdrs, radius, pa, fwhm=3.5, thetas=None):
    """
    Injects a fake planet into a dataset either using a Gaussian PSF or an input PSF

    Inputs:
        frames: array of (N,y,x) for N is the total number of frames
        centers: array of size (N,2) of [x,y] coordiantes of the image center
        inputflux: EITHER array of size N of the peak flux of the fake planet in each frame (will inject a Gaussian PSF)
                   OR array of size (N,psfy,psfx) of template PSFs. The brightnesses should be scaled and the PSFs
                   should be centered at the center of each of the template images
        astr_hdrs: array of size N of the WCS headers
        radius: separation of the planet from the star
        pa: parallactic angle (in degrees) of  planet (if that is a quantity that makes any sense)
        fwhm: fwhm (in pixels) of gaussian
        thetas: ignore PA, supply own thetas (CCW angle from +x axis toward +y)
                array of size N

    Outputs:
        saves result in input "frames" variable
    """

    if thetas is None:
        thetas = np.array([covert_pa_to_image_polar(pa, astr_hdr) for astr_hdr in astr_hdrs])

    for frame, center, inputpsf, theta in zip(frames, centers, inputflux, thetas):
        #calculate the x,y location of the planet for each image
        #theta = covert_pa_to_image_polar(pa, astr_hdr)
        x_pl = radius * np.cos(theta*np.pi/180.) + center[0]
        y_pl = radius * np.sin(theta*np.pi/180.) + center[1]

        #now that we found the planet location, inject it
        #check whether we are injecting a gaussian of a template PSF
        if type(inputpsf) == np.ndarray:
            #shift psf so that center is aligned
            #calculate center of box
            boxsize = inputpsf.shape[0]
            boxcent = (boxsize-1)/2
            #create coordinates to align PSF with image
            xpsf,ypsf = np.meshgrid(np.arange(frame.shape[1]), np.arange(frame.shape[0]))
            xpsf = xpsf - x_pl + boxcent
            ypsf = ypsf - y_pl + boxcent
            #inject into frame
            frame += ndimage.map_coordinates(inputpsf, [ypsf, xpsf], mode='constant', cval=0.0)

        else:
            frame = _inject_gaussian_planet(frame, x_pl, y_pl, inputpsf, fwhm=fwhm)

def _construct_gaussian_disk(x0,y0, xsize,ysize, intensity, angle, fwhm=3.5):
    """
    Constructs a rectangular slab for a disk with a vertical gaussian profile

    Inputs:
        x0,y0: center of disk
        xsize, ysize: x and y dimensions of the output image
        intensity: peak intensity of the disk (whatever units you want)
        angle: orientation of the disk plane (CCW from +x axis) [degrees]
        fwhm: FWHM of guassian profile (in pixels)

    Outputs:
        disk_img: 2d array of size (ysize,xsize) with the image of the disk
    """

    #construct a coordinate system
    x,y = np.meshgrid(np.arange(xsize*1.0), np.arange(ysize*1.0))

    #center at image center
    x -= x0
    y -= y0

    #rotate so x is parallel to the disk plane, y is vertical cuts through the disk
    #so need to do a CW rotation
    rad_angle = angle * np.pi/180.
    xp = x * np.cos(rad_angle) + y * np.sin(rad_angle) + x0
    yp = -x * np.sin(rad_angle) + y * np.cos(rad_angle) + y0

    sigma = fwhm/(2 * np.sqrt(2*np.log(2)))
    disk_img = intensity / (np.sqrt(2*np.pi) * sigma) * np.exp(-(yp-y0)**2/(2*sigma**2))

    return disk_img

def inject_disk(frames, centers, inputfluxes, astr_hdrs, pa, fwhm=3.5):
    """
    Injects a fake disk into a dataset

    Inputs:
        frames: array of (N,y,x) for N is the total number of frames
        centers: array of size (N,2) of [x,y] coordiantes of the image center
        peakflxes: array of size N of the peak flux of the fake disk in each frame
        astr_hdrs: array of size N of the WCS headers
        pa: parallactic angle (in degrees) of disk plane (if that is a quantity that makes any sense)

    Outputs:
        saves result in input "frames" variable
    """

    for frame, center, inputpsf, astr_hdr in zip(frames, centers, inputfluxes, astr_hdrs):
        #calculate the rotation angle in the pixel plane
        theta = covert_pa_to_image_polar(pa, astr_hdr)

        if type(inputpsf) == np.ndarray:
            # inject real data
            # rotate and grab pixels of disk that can be injected into the image
            # assume disk is centered
            xpsf0 = inputpsf.shape[1]/2
            ypsf0 = inputpsf.shape[0]/2
            #grab the pixel numbers for the data
            ximg, yimg = np.meshgrid(np.arange(frame.shape[1]), np.arange(frame.shape[0]))
            #rotate them to extract the disk at the right angle
            ximg -= center[0]
            yimg -= center[1]
            theta_rad = np.radians(theta)
            ximgp = ximg * np.cos(theta_rad) + yimg * np.sin(theta_rad) + xpsf0
            yimgp = -ximg * np.sin(theta_rad) + yimg * np.cos(theta_rad) + ypsf0
            #interpolate and inject datqa
            frame += ndimage.map_coordinates(inputpsf, [yimgp, ximgp])
        else:
            #inject guassian bar into data
            frame += _construct_gaussian_disk(center[0], center[1], frame.shape[1], frame.shape[0], inputpsf, theta, fwhm=fwhm)


def gauss2d(x0, y0, peak, sigma):
    '''
    2d symmetric guassian function for guassfit2d

    Inputs:
        x0,y0: center of gaussian
        peak: peak amplitude of guassian
        sigma: stddev in both x and y directions
    '''
    sigma *= 1.0
    return lambda x,y: peak*np.exp( -(((x-x0)/sigma)**2+((y-y0)/sigma)**2)/2)

def gaussfit2d(frame, xguess, yguess, searchrad=5, guessfwhm=7, guesspeak=1, refinefit=True):
    """
    Fits a 2d gaussian to the data at point (xguess, yguess)

    Inputs:
        frame: the data - Array of size (y,x)
        xguess,yguess: location to fit the 2d guassian to (should be pretty accurate)
        searchrad: 1/2 the length of the box used for the fit
        guessfwhm: approximate fwhm to fit to
        guesspeak: approximate flux
        refinefit: whether to refine the fit of the position of the guess

    Ouputs:
        peakflux: the peakflux of the gaussian
    """
    x0 = np.round(xguess)
    y0 = np.round(yguess)
    #construct our searchbox
    fitbox = np.copy(frame[y0-searchrad:y0+searchrad+1, x0-searchrad:x0+searchrad+1])

    #mask bad pixels
    fitbox[np.where(np.isnan(fitbox))] = 0
 
    #fit a least squares gaussian to refine the fit on the source, otherwise just use the guess
    if refinefit:
        #construct the residual to the fit
        errorfunction = lambda p: np.ravel(gauss2d(*p)(*np.indices(fitbox.shape)) - fitbox)
   
        #do a least squares fit. Note that we use searchrad for x and y centers since we're narrowed it to a box of size
        #(2searchrad+1,2searchrad+1)
        p, success = optimize.leastsq(errorfunction, (searchrad, searchrad, guesspeak, guessfwhm/(2 * np.sqrt(2*np.log(2)))))
        
        xfit = p[0]
        yfit = p[1]
        peakflux = p[2]
        fwhm = p[3] * (2 * np.sqrt(2*np.log(2)))
    else:
        xfit = xguess-x0 + searchrad
        yfit = yguess-y0 + searchrad
   
    #ok now, to really calculate fwhm and flux, because we really need that right, we're going
    # to use what's in the GPI DRP pipeline to measure satellite spot fluxes instead of
    # a least squares gaussian fit. Apparently my least squares fit relatively underestimates
    # the flux so it's not consistent.
    # grab a radial profile of the fit
    rs = np.arange(searchrad)
    thetas = np.arange(0,2*np.pi, 1./searchrad) #divide maximum circumfrence into equal parts
    radprof = [np.mean(ndimage.map_coordinates(fitbox, [thisr*np.sin(thetas)+yfit, thisr*np.cos(thetas)+xfit])) for thisr in rs]
    #now interpolate this radial profile to get fwhm
    try:
        radprof_interp = interp.interp1d(radprof, rs)
        fwhm = 2*radprof_interp(np.max(fitbox)/2)
    except ValueError:
        fwhm = searchrad

    #now calculate flux
    xfitbox, yfitbox = np.meshgrid(np.arange(0,2* searchrad+1, 1.0)-xfit, np.arange(0, 2*searchrad+1, 1.0)-yfit)
    #correlate data with a gaussian to get flux
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    ## attempt to calculate sigma using moments
    #sigmax = np.sqrt(np.nansum(xfitbox*xfitbox*fitbox)/np.nansum(fitbox) - (np.nansum(xfitbox*fitbox)/np.nansum(fitbox))**2)
    #sigmay = np.sqrt(np.nansum(yfitbox*yfitbox*fitbox)/np.nansum(fitbox) - (np.nansum(yfitbox*fitbox)/np.nansum(fitbox))**2)
    #sigma = np.nanmean([sigmax, sigmay])
    #print(sigma, sigmax, sigmay)
    gmask = np.exp(-(xfitbox**2+yfitbox**2)/(2.*sigma**2))
    outofaper = np.where(xfitbox**2 + yfitbox**2 > searchrad**2)
    gmask[outofaper] = 0 
    corrflux = np.nansum(fitbox*gmask)/np.sum(gmask*gmask)

    print("Fitparams", xfit, yfit, corrflux, fwhm)

    return corrflux, fwhm, xfit, yfit

def retrieve_planet_flux(frames, centers, astr_hdrs, sep, pa, searchrad=7, guessfwhm=3.0, guesspeak=1, refinefit=False, thetas=None):
    """
    Retrives the peak flux of the planet from a series of frames given a separation and PA

    Inputs:
        frames: N frames of data - Array of size (N,y,x)
        centers: array of size (N,2) of [x,y] coordiantes of the image center
        astr_hdrs: array of N astr_hdrs
        sep: radial distance in pixels
        PA: parallactic angle in degrees
        searchrad: 1/2 the length of the box used for the fit
        guessfwhm: approximate fwhm to fit to
        guesspeak: approximate flux
        refinefit: whether or not to refine the positioning of the planet
        thetas: ignore PA, supply own thetas (CCW angle from +x axis toward +y)
                array of size N

    Outputs:
        peakfluxes: array of N peak planet fluxes
    """
    peakfluxes = []

   
    if thetas is None:
        thetas = np.array([covert_pa_to_image_polar(pa, astr_hdr) for astr_hdr in astr_hdrs])        

    #loop over all of them
    for frame, center, theta in zip(frames, centers, thetas):
        #find the pixel location on this image
        #theta = covert_pa_to_image_polar(pa, astr_hdr)
        x = sep*np.cos(np.radians(theta)) + center[0]
        y = sep*np.sin(np.radians(theta)) + center[1]
        print(x,y)
        #calculate the flux
        flux, fwhm, xfit, yfit = gaussfit2d(frame, x, y, searchrad=searchrad, guessfwhm=guessfwhm, guesspeak=guesspeak, refinefit=refinefit)
        
        peakfluxes.append(flux)

    return np.array(peakfluxes)
