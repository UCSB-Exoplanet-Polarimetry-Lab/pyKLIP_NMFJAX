import numpy as np
from scipy import optimize

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

def inject_planet(frames, centers, peakfluxes, astr_hdrs, radius, pa, fwhm=3.5):
    """
    Injects a fake planet into a dataset

    Inputs:
        frames: array of (N,y,x) for N is the total number of frames
        centers: array of size (N,2) of [x,y] coordiantes of the image center
        peakflxes: array of size N of the peak flux of the fake planet in each frame
        astr_hdrs: array of size N of the WCS headers
        radius: separation of the planet from the star
        pa: parallactic angle (in degrees) of  planet (if that is a quantity that makes any sense)

    Outputs:
        saves result in input "frames" variable
    """

    for frame, center, peakflux, astr_hdr in zip(frames, centers, peakfluxes, astr_hdrs):
        #calculate the x,y location of the planet for each image
        theta = covert_pa_to_image_polar(pa, astr_hdr)

        x_pl = radius * np.cos(theta*np.pi/180.) + center[0]
        y_pl = radius * np.sin(theta*np.pi/180.) + center[1]

        #now that we found the planet location, inject it
        frame = _inject_gaussian_planet(frame, x_pl, y_pl, peakflux, fwhm=fwhm)

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
    x,y = np.meshgrid(np.arange(ysize*1.0), np.arange(xsize*1.0))

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

def inject_disk(frames, centers, peakfluxes, astr_hdrs, pa, fwhm=3.5):
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

    for frame, center, peakflux, astr_hdr in zip(frames, centers, peakfluxes, astr_hdrs):
        #calculate the x,y location of the planet for each image
        theta = covert_pa_to_image_polar(pa, astr_hdr)

        #now that we found the planet location, inject it
        frame += _construct_gaussian_disk(center[0], center[1], frame.shape[1], frame.shape[0], peakflux, theta, fwhm=fwhm)


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

def gaussfit2d(frame, xguess, yguess, searchrad=5, guessfwhm=3.0, guesspeak=1):
    """
    Fits a 2d gaussian to the data at point (xguess, yguess)

    Inputs:
        frame: the data - Array of size (y,x)
        xguess,yguess: location to fit the 2d guassian to (should be pretty accurate)
        searchrad: 1/2 the length of the box used for the fit
        guessfwhm: approximate fwhm to fit to
        guesspeak: approximate flux

    Ouputs:
        peakflux: the peakflux of the gaussian
    """
    x0 = np.round(xguess)
    y0 = np.round(yguess)
    #construct our searchbox
    fitbox = frame[y0-searchrad:y0+searchrad+1, x0-searchrad:x0+searchrad+1]
    #construct the residual to the fit
    errorfunction = lambda p: np.ravel(gauss2d(*p)(*np.indices(fitbox.shape)) - fitbox)

    #mask bad pixels
    fitbox[np.where(np.isnan(fitbox))] = 0

    #do a least squares fit. Note that we use searchrad for x and y centers since we're narrowed it to a box of size
    #(2searchrad+1,2searchrad+1)
    p, success = optimize.leastsq(errorfunction, (searchrad, searchrad, guesspeak, guessfwhm/(2 * np.sqrt(2*np.log(2)))))

    xfit = p[0]
    yfit = p[1]
    peakflux = p[2]
    fwhm = p[3] * (2 * np.sqrt(2*np.log(2)))

    print("Fitparams", xfit, yfit, peakflux, fwhm)

    return peakflux


def retrieve_planet_flux(frames, centers, astr_hdrs, sep, pa, searchrad=5, guessfwhm=3.0, guesspeak=1):
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

    Outputs:
        peakfluxes: array of N peak planet fluxes
    """
    peakfluxes = []
    #loop over all of them
    for frame, center, astr_hdr in zip(frames, centers, astr_hdrs):
        #find the pixel location on this image
        theta = covert_pa_to_image_polar(pa, astr_hdr)
        x = sep*np.cos(np.radians(theta)) + center[0]
        y = sep*np.sin(np.radians(theta)) + center[1]

        #calculate the flux
        flux = gaussfit2d(frame, x, y, searchrad=searchrad, guessfwhm=guessfwhm, guesspeak=guesspeak)

        peakfluxes.append(flux)

    return np.array(peakfluxes)
