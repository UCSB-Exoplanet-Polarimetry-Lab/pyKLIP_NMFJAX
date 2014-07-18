import numpy as np
import numpy.linalg as la
import scipy.ndimage as ndimage


def klip_math(sci, ref_psfs, numbasis):
    """
    Helper function for KLIP that does the linear algebra
    
    Inputs:
        sci: array of length p containing the science data
        ref_psfs: N x p array of the N reference PSFs that 
                  characterizes the PSF of the p pixels
        numbasis: number of KLIP basis vectors to use

    Outputs:
        sub_img: array of lenght p that is the PSF subtracted data

    TODO:
        make numbasis to be any number of KLIP cutoffs and return all of them
    """
    #import pdb

    #for the science image, subtract the mean and mask bad pixels
    sci_mean_sub = sci - np.nanmean(sci)
    sci_nanpix = np.where(np.isnan(sci_mean_sub))
    sci_mean_sub[sci_nanpix] = 0

    #do the same for the reference PSFs
    #playing some tricks to vectorize the subtraction
    ref_psfs_mean_sub = ref_psfs - np.nanmean(ref_psfs, axis=1)[:, None]
    ref_psfs_mean_sub[np.where(np.isnan(ref_psfs_mean_sub))] = 0

    #calculate the covariance matrix for the reference PSFs
    #note that numpy.cov normalizes by p-1 to get the NxN covariance matrix
    #we have to correct for that a few lines down when consturcting the KL 
    #vectors since that's not part of the equation in the KLIP paper
    covar_psfs = np.cov(ref_psfs_mean_sub)

    #calculate eigenvalues and eigenvectors of covariance matrix
    evals, evecs = la.eigh(covar_psfs)  #function for symmetric matrices

    #sort the eigenvalues and eigenvectors (unfortunately smallest first)
    eig_args_all = np.argsort(evals)

    #calculate the KL basis vectors
    kl_basis = np.dot(ref_psfs_mean_sub.T, evecs)
    kl_basis = kl_basis * (1. / np.sqrt(evals * (np.size(sci) - 1)))[None, :]  #multiply a value from each row

    #pick the largest however many to model PSF
    tot_basis = np.size(evals)
    #truncation either based on user input or maximum number of PSFs
    trunc_basis = np.min([numbasis, tot_basis])
    #remember that sorting sorted the smallest eigenvalues first
    eig_args = eig_args_all[tot_basis - trunc_basis: tot_basis]
    kl_basis = kl_basis[:, eig_args]

    #project KL vectors onto science image to construct model PSF
    inner_products = np.dot(sci_mean_sub, kl_basis)
    klip_psf = np.dot(inner_products, kl_basis.T)

    #subtract from original image to get final image
    sub_img = sci_mean_sub - klip_psf

    #restore NANs
    sub_img[sci_nanpix] = np.nan

    #pdb.set_trace()

    return sub_img


def align_and_scale(img, new_center, old_center=None, scale_factor=1):
    """
    Helper function that realigns and/or scales the image

    Inputs:
        img: 2D image to perform manipulation on
        new_center: 2 element tuple (xpos, ypos) of new image center
        old_center: 2 element tuple (xpos, ypos) of old image center
        scale_factor: how much the stretch/contract the image. Will we
                      scaled w.r.t the new_center (done after relaignment).
                      We will adopt the convention
                        >1: stretch image (shorter to longer wavelengths)
                        <1: contract the image (longer to shorter wvs)

    Outputs:
        resampled_img: shifted and/or scaled 2D image
    """
    #import scipy.interpolate as interp
    #import pdb

    #create the coordinate system of the image to manipulate for the transform
    dims = img.shape
    x, y = np.meshgrid(np.arange(dims[1], dtype=np.float32), np.arange(dims[0], dtype=np.float32))

    #if old_center is specified, realign the images
    if old_center is not None:
        dx = new_center[0] - old_center[0]
        dy = new_center[1] - old_center[1]
        x -= dx
        y -= dy

    #if scale_factor is specified, scale the images
    if scale_factor != 1:
        #conver to polar for scaling
        r = np.sqrt((x - new_center[0]) ** 2 + (y - new_center[1]) ** 2)
        theta = np.arctan2(y - new_center[1], x - new_center[0])  #theta range is [-pi,pi]

        r /= scale_factor

        #convert back to cartesian
        x = r * np.cos(theta) + new_center[0]
        y = r * np.sin(theta) + new_center[1]

    #resample image based on new coordinates
    #scipy uses y,x convention when meshgrid uses x,y
    #stupid scipy functions can't work with masked arrays (NANs)
    #and trying to use interp2d with sparse arrays is way to slow
    #hack my way out of this by picking a really small value for NANs and try to detect them after the interpolation
    minval = np.min([np.nanmin(img), 0.0])
    nanpix = np.where(np.isnan(img))
    img_copy = np.copy(img)
    img_copy[nanpix] = minval * 2.0
    resampled_img = ndimage.map_coordinates(img_copy, [y, x], cval=np.nan)
    resampled_img[np.where(resampled_img <= minval)] = np.nan
    resampled_img[nanpix] = np.nan

    #broken attempt at using sparse arrays with interp2d. Warning: takes forever to run
    #good_dat = np.where(~(np.isnan(img)))
    ##recreate old coordinate system
    #x0,y0 = np.meshgrid(np.arange(dims[0], dtype=np.float32), np.arange(dims[1], dtype=np.float32))
    #interpolated = interp.interp2d(x0[good_dat], y0[good_dat], img[good_dat], kind='cubic')
    #resampled_img = np.ones(img.shape) + np.nan
    #resampled_img[good] = interpolated(y[good],x[good])

    return resampled_img


def rotate(img, angle, center, old_center=None):
    """
    Rotate an image by the given angle about the given center.
    Optional: can shift the image from the oldcenter first before rotating
    """

    #create the coordinate system of the image to manipulate for the transform
    dims = img.shape
    x, y = np.meshgrid(np.arange(dims[1], dtype=np.float32), np.arange(dims[0], dtype=np.float32))

    if old_center is not None:
        dx = center[0] - old_center[0]
        dy = center[1] - old_center[1]
        x -= dx
        y -= dy
        center = old_center

    #do rotation. CW rotation formula to get a CCW of the image
    xp = (x-center[0])*np.cos(angle) + (y-center[1])*np.sin(angle) + center[0]
    yp = -(x-center[0])*np.sin(angle) + (y-center[1])*np.cos(angle) + center[1]

    #resample image based on new coordinates
    #scipy uses y,x convention when meshgrid uses x,y
    #stupid scipy functions can't work with masked arrays (NANs)
    #and trying to use interp2d with sparse arrays is way to slow
    #hack my way out of this by picking a really small value for NANs and try to detect them after the interpolation
    minval = np.min([np.nanmin(img), 0.0])
    nanpix = np.where(np.isnan(img))
    img_copy = np.copy(img)
    img_copy[nanpix] = minval * 2.0
    resampled_img = ndimage.map_coordinates(img_copy, [yp, xp], cval=np.nan)
    resampled_img[np.where(resampled_img <= minval)] = np.nan

    return resampled_img


def klip_adi(imgs, centers, parangs, annuli=5, subsections=4, movement=3, numbasis=50):
    """
    KLIP PSF Subtraction using angular differential imaging

    Inputs:
        imgs: array of 2D images for ADI. Shape of array (N,y,x)
        centers: N by 2 array of (x,y) coordinates of image centers
        parangs: N legnth array detailing parallactic angle of each image
        anuuli: number of annuli to use for KLIP
        subsections: number of sections to break each annuli into
        movement: minimum amount of movement (in pixels) of an astrophysical souce
                  to consider using that image for a refernece PSF
        numbasis: number of KL basis vectors to use

    Ouput:
        sub_imgs: array of 2D images (PSF subtracted)
    """

    #figure out error checking later..

    #save all bad pixels
    allnans = np.where(np.isnan(imgs))

    #use first image to figure out how to divide the annuli
    #TODO: should be smart about this in the future. Going to hard code some guessing
    #need to make the next 10 lines or so much smarter
    dims = imgs.shape
    x, y = np.meshgrid(np.arange(dims[2] * 1.0), np.arange(dims[1] * 1.0))
    nanpix = np.where(np.isnan(imgs[0]))
    OWA = np.sqrt(np.min((x[nanpix] - centers[0][0]) ** 2 + (y[nanpix] - centers[0][1]) ** 2))
    IWA = 10  #because I'm lazy
    dr = float(OWA - IWA) / (annuli - 1)

    #error checking for too small of annuli go here

    #calculate the annuli
    rad_bounds = [(dr * rad + IWA, dr * (rad + 1) + IWA) for rad in range(annuli)]
    #last annulus should mostly emcompass everything
    rad_bounds[annuli - 1] = (rad_bounds[annuli - 1][0], imgs[0].shape[0] / 2)

    #divide annuli into subsections
    dphi = 2 * np.pi / subsections
    phi_bounds = [(dphi * phi_i - np.pi, dphi * (phi_i + 1) - np.pi) for phi_i in range(subsections)]

    #before we start, create the output array in flattened form
    sub_imgs = np.zeros([dims[0], dims[1] * dims[2]])


    #begin KLIP process for each image
    for img_num, (center, pa) in enumerate(zip(centers, parangs)):

        #align images
        recentered = np.array([align_and_scale(frame, center, oldcenter)
                               for frame, oldcenter in zip(imgs, centers)])

        #create coordinate system 
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        phi = np.arctan2(y - center[1], x - center[0])

        #flatten img dimension
        flattened = imgs.reshape((dims[0], dims[1] * dims[2]))
        r = r.reshape(dims[1] * dims[2])
        phi = phi.reshape(dims[1] * dims[2])

        #iterate over the different sections
        for radstart, radend in rad_bounds:
            for phistart, phiend in phi_bounds:
                #grab the pixel location of the section we are going to anaylze
                section_ind = np.where((r >= radstart) & (r < radend) & (phi >= phistart) & (phi < phiend))
                if np.size(section_ind) == 0:
                    continue
                #grab the files suitable for reference PSF
                avg_rad = (radstart + radend) / 2.0
                file_ind = np.where((parangs - pa) * avg_rad > movement)
                if np.size(file_ind) < 2:
                    sub_imgs[img_num, section_ind] = np.zeros(np.size(section_ind))
                    continue
                ref_psfs = flattened[file_ind[0], :]
                ref_psfs = ref_psfs[:, section_ind[0]]
                print(ref_psfs.shape)
                sub_imgs[img_num, section_ind] = klip_math(flattened[img_num, section_ind][0], ref_psfs, numbasis)

    #finished. Let's reshape the output images
    sub_imgs = sub_imgs.reshape((dims[0], dims[1], dims[2]))
    sub_imgs[allnans] = np.nan

    #derotate images
    #sub_imgs = np.array([rotate(img, pa, (140,140), center) for img,pa,center in zip(sub_imgs, parangs, centers)])

    return sub_imgs


