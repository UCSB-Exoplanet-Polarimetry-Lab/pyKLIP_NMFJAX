import numpy as np
import scipy.linalg as la
import scipy.ndimage as ndimage


def klip_math(sci, ref_psfs, numbasis, covar_psfs=None):
    """
    Helper function for KLIP that does the linear algebra
    
    Inputs:
        sci: array of length p containing the science data
        ref_psfs: N x p array of the N reference PSFs that 
                  characterizes the PSF of the p pixels
        numbasis: number of KLIP basis vectors to use (can be an int or an array of ints of length b)
        covar_psfs: covariance matrix of reference psfs passed in so you don't have to calculate it here

    Outputs:
        sub_img_rows_selected: array of shape (p,b) that is the PSF subtracted data for each of the b KLIP basis
                               cutoffs. If numbasis was an int, then sub_img_row_selected is just an array of length p

    TODO:
        make numbasis to be any number of KLIP cutoffs and return all of them
    """
    #import pdb

    #for the science image, subtract the mean and mask bad pixels
    sci_mean_sub = sci - np.nanmean(sci)
    # sci_nanpix = np.where(np.isnan(sci_mean_sub))
    # sci_mean_sub[sci_nanpix] = 0

    #do the same for the reference PSFs
    #playing some tricks to vectorize the subtraction
    ref_psfs_mean_sub = ref_psfs - np.nanmean(ref_psfs, axis=1)[:, None]
    ref_psfs_mean_sub[np.where(np.isnan(ref_psfs_mean_sub))] = 0

    #calculate the covariance matrix for the reference PSFs
    #note that numpy.cov normalizes by p-1 to get the NxN covariance matrix
    #we have to correct for that a few lines down when consturcting the KL 
    #vectors since that's not part of the equation in the KLIP paper
    if covar_psfs is None:
        covar_psfs = np.cov(ref_psfs_mean_sub)

    #maximum number of KL modes
    tot_basis = covar_psfs.shape[0]

    #only pick numbasis requested that are valid. We can't compute more KL basis than there are reference PSFs
    #do numbasis - 1 for ease of indexing since index 0 is using 1 KL basis vector
    numbasis = np.clip(numbasis - 1, 0, tot_basis-1)  # clip values, for output consistency we'll keep duplicates
    max_basis = np.max(numbasis) + 1  # maximum number of eigenvectors/KL basis we actually need to use/calculate

    #calculate eigenvalues and eigenvectors of covariance matrix, but only the ones we need (up to max basis)
    evals, evecs = la.eigh(covar_psfs, eigvals=(tot_basis-max_basis, tot_basis-1))

    #scipy.linalg.eigh spits out the eigenvalues/vectors smallest first so we need to reverse
    #we're going to recopy them to hopefully improve caching when doing matrix multiplication
    evals = np.copy(evals[::-1])
    evecs = np.copy(evecs[:,::-1], order='F') #fortran order to improve memory caching in matrix multiplication

    #calculate the KL basis vectors
    kl_basis = np.dot(ref_psfs_mean_sub.T, evecs)
    kl_basis = kl_basis * (1. / np.sqrt(evals * (np.size(sci) - 1)))[None, :]  #multiply a value for each row

    #sort to KL basis in descending order (largest first)
    #kl_basis = kl_basis[:,eig_args_all]

    #duplicate science image by the max_basis to do simultaneous calculation for different k_KLIP
    sci_mean_sub_rows = np.tile(sci_mean_sub, (max_basis, 1))
    sci_rows_selected = np.tile(sci_mean_sub, (np.size(numbasis), 1)) # this is the output image which has less rows

    #bad pixel mask
    #do it first for the image we're just doing computations on but don't care about the output
    sci_nanpix = np.where(np.isnan(sci_mean_sub_rows))
    sci_mean_sub_rows[sci_nanpix] = 0
    #now do it for the output image
    sci_nanpix = np.where(np.isnan(sci_rows_selected))
    sci_rows_selected[sci_nanpix] = 0

    # do the KLIP equation, but now all the different k_KLIP simultaneously
    # calculate the inner product of science image with each of the different kl_basis vectors
    #TODO: can we optimize this so it doesn't have to multiply all the rows because in the next lines we only select some of them
    inner_products = np.dot(sci_mean_sub_rows, np.require(kl_basis, requirements=['F']))
    # select the KLIP modes we want for each level of KLIP by multiplying by lower diagonal matrix
    inner_products = inner_products * np.tril(np.ones([max_basis, max_basis]))
    # make a KLIP PSF for each amount of klip basis, but only for the amounts of klip basis we actually output
    klip_psf = np.dot(inner_products[numbasis,:], kl_basis.T)
    # make subtracted image for each number of klip basis
    sub_img_rows_selected = sci_rows_selected - klip_psf

    #restore NaNs
    sub_img_rows_selected[sci_nanpix] = np.nan

    return sub_img_rows_selected.transpose() # need to flip them so the output is shaped (p,b)


    #old code that only did one number of KL basis for truncation
    # #truncation either based on user input or maximum number of PSFs
    # trunc_basis = np.min([numbasis, tot_basis])
    # #eigenvalues are ordered largest first now
    # eig_args = eig_args_all[0: trunc_basis]
    # kl_basis = kl_basis[:, eig_args]
    #
    # #project KL vectors onto science image to construct model PSF
    # inner_products = np.dot(sci_mean_sub, kl_basis)
    # klip_psf = np.dot(inner_products, kl_basis.T)
    #
    # #subtract from original image to get final image
    # sub_img = sci_mean_sub - klip_psf
    #
    # #restore NANs
    # sub_img[sci_nanpix] = np.nan
    #
    # #pdb.set_trace()
    #
    # return sub_img


def estimate_movement(radius, parang0=None, parangs=None, wavelength0=None, wavelengths=None):
    """
    Estimates the movement of a hypothetical astrophysical source in ADI and/or SDI at the given radius and
    given reference parallactic angle (parang0) and reference wavelegnth (wavelength0)

    Inputs:
        radius: the radius from the star of the hypothetical astrophysical source
        parang0: the parallactic angle of the reference image (in degrees)
        parangs: array of length N of the parallactic angle of all N images (in degrees)
        wavelength0: the wavelength of the reference image
        wavelengths: array of length N of the wavelengths of all N images
        NOTE: we expect parang0 and parangs to be either both defined or both None.
                Same with wavelength0 and wavelengths

    Output:
        moves: array of length N of the distance an astrophysical source would have moved from the
               reference image
    """
    #default no movement parameters
    dtheta = 0 # how much the images moved in theta (polar coordinate)
    scale_fac = 1 # how much the image moved radially (r/radius)

    if parang0 is not None:
        dtheta = np.radians(parang0 - parangs)
    if wavelength0 is not None:
        scale_fac = (wavelength0/wavelengths)

    #define cartesean coordinate system where astrophysical source is at (x,y) = (r,0)
    x0 = radius
    y0 = 0.

    #find x,y location of astrophysical source for the rest of the images
    r = radius * scale_fac
    x = r * np.cos(dtheta)
    y = r * np.sin(dtheta)

    moves = np.sqrt((x-x0)**2 + (y-y0)**2)
    return moves

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
                        This means scale factor should be lambda_0/lambda
                        where lambda_0 is the wavelength you want to scale to
    Outputs:
        resampled_img: shifted and/or scaled 2D image
    """
    #import scipy.interpolate as interp
    #import pdb

    #create the coordinate system of the image to manipulate for the transform
    dims = img.shape
    x, y = np.meshgrid(np.arange(dims[1], dtype=np.float32), np.arange(dims[0], dtype=np.float32))
    mod_flag = 0 #check how many modifications we are making

    #if old_center is specified, realign the images
    if ((old_center is not None) & ~(np.array_equal(new_center, old_center))):
        dx = new_center[0] - old_center[0]
        dy = new_center[1] - old_center[1]
        x -= dx
        y -= dy
        mod_flag += 1

    #if scale_factor is specified, scale the images
    if scale_factor != 1:
        #conver to polar for scaling
        r = np.sqrt((x - new_center[0]) ** 2 + (y - new_center[1]) ** 2)
        theta = np.arctan2(y - new_center[1], x - new_center[0])  #theta range is [-pi,pi]

        r /= scale_factor

        #convert back to cartesian
        x = r * np.cos(theta) + new_center[0]
        y = r * np.sin(theta) + new_center[1]
        mod_flag += 1

    #if nothing is to be changed, return a copy of the image
    if mod_flag == 0:
        return np.copy(img)

    #resample image based on new coordinates
    #scipy uses y,x convention when meshgrid uses x,y
    #stupid scipy functions can't work with masked arrays (NANs)
    #and trying to use interp2d with sparse arrays is way to slow
    #hack my way out of this by picking a really small value for NANs and try to detect them after the interpolation
    minval = np.min([np.nanmin(img), 0.0])
    nanpix = np.where(np.isnan(img))
    img_copy = np.copy(img)
    img_copy[nanpix] = minval * 2.0
    resampled_img = ndimage.map_coordinates(img_copy, [y, x], cval=minval * 2.0)
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


def rotate(img, angle, center, new_center=None, flipx=True, astr_hdr=None):
    """
    Rotate an image by the given angle about the given center.
    Optional: can shift the image to a new image center after rotation. Also can reverse x axis for those left
              handed astronomy coordinate systems

    Inputs:
        img: a 2D image
        angle: angle CCW to rotate by (degrees)
        center: 2 element list [x,y] that defines the center to rotate the image to respect to
        new_center: 2 element list [x,y] that defines the new image center after rotation
        flipx: default is True, which reverses x axis.
        astr_hdr: wcs astrometry header for the image
    Outputs:
        resampled_img: new 2D image
    """
    #convert angle to radians
    angle_rad = np.radians(angle)

    #create the coordinate system of the image to manipulate for the transform
    dims = img.shape
    x, y = np.meshgrid(np.arange(dims[1], dtype=np.float32), np.arange(dims[0], dtype=np.float32))

    #if necessary, move coordinates to new center
    if new_center is not None:
        dx = new_center[0] - center[0]
        dy = new_center[1] - center[1]
        x -= dx
        y -= dy

    #flip x if needed to get East left of North
    if flipx is True:
        x = x[:, ::-1]

    #do rotation. CW rotation formula to get a CCW of the image
    xp = (x-center[0])*np.cos(angle_rad) + (y-center[1])*np.sin(angle_rad) + center[0]
    yp = -(x-center[0])*np.sin(angle_rad) + (y-center[1])*np.cos(angle_rad) + center[1]


    #resample image based on new coordinates
    #scipy uses y,x convention when meshgrid uses x,y
    #stupid scipy functions can't work with masked arrays (NANs)
    #and trying to use interp2d with sparse arrays is way to slow
    #hack my way out of this by picking a really small value for NANs and try to detect them after the interpolation
    #then redo the transformation setting NaN to zero to reduce interpolation effects, but using the mask we derived
    minval = np.min([np.nanmin(img), 0.0])
    nanpix = np.where(np.isnan(img))
    img_copy = np.copy(img)
    img_copy[nanpix] = minval * 5.0
    resampled_img_mask = ndimage.map_coordinates(img_copy, [yp, xp], cval=minval * 5.0)
    img_copy[nanpix] = 0
    resampled_img = ndimage.map_coordinates(img_copy, [yp, xp], cval=np.nan)
    resampled_img[np.where(resampled_img_mask < minval)] = np.nan

    #edit the astrometry header if given to compensate for orientation
    if astr_hdr is not None:
        _rotate_wcs_hdr(astr_hdr, angle, flipx=flipx)

    return resampled_img


def _rotate_wcs_hdr(wcs_header, rot_angle, flipx=False, flipy=False):
    """
    Modifies the wcs header when rotating/flipping an image.

    Inputs:
        wcs_header: wcs astrometry header
        rot_angle: in degrees CCW, the specified rotation desired
        flipx: after the rotation, reverse x axis? Yes if True
        flipy: after the rotation, reverse y axis? Yes if True
    """
    wcs_header.rotateCD(rot_angle)
    if flipx is True:
        wcs_header.wcs.cd[:,0] *= -1
    if flipy is True:
        wcs_header.wcs.cd[:,1] *= -1


def klip_adi(imgs, centers, parangs, IWA, annuli=5, subsections=4, minmove=3, numbasis=None, aligned_center=None,
             minrot=0):
    """
    KLIP PSF Subtraction using angular differential imaging

    Inputs:
        imgs: array of 2D images for ADI. Shape of array (N,y,x)
        centers: N by 2 array of (x,y) coordinates of image centers
        parangs: N legnth array detailing parallactic angle of each image
        IWA: inner working angle (in pixels)
        anuuli: number of annuli to use for KLIP
        subsections: number of sections to break each annuli into
        movement: minimum amount of movement (in pixels) of an astrophysical source
                  to consider using that image for a refernece PSF
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        aligned_center: array of 2 elements [x,y] that all the KLIP subtracted images will be centered on for image
                        registration
        minrot: minimum PA rotation (in degrees) to be considered for use as a reference PSF (good for disks)

    Ouput:
        sub_imgs: array of [array of 2D images (PSF subtracted)] using different number of KL basis vectors as
                    specified by numbasis. Shape of (b,N,y,x). Exception is if b==1. Then sub_imgs has the first
                    array stripped away and is shape of (N,y,x).
    """
    #figure out error checking later..
    import pdb

    #defaullt numbasis if none
    if numbasis is None:
        totalimgs = imgs.shape[0]
        numbasis = np.arange(1, totalimgs + 5, 5)
        print(numbasis)
    else:
        if hasattr(numbasis, "__len__"):
            numbasis = np.array(numbasis)
        else:
            numbasis = np.array([numbasis])

    if aligned_center is None:
        aligned_center = [int(imgs.shape[2]//2), int(imgs.shape[1]//2)]

    #save all bad pixels
    allnans = np.where(np.isnan(imgs))

    #use first image to figure out how to divide the annuli
    #TODO: should be smart about this in the future. Going to hard code some guessing
    #need to make the next 10 lines or so much smarter
    dims = imgs.shape
    x, y = np.meshgrid(np.arange(dims[2] * 1.0), np.arange(dims[1] * 1.0))
    nanpix = np.where(np.isnan(imgs[0]))
    OWA = np.sqrt(np.min((x[nanpix] - centers[0][0]) ** 2 + (y[nanpix] - centers[0][1]) ** 2))
    dr = float(OWA - IWA) / (annuli)

    #error checking for too small of annuli go here

    #calculate the annuli
    rad_bounds = [(dr * rad + IWA, dr * (rad + 1) + IWA) for rad in range(annuli)]
    #last annulus should mostly emcompass everything
    rad_bounds[annuli - 1] = (rad_bounds[annuli - 1][0], imgs[0].shape[0] / 2)

    #divide annuli into subsections
    dphi = 2 * np.pi / subsections
    phi_bounds = [(dphi * phi_i - np.pi, dphi * (phi_i + 1) - np.pi) for phi_i in range(subsections)]

    #before we start, create the output array in flattened form
    sub_imgs = np.zeros([dims[0], dims[1] * dims[2], numbasis.shape[0]])


    #begin KLIP process for each image
    for img_num, pa in enumerate(parangs):
        recentered = np.array([align_and_scale(frame, aligned_center, oldcenter)
                               for frame, oldcenter in zip(imgs, centers)])

        #create coordinate system 
        r = np.sqrt((x - aligned_center[0]) ** 2 + (y - aligned_center[1]) ** 2)
        phi = np.arctan2(y - aligned_center[1], x - aligned_center[0])

        #flatten img dimension
        flattened = recentered.reshape((dims[0], dims[1] * dims[2]))
        r.shape = (dims[1] * dims[2])
        phi.shape = (dims[1] * dims[2])

        #iterate over the different sections
        for radstart, radend in rad_bounds:
            for phistart, phiend in phi_bounds:
                #grab the pixel location of the section we are going to anaylze
                section_ind = np.where((r >= radstart) & (r < radend) & (phi >= phistart) & (phi < phiend))
                if np.size(section_ind) == 0:
                    continue
                #grab the files suitable for reference PSF
                avg_rad = (radstart + radend) / 2.0
                moves = estimate_movement(avg_rad, parang0=pa, parangs=parangs)
                file_ind = np.where((moves >= minmove) & (np.abs(parangs - pa) > minrot))
                if np.size(file_ind) < 2:
                    print("less than 2 reference PSFs available, skipping...")
                    sub_imgs[img_num, section_ind] = np.zeros(np.size(section_ind))
                    continue
                ref_psfs = flattened[file_ind[0], :]
                ref_psfs = ref_psfs[:, section_ind[0]]
                #print(sub_imgs.shape)
                #print(sub_imgs[img_num, section_ind, :].shape)
                sub_imgs[img_num, section_ind, :] = klip_math(flattened[img_num, section_ind][0], ref_psfs, numbasis)

    #finished. Let's reshape the output images
    #move number of KLIP modes as leading axis (i.e. move from shape (N,y,x,b) to (b,N,y,x)
    sub_imgs = np.rollaxis(sub_imgs.reshape((dims[0], dims[1], dims[2], numbasis.shape[0])), 3)
    #if we only passed in one value for numbasis (i.e. only want one PSF subtraction), strip off the number of basis)
    if sub_imgs.shape[0] == 1:
        sub_imgs = sub_imgs[0]
    #restore bad pixels
    sub_imgs[:, allnans[0], allnans[1], allnans[2]] = np.nan

    #derotate images
    #sub_imgs = np.array([rotate(img, pa, (140,140), center) for img,pa,center in zip(sub_imgs, parangs, centers)])

    return sub_imgs


