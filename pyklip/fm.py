#KLIP Forward Modelling
import pyklip.klip as klip
import numpy as np
import scipy.linalg as la
from scipy.stats import norm

import ctypes
import itertools
import multiprocessing as mp
from pyklip.parallelized import _arraytonumpy, _align_and_scale

def klip_math(sci, refs, models, numbasis, covar_psfs=None, return_basis=False):
    """
    linear algebra of KLIP with linear perturbation
    disks and point sources

    Args:
        sci: array of length p containing the science data
        refs: N x p array of the N reference images that
                  characterizes the extended source with p pixels
        models: N x p array of the N models corresponding to reference images
        numbasis: number of KLIP basis vectors to use (can be an int or an array of ints of length b)
        covmat: covariance matrix of reference images (for large N, useful)
        return_basis: If true, return KL basis vectors (used when onesegment==True)

    Returns:
        sub_img_rows_selected: array of shape (p,b) that is the PSF subtracted data for each of the b KLIP basis
                               cutoffs. If numbasis was an int, then sub_img_row_selected is just an array of length p

    """
    #remove means and nans
    sci_mean_sub = sci - np.nanmean(sci)
    sci_nanpix = np.where(np.isnan(sci_mean_sub))
    sci_mean_sub[sci_nanpix] = 0

    refs_mean_sub = refs - np.nanmean(refs, axis=1)[:, None]
    refs_mean_sub[np.where(np.isnan(refs_mean_sub))] = 0

    models_mean_sub = models# - np.nanmean(models, axis=1)[:,None] should this be the case?
    models_mean_sub[np.where(np.isnan(models_mean_sub))] = 0

    # calculate the covariance matrix for the reference PSFs
    # note that numpy.cov normalizes by p-1 to get the NxN covariance matrix
    # we have to correct for that a few lines down when consturcting the KL
    # vectors since that's not part of the equation in the KLIP paper
    if covar_psfs is None:
        covar_psfs = np.cov(refs_mean_sub)

    tot_basis = covar_psfs.shape[0]
    numbasis = np.clip(numbasis - 1, 0, tot_basis-1)
    max_basis = np.max(numbasis) + 1

    evals, evecs = la.eigh(covar_psfs, eigvals = (tot_basis-max_basis, tot_basis-1))
    evals = np.copy(evals[::-1])
    evecs = np.copy(evecs[:,::-1])

    KL_basis = np.dot(refs_mean_sub.T,evecs)
    KL_basis = KL_basis * (1. / np.sqrt(evals *(np.size(sci) -1)))[None,:] #should scaling be there?
    #Laurent's paper specifically removes 1/sqrt(Npix-1)

    sci_mean_sub_rows = np.tile(sci_mean_sub, (max_basis,1))
    sci_rows_selected = np.tile(sci_mean_sub, (np.size(numbasis),1))

    sci_nanpix = np.where(np.isnan(sci_mean_sub_rows))
    sci_mean_sub_rows[sci_nanpix] = 0
    sci_nanpix = np.where(np.isnan(sci_rows_selected))
    sci_rows_selected[sci_nanpix] = 0

    #expansion of the covariance matrix
    CAdeltaI = np.dot(refs_mean_sub,models_mean_sub.T) + np.dot(models_mean_sub,refs_mean_sub.T)

    Nrefs = evals.shape[0]
    KL_perturb = np.zeros(KL_basis.shape)
    Pert1 = np.zeros(KL_basis.shape)
    Pert2 = np.zeros(KL_basis.shape)
    CoreffsKL = np.zeros((Nrefs,Nrefs))
    Mult = np.zeros((Nrefs,Nrefs))
    cross = np.zeros((Nrefs,Nrefs))

    for i in range(Nrefs):
        for j in range(Nrefs):
            cross[i,j] = np.transpose(evecs[:,j]).dot(CAdeltaI).dot(evecs[:,i]) #I don't think transpose does anything here

    for i in range(Nrefs):
        for j in range(Nrefs):
            if j == i:
                Mult[i,j] = -1./(2.*evals[j])
            else:
                Mult[i,j] = np.sqrt(evals[j]/evals[i])/(evals[i]-evals[j])
        CoreffsKL[i,:] = cross[i]*Mult[i]


    for j in range(Nrefs):
        Pert1[:,j] = np.dot(CoreffsKL[j],KL_basis.T).T
        Pert2[:,j] = (1./np.sqrt(evals[j])*np.transpose(evecs[:,j]).dot(models_mean_sub))
        KL_perturb[:,j] = (Pert1[:,j]+Pert2[:,j])

    KL_pert = KL_perturb + KL_basis #this makes KL_perturb too large (on order of np.size(sci)-1)
    #Perhaps scaling issue? or my coding failure?g

    inner_products = np.dot(sci_mean_sub_rows, KL_pert)
    lower_tri = np.tril(np.ones([max_basis,max_basis]))
    inner_products = inner_products * lower_tri
    klip = np.dot(inner_products[numbasis,:], KL_pert.T)

    sub_img_rows_selected = sci_rows_selected - klip
    sub_img_rows_selected[sci_nanpix] = np.nan

    if return_basis is True:
        return sub_img_rows_selected.transpose(), KL_basis.transpose()
    else:
        return sub_img_rows_selected.transpose()

def klip_adi(imgs, models, centers, parangs, IWA, annuli=5, subsections=4, movement=3, numbasis=None, aligned_center=None, minrot=0): #Zack
    """
    KLIP PSF Subtraction using angular differential imaging, with expansion of covariance matrix

    Args:
        imgs: array of 2D images for ADI. Shape of array (N,y,x)
        models: array of 2D models for ADI corresponding to imgs. Shape of array (N,y,x)
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

    Returns:
        sub_imgs: array of [array of 2D images (PSF subtracted)] using different number of KL basis vectors as
                    specified by numbasis. Shape of (b,N,y,x). Exception is if b==1. Then sub_imgs has the first
                    array stripped away and is shape of (N,y,x).
    """

    if numbasis is None:
        totalimgs = imgs.shape[0]
        numbasis = np.arange(1,totalimgs + 5,5)
        print numbasis
    else:
        if hasattr(numbasis,"__len__"):
            numbasis = np.array(numbasis)
        else:
            numbasis = np.array([numbasis])

    if aligned_center is None:
        aligned_center = [int(imgs.shape[2]//2),int(imgs.shape[1]//2)]

    allnans = np.where(np.isnan(imgs))

    #annuli
    dims = imgs.shape
    x,y = np.meshgrid(np.arange(dims[2] * 1.0),np.arange(dims[1]*1.0))
    nanpix = np.where(np.isnan(imgs[0]))
    OWA = np.sqrt(np.min((x[nanpix] - centers[0][0]) ** 2 + (y[nanpix] - centers[0][1]) ** 2))
    dr = float(OWA - IWA) / (annuli)

    rad_bounds = [(dr * rad + IWA, dr * (rad + 1) + IWA) for rad in range(annuli)]
    rad_bounds[annuli - 1] = (rad_bounds[annuli - 1][0], imgs[0].shape[0] / 2)

    dphi = 2 * np.pi / subsections
    phi_bounds = [[dphi * phi_i - np.pi, dphi * (phi_i + 1) - np.pi] for phi_i in range(subsections)]
    phi_bounds[-1][1] = 2. * np.pi

    sub_imgs = np.zeros([dims[0], dims[1] * dims[2], numbasis.shape[0]])

    #begin KLIP process for each image
    for img_num, pa in enumerate(parangs):
        recenteredimgs = np.array([klip.align_and_scale(frame, aligned_center, oldcenter) for frame, oldcenter in zip(imgs, centers)])
        recenteredmodels = np.array([klip.align_and_scale(frame, aligned_center, oldcenter) for frame, oldcenter in zip(models, centers)])

        #create coordinate system
        r = np.sqrt((x - aligned_center[0]) ** 2 + (y - aligned_center[1]) ** 2)
        phi = np.arctan2(y - aligned_center[1], x - aligned_center[0])

        #flatten img dimension
        flattenedimgs = recenteredimgs.reshape((dims[0], dims[1] * dims[2]))
        flattenedmodels = recenteredmodels.reshape((dims[0],dims[1] * dims[2]))

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
                moves = klip.estimate_movement(avg_rad, parang0=pa, parangs=parangs)
                file_ind = np.where((moves >= movement) & (np.abs(parangs - pa) > minrot))

                if np.size(file_ind) < 2:
                    print("less than 2 reference PSFs available, skipping...")
                    print((sub_imgs[img_num, section_ind]).shape)
                    print(np.zeros(np.size(section_ind)).shape)
#                    sub_imgs[img_num, section_ind] = np.zeros(np.size(section_ind))
                    continue
                images_math = flattenedimgs[file_ind[0], :]
                images_math = images_math[:, section_ind[0]]
                models_math = flattenedmodels[file_ind[0], :]
                models_math = models_math[:, section_ind[0]]

                sub_imgs[img_num, section_ind, :] = klip_math(flattenedimgs[img_num, section_ind][0], images_math, models_math, numbasis)

    #move number of KLIP modes as leading axis (i.e. move from shape (N,y,x,b) to (b,N,y,x)
    sub_imgs = np.rollaxis(sub_imgs.reshape((dims[0], dims[1], dims[2], numbasis.shape[0])), 3)
    #if we only passed in one value for numbasis (i.e. only want one PSF subtraction), strip off the number of basis)
    sub_imgs[:,allnans[0],allnans[1],allnans[2]] = np.nan
    if sub_imgs.shape[0] == 1:
        sub_imgs = sub_imgs[0]

    #derotate
        #img_list = []
        #for a in sub_imgs:
        #    img_list.append(np.array([rotate(img,pa,(140,140),center) for img, pa,center in zip(a,parangs,centers)]))
        #sub_imgs = np.asarray(img_list)

    #all of the image centers are now at aligned_center
    centers[:,0] = aligned_center[0]
    centers[:,1] = aligned_center[1]

    return sub_imgs


#####################################################################
################# Begin Parallelized Framework ######################
#####################################################################

def _tpool_init(original_imgs, original_imgs_shape, aligned_imgs, aligned_imgs_shape, output_imgs, output_imgs_shape,
                pa_imgs, wvs_imgs, centers_imgs, interm_imgs, interm_imgs_shape, aux_imgs, aux_imgs_shape):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array) - output_imgs does not need to be mp.Array and can be anything

    Args:
        original_imgs: original images from files to read and align&scale.
        original_imgs_shape: (N,y,x), N = number of frames = num files * num wavelengths
        aligned: aligned and scaled images for processing.
        aligned_imgs_shape: (wv, N, y, x), wv = number of wavelengths per datacube
        output_imgs: outputs after KLIP-FM processing (NOTE: Is a list of whatevers - top level strucutre must be list!)
        output_imgs_shape: a list of shapes (if applicable) (NOTE: must be list!)
        pa_imgs, wvs_imgs: arrays of size N with the PA and wavelength
        centers_img: array of shape (N,2) with [x,y] image center for image frame
        interm_imgs: intermediate data product shape - what is saved on a sector to sector basis before combining to
                     form the output of that sector
        interm_imgs_shape: shape of interm_imgs
        aux_imgs: auxilliary data
        aux_imgs_shape: shape of aux_imgs
    """
    global original, original_shape, aligned, aligned_shape, outputs, outputs_shape, img_pa, img_wv, img_center,\
        interm, interm_shape,aux, aux_shape
    # original images from files to read and align&scale. Shape of (N,y,x)
    original = original_imgs
    original_shape = original_imgs_shape
    # aligned and scaled images for processing. Shape of (wv, N, y, x)
    aligned = aligned_imgs
    aligned_shape = aligned_imgs_shape
    # output images after KLIP processing
    outputs = output_imgs
    outputs_shape = output_imgs_shape
    # parameters for each image (PA, wavelegnth, image center)
    img_pa = pa_imgs
    img_wv = wvs_imgs
    img_center = centers_imgs

    #intermediate and auxilliary data to store
    interm = interm_imgs
    interm_shape = interm_imgs_shape
    aux = aux_imgs
    aux_shape = aux_imgs_shape


def _align_and_scale_subset(thread_index, aligned_center):
    """
    Aligns and scales a subset of images

    Args:
        thread_index: index of thread, break-up algin and scale equally among threads
        algined_center: center to align things to

    Returns:
        None
    """
    original_imgs = _arraytonumpy(original, original_shape)
    wvs_imgs = _arraytonumpy(img_wv)
    centers_imgs = _arraytonumpy(img_center, (np.size(wvs_imgs),2))
    aligned_imgs = _arraytonumpy(aligned, aligned_shape)

    unique_wvs = np.unique(wvs_imgs)

    # calculate all possible combinations of images and wavelengths to scale to
    # this ordering should hopefully have better cache optimization?
    combos = [combo for combo in itertools.product(np.arange(original_imgs.shape[0]), np.arange(np.size(unique_wvs)))]

    # figure out which ones this thread should do
    numframes_todo = np.round(len(combos)/mp.cpu_count())
    # the last thread needs to finish all of them
    if thread_index == mp.cpu_count() - 1:
        combos_todo = combos[thread_index*numframes_todo:]
    else:
        combos_todo = combos[thread_index*numframes_todo:(thread_index+1)*numframes_todo]

    print(len(combos), len(combos_todo))

    for img_index, ref_wv_index in combos_todo:
        aligned_imgs[ref_wv_index,img_index,:,:] = klip.align_and_scale(original_imgs[img_index], aligned_center,
                                                centers_imgs[img_index], unique_wvs[ref_wv_index]/wvs_imgs[img_index])
    return


def klip_parallelized(imgs, centers, parangs, wvs, IWA, fm_class, mode='ADI+SDI', annuli=5, subsections=4, movement=3,
                      numbasis=None, aligned_center=None, numthreads=None, minrot=0, maxrot=360, spectrum=None
                      ):
    """
    multithreaded KLIP PSF Subtraction

    Args:
        imgs: array of 2D images for ADI. Shape of array (N,y,x)
        centers: N by 2 array of (x,y) coordinates of image centers
        parangs: N length array detailing parallactic angle of each image
        wvs: N length array of the wavelengths
        IWA: inner working angle (in pixels)
        fm_class: class that implements the the forward modelling functionality
        mode: one of ['ADI', 'SDI', 'ADI+SDI'] for ADI, SDI, or ADI+SDI
        anuuli: Annuli to use for KLIP. Can be a number, or a list of 2-element tuples (a, b) specifying
                the pixel bondaries (a <= r < b) for each annulus
        subsections: Sections to break each annuli into. Can be a number of a list of 2-element tuples (a, b) specifying
                     the PA boundaries (a <= PA < b) for each sectgion
        movement: minimum amount of movement (in pixels) of an astrophysical source
                  to consider using that image for a refernece PSF
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        aligned_center: array of 2 elements [x,y] that all the KLIP subtracted images will be centered on for image
                        registration
        numthreads: number of threads to use. If none, defaults to using all the cores of the cpu

        minrot: minimum PA rotation (in degrees) to be considered for use as a reference PSF (good for disks)
        maxrot: maximum PA rotation (in degrees) to be considered for use as a reference PSF (temporal variability)

        spectrum: if not None, a array of length N with the flux of the template spectrum at each wavelength. Uses
                    minmove to determine the separation from the center of the segment to determine contamination and
                    the size of the PSF (TODO: make PSF size another quanitity)
                    (e.g. minmove=3, checks how much containmination is within 3 pixels of the hypothetical source)
                    if smaller than 10%, (hard coded quantity), then use it for reference PSF



    Returns:
        sub_imgs: array of [array of 2D images (PSF subtracted)] using different number of KL basis vectors as
                    specified by numbasis. Shape of (b,N,y,x).
    """

    ################## Interpret input arguments ####################

    # defaullt numbasis if none
    if numbasis is None:
        totalimgs = imgs.shape[0]
        maxbasis = np.min([totalimgs, 100]) #only going up to 100 KL modes by default
        numbasis = np.arange(1, maxbasis + 5, 5)
        print("KL basis not specified. Using default.", numbasis)
    else:
        if hasattr(numbasis, "__len__"):
            numbasis = np.array(numbasis)
        else:
            numbasis = np.array([numbasis])

    if numthreads is None:
        numthreads = mp.cpu_count()

    # default aligned_center if none:
    if aligned_center is None:
        aligned_center = [np.mean(centers[:,0]), np.mean(centers[:,1])]

    # save all bad pixels
    allnans = np.where(np.isnan(imgs))

    # use first image to figure out how to divide the annuli
    # TODO: what to do with OWA
    # need to make the next 10 lines or so much smarter
    dims = imgs.shape
    x, y = np.meshgrid(np.arange(dims[2] * 1.0), np.arange(dims[1] * 1.0))
    nanpix = np.where(np.isnan(imgs[0]))
    OWA = np.sqrt(np.min((x[nanpix] - centers[0][0]) ** 2 + (y[nanpix] - centers[0][1]) ** 2))
    dr = float(OWA - IWA) / (annuli)

    # error checking for too small of annuli go here

    # calculate the annuli
    rad_bounds = [(dr * rad + IWA, dr * (rad + 1) + IWA) for rad in range(annuli)]
    #last annulus should mostly emcompass everything
    rad_bounds[annuli - 1] = (rad_bounds[annuli - 1][0], imgs[0].shape[0])

    # divide annuli into subsections
    dphi = 2 * np.pi / subsections
    phi_bounds = [[dphi * phi_i - np.pi, dphi * (phi_i + 1) - np.pi] for phi_i in range(subsections)]
    phi_bounds[-1][1] = 2. * np.pi


    # calculate how many iterations we need to do
    global tot_iter
    tot_iter = np.size(np.unique(wvs)) * len(phi_bounds) * len(rad_bounds)
    tot_sectors = len(phi_bounds) * len(rad_bounds)


    ########################### Create Shared Memory ###################################

    # implement the thread pool
    # make a bunch of shared memory arrays to transfer data between threads
    # make the array for the original images and initalize it
    original_imgs = mp.Array(ctypes.c_double, np.size(imgs))
    original_imgs_shape = imgs.shape
    original_imgs_np = _arraytonumpy(original_imgs, original_imgs_shape)
    original_imgs_np[:] = imgs
    # make array for recentered/rescaled image for each wavelength
    unique_wvs = np.unique(wvs)
    recentered_imgs = mp.Array(ctypes.c_double, np.size(imgs)*np.size(unique_wvs))
    recentered_imgs_shape = (np.size(unique_wvs),) + imgs.shape

    # remake the PA, wv, and center arrays as shared arrays
    pa_imgs = mp.Array(ctypes.c_double, np.size(parangs))
    pa_imgs_np = _arraytonumpy(pa_imgs)
    pa_imgs_np[:] = parangs
    wvs_imgs = mp.Array(ctypes.c_double, np.size(wvs))
    wvs_imgs_np = _arraytonumpy(wvs_imgs)
    wvs_imgs_np[:] = wvs
    centers_imgs = mp.Array(ctypes.c_double, np.size(centers))
    centers_imgs_np = _arraytonumpy(centers_imgs, centers.shape)
    centers_imgs_np[:] = centers

    # CREATE Custom Shared Memory Arrays: outputs and aux here
    # make output array which is generated by the fm_class to be use case specific
    output_imgs, output_imgs_shape = fm_class.alloc_output()
    output_imgs_np = _arraytonumpy(output_imgs)
    output_imgs_np[:] = np.nan
    aux_data, aux_shape = fm_class.alloc_aux()


    # align and scale the images for each image. Use map to do this asynchronously
    tpool = mp.Pool(processes=numthreads, initializer=_tpool_init,
                   initargs=(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs,
                             output_imgs_shape, pa_imgs, wvs_imgs, centers_imgs, output_imgs, output_imgs_shape,
                             aux_data, aux_shape), maxtasksperchild=50)


    print("Begin align and scale images for each wavelength")
    aligned_outputs = []
    for threadnum in range(numthreads):
        #multitask this
        aligned_outputs += [tpool.apply_async(_align_and_scale_subset, args=(threadnum, aligned_center))]

        #save it to shared memory
    for aligned_output in aligned_outputs:
            aligned_output.wait()
            print("got one")

    import pdb
    pdb.set_trace()

    # list to store each threadpool task
    tpool_outputs = []
    first_pass = True # first pass after algin and scale
    sector_job_queued = np.zeros(tot_sectors) # count for jobs in the tpool queue for each sector
    # as each is finishing, queue up the aligned data to be processed with KLIP
    for sector_index, ((radstart, radend),(phistart,phiend)) in enumerate(itertools.product(rad_bounds, phi_bounds)):
        # calculate sector size
        # create a coordinate system.
        x, y = np.meshgrid(np.arange(original_shape[2] * 1.0), np.arange(original_shape[1] * 1.0))
        x.shape = (x.shape[0] * x.shape[1]) #Flatten
        y.shape = (y.shape[0] * y.shape[1])
        r = np.sqrt((x - aligned_center[0])**2 + (y - aligned_center[1])**2)
        phi = np.arctan2(y - aligned_center[1], x - aligned_center[0])

        #grab the pixel location of the section we are going to anaylze
        phi_rotate = ((phi) % (2.0 * np.pi)) - np.pi
        section_ind = np.where((r >= radstart) & (r < radend) & (phi_rotate >= phistart) & (phi_rotate < phiend))

        interm_data, interm_shape = fm_class.alloc_interm(np.size(section_ind), original_imgs.shape[0])

        for wv_index, wv_value in unique_wvs:

            # pick out the science images that need PSF subtraction for this wavelength
            scidata_indicies = np.where(wvs == wv_value)[0]

            # perform KLIP asynchronously for each group of files of a specific wavelength and section of the image
            sector_job_queued[sector_index] += scidata_indicies.shape[0]
            tpool_outputs += [tpool.apply_async(_klip_section_multifile_perfile,
                                                args=(file_index, sector_index, radstart, radend, phistart, phiend,
                                                      parang, wv_value, wv_index, (radstart + radend) / 2., numbasis,
                                                      movement, aligned_center, minrot, maxrot, mode, spectrum,
                                                      fm_class, interm_data, interm_shape, aux_data, aux_shape))
                                for file_index,parang in zip(scidata_indicies, pa_imgs_np[scidata_indicies])]


        # Run post processing on this sector here
        # Can be multithreaded code using the threadpool defined above
        # Check tpool job outputs. It there is stuff, go do things with it
        while len(tpool_outputs) > 0:
            finished_sector_index = tpool_outputs.pop(0)[0]
            sector_job_queued[finished_sector_index] -= 1

            # if this is the last job finished for this sector,
            # do something here?

        # run custom function to handle end of sector post-processing analysis

    #harness the data!
    #check make sure we are completely unblocked before outputting the data
    print("Total number of tasks for KLIP processing is {0}".format(tot_iter))
    for index, out in enumerate(tpool_outputs):
        out.wait()
        if (index + 1) % 10 == 0:
            print("{0:.4}% done ({1}/{2} completed)".format((index+1)*100.0/tot_iter, index, tot_iter))



    #close to pool now and make sure there's no processes still running (there shouldn't be or else that would be bad)
    print("Closing threadpool")
    tpool.close()
    tpool.join()

    #finished. Let's reshape the output images
    #move number of KLIP modes as leading axis (i.e. move from shape (N,y,x,b) to (b,N,y,x)
    sub_imgs = _arraytonumpy(output_imgs, output_imgs_shape)
    sub_imgs = np.rollaxis(sub_imgs.reshape((dims[0], dims[1], dims[2], numbasis.shape[0])), 3)

    #restore bad pixels
    sub_imgs[:, allnans[0], allnans[1], allnans[2]] = np.nan

    #scrapping this behavior for now because I don't feel like dealing with edge cases
    ## if we only passed in one value for numbasis (i.e. only want one PSF subtraction), strip off that axis)
    #if sub_imgs.shape[0] == 1:
    #    sub_imgs = sub_imgs[0]

    #all of the image centers are now at aligned_center
    centers[:,0] = aligned_center[0]
    centers[:,1] = aligned_center[1]

    # Output for the sole PSFs
    return sub_imgs



def _klip_section_multifile_perfile(img_num, sector_index, radstart, radend, phistart, phiend, parang, wavelength,
                                    wv_index, avg_rad, numbasis, minmove, ref_center, minrot, maxrot, mode, spectrum):
    """
    Imitates the rest of _klip_section for the multifile code. Does the rest of the PSF reference selection

    Args:
        img_num: file index for the science image to process
        sector: index for the section of the image. Used for return purposes only
        radstart: radial distance of inner edge of annulus
        radend: radial distance of outer edge of annulus
        phistart: start of azimuthal sector (in radians)
        phiend: end of azimuthal sector (in radians)
        parang: PA of science iamge
        wavelength: wavelength of science image
        wv_index: array index of the wavelength of the science image
        avg_rad: average radius of this annulus
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        minmove: minimum movement between science image and PSF reference image to use PSF reference image (in pixels)
        maxmove:minimum movement (opposite of minmove) - CURRENTLY NOT USED
        mode: one of ['ADI', 'SDI', 'ADI+SDI'] for ADI, SDI, or ADI+SDI
        spectrum: if not None, a array of length N with the flux of the template spectrum at each wavelength. Uses
                    minmove to determine the separation from the center of the segment to determine contamination and
                    the size of the PSF (TODO: make PSF size another quanitity)
                    (e.g. minmove=3, checks how much containmination is within 3 pixels of the hypothetical source)
                    if smaller than 10%, (hard coded quantity), then use it for reference PSF

    Return:
        sector_index: used for tracking jobs
        Saves image to output array defined in _tpool_init()
    """
    #create a coordinate system. Can use same one for all the images because they have been aligned and scaled
    x, y = np.meshgrid(np.arange(original_shape[2] * 1.0), np.arange(original_shape[1] * 1.0))
    x.shape = (x.shape[0] * x.shape[1]) #Flatten
    y.shape = (y.shape[0] * y.shape[1])
    r = np.sqrt((x - ref_center[0])**2 + (y - ref_center[1])**2)
    phi = np.arctan2(y - ref_center[1], x - ref_center[0])

    #grab the pixel location of the section we are going to anaylze based on the parallactic angle of the image
    phi_rotate = ((phi + np.radians(parang)) % (2.0 * np.pi)) - np.pi
    section_ind = np.where((r >= radstart) & (r < radend) & (phi_rotate >= phistart) & (phi_rotate < phiend))
    if np.size(section_ind) <= 1:
        print("section is too small ({0} pixels), skipping...".format(np.size(section_ind)))
        return False

    #load aligned images for this wavelength
    aligned_imgs = _arraytonumpy(aligned, (aligned_shape[0], aligned_shape[1], aligned_shape[2] * aligned_shape[3]))[wv_index]
    ref_psfs = aligned_imgs[:,  section_ind[0]]

    #do the same for the reference PSFs
    #playing some tricks to vectorize the subtraction of the mean for each row
    ref_psfs_mean_sub = ref_psfs - np.nanmean(ref_psfs, axis=1)[:, None]
    ref_psfs_mean_sub[np.where(np.isnan(ref_psfs_mean_sub))] = 0

    #calculate the covariance matrix for the reference PSFs
    #note that numpy.cov normalizes by p-1 to get the NxN covariance matrix
    #we have to correct for that in the klip.klip_math routine when consturcting the KL
    #vectors since that's not part of the equation in the KLIP paper
    covar_psfs = np.cov(ref_psfs_mean_sub)
    #also calculate correlation matrix since we'll use that to select reference PSFs
    covar_diag = np.diagflat(1./np.sqrt(np.diag(covar_psfs)))
    corr_psfs = np.dot( np.dot(covar_diag, covar_psfs ), covar_diag)


    # grab the files suitable for reference PSF
    # load shared arrays for wavelengths and PAs
    wvs_imgs = _arraytonumpy(img_wv)
    pa_imgs = _arraytonumpy(img_pa)
    # calculate average movement in this section for each PSF reference image w.r.t the science image
    moves = klip.estimate_movement(avg_rad, parang, pa_imgs, wavelength, wvs_imgs, mode)
    # check all the PSF selection criterion
    # enough movement of the astrophyiscal source
    if spectrum is None:
        goodmv = (moves >= minmove)
    else:
        # optimize the selection based on the spectral template rather than just an exclusion principle
        goodmv = (spectrum * norm.sf(moves-minmove/2.355, scale=minmove/2.355) <= 0.1 * spectrum[wv_index])

    # enough field rotation
    if minrot > 0:
        goodmv = (goodmv) & (np.abs(pa_imgs - parang) >= minrot)

    # if no SDI, don't use other wavelengths
    if "SDI" not in mode.upper():
        goodmv = (goodmv) & (wvs_imgs == wavelength)
    # if no ADI, don't use other parallactic angles
    if "ADI" not in mode.upper():
        goodmv = (goodmv) & (pa_imgs == parang)

    # if minrot > 0:
    #     file_ind = np.where((moves >= minmove) & (np.abs(pa_imgs - parang) >= minrot))
    # else:
    #     file_ind = np.where(moves >= minmove)
    # select the good reference PSFs
    file_ind = np.where(goodmv)
    if np.size(file_ind[0]) < 2:
        print("less than 2 reference PSFs available for minmove={0}, skipping...".format(minmove))
        return False
    # pick out a subarray. Have to play around with indicies to get the right shape to index the matrix
    covar_files = covar_psfs[file_ind[0].reshape(np.size(file_ind), 1), file_ind[0]]

    # pick only the most correlated reference PSFs if there's more than enough PSFs
    maxbasis_requested = np.max(numbasis)
    maxbasis_possible = np.size(file_ind)
    if maxbasis_possible > maxbasis_requested:
        xcorr = corr_psfs[img_num, file_ind[0]]  # grab the x-correlation with the sci img for valid PSFs
        sort_ind = np.argsort(xcorr)
        closest_matched = sort_ind[-maxbasis_requested:]  # sorted smallest first so need to grab from the end
        # grab the new and smaller covariance matrix
        covar_files = covar_files[closest_matched.reshape(np.size(closest_matched), 1), closest_matched]
        # grab smaller set of reference PSFs
        ref_psfs_selected = ref_psfs[file_ind[0][closest_matched], :]

    else:
        # else just grab the reference PSFs for all the valid files
        ref_psfs_selected = ref_psfs[file_ind[0], :]



    aligned_imgs = _arraytonumpy(aligned, (aligned_shape[0], aligned_shape[1], aligned_shape[2] * aligned_shape[3]))[wv_index]
    output_imgs = _arraytonumpy(outputs, (outputs_shape[0], outputs_shape[1]*outputs_shape[2], outputs_shape[3]))

    try:
        klipped = klip.klip_math(aligned_imgs[img_num, section_ind[0]], ref_psfs_selected, numbasis, covar_psfs=covar_files)
    except (ValueError, RuntimeError, TypeError) as err:
        print("({0}): {1}".format(err.errno, err.strerror))
        return False

    # write to output
    output_imgs[img_num, section_ind[0], :] = klipped

    return sector_index
