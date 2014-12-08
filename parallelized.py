import pyklip.klip as klip
import multiprocessing as mp
import ctypes
import numpy as np
import cProfile
import os
import itertools

def _tpool_init(original_imgs, original_imgs_shape, aligned_imgs, aligned_imgs_shape, output_imgs, output_imgs_shape,
               pa_imgs, wvs_imgs, centers_imgs):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array).

    Inputs:
        original_imgs: original images from files to read and align&scale.
        original_imgs_shape: (N,y,x), N = number of frames = num files * num wavelengths
        aligned: aligned and scaled images for processing.
        aligned_imgs_shape: (wv, N, y, x), wv = number of wavelengths per datacube
        output_imgs: output images after KLIP processing
        output_imgs_shape: (b, N, y, x), b = number of different KL basis cutoffs for KLIP routine
        pa_imgs, wvs_imgs: arrays of size N with the PA and wavelength
        centers_img: array of shape (N,2) with [x,y] image center for image frame
    """
    global original, original_shape, aligned, aligned_shape, output, output_shape, img_pa, img_wv, img_center
    #original images from files to read and align&scale. Shape of (N,y,x)
    original = original_imgs
    original_shape = original_imgs_shape
    #aligned and scaled images for processing. Shape of (wv, N, y, x)
    aligned = aligned_imgs
    aligned_shape = aligned_imgs_shape
    #output images after KLIP processing
    output = output_imgs
    output_shape = output_imgs_shape
    #parameters for each image (PA, wavelegnth, image center)
    img_pa = pa_imgs
    img_wv = wvs_imgs
    img_center = centers_imgs


def _arraytonumpy(shared_array, shape=None):
    """
    Covert a shared array to a numpy array
    Input:
        shared_array: a multiprocessing.Array array
        shape: a shape for the numpy array. otherwise, will assume a 1d array

    Output:
        numpy_array: numpy array for vectorized operation. still points to the same memory!
    """
    numpy_array = np.frombuffer(shared_array.get_obj())
    if shape is not None:
        numpy_array.shape = shape

    return numpy_array


def _align_and_scale(iterable_arg):
    """
    Aligns and scales the set of original images about a reference center and scaled to a reference wavelength.
    Note: is a helper function to only be used after initializing the threadpool!

    Inputs:
        iterable_arg: a tuple of two elements:
            ref_wv_iter: a tuple of two elements. First is the index of the reference wavelength (between 0 and 36).
                         second is the value of the reference wavelength. This is to determine scaling
            ref_center: a two-element array with the [x,y] center position to align all the images to.

    Ouputs:
        just returns ref_wv_iter again
    """

    #extract out arguments from the iteration argument
    ref_wv_iter = iterable_arg[0]
    ref_center = iterable_arg[1]
    ref_wv_index = ref_wv_iter[0]
    ref_wv = ref_wv_iter[1]

    original_imgs = _arraytonumpy(original, original_shape)
    wvs_imgs = _arraytonumpy(img_wv)
    centers_imgs = _arraytonumpy(img_center, (np.size(wvs_imgs),2))

    aligned_imgs = _arraytonumpy(aligned, aligned_shape)
    aligned_imgs[ref_wv_index, :, :, :] =  np.array([klip.align_and_scale(frame, ref_center, old_center, ref_wv/old_wv)
                                        for frame, old_center, old_wv in zip(original_imgs, centers_imgs, wvs_imgs)])
    #print aligned_imgs.shape

    return ref_wv_index, ref_wv


def _klip_section(img_num, parang, wavelength, wv_index, numbasis, radstart, radend, phistart, phiend, minmove,
                  ref_center):
    """
    DEPRECIATED. Still being preserved in case we want to change size of atomization. But will need some fixing

    Runs klip on a section of an image as given by the geometric parameters. Helper fucntion of klip routines and
    requires thread pool to be initialized! Currently is designed only for ADI+SDI. Not yet that flexible.

    Inputs:
        img_num: file index for the science image to process
        parang: PA of science iamge
        wavelength: wavelength of science image
        wv_index: array index of the wavelength of the science image
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        avg_rad: average radius of this annulus
        radstart: inner radius of the annulus (in pixels)
        radend: outer radius of the annulus (in pixels)
        phistart: lower bound in CCW angle from x axis for the start of the section
        phiend: upper boundin CCW angle from y axis for the end of the section
        minmove: minimum movement between science image and PSF reference image to use PSF reference image (in pixels)
        ref_center: 2 element list for the center of the science frames. Science frames should all be aligned.

    Ouputs:
        Returns True on success and False on failure.
        Output images are stored in output array as defined by _tpool_init()
    """
    global output, aligned

    #create a coordinate system
    x, y = np.meshgrid(np.arange(original_shape[2] * 1.0), np.arange(original_shape[1] * 1.0))
    x.shape = (x.shape[0] * x.shape[1])
    y.shape = (y.shape[0] * y.shape[1])
    r = np.sqrt((x - ref_center[0])**2 + (y - ref_center[1])**2)
    phi = np.arctan2(y - ref_center[1], x - ref_center[0])

    #grab the pixel location of the section we are going to anaylze
    section_ind = np.where((r >= radstart) & (r < radend) & (phi >= phistart) & (phi < phiend))
    if np.size(section_ind) == 0:
        print("section is empty, skipping...")
        return False

    #grab the files suitable for reference PSF
    #load shared arrays for wavelengths and PAs
    wvs_imgs = _arraytonumpy(img_wv)
    pa_imgs = _arraytonumpy(img_pa)
    #calculate average movement in this section
    avg_rad = (radstart + radend) / 2.0
    moves = klip.estimate_movement(avg_rad, parang, pa_imgs, wavelength, wvs_imgs)
    file_ind = np.where(moves >= minmove)
    if np.size(file_ind[0]) < 2:
        print("less than 2 reference PSFs available for minmove={0}, skipping...".format(minmove))
        return False

    #load aligned images and make reference PSFs
    aligned_imgs = _arraytonumpy(aligned, (aligned_shape[0], aligned_shape[1], aligned_shape[2]*aligned_shape[3]))[wv_index]
    ref_psfs = aligned_imgs[file_ind[0], :]
    ref_psfs = ref_psfs[:,  section_ind[0]]
    #ref_psfs = ref_psfs[:, section_ind]
    #print(img_num, avg_rad, ref_psfs.shape)
    #print(sub_imgs.shape)
    #print(sub_imgs[img_num, section_ind, :].shape)

    #write to output
    output_imgs = _arraytonumpy(output, (output_shape[0], output_shape[1]*output_shape[2], output_shape[3]))
    klipped = klip.klip_math(aligned_imgs[img_num, section_ind], ref_psfs, numbasis)
    output_imgs[img_num, section_ind, :] = klipped
    return True


def _klip_section_profiler(img_num, parang, wavelength, wv_index, numbasis, radstart, radend, phistart, phiend, minmove,
                           ref_center=None):
    """
    DEPRECIATED. Still being preserved in case we want to change size of atomization. But will need some fixing

    Profiler wrapper for _klip_section. Outputs a file openable by pstats.Stats for each annulus wavelength.
    However there is no guarentee which wavelength and which subsection of the annulus is saved to disk.

    Inputs: Same arguments as _klip_section
    """
    cProfile.runctx("_klip_section(img_num, parang, wavelength, wv_index, numbasis, radstart, radend, phistart, phiend,"
                    " minmove, ref_center)", globals(), locals(), 'profile-{0}.out'.format(int(radstart+radend)/2))
    return True


def _klip_section_multifile_profiler(scidata_indicies, wavelength, wv_index, numbasis, radstart, radend, phistart,
                                     phiend, minmove, ref_center=None, minrot=0):
    """
    Profiler wrapper for _klip_section_multifile. Outputs a file openable by pstats.Stats for each annulus wavelength.
    However there is no guarentee which wavelength and which subsection of the annulus is saved to disk. There
    is the ability to output a profiler file for each subsection and wavelength but it's too many files and who
    actually looks at all of them.

    Inputs: Same arguments as _klip_section_multifile()
    """
    cProfile.runctx("_klip_section_multifile(scidata_indicies, wavelength, wv_index, numbasis, radstart, radend, "
                    "phistart, phiend, minmove, ref_center, minrot)", globals(), locals(),
                    'profile-{0}.out'.format(int(radstart + radend)/2))
    return True


def _klip_section_multifile(scidata_indicies, wavelength, wv_index, numbasis, radstart, radend, phistart, phiend,
                            minmove, ref_center, minrot):
    """
    Runs klip on a section of the image for all the images of a given wavelength.
    Bigger size of atomization of work than _klip_section but saves computation time and memory. Currently no need to
    break it down even smaller when running on machines on the order of 32 cores.

    Inputs:
        scidata_indicies: array of file indicies that are the science images for this wavelength
        wavelength: value of the wavelength we are processing
        wv_index: index of the wavelenght we are processing
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        radstart: inner radius of the annulus (in pixels)
        radend: outer radius of the annulus (in pixels)
        phistart: lower bound in CCW angle from x axis for the start of the section
        phiend: upper boundin CCW angle from y axis for the end of the section
        minmove: minimum movement between science image and PSF reference image to use PSF reference image (in pixels)
        ref_center: 2 element list for the center of the science frames. Science frames should all be aligned.
        minrot: minimum PA rotation (in degrees) to be considered for use as a reference PSF (good for disks)

    Outputs:
        returns True on success, False on failure. Does not return whether KLIP on each individual image was sucessful.
        Saves data to output array as defined in _tpool_init()
    """

    #create a coordinate system. Can use same one for all the images because they have been aligned and scaled
    x, y = np.meshgrid(np.arange(original_shape[2] * 1.0), np.arange(original_shape[1] * 1.0))
    #JB question: x = x[:]?
    x.shape = (x.shape[0] * x.shape[1])
    y.shape = (y.shape[0] * y.shape[1])
    r = np.sqrt((x - ref_center[0])**2 + (y - ref_center[1])**2)
    phi = np.arctan2(y - ref_center[1], x - ref_center[0])

    #grab the pixel location of the section we are going to anaylze
    section_ind = np.where((r >= radstart) & (r < radend) & (phi >= phistart) & (phi < phiend))
    if np.size(section_ind) <= 1:
        print("section is too small ({0} pixels), skipping...".format(np.size(section_ind)))
        return False

    #export some of klip.klip_math functions to here to minimize computation repeats

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

    #grab the parangs
    parangs = _arraytonumpy(img_pa)
    for file_index,parang in zip(scidata_indicies, parangs[scidata_indicies]):
        try:
            _klip_section_multifile_perfile(file_index, section_ind, ref_psfs_mean_sub, covar_psfs, parang, wavelength,
                                            wv_index, (radstart + radend) / 2.0, numbasis, minmove, minrot)
        except (ValueError, RuntimeError, TypeError) as err:
            print("({0}): {1}".format(err.errno, err.strerror))
            return False

 #   [_klip_section_multifile_perfile(file_index, section_ind, ref_psfs_mean_sub, covar_psfs,
  #                                   parang, wavelength, wv_index, (radstart + radend) / 2.0, numbasis, minmove)
  #      for file_index,parang in zip(scidata_indicies, parangs[scidata_indicies])]

    return True


def _klip_section_multifile_perfile(img_num, section_ind, ref_psfs, covar, parang, wavelength, wv_index, avg_rad, numbasis, minmove, minrot):
    """
    Imitates the rest of _klip_section for the multifile code. Does the rest of the PSF reference selection

    Inputs:
        img_num: file index for the science image to process
        section_ind: np.where(pixels are in this section of the image). Note: coordinate system is collapsed into 1D
        ref_psfs: reference psf images of this section
        covar: the covariance matrix of the reference PSFs. Shape of (N,N)
        parang: PA of science iamge
        wavelength: wavelength of science image
        wv_index: array index of the wavelength of the science image
        avg_rad: average radius of this annulus
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        minmove: minimum movement between science image and PSF reference image to use PSF reference image (in pixels)

    Outputs:
        return True on success, False on failure.
        Saves image to output array defined in _tpool_init()
    """
    #grab the files suitable for reference PSF
    #load shared arrays for wavelengths and PAs
    wvs_imgs = _arraytonumpy(img_wv)
    pa_imgs = _arraytonumpy(img_pa)
    #calculate average movement in this section for each PSF reference image w.r.t the science image
    moves = klip.estimate_movement(avg_rad, parang, pa_imgs, wavelength, wvs_imgs)
    if minrot > 0:
        file_ind = np.where((moves >= minmove) & (np.abs(pa_imgs - parang) >= minrot))
    else:
        file_ind = np.where(moves >= minmove)
    if np.size(file_ind[0]) < 2:
        print("less than 2 reference PSFs available for minmove={0}, skipping...".format(minmove))
        return False

    #pick out a subarray. Have to play around with indicies to get the right shape to index the matrix
    covar_files = covar[file_ind[0].reshape(np.size(file_ind), 1), file_ind[0]]

    #pick only the most correlated reference PSFs if there's more than enough PSFs
    maxbasis_requested = np.max(numbasis)
    maxbasis_possible = np.size(file_ind)
    if maxbasis_possible > maxbasis_requested:
        xcorr = covar[img_num, file_ind[0]]  # grab the x-correlation with the sci img for valid PSFs
        sort_ind = np.argsort(xcorr)
        closest_matched = sort_ind[-maxbasis_requested:]  # sorted smallest first so need to grab from the end
        # grab the new and smaller covariance matrix
        covar_files = covar_files[closest_matched.reshape(np.size(closest_matched), 1), closest_matched]
        # grab smaller set of reference PSFs
        ref_psfs_selected = ref_psfs[file_ind[0][closest_matched], :]
    else:
        # else just grab the reference PSFs for all the valid files
        ref_psfs_selected = ref_psfs[file_ind[0], :]

    #load input/output data
    aligned_imgs = _arraytonumpy(aligned, (aligned_shape[0], aligned_shape[1], aligned_shape[2]*aligned_shape[3]))[wv_index]
    output_imgs = _arraytonumpy(output, (output_shape[0], output_shape[1]*output_shape[2], output_shape[3]))
    #run KLIP
    try:
        klipped = klip.klip_math(aligned_imgs[img_num, section_ind[0]], ref_psfs_selected, numbasis, covar_files)
    except (ValueError, RuntimeError, TypeError) as err:
        print("({0}): {1}".format(err.errno, err.strerror))
        return False

    #write to output
    output_imgs[img_num, section_ind[0], :] = klipped
    return True


def rotate_imgs(imgs, angles, centers, new_center=None, numthreads=None, flipx=True, hdrs=None):
    """
    derotate a sequences of images by their respective angles

    Inputs:
        imgs: array of shape (N,y,x) containing N images
        angles: array of length N with the angle to rotate each frame. Each angle should be CW in degrees.
                (TODO: fix this angle convention)
        centers: array of shape N,2 with the [x,y] center of each frame
        new_centers: a 2-element array with the new center to register each frame. Default is middle of image
        numthreads: number of threads to be used
        flipx: flip the x axis to get a left handed coordinate system (oh astronomers...)
        hdrs: array of N wcs astrometry headers

    Output:
        derotated: array of shape (N,y,x) containing the derotated images
    """

    tpool = mp.Pool(processes=numthreads)

    #klip.rotate(img, -angle, oldcenter, [152,152]) for img, angle, oldcenter
    #multithreading the rotation for each image
    if hdrs is None:
        tasks = [tpool.apply_async(klip.rotate, args=(img, angle, center, new_center, flipx, None))
                 for img, angle, center in zip(imgs, angles, centers)]
    else:
        tasks = [tpool.apply_async(klip.rotate, args=(img, angle, center, new_center, flipx, None))
                 for img, angle, center in zip(imgs, angles, centers)]
        #lazy hack around the fact that wcs objects don't preserve wcs.cd fields when sent to other processes
        #so let's just do it manually outside of the rotation
        [klip._rotate_wcs_hdr(astr_hdr, angle, flipx=flipx) for angle, astr_hdr in zip(angles, hdrs)]

    #reform back into a giant array
    derotated = np.array([task.get() for task in tasks])

    tpool.close()

    return derotated


def klip_adi_plus_sdi(imgs, centers, parangs, wvs, IWA, annuli=5, subsections=4, movement=3, numbasis=None,
                      aligned_center=None, numthreads=None, minrot=0):
    """
    KLIP PSF Subtraction using angular differential imaging

    Inputs:
        imgs: array of 2D images for ADI. Shape of array (N,y,x)
        centers: N by 2 array of (x,y) coordinates of image centers
        parangs: N length array detailing parallactic angle of each image
        wvs: N length array of the wavelengths
        IWA: inner working angle (in pixels)
        anuuli: number of annuli to use for KLIP
        subsections: number of sections to break each annuli into
        movement: minimum amount of movement (in pixels) of an astrophysical source
                  to consider using that image for a refernece PSF
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        aligned_center: array of 2 elements [x,y] that all the KLIP subtracted images will be centered on for image
                        registration
        numthreads: number of threads to use. If none, defaults to using all the cores of the cpu
        minrot: minimum PA rotation (in degrees) to be considered for use as a reference PSF (good for disks)

    Ouput:
        sub_imgs: array of [array of 2D images (PSF subtracted)] using different number of KL basis vectors as
                    specified by numbasis. Shape of (b,N,y,x).
    """

    #defaullt numbasis if none
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

    #default aligned_center if none:
    if aligned_center is None:
        aligned_center = [int(imgs.shape[2]//2), int(imgs.shape[1]//2)]

    #save all bad pixels
    allnans = np.where(np.isnan(imgs))

    #use first image to figure out how to divide the annuli
    #TODO: what to do with OWA
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
    rad_bounds[annuli - 1] = (rad_bounds[annuli - 1][0], imgs[0].shape[0])

    #divide annuli into subsections
    dphi = 2 * np.pi / subsections
    phi_bounds = [[dphi * phi_i - np.pi, dphi * (phi_i + 1) - np.pi] for phi_i in range(subsections)]
    phi_bounds[-1][1] = 2. * np.pi

    #calculate how many iterations we need to do
    global tot_iter
    tot_iter = np.size(np.unique(wvs)) * len(phi_bounds) * len(rad_bounds)

    #before we start, create the output array in flattened form
    #sub_imgs = np.zeros([dims[0], dims[1] * dims[2], numbasis.shape[0]])

    #implement the thread pool
    #make a bunch of shared memory arrays to transfer data between threads
    #make the array for the original images and initalize it
    original_imgs = mp.Array(ctypes.c_double, np.size(imgs))
    original_imgs_shape = imgs.shape
    original_imgs_np = _arraytonumpy(original_imgs, original_imgs_shape)
    original_imgs_np[:] = imgs
    #make array for recentered/rescaled image for each wavelength
    unique_wvs = np.unique(wvs)
    recentered_imgs = mp.Array(ctypes.c_double, np.size(imgs)*np.size(unique_wvs))
    recentered_imgs_shape = (np.size(unique_wvs),) + imgs.shape
    #make output array which also has an extra dimension for the number of KL modes to use
    output_imgs = mp.Array(ctypes.c_double, np.size(imgs)*np.size(numbasis))
    output_imgs_np = _arraytonumpy(output_imgs)
    output_imgs_np[:] = np.nan
    output_imgs_shape = imgs.shape + numbasis.shape
    #remake the PA, wv, and center arrays as shared arrays
    pa_imgs = mp.Array(ctypes.c_double, np.size(parangs))
    pa_imgs_np = _arraytonumpy(pa_imgs)
    pa_imgs_np[:] = parangs
    wvs_imgs = mp.Array(ctypes.c_double, np.size(wvs))
    wvs_imgs_np = _arraytonumpy(wvs_imgs)
    wvs_imgs_np[:] = wvs
    centers_imgs = mp.Array(ctypes.c_double, np.size(centers))
    centers_imgs_np = _arraytonumpy(centers_imgs, centers.shape)
    centers_imgs_np[:] = centers
    tpool = mp.Pool(processes=numthreads, initializer=_tpool_init,
                   initargs=(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs,
                             output_imgs_shape, pa_imgs, wvs_imgs, centers_imgs), maxtasksperchild=50)

    #align and scale the images for each image. Use map to do this asynchronously
    print("Begin align and scale images for each wavelength")
    realigned_index = tpool.imap_unordered(_align_and_scale, zip(enumerate(unique_wvs), itertools.repeat(aligned_center)))


    #list to store each threadpool task
    outputs = []
    #as each is finishing, queue up the aligned data to be processed with KLIP
    for wv_index, wv_value in realigned_index:
        print("Wavelength {1:.4} with index {0} has finished align and scale. Queuing for KLIP".format(wv_index, wv_value))

        #pick out the science images that need PSF subtraction for this wavelength
        scidata_indicies = np.where(wvs == wv_value)[0]

        # commented out code to do _klip_section instead of _klip_section_multifile
        # outputs += [tpool.apply_async(_klip_section_profiler, args=(file_index, parang, wv_value, wv_index, numbasis,
        #                                                   radstart, radend, phistart, phiend, movement))
        #                     for phistart,phiend in phi_bounds
        #                 for radstart, radend in rad_bounds
        #             for file_index,parang in zip(scidata_indicies, parangs[scidata_indicies])]

        #perform KLIP asynchronously for each group of files of a specific wavelenght and section of the image
        outputs += [tpool.apply_async(_klip_section_multifile, args=(scidata_indicies, wv_value, wv_index, numbasis,
                                                                     radstart, radend, phistart, phiend, movement,
                                                                     aligned_center, minrot))
                    for phistart,phiend in phi_bounds
                    for radstart, radend in rad_bounds]

    #harness the data!
    #check make sure we are completely unblocked before outputting the data
    print("Total number of tasks for KLIP processing is {0}".format(tot_iter))
    for index, out in enumerate(outputs):
        out.wait()
        if (index + 1) % 10 == 0:
            print("{0:.4}% done ({1}/{2} completed)".format(index*100.0/tot_iter, index, tot_iter))

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

    return sub_imgs

def klip_dataset(dataset, mode='ADI+SDI', outputdir=".", fileprefix="", annuli=5, subsections=4, movement=3, numbasis=None,
                 numthreads=None, minrot=0, calibrate_flux=False, aligned_center=None):
    """
    run klip on a dataset class outputted by an implementation of Instrument.Data

    Inputs:
        dataset: an implementation of Instrument.Data (see instruments/ subfolder)
        mode: one of ['ADI', 'SDI', 'ADI+SDI'] for ADI, SDI, or ADI+SDI
        outputdir: directory to save output files
        fileprefix: filename prefix for saved files
        anuuli: number of annuli to use for KLIP
        subsections: number of sections to break each annuli into
        movement: minimum amount of movement (in pixels) of an astrophysical source
                  to consider using that image for a refernece PSF
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        numthreads: number of threads to use. If none, defaults to using all the cores of the cpu
        minrot: minimum PA rotation (in degrees) to be considered for use as a reference PSF (good for disks)
        calibrate_flux: if True calibrate flux of the dataset, otherwise leave it be
        aligned_center: array of 2 elements [x,y] that all the KLIP subtracted images will be centered on for image
                        registration

    Output
        Saved files in the output directory
        Returns: rot_imgs: (b, N, wv, y, x) 5D cube of KL cutoff modes (b), number of images (N), wavelengths (wv),
                            and spatial dimensions. Images are derotated.
    """
    #defaullt numbasis if none
    if numbasis is None:
        totalimgs = dataset.input.shape[0]
        maxbasis = np.min([totalimgs, 100]) #only going up to 100 KL modes by default
        numbasis = np.arange(1, maxbasis + 5, 5)
        print("KL basis not specified. Using default.", numbasis)
    else:
        if hasattr(numbasis, "__len__"):
            numbasis = np.array(numbasis)
        else:
            numbasis = np.array([numbasis])

    # JB: Add center of the image in the header because the sat-spot are gone so we can't get it after klip.
    if aligned_center is None:
        c = [int(dataset.input.shape[2]//2),int(dataset.input.shape[1]//2)]
    else:
        c = [aligned_center[0],aligned_center[1]]

    #run KLIP
    if mode == 'ADI+SDI':
        print("Beginning ADI+SDI KLIP")
        klipped_imgs = klip_adi_plus_sdi(dataset.input, dataset.centers, dataset.PAs, dataset.wvs,
                                         dataset.IWA, annuli=annuli, subsections=subsections, movement=movement,
                                         numbasis=numbasis, numthreads=numthreads, minrot=minrot,
                                         aligned_center=aligned_center)
        dataset.output = klipped_imgs
        if calibrate_flux == True:
            dataset.calibrate_output()

        #TODO: handling of only a single numbasis
        #derotate all the images
        #first we need to flatten so it's just a 3D array
        oldshape = klipped_imgs.shape
        dataset.output = dataset.output.reshape(oldshape[0]*oldshape[1], oldshape[2], oldshape[3])
        #we need to duplicate PAs and centers for the different KL mode cutoffs we supplied
        flattend_parangs = np.tile(dataset.PAs, oldshape[0])
        flattened_centers = np.tile(dataset.centers.reshape(oldshape[1]*2), oldshape[0]).reshape(oldshape[1]*oldshape[0],2)

        #parallelized rotate images
        print("Derotating Images...")
        rot_imgs = rotate_imgs(dataset.output, flattend_parangs, flattened_centers, numthreads=numthreads, flipx=True,
                               hdrs=dataset.wcs)

        #reconstruct datacubes, need to obtain wavelength dimension size
        num_wvs = np.size(np.unique(dataset.wvs)) # assuming all datacubes are taken in same band

        #give rot_imgs dimensions of (num KLmode cutoffs, num cubes, num wvs, y, x)
        rot_imgs = rot_imgs.reshape(oldshape[0], oldshape[1]/num_wvs, num_wvs, oldshape[2], oldshape[3])

        dataset.output = rot_imgs

        #valid output path and write iamges
        outputdirpath = os.path.realpath(outputdir)
        print("Writing Images to directory {0}".format(outputdirpath))


        #collapse in time and wavelength to examine KL modes
        KLmode_cube = np.nanmean(dataset.output, axis=(1,2))
        dataset.savedata(outputdirpath + '/' + fileprefix + "-KLmodes-all.fits", KLmode_cube, dataset.wcs[0], center=c)

        #for each KL mode, collapse in time to examine spectra
        KLmode_spectral_cubes = np.nanmean(dataset.output, axis=1)
        for KLcutoff, spectral_cube in zip(numbasis, KLmode_spectral_cubes):
            dataset.savedata(outputdirpath + '/' + fileprefix + "-KL{0}-speccube.fits".format(KLcutoff), spectral_cube,
                                  dataset.wcs[0], center=c)
    elif mode == 'ADI':
        #ADI is not parallelized
        dataset.output = klip.klip_adi(dataset.input, dataset.centers, dataset.PAs,
                                         dataset.IWA, annuli=annuli, subsections=subsections, movement=movement,
                                         numbasis=numbasis, minrot=minrot, aligned_center=aligned_center)
        #TODO: handling of only a single numbasis
        #derotate all the images
        #first we need to flatten so it's just a 3D array
        oldshape = dataset.output.shape
        dataset.output = dataset.output.reshape(oldshape[0]*oldshape[1], oldshape[2], oldshape[3])
        #we need to duplicate PAs and centers for the different KL mode cutoffs we supplied
        flattend_parangs = np.tile(dataset.PAs, oldshape[0])
        flattened_centers = np.tile(dataset.centers.reshape(oldshape[1]*2), oldshape[0]).reshape(oldshape[1]*oldshape[0],2)

        #parallelized rotate images
        print("Derotating Images...")
        rot_imgs = rotate_imgs(dataset.output, flattend_parangs, flattened_centers, numthreads=numthreads, flipx=True,
                               hdrs=dataset.wcs)

        #give rot_imgs dimensions of (num KLmode cutoffs, num cubes, num wvs, y, x)
        rot_imgs = rot_imgs.reshape(oldshape[0], oldshape[1], oldshape[2], oldshape[3])

        dataset.output = rot_imgs

        #valid output path and write iamges
        outputdirpath = os.path.realpath(outputdir)
        print("Writing Images to directory {0}".format(outputdirpath))

        #collapse in time and wavelength to examine KL modes
        KLmode_cube = np.nanmean(dataset.output, axis=(1))
        dataset.savedata(outputdirpath + '/' + fileprefix + "-KLmodes-all.fits", KLmode_cube, dataset.wcs[0], center=c)
    elif mode == 'SDI':
        print("SDI Not Yet Implemented")
        return
    else:
        print("Invalid mode. Either ADI, SDI, or ADI+SDI")
        return

    #pyfits.writeto("out1.fits", klipped_imgs[-1], clobber=True)


