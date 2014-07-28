import klip
import multiprocessing as mp
import ctypes
import numpy as np
import cProfile
import os

def _tpool_init(original_imgs, original_imgs_shape, aligned_imgs, aligned_imgs_shape, output_imgs, output_imgs_shape,
               pa_imgs, wvs_imgs, centers_imgs):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array).
    """
    global original, original_shape, aligned, aligned_shape, output, output_shape, img_pa, img_wv, img_center
    original = original_imgs
    original_shape = original_imgs_shape
    aligned = aligned_imgs
    aligned_shape = aligned_imgs_shape
    output = output_imgs
    output_shape = output_imgs_shape
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


def _align_and_scale(ref_wv_iter, ref_center=None):
    """
    Aligns and scales the set of original images about a reference center and scaled to a reference wavelength.
    Note: is a helper function to only be used after initializing the threadpool!

    Inputs:
        ref_wv_iter: a tuple of two elements. First is the index of the reference wavelength (between 0 and 36).
                     second is the value of the reference wavelength. This is to determine scaling
        ref_center: a two-element array with the [x,y] cetner position to align all the images to. Default is [140,140]

    Ouputs:
        just returns ref_wv_iter again
    """
    if ref_center is None:
        ref_center = [140, 140]

    #extract out wavelength information from the iteration argument
    ref_wv_index = ref_wv_iter[0]
    ref_wv = ref_wv_iter[1]

    original_imgs = _arraytonumpy(original, original_shape)
    wvs_imgs = _arraytonumpy(img_wv)
    centers_imgs = _arraytonumpy(img_center, (np.size(wvs_imgs),2))

    aligned_imgs = _arraytonumpy(aligned, aligned_shape)
    aligned_imgs[ref_wv_index, :, :, :] =  np.array([klip.align_and_scale(frame, ref_center, old_center, ref_wv/old_wv)
                                        for frame, old_center, old_wv in zip(original_imgs, centers_imgs, wvs_imgs)])

    return ref_wv_index, ref_wv


def _klip_section(img_num, parang, wavelength, wv_index, numbasis, radstart, radend, phistart, phiend, minmove, ref_center=None):
    """
    Runs klip on a section of an image as given by the geometric parameters. Helper fucntion of klip routines and
    requires thread pool to be initialized! Currently is designed only for ADI+SDI. Not yet that flexible.
    """
    global output, aligned
    if ref_center is None:
        ref_center = [140, 140]

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


def _klip_section_profiler(img_num, parang, wavelength, wv_index, numbasis, radstart, radend, phistart, phiend, minmove, ref_center=None):
    """
    Profiler wrapper for _klip_section
    """
    cProfile.runctx("_klip_section(img_num, parang, wavelength, wv_index, numbasis, radstart, radend, phistart, phiend, minmove, ref_center)",
                    globals(), locals(), 'profile-{0}.out'.format(os.getpid()))
    return True


def _klip_section_multifile_profiler(scidata_indicies, wavelength, wv_index, numbasis, radstart, radend, phistart, phiend, minmove, ref_center=None):
    """
    Profiler wrapper for _klip_section_multifile
    """
    cProfile.runctx("_klip_section_multifile(scidata_indicies, wavelength, wv_index, numbasis, radstart, radend, phistart, phiend, minmove, ref_center)",
                    globals(), locals(), 'profile2-{0}.out'.format((radstart+radend)/2.0))
    return True


def _klip_section_multifile(scidata_indicies, wavelength, wv_index, numbasis, radstart, radend, phistart, phiend, minmove, ref_center=None):
    """
    Runs klip on a section for all the images of a given wavelength. Bigger size of atomization of work than _klip_section
    but saves computation time and memory. Currently no need to break it down even smaller
    """
    if ref_center is None:
        ref_center = [140, 140]

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

    #export some of klip.klip_math functions to here to minimize computation repeats

    #load aligned images for this wavelength
    aligned_imgs = _arraytonumpy(aligned, (aligned_shape[0], aligned_shape[1], aligned_shape[2]*aligned_shape[3]))[wv_index]
    ref_psfs = aligned_imgs[:,  section_ind[0]]

    #do the same for the reference PSFs
    #playing some tricks to vectorize the subtraction
    ref_psfs_mean_sub = ref_psfs - np.nanmean(ref_psfs, axis=1)[:, None]
    ref_psfs_mean_sub[np.where(np.isnan(ref_psfs_mean_sub))] = 0

    #calculate the covariance matrix for the reference PSFs
    #note that numpy.cov normalizes by p-1 to get the NxN covariance matrix
    #we have to correct for that in the klip.klip_math routine when consturcting the KL
    #vectors since that's not part of the equation in the KLIP paper
    covar_psfs = np.cov(ref_psfs_mean_sub)

    #grab the parangs
    parangs = _arraytonumpy(img_pa)
    [_klip_section_multifile_perfile(file_index, section_ind, ref_psfs_mean_sub, covar_psfs,
                                     parang, wavelength, wv_index, (radstart + radend) / 2.0, numbasis, minmove)
        for file_index,parang in zip(scidata_indicies, parangs[scidata_indicies])]

    return True


def _klip_section_multifile_perfile(img_num, section_ind, ref_psfs, covar, parang, wavelength, wv_index, avg_rad, numbasis, minmove):
    """
    Imitates the rest of _klip_section for the multifile code
    """
    #grab the files suitable for reference PSF
    #load shared arrays for wavelengths and PAs
    wvs_imgs = _arraytonumpy(img_wv)
    pa_imgs = _arraytonumpy(img_pa)
    #calculate average movement in this section
    moves = klip.estimate_movement(avg_rad, parang, pa_imgs, wavelength, wvs_imgs)
    file_ind = np.where(moves >= minmove)
    if np.size(file_ind[0]) < 2:
        print("less than 2 reference PSFs available for minmove={0}, skipping...".format(minmove))
        return False

    #pick out a subarray. Have to play around with indicies to get the right shape to index the matrix
    covar_files = covar[file_ind[0].reshape(np.size(file_ind),1), file_ind[0]]

    #load input/output data
    aligned_imgs = _arraytonumpy(aligned, (aligned_shape[0], aligned_shape[1], aligned_shape[2]*aligned_shape[3]))[wv_index]
    output_imgs = _arraytonumpy(output, (output_shape[0], output_shape[1]*output_shape[2], output_shape[3]))
    #run KLIP
    try:
        klipped = klip.klip_math(aligned_imgs[img_num, section_ind[0]], ref_psfs[file_ind[0],:], numbasis, covar_files)
    except (ValueError, RuntimeError) as err:
        print(err)

    #write to output
    output_imgs[img_num, section_ind[0], :] = klipped
    return True


def derotate_imgs(imgs, angles, centers, new_center=None, numthreads=None):
    """
    derotate a sequences of images by their respective angles

    Inputs:
        imgs: array of shape (N,y,x) containing N images
        angles: array of length N with the angle to rotate each frame. Each angle should be CW in degrees. (TODO: fix this angle convention)
        centers: array of shape N,2 with the [x,y] center of each frame
        new_centers: a 2-element array with the new center to register each frame. Default is [140,140]
        numthreads: number of threads to be used

    Output:
        derotated: array of shape (N,y,x) containing the derotated images
    """
    if new_center is None:
        new_center = [140, 140]

    tpool = mp.Pool(processes=numthreads)

    #klip.rotate(img, -angle, oldcenter, [152,152]) for img, angle, oldcenter
    tasks = [tpool.apply_async(klip.rotate, args=(img, -angle, center, new_center))
             for img, angle, center in zip(imgs, angles, centers)]

    derotated = np.array([task.get() for task in tasks])

    return derotated



def _check_output(enumerated_input):
    """
    Helper function that checks and prints progress of the multiprocessing

    Input:
        enumerated_input: 2 element tuple output of enumerate. First is the index, second is the output of apply_async
    """
    global tot_iter
    index = enumerated_input[0] + 1
    output = enumerated_input[1]
    output.wait()
    if index % 50 == 0:
        print("{0} percent done".format(float(index)*100/tot_iter))

def klip_adi_plus_sdi(imgs, centers, parangs, wvs, annuli=5, subsections=4, movement=3, numbasis=None, numthreads=None):
    """
    KLIP PSF Subtraction using angular differential imaging

    Inputs:
        imgs: array of 2D images for ADI. Shape of array (N,y,x)
        centers: N by 2 array of (x,y) coordinates of image centers
        parangs: N legnth array detailing parallactic angle of each image
        anuuli: number of annuli to use for KLIP
        subsections: number of sections to break each annuli into
        movement: minimum amount of movement (in pixels) of an astrophysical source
                  to consider using that image for a refernece PSF
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        numthreads:

    Ouput:
        sub_imgs: array of [array of 2D images (PSF subtracted)] using different number of KL basis vectors as
                    specified by numbasis. Shape of (b,N,y,x). Exception is if b==1. Then sub_imgs has the first
    """

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
    dr = float(OWA - IWA) / (annuli)

    #error checking for too small of annuli go here

    #calculate the annuli
    rad_bounds = [(dr * rad + IWA, dr * (rad + 1) + IWA) for rad in range(annuli)]
    #last annulus should mostly emcompass everything
    rad_bounds[annuli - 1] = (rad_bounds[annuli - 1][0], imgs[0].shape[0] / 2)

    #divide annuli into subsections
    dphi = 2 * np.pi / subsections
    phi_bounds = [(dphi * phi_i - np.pi, dphi * (phi_i + 1) - np.pi) for phi_i in range(subsections)]

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
    realigned_index = tpool.imap_unordered(_align_and_scale, enumerate(unique_wvs))

    outputs = []
    #as each is finishing, queue up the aligned data to be processed with KLIP
    for wv_index,wv_value in realigned_index:
        print(wv_index, wv_value)

        scidata_indicies = np.where(wvs == wv_value)[0]

        #def klip_section(img_num, parang, wavelength, wv_index, numbasis, radstart, radend, phistart, phiend, minmove, ref_center=None):
        # outputs += [tpool.apply_async(_klip_section_profiler, args=(file_index, parang, wv_value, wv_index, numbasis,
        #                                                   radstart, radend, phistart, phiend, movement))
        #                     for phistart,phiend in phi_bounds
        #                 for radstart, radend in rad_bounds
        #             for file_index,parang in zip(scidata_indicies, parangs[scidata_indicies])]
        outputs += [tpool.apply_async(_klip_section_multifile_profiler, args=(scidata_indicies, wv_value, wv_index, numbasis,
                                                          radstart, radend, phistart, phiend, movement))
                        for phistart,phiend in phi_bounds
                    for radstart, radend in rad_bounds]

    #harness the data!
    #check make sure we are completely unblocked before outputting the data
    #[out.wait() for out in outputs]
    # map(_check_output, enumerate(outputs))
    for index,out in enumerate(outputs):
        out.wait()
        if index % 50 == 0:
            print("{0} percent done".format(float(index)*100/tot_iter))
    # TODO: make the process of waiting for all threads to finish better and print progress in both python2 and python3


    #close to pool now and wait for all processes to finish
    print("Calling threadpool close()")
    tpool.close()
    print("Waiting for jobs to finish")
    tpool.join()

    #finished. Let's reshape the output images
    #move number of KLIP modes as leading axis (i.e. move from shape (N,y,x,b) to (b,N,y,x)
    sub_imgs = _arraytonumpy(output_imgs, output_imgs_shape)
    sub_imgs = np.rollaxis(sub_imgs.reshape((dims[0], dims[1], dims[2], numbasis.shape[0])), 3)
    #if we only passed in one value for numbasis (i.e. only want one PSF subtraction), strip off the number of basis)
    if sub_imgs.shape[0] == 1:
        sub_imgs = sub_imgs[0]
    #restore bad pixels
    sub_imgs[:, allnans[0], allnans[1], allnans[2]] = np.nan

    #derotate images
    #sub_imgs = np.array([rotate(img, pa, (140,140), center) for img,pa,center in zip(sub_imgs, parangs, centers)])

    return sub_imgs
