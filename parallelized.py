import klip
import multiprocessing as mp
import ctypes
import numpy as np
import numpy.linalg as la
import scipy.ndimage as ndimage

def tpool_init(original_imgs, original_imgs_shape, aligned_imgs, aligned_imgs_shape, output_imgs, output_imgs_shape,
               pa_imgs, wvs_imgs, centers_imgs):
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


def arraytonumpy(regular_array, shape=None):
    numpy_array = np.frombuffer(regular_array.get_obj())
    if shape is not None:
        numpy_array.shape = shape

    return numpy_array


def align_and_scale(ref_wv_iter, ref_center = None):
    """
    Aligns and scales the set of original images about a reference center and scaled to a reference wavelength
    """
    if ref_center is None:
        ref_center = [140, 140]

    #extract out wavelength information from the iteration argument
    ref_wv_index = ref_wv_iter[0]
    ref_wv = ref_wv_iter[1]

    original_imgs = arraytonumpy(original, original_shape)
    wvs_imgs = arraytonumpy(img_wv)
    centers_imgs = arraytonumpy(img_center, (np.size(wvs_imgs),2))

    recentered = np.array([klip.align_and_scale(frame, ref_center, old_center, ref_wv)
                               for frame, old_center, old_wv in zip(original_imgs, centers_imgs, wvs_imgs)])

    aligned_imgs = arraytonumpy(aligned, aligned_shape)
    aligned_imgs[ref_wv_index] = recentered

    return ref_wv_index, ref_wv

def klip_section(img_num, parang, wavelength, wv_index, numbasis, radstart, radend, phistart, phiend, minmove, ref_center=None):
    """
    Runs klip on a section of an image as given by the geometric parameters
    """
    if ref_center is None:
        ref_center = [140, 140]

    #create a coordinate system
    x, y = np.meshgrid(np.arange(original_shape[2] * 1.0), np.arange(original_shape[1] * 1.0))
    r = np.sqrt((x - ref_center[0])**2 + (y - ref_center[1])**2)
    phi = np.arctan2(y - ref_center[1], x - ref_center[0])

    #grab the pixel location of the section we are going to anaylze
    section_ind = np.where((r >= radstart) & (r < radend) & (phi >= phistart) & (phi < phiend))
    if np.size(section_ind) == 0:
        print("section is empty, skipping...")
        return False

    #grab the files suitable for reference PSF
    #load shared arrays for wavelengths and PAs
    wvs_imgs = arraytonumpy(img_wv)
    pa_imgs = arraytonumpy(img_pa)
    #calculate average movement in this section
    avg_rad = (radstart + radend) / 2.0
    file_ind = np.where(np.sqrt((np.radians(pa_imgs - parang) * avg_rad)**2 + ((wvs_imgs/wavelength - 1.0) * avg_rad)**2) > minmove)
    if np.size(file_ind) < 2:
        print("less than 2 reference PSFs available, skipping...")
        return False

    #load aligned images and make reference PSFs
    aligned_imgs = arraytonumpy(aligned, aligned_shape)[wv_index]
    ref_psfs = aligned_imgs[file_ind[0], :]
    ref_psfs = ref_psfs[:, section_ind[0]]
    #print(img_num, avg_rad, ref_psfs.shape)
    #print(sub_imgs.shape)
    #print(sub_imgs[img_num, section_ind, :].shape)

    #write to output
    output_imgs = arraytonumpy(output, output_shape)
    output_imgs[img_num, section_ind, :] = klip.klip_math(aligned_imgs[img_num, section_ind][0], ref_psfs, numbasis)
    return True

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

    #before we start, create the output array in flattened form
    #sub_imgs = np.zeros([dims[0], dims[1] * dims[2], numbasis.shape[0]])

    #implement the thread pool
    #make a bunch of shared memory arrays to transfer data between threads
    #make the array for the original images and initalize it
    original_imgs = mp.Array(ctypes.c_double, np.size(imgs))
    original_imgs_shape = imgs.shape
    original_imgs_np = arraytonumpy(original_imgs, original_imgs_shape)
    original_imgs_np = imgs
    #make array for recentered/rescaled image for each wavelength
    unique_wvs = np.unique(wvs)
    recentered_imgs = mp.Array(ctypes.c_double, np.size(imgs)*np.size(unique_wvs))
    recentered_imgs_shape = (np.size(unique_wvs),) + imgs.shape
    #make output array which also has an extra dimension for the number of KL modes to use
    output_imgs = mp.Array(ctypes.c_double, np.size(imgs)*np.size(numbasis))
    output_imgs_shape = imgs.shape + numbasis.shape
    #remake the PA, wv, and center arrays as shared arrays
    pa_imgs = mp.Array(ctypes.c_double, np.size(parangs))
    pa_imgs_np = arraytonumpy(pa_imgs)
    pa_imgs_np = parangs
    wvs_imgs = mp.Array(ctypes.c_double, np.size(wvs))
    wvs_imgs_np = arraytonumpy(wvs_imgs)
    wvs_imgs_np = wvs
    centers_imgs = mp.Array(ctypes.c_double, np.size(centers))
    centers_imgs_np = arraytonumpy(centers_imgs, centers.shape)
    centers_imgs_np = centers
    tpool = mp.Pool(processes=numthreads, initializaer=tpool_init,
                   initargs=(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs,
                             output_imgs_shape, pa_imgs, wvs_imgs, centers_imgs))

    #align and scale the images for each image. Use map to do this asynchronously
    realigned_index = tpool.imap_unordered(align_and_scale, enumerate(unique_wvs))

    outputs = []
    #as each is finishing, queue up the aligned data to be processed with KLIP
    for wv_index,wv_value in realigned_index:
        print(wv_index, wv_value)

        scidata_indicies = np.where(wvs == wv_value)[0]

        #def klip_section(img_num, parang, wavelength, wv_index, numbasis, radstart, radend, phistart, phiend, minmove, ref_center=None):
        outputs += [tpool.apply_async(klip_section, args=(file_index, parang, wv_value, wv_index, numbasis,
                                                          radstart, radend, phistart, phiend, movement))
                            for phistart,phiend in phi_bounds
                        for radstart, radend in rad_bounds
                    for file_index,parang in zip(scidata_indicies, parangs[scidata_indicies])]

    #harness the data!
    #check make sure we are completely unblocked before outputting the data
    [out.get() for out in outputs]

    #finished. Let's reshape the output images
    #move number of KLIP modes as leading axis (i.e. move from shape (N,y,x,b) to (b,N,y,x)
    sub_imgs = arraytonumpy(output_imgs, output_imgs_shape)
    sub_imgs = np.rollaxis(sub_imgs.reshape((dims[0], dims[1], dims[2], numbasis.shape[0])), 3)
    #if we only passed in one value for numbasis (i.e. only want one PSF subtraction), strip off the number of basis)
    if sub_imgs.shape[0] == 1:
        sub_imgs = sub_imgs[0]
    #restore bad pixels
    sub_imgs[:, allnans[0], allnans[1], allnans[2]] = np.nan

    #derotate images
    #sub_imgs = np.array([rotate(img, pa, (140,140), center) for img,pa,center in zip(sub_imgs, parangs, centers)])

    return sub_imgs
