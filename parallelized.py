import pyklip.klip as klip
import multiprocessing as mp
import ctypes
import numpy as np
import cProfile
import os
import itertools
import fakes

import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import spectra_management as spec

def _tpool_init(original_imgs, original_imgs_shape, aligned_imgs, aligned_imgs_shape, output_imgs, output_imgs_shape,
               pa_imgs, wvs_imgs, centers_imgs,ori_PSFs_shared,rec_PSFs_shared,out_PSFs_shared, seg_index_shared, seg_basis_shared, seg_shape_shared):
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
        ori_PSFs_shared: original images containing the sole PSFs. Same shape as original_imgs.
        rec_PSFs_shared: aligned and scaled images of the sole PSFs for processing. Same shape as aligned.
        out_PSFs_shared: output images of the sole PSFs after KLIP processing. Same shape as output_imgs.
    """
    global original, original_shape, aligned, aligned_shape, output, output_shape, img_pa, img_wv, img_center, ori_PSFs_thread,rec_PSFs_thread,out_PSFs_thread, seg_index, seg_basis, seg_shape
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
    #Management of the sole PSFs arrays
    ori_PSFs_thread = ori_PSFs_shared
    rec_PSFs_thread = rec_PSFs_shared
    out_PSFs_thread = out_PSFs_shared

    #array for basis vectors
    seg_index = seg_index_shared
    seg_basis = seg_basis_shared
    seg_shape = seg_shape_shared

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

    # Apply align and scale to the sole PSFs as well.
    if ori_PSFs_thread is not None:
        ori_PSFs_thread_np = _arraytonumpy(ori_PSFs_thread, original_shape)
        rec_PSFs_thread_np = _arraytonumpy(rec_PSFs_thread, aligned_shape)
        rec_PSFs_thread_np[ref_wv_index, :, :, :] =  np.array([klip.align_and_scale(frame_PSF, ref_center, old_center, ref_wv/old_wv)
                                            for frame_PSF, old_center, old_wv in zip(ori_PSFs_thread_np, centers_imgs, wvs_imgs)])
    
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
                            minmove, ref_center, minrot, maxrot, spectrum, mode, onesegment, companion_theta):
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
        maxrot: maximum PA rotation (in degrees) to be considered for use as a reference PSF (temporal variability)
        spectrum: if not None, optimizes the choosing the reference PSFs based on the spectrum
                        shape. Currently only supports "methane" in H band.
        mode: one of ['ADI', 'SDI', 'ADI+SDI'] for ADI, SDI, or ADI+SDI

    Outputs:
        returns True on success, False on failure. Does not return whether KLIP on each individual image was sucessful.
        Saves data to output array as defined in _tpool_init()
    """

    #create a coordinate system. Can use same one for all the images because they have been aligned and scaled
    x, y = np.meshgrid(np.arange(original_shape[2] * 1.0), np.arange(original_shape[1] * 1.0))
    x.shape = (x.shape[0] * x.shape[1]) #Flatten
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
    # Do the same for the sole PSFs (~fake planet)
    if rec_PSFs_thread is not None:
        rec_PSFs_thread_np = _arraytonumpy(rec_PSFs_thread, (aligned_shape[0], aligned_shape[1], aligned_shape[2] * aligned_shape[3]))[wv_index]
        PSFsarea_thread_np = rec_PSFs_thread_np[:,  section_ind[0]]
    else:
        PSFsarea_thread_np = None

    #do the same for the reference PSFs
    #playing some tricks to vectorize the subtraction of the mean for each row
    ref_psfs_mean_sub = ref_psfs - np.nanmean(ref_psfs, axis=1)[:, None]
    ref_psfs_mean_sub[np.where(np.isnan(ref_psfs_mean_sub))] = 0

    # Replace the nans of the sole PSFs (~fake planet) area by zeros.
    # We don't want to subtract the mean here. Well at least JB thinks so...
    if PSFsarea_thread_np is not None:
        PSFsarea_thread_np[np.where(np.isnan(PSFsarea_thread_np))] = 0

    #calculate the covariance matrix for the reference PSFs
    #note that numpy.cov normalizes by p-1 to get the NxN covariance matrix
    #we have to correct for that in the klip.klip_math routine when consturcting the KL
    #vectors since that's not part of the equation in the KLIP paper
    covar_psfs = np.cov(ref_psfs_mean_sub)
    #also calculate correlation matrix since we'll use that to select reference PSFs
    covar_diag = np.diagflat(1./np.sqrt(np.diag(covar_psfs)))
    corr_psfs = np.dot( np.dot(covar_diag, covar_psfs ), covar_diag)

    #grab the parangs
    parangs = _arraytonumpy(img_pa)
    if onesegment is not True:
        for file_index,parang in zip(scidata_indicies, parangs[scidata_indicies]):
            try:
                _klip_section_multifile_perfile(file_index, section_ind, ref_psfs_mean_sub, covar_psfs, corr_psfs,
                                                parang, wavelength, wv_index, (radstart + radend) / 2.0, numbasis, minmove,
                                                minrot, maxrot, mode,
                                                PSFsarea_thread_np = PSFsarea_thread_np, spectrum=spectrum)
            except (ValueError, RuntimeError, TypeError) as err:
                print("({0}): {1}".format(err.errno, err.strerror))
                return False
    else:
        # section_ind must be re-defined for each image due to rotation of astrophysical source
        for file_index,parang in zip(scidata_indicies, parangs[scidata_indicies]):
            try:
                this_phi = ((phi + (np.pi - (((-1.0* parang) * (np.pi/180.0))- (companion_theta + np.pi)))) % (2.0 * np.pi)) - np.pi
                section_ind = np.where((r >= radstart) & (r < radend) & (this_phi >= phistart) & (this_phi < phiend))

                aligned_imgs = _arraytonumpy(aligned, (aligned_shape[0], aligned_shape[1], aligned_shape[2] * aligned_shape[3]))[wv_index]
                ref_psfs = aligned_imgs[:,  section_ind[0]]
   
                PSFsarea_thread_np = None

                ref_psfs_mean_sub = ref_psfs - np.nanmean(ref_psfs, axis=1)[:, None]
                ref_psfs_mean_sub[np.where(np.isnan(ref_psfs_mean_sub))] = 0

                covar_psfs = np.cov(ref_psfs_mean_sub)
                covar_diag = np.diagflat(1./np.sqrt(np.diag(covar_psfs)))
                corr_psfs = np.dot( np.dot(covar_diag, covar_psfs ), covar_diag)

                _klip_section_multifile_perfile(file_index, section_ind, ref_psfs_mean_sub, covar_psfs, corr_psfs,
                                                parang, wavelength, wv_index, (radstart + radend) / 2.0, numbasis, minmove,
                                                minrot, maxrot, mode,
                                                PSFsarea_thread_np = PSFsarea_thread_np, spectrum=spectrum, onesegment=onesegment)

            except (ValueError, RuntimeError, TypeError) as err:
                print("({0}): {1}".format(err.errno, err.strerror))
                return False

 #   [_klip_section_multifile_perfile(file_index, section_ind, ref_psfs_mean_sub, covar_psfs,
  #                                   parang, wavelength, wv_index, (radstart + radend) / 2.0, numbasis, minmove)
  #      for file_index,parang in zip(scidata_indicies, parangs[scidata_indicies])]

    return True


def _klip_section_multifile_perfile(img_num, section_ind, ref_psfs, covar,  corr, parang, wavelength, wv_index, avg_rad,
                                    numbasis, minmove, minrot, maxrot, mode, PSFsarea_thread_np = None, spectrum=None, onesegment=False):
    """
    Imitates the rest of _klip_section for the multifile code. Does the rest of the PSF reference selection

    Inputs:
        img_num: file index for the science image to process
        section_ind: np.where(pixels are in this section of the image). Note: coordinate system is collapsed into 1D
        ref_psfs: reference psf images of this section
        covar: the covariance matrix of the reference PSFs. Shape of (N,N)
        corr: the correlation matrix of the refernece PSFs. Shape of (N,N)
        parang: PA of science iamge
        wavelength: wavelength of science image
        wv_index: array index of the wavelength of the science image
        avg_rad: average radius of this annulus
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        minmove: minimum movement between science image and PSF reference image to use PSF reference image (in pixels)
        mode: one of ['ADI', 'SDI', 'ADI+SDI'] for ADI, SDI, or ADI+SDI
        PSFsarea_thread_np: Ignored if None. Should be the same as ref_psfs but with the sole PSFs.
        spectrum: if not None, optimizes the choosing the reference PSFs based on the spectrum
                        shape. Currently only supports "methane" in H band.

    Outputs:
        return True on success, False on failure.
        Saves image to output array defined in _tpool_init()
    """
    #grab the files suitable for reference PSF
    #load shared arrays for wavelengths and PAs
    wvs_imgs = _arraytonumpy(img_wv)
    pa_imgs = _arraytonumpy(img_pa)
    #calculate average movement in this section for each PSF reference image w.r.t the science image
    moves = klip.estimate_movement(avg_rad, parang, pa_imgs, wavelength, wvs_imgs, mode)
    #check all the PSF selection criterion
    #enough movement of the astrophyiscal source
    goodmv = (moves >= minmove)
    #enough field rotation
    if minrot > 0:
        goodmv = (goodmv) & (np.abs(pa_imgs - parang) >= minrot)
    #optimization for different spectrum
    if "SDI" not in mode.upper():
        goodmv = (goodmv) & (wvs_imgs == wavelength)
    if spectrum is not None:
        if spectrum.lower() == "methane":
            goodmv = (goodmv) | ((wvs_imgs > 1.64) & (wvs_imgs < 1.8)) | ((wvs_imgs < 1.19) & (wvs_imgs > 1.1))
    if "ADI" not in mode.upper():
        goodmv = (goodmv) & (pa_imgs == parang)
    #if minrot > 0:
    #    file_ind = np.where((moves >= minmove) & (np.abs(pa_imgs - parang) >= minrot))
    #else:
    #    file_ind = np.where(moves >= minmove)
    #select the good reference PSFs
    file_ind = np.where(goodmv)
    if np.size(file_ind[0]) < 2:
        print("less than 2 reference PSFs available for minmove={0}, skipping...".format(minmove))
        return False
    #pick out a subarray. Have to play around with indicies to get the right shape to index the matrix
    covar_files = covar[file_ind[0].reshape(np.size(file_ind), 1), file_ind[0]]

    #pick only the most correlated reference PSFs if there's more than enough PSFs
    maxbasis_requested = np.max(numbasis)
    maxbasis_possible = np.size(file_ind)
    if maxbasis_possible > maxbasis_requested:
        xcorr = corr[img_num, file_ind[0]]  # grab the x-correlation with the sci img for valid PSFs
        sort_ind = np.argsort(xcorr)
        closest_matched = sort_ind[-maxbasis_requested:]  # sorted smallest first so need to grab from the end
        # grab the new and smaller covariance matrix
        covar_files = covar_files[closest_matched.reshape(np.size(closest_matched), 1), closest_matched]
        # grab smaller set of reference PSFs
        ref_psfs_selected = ref_psfs[file_ind[0][closest_matched], :]
        if PSFsarea_thread_np is not None:
            PSFsarea_thread_np_selected = PSFsarea_thread_np[file_ind[0][closest_matched], :]
    else:
        # else just grab the reference PSFs for all the valid files
        ref_psfs_selected = ref_psfs[file_ind[0], :]
        if PSFsarea_thread_np is not None:
            PSFsarea_thread_np_selected = PSFsarea_thread_np[file_ind[0], :]

    #load input/output data
    aligned_imgs = _arraytonumpy(aligned, (aligned_shape[0], aligned_shape[1], aligned_shape[2]*aligned_shape[3]))[wv_index]
    output_imgs = _arraytonumpy(output, (output_shape[0], output_shape[1]*output_shape[2], output_shape[3]))
    if rec_PSFs_thread is not None:
        rec_PSFs_thread_np = _arraytonumpy(rec_PSFs_thread, (aligned_shape[0], aligned_shape[1], aligned_shape[2] * aligned_shape[3]))[wv_index]
        out_PSFs_threads_np = _arraytonumpy(out_PSFs_thread, (output_shape[0], output_shape[1]*output_shape[2], output_shape[3]))
    if onesegment is True:
        seg_basis_np = _arraytonumpy(seg_basis, seg_shape)
        seg_index_np = _arraytonumpy(seg_index, (seg_shape[0], seg_shape[2]))
        seg_basis_np[img_num,:] = -9
        seg_index_np[img_num,:] = -9
        seg_index_np[img_num, 0:np.size(section_ind[0])] = section_ind[0]
    #run KLIP
    try:
        if rec_PSFs_thread is not None:
            klipped,klipped_solePSFs = klip.klip_math(aligned_imgs[img_num, section_ind[0]], ref_psfs_selected, numbasis, covar_psfs=covar_files,
                                     PSFarea_tobeklipped=rec_PSFs_thread_np[img_num, section_ind[0]], PSFsarea_forklipping=PSFsarea_thread_np_selected)
        else:
            if onesegment is True:
                klipped, basis = klip.klip_math(aligned_imgs[img_num, section_ind[0]], ref_psfs_selected, numbasis, covar_psfs=covar_files, return_basis=True)
            else:
                klipped = klip.klip_math(aligned_imgs[img_num, section_ind[0]], ref_psfs_selected, numbasis, covar_psfs=covar_files)
    except (ValueError, RuntimeError, TypeError) as err:
        print("({0}): {1}".format(err.errno, err.strerror))
        return False

    #write to output
    output_imgs[img_num, section_ind[0], :] = klipped
    if onesegment is True:
        seg_basis_np[img_num, :, 0:np.size(section_ind[0])] = basis
    
    if rec_PSFs_thread is not None:
        out_PSFs_threads_np[img_num, section_ind[0], :] = klipped_solePSFs

    return True


def rotate_imgs(imgs, angles, centers, new_center=None, numthreads=None, flipx=True, hdrs=None,disable_wcs_rotation = False):
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
        if not disable_wcs_rotation:
            [klip._rotate_wcs_hdr(astr_hdr, angle, flipx=flipx) for angle, astr_hdr in zip(angles, hdrs)]

    #reform back into a giant array
    derotated = np.array([task.get() for task in tasks])

    tpool.close()

    return derotated

def high_pass_filter_imgs(imgs, numthreads=None, filtersize=10):
    """
    filters a sequences of images using a FFT

    Inputs:
        imgs: array of shape (N,y,x) containing N images
        numthreads: number of threads to be used
        filtersize: size in Fourier space of the size of the space. In image space, size=img_size/filtersize

    Output:
        filtered: array of shape (N,y,x) containing the filtered images
    """

    tpool = mp.Pool(processes=numthreads)


    tasks = [tpool.apply_async(klip.high_pass_filter, args=(img, filtersize)) for img in imgs]

    #reform back into a giant array
    filtered = np.array([task.get() for task in tasks])

    tpool.close()

    return filtered


def klip_parallelized(imgs, centers, parangs, wvs, IWA, mode='ADI+SDI', onesegment=False, annuli=5, subsections=4, movement=3, numbasis=None,
                      aligned_center=None, numthreads=None, minrot=0, maxrot=360, PSFs = None, out_PSFs=None, spectrum=None,
                      companion_rho = 50.0, companion_theta = 0.0, segment_dr = 10.0, segment_dt = 90.0):
    """
    multithreaded KLIP PSF Subtraction

    Inputs:
        imgs: array of 2D images for ADI. Shape of array (N,y,x)
        centers: N by 2 array of (x,y) coordinates of image centers
        parangs: N length array detailing parallactic angle of each image
        wvs: N length array of the wavelengths
        IWA: inner working angle (in pixels)
        mode: one of ['ADI', 'SDI', 'ADI+SDI'] for ADI, SDI, or ADI+SDI
        anuuli: number of annuli to use for KLIP
        subsections: number of sections to break each annuli into
        movement: minimum amount of movement (in pixels) of an astrophysical source
                  to consider using that image for a refernece PSF
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        aligned_center: array of 2 elements [x,y] that all the KLIP subtracted images will be centered on for image
                        registration
        numthreads: number of threads to use. If none, defaults to using all the cores of the cpu
        minrot: minimum PA rotation (in degrees) to be considered for use as a reference PSF (good for disks)
        maxrot: maximum PA rotation (in degrees) to be considered for use as a reference PSF (temporal variability)
        PSFs: Array of shape similar to imgs. It should contain sole PSFs. It will suffer exactly the same
              transformation as imgs without influencing the KL-modes.
        out_PSFs: Array of shape similar to sub_imgs (the output of this function). It contains the reduced images of PSFs.
                  It should be defined as out_PSFs = np.zeros((np.size(numbasis),)+dataset.input.shape).
        spectrum (only applicable for SDI): if not None, optimizes the choosing the reference PSFs based on the spectrum
                        shape. Currently only supports "methane" in H band.

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
        #aligned_center = [int(imgs.shape[2]//2), int(imgs.shape[1]//2)]
        aligned_center = [np.mean(centers[:,0]), np.mean(centers[:,1])]

    #save all bad pixels
    allnans = np.where(np.isnan(imgs))
    if PSFs is not None:
        allnans_PSFs = np.where(np.isnan(PSFs))

    #use first image to figure out how to divide the annuli
    #TODO: what to do with OWA
    #need to make the next 10 lines or so much smarter
    dims = imgs.shape
    x, y = np.meshgrid(np.arange(dims[2] * 1.0), np.arange(dims[1] * 1.0))
    nanpix = np.where(np.isnan(imgs[0]))
    OWA = np.sqrt(np.min((x[nanpix] - centers[0][0]) ** 2 + (y[nanpix] - centers[0][1]) ** 2))
    dr = float(OWA - IWA) / (annuli)

    #error checking for too small of annuli go here

    if onesegment == False:
        #calculate the annuli
        rad_bounds = [(dr * rad + IWA, dr * (rad + 1) + IWA) for rad in range(annuli)]
        #last annulus should mostly emcompass everything
        rad_bounds[annuli - 1] = (rad_bounds[annuli - 1][0], imgs[0].shape[0])

        #divide annuli into subsections
        dphi = 2 * np.pi / subsections
        phi_bounds = [[dphi * phi_i - np.pi, dphi * (phi_i + 1) - np.pi] for phi_i in range(subsections)]
        phi_bounds[-1][1] = 2. * np.pi
    else:
        # Only one segment, define bounds according companion_rho, _theta, segment_dr, _dt
        rad_bounds = [(np.around(companion_rho - (segment_dr / 2.0)), np.around(companion_rho + (segment_dr / 2.0)))]
        # For phi, need to take into account how PA translated to python arctan2 coordinates
        # I think we have to subtract 270 degrees from the PA, and wrap it to -180 -> 180
        companion_theta_corrected = companion_theta - 270.0
        if companion_theta_corrected < -180:
            companion_theta_corrected += 360.0
        #Convert to radians
        companion_theta = companion_theta_corrected * (np.pi/180.0)
        segment_dt *= (np.pi/180.0)
        phi_bounds = [(((-1.0)*(segment_dt/2.0)), segment_dt/2.0)]
        # will need to redefine phi in klip_multifile so that the discontinuity occurs at companion_theta_corrected + pi
        # phi will be ((phi + (np.pi - (companion_theta_corrected + parallactic_angle))) % (2.0 * np.pi)) - np.pi

        #Also work out how big this segment will be to create the basis and index arrays
        x_tmp, y_tmp = np.meshgrid(np.arange(dims[2] * 1.0), np.arange(dims[1] * 1.0))
        x_tmp.shape = (x_tmp.shape[0] * x_tmp.shape[1]) #Flatten
        y_tmp.shape = (y_tmp.shape[0] * y_tmp.shape[1])
        r_tmp = np.sqrt((x_tmp - aligned_center[0])**2 + (y_tmp - aligned_center[1])**2)
        phi_tmp = np.arctan2(y_tmp - aligned_center[1], x_tmp - aligned_center[0])
        section_ind_size = np.size(np.where((r_tmp >= rad_bounds[0][0]) & (r_tmp < rad_bounds[0][1]) & (phi_tmp >= phi_bounds[0][0]) & (phi_tmp < phi_bounds[0][1])))
        section_ind_size = int(np.round(section_ind_size * 1.1))


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

    if onesegment is True:
        #Create arrays storing the basis vectors and index
        #seg_index_shared size (N, s), as is same for all numbasis
        #seg_basis_shared size (N, b, s)
        seg_shape_shared = (np.size(wvs), np.max(numbasis), section_ind_size)

        seg_index_shared = mp.Array(ctypes.c_double, (np.size(wvs) * section_ind_size))
        #seg_index_np = _arraytonumpy(seg_index_shared, (seg_shape_shared[0], seg_shape_shared[2]))

        seg_basis_shared = mp.Array(ctypes.c_double, (np.size(wvs) * np.max(numbasis) * section_ind_size))
        #print (np.size(wvs) * np.max(numbasis) * section_ind_size)), seg_shape_shared
        #seg_basis_np = _arraytonumpy(seg_basis_shared, seg_shape_shared)
    else:
        seg_index_shared = None
        seg_basis_shared = None
        seg_shape_shared = None

    if PSFs is not None:
        # ori_PSFs_shared are the sole PSFs images. It corresponds to input set of images.
        ori_PSFs_shared = mp.Array(ctypes.c_double, np.size(imgs))
        ori_PSFs_shared_np = _arraytonumpy(ori_PSFs_shared, PSFs.shape)
        ori_PSFs_shared_np[:] = PSFs
        # rec_PSFs_shared contains the PSFs rescaled and realigned for all the different wavelengths.
        rec_PSFs_shared =  mp.Array(ctypes.c_double, np.size(imgs)*np.size(unique_wvs))
        #rec_PSFs_shared_np = _arraytonumpy(rec_PSFs_shared, (np.size(unique_wvs),) + imgs.shape)
        out_PSFs_shared = mp.Array(ctypes.c_double, np.size(imgs)*np.size(numbasis))
        out_PSFs_shared_np = _arraytonumpy(out_PSFs_shared)
        out_PSFs_shared_np[:] = np.nan
    else:
        # If no PSFs then define None variables.
        ori_PSFs_shared = None
        rec_PSFs_shared = None
        out_PSFs_shared = None

    tpool = mp.Pool(processes=numthreads, initializer=_tpool_init,
                   initargs=(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs,

                             output_imgs_shape, pa_imgs, wvs_imgs, centers_imgs,
                             ori_PSFs_shared, rec_PSFs_shared, out_PSFs_shared, seg_index_shared, seg_basis_shared, seg_shape_shared), maxtasksperchild=50)

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

        #perform KLIP asynchronously for each group of files of a specific wavelength and section of the image
        outputs += [tpool.apply_async(_klip_section_multifile, args=(scidata_indicies, wv_value, wv_index, numbasis,
                                                                     radstart, radend, phistart, phiend, movement,
                                                                     aligned_center, minrot, maxrot, spectrum, mode, onesegment, companion_theta))
                    for phistart,phiend in phi_bounds
                    for radstart, radend in rad_bounds]

    #harness the data!
    #check make sure we are completely unblocked before outputting the data
    print("Total number of tasks for KLIP processing is {0}".format(tot_iter))
    for index, out in enumerate(outputs):
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

    if PSFs is not None:
        rec_PSFs_shared_np = _arraytonumpy(rec_PSFs_shared, (np.size(unique_wvs),) + imgs.shape)
        out_PSFs_shared_np = np.rollaxis(out_PSFs_shared_np.reshape((dims[0], dims[1], dims[2], numbasis.shape[0])), 3)
        out_PSFs_shared_np[:, allnans_PSFs[0], allnans_PSFs[1], allnans_PSFs[2]] = np.nan

    #JB's debug
    #plt.imshow(ori_PSFs_shared_np[100,:,:],interpolation = 'nearest')
    #plt.show()
    #plt.imshow(rec_PSFs_shared_np[1,100,:,:],interpolation = 'nearest')
    #plt.show()
    #plt.imshow(out_PSFs_shared_np[1,100,:,:],interpolation = 'nearest')
    #plt.show()

    #scrapping this behavior for now because I don't feel like dealing with edge cases
    ## if we only passed in one value for numbasis (i.e. only want one PSF subtraction), strip off that axis)
    #if sub_imgs.shape[0] == 1:
    #    sub_imgs = sub_imgs[0]

    #all of the image centers are now at aligned_center
    centers[:,0] = aligned_center[0]
    centers[:,1] = aligned_center[1]

    # Output for the sole PSFs
    if PSFs is not None:
        out_PSFs[:] = out_PSFs_shared_np

    if onesegment is not True:
        return sub_imgs, None, None
    else:
        seg_basis_np = _arraytonumpy(seg_basis_shared, seg_shape_shared)
        seg_index_np = _arraytonumpy(seg_index_shared, (seg_shape_shared[0], seg_shape_shared[2]))
        return sub_imgs, seg_basis_np, seg_index_np


class psfs_management:
    """
    Structure to tune PSFs calculation inside klip. To give as input parameter of klip_dataset().

    TODO: 1/ Specifying fake planets position.
          2/ It would be good to parameterize the planet spectrum like for the star instead of specifying a file.
          3/ Make it work for ADI only

    Fields:
        calculate_PSFs: Integer activating the computation of the PSF through KLIP. It injects fake planets into the
                        original cubes and apply KLIP normally.
                            If 0, regular klip without psfs calculation
                            If 1, psf calculation using satellite spots PSFs
                            If 2, psf calculation gaussian PSFs (I the sat spots are too noisy)
                        In addition it creates another set of cubes with the
                        sole planets (no speckles) and apply the same transformation on this cube as on the first one.
                        It will create an extra output fileprefix + "-KLmodes-all-PSFs.fits" which is the equivalent of
                        fileprefix + "-KLmodes-all.fits" but built with the sole PSFs dataset.
        fake_planet_flux_ratio: Ratio of the flux of the planets over the flux of the satellite spots.
                        The flux is defined as the integral of the spectrum over the spectral channel (basically a sum
                        of the spectrum). This ratio is not corrected for transmission. It is the ratio of the flux in
                        gpi cubes as they appear but not the real ratio of the planet and the star.
        spectrum: Spectrum of the planet to be included. psfs_management.spectrum is ignored if the spectrum is defined
                through a template. See the following fields.
        pipeline_directory: directory of GPI pipeline on your computer.
        star_type: Type of the star of the dataset. The type is then translated in a temperature using pickles lookup table.
                Then the temperature is used to interpolate a spectrum using the pickles catalog.
                The type is ignored if a star_temperature is defined.
        star_temperature: Temperature of the star of the dataset. Overwrite star_type.
                        The temperature is used to interpolate a spectrum using the pickles catalog.
        planet_spec_filename: filename of the planet spectrum to considered. Mark Marley's files should be used.

    Functions:
    """
    def __init__(self, calculate_PSFs=0,
                 fake_planet_flux_ratio= 1.0,
                 spectrum = [],
                 pipeline_directory='',
                 star_type = None,
                 star_temperature=None,
                 planet_spec_filename='',
                 angle_offset = 0.0):
        self.calculate_PSFs = calculate_PSFs
        self.fake_planet_flux_ratio = fake_planet_flux_ratio
        self.angle_offset = angle_offset

        self.spectrum = spectrum # must contain positive values

        self.pipeline_directory = pipeline_directory
        self.star_type = star_type
        self.star_temperature = star_temperature
        self.planet_spec_filename = planet_spec_filename

def klip_dataset(dataset, mode='ADI+SDI', onesegment=False, outputdir=".", fileprefix="",
                 annuli=5, subsections=4, movement=3, numbasis=None, numthreads=None,
                 minrot=0, calibrate_flux=False, aligned_center=None, psfs_struct=None,
                 sat_spot_psf = False,spectrum=None, highpass=False,
                 companion_rho = 50.0, companion_theta = 0.0, segment_dr = 10.0, segment_dt = 90.0):
    """
    run klip on a dataset class outputted by an implementation of Instrument.Data

    Inputs:
        dataset:        an implementation of Instrument.Data (see instruments/ subfolder)
        mode:           one of ['ADI', 'SDI', 'ADI+SDI'] for ADI, SDI, or ADI+SDI
        onesegment:     False (default) - pefrom KLIP on entire image, True - perform KLIP on one segment defined
                        by segment_dr, segment_dt, centered on (companion_rho, companion_theta). ONLY WORKS ON ADI+SDI
        outputdir:      directory to save output files
        fileprefix:     filename prefix for saved files
        anuuli:         number of annuli to use for KLIP
        subsections:    number of sections to break each annuli into
        movement:       minimum amount of movement (in pixels) of an astrophysical source
                        to consider using that image for a refernece PSF
        numbasis:       number of KL basis vectors to use (can be a scalar or list like). Length of b
        numthreads:     number of threads to use. If none, defaults to using all the cores of the cpu
        minrot:         minimum PA rotation (in degrees) to be considered for use as a reference PSF (good for disks)
        calibrate_flux: if True calibrate flux of the dataset, otherwise leave it be
        aligned_center: array of 2 elements [x,y] that all the KLIP subtracted images will be centered on for image
                        registration
        psfs_struct:    Structure to tune PSFs calculation inside klip. See documentation of psfs_management class.
                        This is because this version is not definitive.
        sat_spot_psf:   Save a original psf and radial psf cube (invariant by rotation). These PSFs are not klipped.
                        spectrum (only applicable for SDI): if not None, optimizes the choosing the reference PSFs based on the spectrum
                        shape. Currently only supports "methane" in H band.
        highpass:       if True, run a high pass filter
        companion_rho:  Separation of known companion (pixels, used if onesegment==True)
        companion_theta: PA of known companion (degrees, used if onesegment==True)
        segment_dr:     Radial width of segment (pixels)
        segment_dt:     Azimuthal width of segment (degrees)

    Output
        Saved files in the output directory
        Returns: nothing, but saves to dataset.output: (b, N, wv, y, x) 5D cube of KL cutoff modes (b), number of images
                            (N), wavelengths (wv), and spatial dimensions. Images are derotated.
                            For ADI only, the wv is omitted so only 4D cube
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

    if highpass:
        dataset.input = high_pass_filter_imgs(dataset.input)

    # Default value of psfs_struct. Don't do anything
    if psfs_struct is None:
        psfs_struct = psfs_management()

    #if no outputdir specified, then current working directory (don't want to write to '/'!)
    if outputdir == "":
        outputdir = "."

    #save klip parameters as a string
    klipparams = "mode={mode},annuli={annuli},subsect={subsections},minmove={movement}," \
                 "numbasis={numbasis}/{maxbasis},minrot={minrot},calibflux={calibrate_flux},spectrum={spectrum}," \
                 "highpass={highpass}".format(mode=mode, annuli=annuli, subsections=subsections, movement=movement,
                                              numbasis="{numbasis}", maxbasis=np.max(numbasis), minrot=minrot,
                                              calibrate_flux=calibrate_flux, spectrum=spectrum, highpass=highpass)
    dataset.klipparams = klipparams

    #run KLIP
    if mode == 'ADI+SDI':
        print("Beginning ADI+SDI KLIP")

        if sat_spot_psf or psfs_struct.calculate_PSFs == 1:
            print("Calculating the planet PSF from the satellite spots...")
            dataset.generate_psf_cube(20)
            ## TODO remove next lines because it is already done at the end of klip (down several lines on this page)
            #dataset.get_radial_psf(save =  os.path.realpath(outputdir) + '/' + fileprefix) #outputdirpath
            #print("coucou")
            #return

        # If fake planets have to be injected into the datacube for PSFs computation then we do it here
        if psfs_struct.calculate_PSFs:
            print("Injecting fake planets...")

            # Calculate the radii of the annuli like in klip_adi_plus_sdi using the first image
            # We want to inject one planet per section where klip is independently applied.
            dims = dataset.input.shape
            x_grid, y_grid = np.meshgrid(np.arange(dims[2] * 1.0), np.arange(dims[1] * 1.0))
            nanpix = np.where(np.isnan(dataset.input[0]))
            OWA = np.sqrt(np.min((x_grid[nanpix] - dataset.centers[0][0]) ** 2 + (y_grid[nanpix] - dataset.centers[0][1]) ** 2))
            dr = float(OWA - dataset.IWA) / (annuli)

            # calculate the annuli mean radius where the fake planets are going to be.
            annuli_radii = np.array([dr * annuli_it + dataset.IWA + dr/2.for annuli_it in range(annuli)])
            # No PSFs in the last annulus which will emcompass everything

            # New array for data with sole PSFs
            PSFs = np.zeros(dataset.input.shape)
            PSFs[np.where(np.isnan(dataset.input))] = np.nan
            # Define an array that will contain the reduced PSFs dataset after klip_adi_plus_sdi().
            out_PSFs = np.zeros((np.size(numbasis),)+dataset.input.shape)

            # Extract the 37 points spectrum for the 37*N_cubes vector dataset.wvs.
            unique_wvs = np.unique(dataset.wvs)
            # Shoudl give numwaves=37
            numwaves = np.size(unique_wvs)
            # Number of cubes in dataset
            N_cubes = int(dataset.input.shape[0])/int(numwaves)
            # Peak value of the fake planet for each slice. To be define below.
            inputflux = np.zeros(dataset.input.shape[0])

            # Check if the user wants a spectrum from a star and planet template.
            # ie if the required fields of psfs_struct are defined.
            spec_from_template = psfs_struct.pipeline_directory !='' and (psfs_struct.star_type != None or psfs_struct.star_temperature != None) and psfs_struct.planet_spec_filename != ''

            # Define the peak value of the fake planet for each slice depending if spectrum is defined
            if not spec_from_template:
                if np.size(psfs_struct.spectrum) == 37:
                    for k in range(N_cubes):
                        inputflux[37*k:37*(k+1)] = psfs_struct.spectrum*psfs_struct.fake_planet_flux_ratio*(np.mean(dataset.spot_flux[37*k:37*(k+1)])/np.mean(psfs_struct.spectrum))
                elif np.size(psfs_struct.spectrum) == 0:
                    for k in range(N_cubes):
                        inputflux[37*k:37*(k+1)] = psfs_struct.fake_planet_flux_ratio*dataset.spot_flux[37*k:37*(k+1)]
                elif np.size(psfs_struct.spectrum) != 0:
                    print("Error size of spectrum in klip_dataset()")
                    return 0
                if 0: # plot individual spectra for debugging.
                    plt.figure(2)
                    plt.plot(dataset.spot_flux[37*k:37*(k+1)],'r')
                    plt.plot(inputflux[37*k:37*(k+1)],'b')
                    ax = plt.gca()
                    ax.legend(["sat spot","planet"]) #, loc = 'upper left'
                    print('psfs_struct.fake_planet_flux_ratio = ',psfs_struct.fake_planet_flux_ratio)
                    plt.show()

            # Define the peak value of the fake planet for each slice depending if a star and a planet type is given.
            filter = dataset.prihdrs[0]['IFSFILT'].split('_')[1]
            if spec_from_template:
                # Interpolate a spectrum of the star based on its spectral type/temperature
                wv,star_sp = spec.get_star_spectrum(psfs_struct.pipeline_directory,filter,psfs_struct.star_type,psfs_struct.star_temperature)
                # Interpolate the spectrum of the planet based on the given filename
                wv,planet_sp = spec.get_planet_spectrum(psfs_struct.planet_spec_filename,filter)
                ratio_spec_models = planet_sp/star_sp
                for k in range(N_cubes):
                    inputflux[37*k:37*(k+1)] = dataset.spot_flux[37*k:37*(k+1)]*ratio_spec_models
                    inputflux[37*k:37*(k+1)] *= psfs_struct.fake_planet_flux_ratio*(np.mean(dataset.spot_flux[37*k:37*(k+1)])/np.mean(inputflux[37*k:37*(k+1)]))

                    if 0: # plot individual spectra for debugging.
                        if 0:
                            plt.figure(1)
                            plt.plot(wv,star_sp,'r')
                            plt.plot(wv,planet_sp,'g')
                            ax = plt.gca()
                            ax.legend(["star","planet"]) #, loc = 'upper left'
                        plt.figure(2)
                        plt.plot(wv,dataset.spot_flux[37*k:37*(k+1)],'r')
                        plt.plot(wv,inputflux[37*k:37*(k+1)],'b')
                        ax = plt.gca()
                        ax.legend(["sat spot","planet"]) #, loc = 'upper left'
                        print('psfs_struct.fake_planet_flux_ratio = ',psfs_struct.fake_planet_flux_ratio)
                        plt.show()

            # Manage the shape of the PSF.
            # If psfs_struct.calculate_PSFs then the PSF is calculated from the sat spots.
            # Otherwise a gaussian PSF will be used later on.
            if psfs_struct.calculate_PSFs == 1:
                inputpsfs = np.tile(dataset.psfs,(N_cubes,1,1))
                for l in range(dataset.input.shape[0]):
                    #peak value of the psfs is normalized to unity.
                    inputpsfs[l,:,:] /= np.nanmax(inputpsfs[l,:,:])
                    # then multiplied by the peak flux
                    inputpsfs[l,:,:] *= inputflux[l]
            elif psfs_struct.calculate_PSFs == 2:
                # If inputpsfs is a vector and not a cube fakes.inject_planet will know that it has to use gaussian PSFs.
                inputpsfs = inputflux
            elif psfs_struct.calculate_PSFs != 0:
                print("Error calculate_PSFs parameter has invalid value in klip_dataset()")
                return 0


            # Get parallactic angle of where to put fake planets
            #PSF_dist = 20 # Distance between PSFs. Actually length of an arc between 2 consecutive PSFs.
            #delta_pa = 180/np.pi*PSF_dist/radius
            delta_th = 360/subsections
            th_list = np.arange(-180+delta_th/2.,180.1-delta_th/2.,delta_th) + psfs_struct.angle_offset
            pa_list = fakes.covert_polar_to_image_pa(th_list, dataset.wcs[0])
            #print(annuli_radii,th_list,pa_list)
            #return
            # Loop for injecting fake planets. One planet per section of the image.
            # Too many hard-coded parameters because still work in progress.
            for annuli_id, radius in enumerate(annuli_radii):
                for pa_id, pa in enumerate(pa_list):
                    fakes.inject_planet(PSFs, dataset.centers, inputpsfs, dataset.wcs, radius, pa)
                    fakes.inject_planet(dataset.input, dataset.centers, inputpsfs, dataset.wcs, radius, pa)
            # Save fits for debugging on JB's computer
            if 0:
                #print(dataset.spot_flux)
                pyfits.writeto("/Users/jruffio/gpi/pyklip/outputs/tmpPSFs.fits", PSFs, clobber=True)
                pyfits.writeto("/Users/jruffio/gpi/pyklip/outputs/tmpINPUT.fits", dataset.input, clobber=True)


            fakePlparams = "fake_planet_flux_ratio={pfake_planet_flux_ratio};" \
                           "angle_offset={pangle_offset};" \
                           "spectrum={pspectrum};" \
                           "star_type={pstar_type};" \
                           "star_temperature={pstar_temperature};" \
                           "planet_spec_filename={pplanet_spec_filename};" \
                           "planet_angles={pplanet_angles};" \
                           "planet_radii={pplanet_radii}".format(pfake_planet_flux_ratio=psfs_struct.fake_planet_flux_ratio,
                                                                  pangle_offset=psfs_struct.angle_offset,
                                                                  pspectrum=str(psfs_struct.spectrum),
                                                                  pstar_type=psfs_struct.star_type,
                                                                  pstar_temperature=psfs_struct.star_temperature,
                                                                  pplanet_spec_filename=psfs_struct.planet_spec_filename,
                                                                  pplanet_angles=str((90.+pa_list).tolist()),
                                                                  pplanet_radii=str(annuli_radii.tolist()))
            print(fakePlparams)
        else:
            # Define the PSFs variables as None so that klip is applied normally without fake planets injections.
            PSFs = None
            out_PSFs = None
            fakePlparams = None

        klipped_imgs, seg_basis_array, seg_index_array = klip_parallelized(dataset.input, dataset.centers, dataset.PAs, dataset.wvs,
                                         dataset.IWA, mode=mode, annuli=annuli, subsections=subsections,
                                         movement=movement, numbasis=numbasis, numthreads=numthreads, minrot=minrot,
                                         aligned_center=aligned_center, PSFs = PSFs,out_PSFs=out_PSFs, spectrum=spectrum, onesegment=onesegment,
                                         companion_rho = companion_rho, companion_theta = companion_theta, segment_dr = segment_dr, segment_dt = segment_dt)
        #JB's debug
        #if psfs_struct.calculate_PSFs:
        #    pyfits.writeto("/Users/jruffio/gpi/pyklip/outputs/solePSFs.fits", out_PSFs, clobber=True)


        if onesegment is True:
            #Save basis vectors, indicies to disk
            pyfits.writeto("test_seg_basis.fits", seg_basis_array, clobber=True)
            pyfits.writeto("test_seg_index.fits", seg_index_array, clobber=True)

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

        #align center to center of image if not specified
        if aligned_center is None:
            aligned_center = [int(dataset.input.shape[2]//2), int(dataset.input.shape[1]//2)]

        #parallelized rotate images
        print("Derotating Images...")
        rot_imgs = rotate_imgs(dataset.output, flattend_parangs, flattened_centers, numthreads=numthreads, flipx=True,
                               hdrs=dataset.wcs, new_center=aligned_center)

        #reconstruct datacubes, need to obtain wavelength dimension size
        num_wvs = np.size(np.unique(dataset.wvs)) # assuming all datacubes are taken in same band

        #give rot_imgs dimensions of (num KLmode cutoffs, num cubes, num wvs, y, x)
        rot_imgs = rot_imgs.reshape(oldshape[0], oldshape[1]/num_wvs, num_wvs, oldshape[2], oldshape[3])

        #save modified data and centers
        dataset.output = rot_imgs
        dataset.centers[:,0] = aligned_center[0]
        dataset.centers[:,1] = aligned_center[1]

        #valid output path and write iamges
        outputdirpath = os.path.realpath(outputdir)
        print("Writing Images to directory {0}".format(outputdirpath))

        if sat_spot_psf or psfs_struct.calculate_PSFs == 1:
            # Save the original PSF calculated from combining the sat spots
            dataset.savedata(outputdirpath + '/' + fileprefix+"-original_PSF_cube.fits", dataset.psfs,
                                      astr_hdr=dataset.wcs[0])
            # Calculate and save the rotationally invariant psf (ie smeared out/averaged).
            dataset.get_radial_psf(save = outputdirpath + '/' + fileprefix)

        # apply same transformation on out_PSFs than on dataset.output
        if psfs_struct.calculate_PSFs:
            out_PSFs = out_PSFs.reshape(oldshape[0]*oldshape[1], oldshape[2], oldshape[3])
            rot_out_PSFs = rotate_imgs(out_PSFs, flattend_parangs, flattened_centers, numthreads=numthreads, flipx=True,
                                   hdrs=dataset.wcs,disable_wcs_rotation = True)
            rot_out_PSFs = rot_out_PSFs.reshape(oldshape[0], oldshape[1]/num_wvs, num_wvs, oldshape[2], oldshape[3])
            KLmode_cube_PSFs = np.nanmean(rot_out_PSFs, axis=(1,2))
            dataset.savedata(outputdirpath + '/' + fileprefix + "-KLmodes-all-solePSFs.fits", KLmode_cube_PSFs,
                                      klipparams.format(numbasis=str(numbasis)), fakePlparams = fakePlparams,
                                      astr_hdr=dataset.wcs[0], center=dataset.centers[0])
            #pyfits.writeto("/Users/jruffio/gpi/pyklip/outputs/KLmode_cube_PSFs.fits", KLmode_cube_PSFs, clobber=True)

            #for each KL mode, collapse in time to examine spectra
            KLmode_spectral_cubes_PSFs = np.nanmean(rot_out_PSFs, axis=1)
            for KLcutoff, spectral_cube_PSFs in zip(numbasis, KLmode_spectral_cubes_PSFs):
                dataset.savedata(outputdirpath + '/' + fileprefix + "-KL{0}-speccube-solePSFs.fits".format(KLcutoff), spectral_cube_PSFs,
                                      klipparams.format(numbasis=str(numbasis)), fakePlparams = fakePlparams,
                                      astr_hdr=dataset.wcs[0], center=dataset.centers[0])

            #collapse in time and wavelength to examine KL modes
            KLmode_cube = np.nanmean(dataset.output, axis=(1,2))
            dataset.savedata(outputdirpath + '/' + fileprefix + "-KLmodes-all-PSFs.fits", KLmode_cube,
                                      klipparams.format(numbasis=str(numbasis)), fakePlparams = fakePlparams,
                                      astr_hdr=dataset.wcs[0], center=dataset.centers[0])

            #for each KL mode, collapse in time to examine spectra
            KLmode_spectral_cubes = np.nanmean(dataset.output, axis=1)
            for KLcutoff, spectral_cube in zip(numbasis, KLmode_spectral_cubes):
                dataset.savedata(outputdirpath + '/' + fileprefix + "-KL{0}-speccube-PSFs.fits".format(KLcutoff), spectral_cube,
                                      klipparams.format(numbasis=str(numbasis)), fakePlparams = fakePlparams,
                                      astr_hdr=dataset.wcs[0], center=dataset.centers[0])
        else:
            #collapse in time and wavelength to examine KL modes
            KLmode_cube = np.nanmean(dataset.output, axis=(1,2))
            dataset.savedata(outputdirpath + '/' + fileprefix + "-KLmodes-all.fits", KLmode_cube,
                             klipparams.format(numbasis=str(numbasis)))

            #for each KL mode, collapse in time to examine spectra
            KLmode_spectral_cubes = np.nanmean(dataset.output, axis=1)
            for KLcutoff, spectral_cube in zip(numbasis, KLmode_spectral_cubes):
                dataset.savedata(outputdirpath + '/' + fileprefix + "-KL{0}-speccube.fits".format(KLcutoff), spectral_cube,
                                      klipparams.format(numbasis=KLcutoff))
    elif mode == 'ADI':
        #ADI is not parallelized
        dataset.output = klip_parallelized(dataset.input, dataset.centers, dataset.PAs, dataset.wvs,
                                         dataset.IWA, mode=mode, annuli=annuli, subsections=subsections,
                                         movement=movement, numbasis=numbasis, numthreads=numthreads, minrot=minrot,
                                         aligned_center=aligned_center)
        if calibrate_flux == True:
            dataset.calibrate_output()

        #TODO: handling of only a single numbasis
        #derotate all the images
        #first we need to flatten so it's just a 3D array
        oldshape = dataset.output.shape
        dataset.output = dataset.output.reshape(oldshape[0]*oldshape[1], oldshape[2], oldshape[3])
        #we need to duplicate PAs and centers for the different KL mode cutoffs we supplied
        flattend_parangs = np.tile(dataset.PAs, oldshape[0])
        flattened_centers = np.tile(dataset.centers.reshape(oldshape[1]*2), oldshape[0]).reshape(oldshape[1]*oldshape[0],2)

        #align center to center of image if not specified
        if aligned_center is None:
            aligned_center = [int(dataset.input.shape[2]//2), int(dataset.input.shape[1]//2)]

        #parallelized rotate images
        print("Derotating Images...")
        rot_imgs = rotate_imgs(dataset.output, flattend_parangs, flattened_centers, numthreads=numthreads, flipx=True,
                               hdrs=dataset.wcs, new_center=aligned_center)

        #give rot_imgs dimensions of (num KLmode cutoffs, num cubes, num wvs, y, x)
        rot_imgs = rot_imgs.reshape(oldshape[0], oldshape[1], oldshape[2], oldshape[3])

        dataset.output = rot_imgs

        #valid output path and write iamges
        outputdirpath = os.path.realpath(outputdir)
        print("Writing Images to directory {0}".format(outputdirpath))

        #collapse in time and wavelength to examine KL modes
        KLmode_cube = np.nanmean(dataset.output, axis=(1))
        dataset.savedata(outputdirpath + '/' + fileprefix + "-KLmodes-all.fits", KLmode_cube,
                         klipparams.format(numbasis=str(numbasis)))

        num_wvs = np.size(np.unique(dataset.wvs)) # assuming all datacubes are taken in same band
        #if we actually have spectral cubes, let's save those too
        if num_wvs > 1:
            oldshape = dataset.output.shape
            wv_imgs = dataset.output.reshape(oldshape[0], oldshape[1]/num_wvs, num_wvs, oldshape[2], oldshape[3])
            KLmode_spectral_cubes = np.nanmean(wv_imgs, axis=1)
            for KLcutoff, spectral_cube in zip(numbasis, KLmode_spectral_cubes):
                dataset.savedata(outputdirpath + '/' + fileprefix + "-KL{0}-speccube.fits".format(KLcutoff), spectral_cube,
                                  klipparams.format(numbasis=KLcutoff))

    elif mode == 'SDI':
        dataset.output = klip_parallelized(dataset.input, dataset.centers, dataset.PAs, dataset.wvs,
                                         dataset.IWA, mode=mode, annuli=annuli, subsections=subsections,
                                         movement=movement, numbasis=numbasis, numthreads=numthreads, minrot=minrot,
                                         aligned_center=aligned_center, spectrum=spectrum)
        if calibrate_flux == True:
            dataset.calibrate_output()

        #TODO: handling of only a single numbasis
        #derotate all the images
        #first we need to flatten so it's just a 3D array
        oldshape = dataset.output.shape
        dataset.output = dataset.output.reshape(oldshape[0]*oldshape[1], oldshape[2], oldshape[3])
        #we need to duplicate PAs and centers for the different KL mode cutoffs we supplied
        flattend_parangs = np.tile(dataset.PAs, oldshape[0])
        flattened_centers = np.tile(dataset.centers.reshape(oldshape[1]*2), oldshape[0]).reshape(oldshape[1]*oldshape[0],2)

        #align center to center of image if not specified
        if aligned_center is None:
            aligned_center = [int(dataset.input.shape[2]//2), int(dataset.input.shape[1]//2)]

        #parallelized rotate images
        print("Derotating Images...")
        rot_imgs = rotate_imgs(dataset.output, flattend_parangs, flattened_centers, numthreads=numthreads, flipx=True,
                               hdrs=dataset.wcs, new_center=aligned_center)

        #give rot_imgs dimensions of (num KLmode cutoffs, num cubes, num wvs, y, x)
        rot_imgs = rot_imgs.reshape(oldshape[0], oldshape[1], oldshape[2], oldshape[3])

        dataset.output = rot_imgs

        #valid output path and write iamges
        outputdirpath = os.path.realpath(outputdir)
        print("Writing Images to directory {0}".format(outputdirpath))

        #collapse in time and wavelength to examine KL modes
        KLmode_cube = np.nanmean(dataset.output, axis=(1))
        dataset.savedata(outputdirpath + '/' + fileprefix + "-KLmodes-all.fits", KLmode_cube,
                         klipparams.format(numbasis=str(numbasis)))

        num_wvs = np.size(np.unique(dataset.wvs)) # assuming all datacubes are taken in same band
        #if we actually have spectral cubes, let's save those too
        if num_wvs > 1:
            oldshape = dataset.output.shape
            wv_imgs = dataset.output.reshape(oldshape[0], oldshape[1]/num_wvs, num_wvs, oldshape[2], oldshape[3])
            KLmode_spectral_cubes = np.nanmean(wv_imgs, axis=1)
            for KLcutoff, spectral_cube in zip(numbasis, KLmode_spectral_cubes):
                dataset.savedata(outputdirpath + '/' + fileprefix + "-KL{0}-speccube.fits".format(KLcutoff), spectral_cube,
                                  klipparams.format(numbasis=KLcutoff))

    else:
        print("Invalid mode. Either ADI, SDI, or ADI+SDI")
        return

    #pyfits.writeto("out1.fits", klipped_imgs[-1], clobber=True)


