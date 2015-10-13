#KLIP Forward Modelling
import pyklip.klip as klip
import numpy as np
import scipy.linalg as la
from scipy.stats import norm
import scipy.ndimage as ndimage

import ctypes
import itertools
import multiprocessing as mp
from pyklip.parallelized import _arraytonumpy

#import matplotlib.pyplot as plt

parallel = False


def klip_math(sci, refs, numbasis, covar_psfs=None, model_sci=None, models_ref=None, spec_included=False, spec_from_model=False):
    """
    linear algebra of KLIP with linear perturbation
    disks and point source

    Args:
        sci: array of length p containing the science data
        refs: N x p array of the N reference images that
                  characterizes the extended source with p pixels
        numbasis: number of KLIP basis vectors to use (can be an int or an array of ints of length b)
        covar_psfs: covariance matrix of reference images (for large N, useful). Normalized following numpy normalization in np.cov documentation
        # The following arguments must all be passed in, or none of them for klip_math to work
        models_ref: N x p array of the N models corresponding to reference images. Each model should be normalized to unity (no flux information)
        model_sci: array of size p corresponding to the PSF of the science frame
        Sel_wv: wv x N array of the the corresponding wavelength for each reference PSF
        input_spectrum: array of size wv with the assumed spectrum of the model


    Returns:
        sub_img_rows_selected: array of shape (p,b) that is the PSF subtracted data for each of the b KLIP basis
                               cutoffs. If numbasis was an int, then sub_img_row_selected is just an array of length p
        KL_basis: array of KL basis (shape of [numbasis, p])
        If models_ref is passed in (not None):
            delta_KL_nospec: array of shape (b, wv, p) that is the almost perturbed KL modes just missing spectral info
        Otherwise:
            evals: array of eigenvalues (size of max number of KL basis requested aka nummaxKL)
            evecs: array of corresponding eigenvectors (shape of [p, nummaxKL])


    """
    # remove means and nans
    sci_mean_sub = sci - np.nanmean(sci)
    sci_nanpix = np.where(np.isnan(sci_mean_sub))
    sci_mean_sub[sci_nanpix] = 0

    refs_mean_sub = refs - np.nanmean(refs, axis=1)[:, None]
    refs_mean_sub[np.where(np.isnan(refs_mean_sub))] = 0

    # calculate the covariance matrix for the reference PSFs
    # note that numpy.cov normalizes by p-1 to get the NxN covariance matrix
    # we have to correct for that since that's not part of the equation in the KLIP paper
    if covar_psfs is None:
        # call np.cov to make the covariance matrix
        covar_psfs = np.cov(refs_mean_sub)
    # fix normalization of covariance matrix
    covar_psfs *= (np.size(sci)-1)

    # calculate the total number of KL basis we need based on the number of reference PSFs and number requested
    tot_basis = covar_psfs.shape[0]
    numbasis = np.clip(numbasis - 1, 0, tot_basis-1)
    max_basis = np.max(numbasis) + 1

    # calculate eigenvectors/values of covariance matrix
    evals, evecs = la.eigh(covar_psfs, eigvals = (tot_basis-max_basis, tot_basis-1))
    evals = np.copy(evals[::-1])
    evecs = np.copy(evecs[:,::-1])

    # project on reference PSFs to generate KL modes
    KL_basis = np.dot(refs_mean_sub.T,evecs)
    KL_basis = KL_basis * (1. / np.sqrt(evals))[None,:]
    KL_basis = KL_basis.T # flip dimensions to be consistent with Laurent's paper

    # prepare science frame for KLIP subtraction
    sci_mean_sub_rows = np.tile(sci_mean_sub, (max_basis,1))
    sci_rows_selected = np.tile(sci_mean_sub, (np.size(numbasis),1))

    sci_nanpix = np.where(np.isnan(sci_mean_sub_rows))
    sci_mean_sub_rows[sci_nanpix] = 0
    sci_nanpix = np.where(np.isnan(sci_rows_selected))
    sci_rows_selected[sci_nanpix] = 0

    # run KLIP on this sector and subtract the stellar PSF
    inner_products = np.dot(sci_mean_sub_rows, KL_basis.T)
    lower_tri = np.tril(np.ones([max_basis,max_basis]))
    inner_products = inner_products * lower_tri
    klip = np.dot(inner_products[numbasis,:], KL_basis)
    sub_img_rows_selected = sci_rows_selected - klip
    sub_img_rows_selected[sci_nanpix] = np.nan

    if models_ref is not None:


        if spec_included:
            delta_KL = pertrub_specIncluded(evals, evecs, KL_basis, refs_mean_sub, models_ref)
            return sub_img_rows_selected.transpose(), KL_basis,  delta_KL
        elif spec_from_model:
            delta_KL_nospec = pertrub_nospec_modelsBased(evals, evecs, KL_basis, refs_mean_sub, models_ref)
            return sub_img_rows_selected.transpose(), KL_basis,  delta_KL_nospec
        else:
            delta_KL_nospec = pertrub_nospec(evals, evecs, KL_basis, refs_mean_sub, models_ref)
            return sub_img_rows_selected.transpose(), KL_basis,  delta_KL_nospec


    else:

        return sub_img_rows_selected.transpose(), KL_basis, evals, evecs

def pertrub_specIncluded(evals, evecs, original_KL, refs, models_ref):
    """
    Perturb the KL modes using a model of the PSF but with the spectrum included in the model. Quicker than the others

    Args:
        evals: array of eigenvalues of the reference PSF covariance matrix (array of size numbasis)
        evecs: corresponding eigenvectors (array of size [p, numbasis])
        orignal_KL: unpertrubed KL modes (array of size [numbasis, p])
        refs: N x p array of the N reference images that
                  characterizes the extended source with p pixels
        models_ref: N x p array of the N models corresponding to reference images.
                    Each model should contain spectral informatoin
        model_sci: array of size p corresponding to the PSF of the science frame

    Returns:
        delta_KL_nospec: perturbed KL modes. Shape is (numKL, wv, pix)
    """

    max_basis = original_KL.shape[0]
    N_ref = refs.shape[0]
    N_pix = original_KL.shape[1]

    refs_mean_sub = refs - np.nanmean(refs, axis=1)[:, None]
    refs_mean_sub[np.where(np.isnan(refs_mean_sub))] = 0

    models_mean_sub = models_ref # - np.nanmean(models_ref, axis=1)[:,None] should this be the case?
    models_mean_sub[np.where(np.isnan(models_mean_sub))] = 0

    #print(evals.shape,evecs.shape,original_KL.shape,refs.shape,models_ref.shape)

    evals_tiled = np.tile(evals,(max_basis,1))
    np.fill_diagonal(evals_tiled,np.nan)
    evals_sqrt = np.sqrt(evals)
    evalse_inv_sqrt = 1./evals_sqrt
    evals_ratio = (evalse_inv_sqrt[:,None]).dot(evals_sqrt[None,:])
    beta_tmp = 1./(evals_tiled.transpose()- evals_tiled)
    beta_tmp[np.diag_indices(N_ref)] = -0.5/evals
    beta = evals_ratio*beta_tmp

    C =  models_mean_sub.dot(refs_mean_sub.transpose())+refs_mean_sub.dot(models_mean_sub.transpose())
    alpha = (evecs.transpose()).dot(C).dot(evecs)

    delta_KL = (beta*alpha).dot(original_KL)+(evalse_inv_sqrt[:,None]*evecs.transpose()).dot(models_mean_sub)


    return delta_KL


def pertrub_nospec_modelsBased(evals, evecs, original_KL, refs, models_ref_list):
    """

    :param evals:
    :param evecs:
    :param original_KL:
    :param refs:
    :param models_ref:
    :return:
    """

    max_basis = original_KL.shape[0]
    N_wv,N_ref,N_pix = models_ref_list.shape

    refs_mean_sub = refs - np.nanmean(refs, axis=1)[:, None]
    refs_mean_sub[np.where(np.isnan(refs_mean_sub))] = 0

    # perturbed KL modes
    delta_KL_nospec = np.zeros([max_basis, N_wv, N_pix]) # (numKL,N_ref,N_pix)

    for k,models_ref in enumerate(models_ref_list):
        models_mean_sub = models_ref # - np.nanmean(models_ref, axis=1)[:,None] should this be the case?
        models_mean_sub[np.where(np.isnan(models_mean_sub))] = 0

        evals_tiled = np.tile(evals,(N_ref,1))
        np.fill_diagonal(evals_tiled,np.nan)
        evals_sqrt = np.sqrt(evals)
        evalse_inv_sqrt = 1./evals_sqrt
        evals_ratio = (evalse_inv_sqrt[:,None]).dot(evals_sqrt[None,:])
        beta_tmp = 1./(evals_tiled.transpose()- evals_tiled)
        beta_tmp[np.diag_indices(N_ref)] = -0.5/evals
        beta = evals_ratio*beta_tmp

        C =  models_mean_sub.dot(refs.transpose())+refs.dot(models_mean_sub.transpose())
        alpha = (evecs.transpose()).dot(C).dot(evecs)

        delta_KL = (beta*alpha).dot(original_KL)+(evalse_inv_sqrt[:,None]*evecs.transpose()).dot(models_mean_sub)
        delta_KL_nospec[:,k,:] = delta_KL[:,:]


    return delta_KL_nospec

def pertrub_nospec(evals, evecs, original_KL, refs, models_ref):
    """
    Perturb the KL modes using a model of the PSF but with no assumption on the spectrum. Useful for planets

    Args:
        evals: array of eigenvalues of the reference PSF covariance matrix (array of size numbasis)
        evecs: corresponding eigenvectors (array of size [p, numbasis])
        orignal_KL: unpertrubed KL modes (array of size [numbasis, p])
        Sel_wv: wv x N array of the the corresponding wavelength for each reference PSF
        refs: N x p array of the N reference images that
                  characterizes the extended source with p pixels
        models_ref: N x p array of the N models corresponding to reference images. Each model should be normalized to unity (no flux information)
        model_sci: array of size p corresponding to the PSF of the science frame

    Returns:
        delta_KL_nospec: perturbed KL modes but without the spectral info. delta_KL = spectrum x delta_Kl_nospec.
                         Shape is (numKL, wv, pix)
    """

    max_basis = original_KL.shape[0]
    N_ref = refs.shape[0]
    N_pix = original_KL.shape[1]

    refs_mean_sub = refs - np.nanmean(refs, axis=1)[:, None]
    refs_mean_sub[np.where(np.isnan(refs_mean_sub))] = 0

    models_mean_sub = models_ref # - np.nanmean(models_ref, axis=1)[:,None] should this be the case?
    models_mean_sub[np.where(np.isnan(models_mean_sub))] = 0

    # science PSF models
    #model_sci_mean_sub = model_sci # should be subtracting off the mean?
    #model_nanpix = np.where(np.isnan(model_sci_mean_sub))
    #model_sci_mean_sub[model_nanpix] = 0

    # perturbed KL modes
    delta_KL_nospec = np.zeros([max_basis, N_ref, N_pix]) # (numKL,N_ref,N_pix)

    #plt.figure(1)
    #plt.plot(evals)
    #ax = plt.gca()
    #ax.set_yscale('log')
    #plt.show()

    models_mean_sub_X_refs_mean_sub_T = models_mean_sub.dot(refs_mean_sub.transpose())
    # calculate perturbed KL modes. TODO: make this NOT a freaking for loop
    for k in range(max_basis):
        Zk = np.reshape(original_KL[k,:],(1,original_KL[k,:].size))
        Vk = (evecs[:,k])[:,None]

        DeltaZk_noSpec = -(1/np.sqrt(evals[k]))*(Vk*models_mean_sub_X_refs_mean_sub_T).dot(Vk).dot(Zk)+Vk*models_mean_sub
        # TODO: Make this NOT a for loop
        diagVk_X_models_mean_sub_X_refs_mean_sub_T = Vk*models_mean_sub_X_refs_mean_sub_T
        models_mean_sub_X_refs_mean_sub_T_X_Vk = models_mean_sub_X_refs_mean_sub_T.dot(Vk)
        for j in range(k):
            Zj = original_KL[j, :][None,:]
            Vj = evecs[:, j][:,None]
            DeltaZk_noSpec += np.sqrt(evals[j])/(evals[k]-evals[j])*(diagVk_X_models_mean_sub_X_refs_mean_sub_T.dot(Vj) + Vj*models_mean_sub_X_refs_mean_sub_T_X_Vk).dot(Zj)
        for j in range(k+1, max_basis):
            Zj = original_KL[j, :][None,:]
            Vj = evecs[:, j][:,None]
            DeltaZk_noSpec += np.sqrt(evals[j])/(evals[k]-evals[j])*(diagVk_X_models_mean_sub_X_refs_mean_sub_T.dot(Vj) + Vj*models_mean_sub_X_refs_mean_sub_T_X_Vk).dot(Zj)

        delta_KL_nospec[k] = DeltaZk_noSpec/np.sqrt(evals[k])
    '''
    if 0:  # backup slow version
        # calculate perturbed KL modes. TODO: make this NOT a freaking for loop
        for k in range(max_basis):
            # Define Z_{k}. Variable name: kl_basis_noPl[k,:], Shape: (1, N_pix)
            Zk = np.reshape(original_KL[k,:],(1,original_KL[k,:].size))
            # Define V_{k}. Variable name: eigvec_noPl[:,k], Shape: (1, N_ref)
            Vk = np.reshape(evecs[:,k],(evecs[:,k].size,1))
            # Define bolt{V}_{k}. Variable name: diagV_k = np.diag(eigvec_noPl[:,k]), Shape: (N_ref,N_ref)
            diagVk = np.diag(evecs[:,k])


            DeltaZk_noSpec = -(1/np.sqrt(evals[k]))*diagVk.dot(models_mean_sub).dot(refs_mean_sub.transpose()).dot(Vk).dot(Zk)+diagVk.dot(models_mean_sub)
            # TODO: Make this NOT a for loop
            for j in range(k):
                Zj = np.reshape(original_KL[j, :], (1, original_KL[j, :].size))
                Vj = np.reshape(evecs[:, j], (evecs[:, j].size,1))
                diagVj = np.diag(evecs[:, j])
                DeltaZk_noSpec += np.sqrt(evals[j])/(evals[k]-evals[j])*(diagVk.dot(models_mean_sub).dot(refs_mean_sub.transpose()).dot(Vj) + diagVj.dot(models_mean_sub).dot(refs_mean_sub.transpose()).dot(Vk)).dot(Zj)
            for j in range(k+1, max_basis):
                Zj = np.reshape(original_KL[j, :], (1, original_KL[j, :].size))
                Vj = np.reshape(evecs[:, j], (evecs[:, j].size,1))
                diagVj = np.diag(evecs[:, j])
                DeltaZk_noSpec += np.sqrt(evals[j])/(evals[k]-evals[j])*(diagVk.dot(models_mean_sub).dot(refs_mean_sub.transpose()).dot(Vj) + diagVj.dot(models_mean_sub).dot(refs_mean_sub.transpose()).dot(Vk)).dot(Zj)

            delta_KL_nospec[k] = (Sel_wv/np.sqrt(evals[k])).dot(DeltaZk_noSpec)
    '''

    return delta_KL_nospec


def calculate_fm(delta_KL_nospec, original_KL, numbasis, sci, model_sci, inputflux = None):
    """
    Calculate what the PSF looks up post-KLIP using knowledge of the input PSF, assumed spectrum of the science target,
    and the partially calculated KL modes (\Delta Z_k^\lambda in Laurent's paper). If inputflux is None,
    the spectral dependence has already been folded into delta_KL_nospec (treat it as delta_KL)

    Args:
        delta_KL_nospec: perturbed KL modes but without the spectral info. delta_KL = spectrum x delta_Kl_nospec.
                         Shape is (numKL, wv, pix). If inputflux is None, delta_KL_nospec = delta_KL
        orignal_KL: unpertrubed KL modes (array of size [numbasis, numpix])
        numbasis: array of KL mode cutoffs
        sci: array of size p representing the science data
        model_sci: array of size p corresponding to the PSF of the science frame
        input_spectrum: array of size wv with the assumed spectrum of the model

    Returns:
        fm_psf: array of shape (b,p) showing the forward modelled PSF
        klipped_oversub: array of shape (b, p) showing the effect of oversubtraction as a function of KL modes
        klipped_selfsub: array of shape (b, p) showing the effect of selfsubtraction as a function of KL modes
        Note: psf_FM = model_sci - klipped_oversub - klipped_selfsub to get the FM psf as a function of K Lmodes
              (shape of b,p)
    """
    max_basis = original_KL.shape[0]
    numbasis_index = np.clip(numbasis - 1, 0, max_basis-1)

    # remove means and nans from science image
    sci_mean_sub = sci - np.nanmean(sci)
    sci_nanpix = np.where(np.isnan(sci_mean_sub))
    sci_mean_sub[sci_nanpix] = 0
    sci_mean_sub_rows = np.tile(sci_mean_sub, (max_basis,1))
    sci_rows_selected = np.tile(sci_mean_sub, (np.size(numbasis),1))


    # science PSF models, ready for FM
    # /!\ JB: If subtracting the mean. It should be done here. not in klip_math since we don't use model_sci there.
    model_sci_mean_sub = model_sci # should be subtracting off the mean?
    model_nanpix = np.where(np.isnan(model_sci_mean_sub))
    model_sci_mean_sub[model_nanpix] = 0
    model_sci_mean_sub_rows = np.tile(model_sci_mean_sub, (max_basis,1))
    # model_rows_selected = np.tile(sci_mean_sub, (np.size(numbasis),1)) # don't need this because of python behavior where I don't need to duplicate rows


    # calculate perturbed KL modes based on spectrum
    if inputflux is not None:
        delta_KL = np.dot(inputflux, delta_KL_nospec) # this will take the last dimension of input_spectrum (wv) and sum over the second to last dimension of delta_KL_nospec (wv)
    else:
        delta_KL = delta_KL_nospec

    # Forward model the PSF
    # 3 terms: 1 for oversubtracton (planet attenauted by speckle KL modes),
    # and 2 terms for self subtraction (planet signal leaks in KL modes which get projected onto speckles)
    oversubtraction_inner_products = np.dot(model_sci_mean_sub_rows, original_KL.T)
    selfsubtraction_1_inner_products = np.dot(sci_mean_sub_rows, delta_KL.T)
    selfsubtraction_2_inner_products = np.dot(sci_mean_sub_rows, original_KL.T)

    lower_tri = np.tril(np.ones([max_basis,max_basis]))
    oversubtraction_inner_products = oversubtraction_inner_products * lower_tri
    selfsubtraction_1_inner_products = selfsubtraction_1_inner_products * lower_tri
    selfsubtraction_2_inner_products = selfsubtraction_2_inner_products * lower_tri

    klipped_oversub = np.dot(np.take(oversubtraction_inner_products, numbasis_index, axis=0), original_KL)
    klipped_selfsub = np.dot(np.take(selfsubtraction_1_inner_products, numbasis_index, axis=0), original_KL) + \
                      np.dot(np.take(selfsubtraction_2_inner_products,numbasis_index, axis=0), delta_KL)

    return model_sci - klipped_oversub - klipped_selfsub, klipped_oversub, klipped_selfsub

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
                output_imgs_numstacked,
                pa_imgs, wvs_imgs, centers_imgs, interm_imgs, interm_imgs_shape, fmout_imgs, fmout_imgs_shape):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array) - output_imgs does not need to be mp.Array and can be anything

    Args:
        original_imgs: original images from files to read and align&scale.
        original_imgs_shape: (N,y,x), N = number of frames = num files * num wavelengths
        aligned: aligned and scaled images for processing.
        aligned_imgs_shape: (wv, N, y, x), wv = number of wavelengths per datacube
        output_imgs: PSF subtraceted images
        output_imgs_shape: (N, y, x, b)
        output_imgs_numstacked: number of images stacked together for each pixel due to geometry overlap. Shape of
                                (N, y x). Output without the b dimension
        pa_imgs, wvs_imgs: arrays of size N with the PA and wavelength
        centers_img: array of shape (N,2) with [x,y] image center for image frame
        interm_imgs: intermediate data product shape - what is saved on a sector to sector basis before combining to
                     form the output of that sector. The first dimention should be N (i.e. same thing for each science
                     image)
        interm_imgs_shape: shape of interm_imgs. The first dimention should be N.
        fmout_imgs: array for output of forward modelling. What's stored in here depends on the class
        fmout_imgs_shape: shape of fmout
    """
    global original, original_shape, aligned, aligned_shape, outputs, outputs_shape, outputs_numstacked, img_pa, \
        img_wv, img_center, interm, interm_shape, fmout, fmout_shape
    # original images from files to read and align&scale. Shape of (N,y,x)
    original = original_imgs
    original_shape = original_imgs_shape
    # aligned and scaled images for processing. Shape of (wv, N, y, x)
    aligned = aligned_imgs
    aligned_shape = aligned_imgs_shape
    # output images after KLIP processing
    outputs = output_imgs
    outputs_shape = output_imgs_shape
    outputs_numstacked = output_imgs_numstacked
    # parameters for each image (PA, wavelegnth, image center)
    img_pa = pa_imgs
    img_wv = wvs_imgs
    img_center = centers_imgs

    #intermediate and auxilliary data to store
    interm = interm_imgs
    interm_shape = interm_imgs_shape
    fmout = fmout_imgs
    fmout_shape = fmout_imgs_shape


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
    numframes_todo = int(np.round(len(combos)/mp.cpu_count()))
    leftovers = len(combos) % mp.cpu_count()
    # the last thread needs to finish all of them
    if thread_index == mp.cpu_count() - 1:
        combos_todo = combos[leftovers + thread_index*numframes_todo:]
        print(len(combos), len(combos_todo), leftovers + thread_index*numframes_todo)
    else:
        if thread_index < leftovers:
            leftovers_completed = thread_index
            plusone = 1
        else:
            leftovers_completed = leftovers
            plusone = 0

        combos_todo = combos[leftovers_completed + thread_index*numframes_todo:(thread_index+1)*numframes_todo + leftovers_completed + plusone]
        print(len(combos), len(combos_todo), leftovers_completed + thread_index*numframes_todo, (thread_index+1)*numframes_todo + leftovers_completed + plusone)

    #print(len(combos), len(combos_todo), leftovers, thread_index)

    for img_index, ref_wv_index in combos_todo:
        aligned_imgs[ref_wv_index,img_index,:,:] = klip.align_and_scale(original_imgs[img_index], aligned_center,
                                                centers_imgs[img_index], unique_wvs[ref_wv_index]/wvs_imgs[img_index])
    return


def _get_section_indicies(input_shape, img_center, radstart, radend, phistart, phiend, padding, parang):
    """
    Gets the pixels (via numpy.where) that correspond to this section

    Args:
        input_shape: shape of the image [ysize, xsize] [pixels]
        img_center: [x,y] image center [pxiels]
        radstart: minimum radial distance of sector [pixels]
        radend: maximum radial distance of sector [pixels]
        phistart: minimum azimuthal coordinate of sector [radians]
        phiend: maximum azimuthal coordinate of sector [radians]
        padding: number of pixels to pad to the sector [pixels]
        parang: how much to rotate phi due to field rotation [IN DEGREES]

    Returns:
        sector_ind: the pixel coordinates that corespond to this sector
    """
    # create a coordinate system.
    x, y = np.meshgrid(np.arange(input_shape[1] * 1.0), np.arange(input_shape[0] * 1.0))
    x.shape = (x.shape[0] * x.shape[1]) # Flatten
    y.shape = (y.shape[0] * y.shape[1])
    r = np.sqrt((x - img_center[0])**2 + (y - img_center[1])**2)
    phi = np.arctan2(y - img_center[1], x - img_center[0])

    # incorporate padding
    radstart -= padding
    radend += padding
    phistart = (phistart - padding/np.mean([radstart, radend])) % (2 * np.pi)
    phiend = (phiend + padding/np.mean([radstart, radend])) % (2 * np.pi)

    # grab the pixel location of the section we are going to anaylze
    phi_rotate = ((phi + np.radians(parang)) % (2.0 * np.pi))
    # normal case where there's no 2 pi wrap
    if phistart < phiend:
        section_ind = np.where((r >= radstart) & (r < radend) & (phi_rotate >= phistart) & (phi_rotate < phiend))
    # 2 pi wrap case
    else:
        section_ind = np.where((r >= radstart) & (r < radend) & ((phi_rotate >= phistart) | (phi_rotate < phiend)))

    return section_ind



def _save_rotated_section(input_shape, sector, sector_ind, output_img, output_img_numstacked, angle, radstart, radend, phistart, phiend, padding, img_center, flipx=True,
                         new_center=None):
    """
    Rotate and save sector in output image at desired ranges

    Args:
        input_shape: shape of input_image
        sector: data in the sector to save to output_img
        sector_ind: index into input img (corresponding to input_shape) for the original sector
        output_img: the array to save the data to
        output_img_numstacked: array to increment region where we saved output to to bookkeep stacking. None for
                               skipping bookkeeping
        angle: angle that the sector needs to rotate (I forget the convention right now)

        The next 6 parameters define the sector geometry in input image coordinates
        radstart: radius from img_center of start of sector
        radend: radius from img_center of end of sector
        phistart: azimuthal start of sector
        phiend: azimuthal end of sector
        padding: amount of padding around each sector
        img_center: center of image in input image coordinate

        flipx: if true, flip the x coordinate to switch coordinate handiness
        new_center: if not none, center of output_img. If none, center stays the same
    """
    # convert angle to radians
    angle_rad = np.radians(angle)

    #wrap phi
    phistart %= 2 * np.pi
    phiend %= 2 * np.pi

    #incorporate padding
    radstart_padded = radstart - padding
    radend_padded = radend + padding
    phistart_padded = (phistart - padding/np.mean([radstart, radend])) % (2 * np.pi)
    phiend_padded = (phiend + padding/np.mean([radstart, radend])) % (2 * np.pi)

    # create the coordinate system of the image to manipulate for the transform
    dims = input_shape
    x, y = np.meshgrid(np.arange(dims[1], dtype=np.float32), np.arange(dims[0], dtype=np.float32))

    # if necessary, move coordinates to new center
    if new_center is not None:
        dx = new_center[0] - img_center[0]
        dy = new_center[1] - img_center[1]
        x -= dx
        y -= dy

    # flip x if needed to get East left of North
    if flipx is True:
        x = img_center[0] - (x - img_center[0])

    # do rotation. CW rotation formula to get a CCW of the image
    xp = (x-img_center[0])*np.cos(angle_rad) + (y-img_center[1])*np.sin(angle_rad) + img_center[0]
    yp = -(x-img_center[0])*np.sin(angle_rad) + (y-img_center[1])*np.cos(angle_rad) + img_center[1]

    if new_center is None:
        new_center = img_center

    rp = np.sqrt((xp - new_center[0])**2 + (yp - new_center[1])**2)
    phip = (np.arctan2(yp-new_center[1], xp-new_center[0]) + angle_rad) % (2 * np.pi)

    # grab sectors based on whether the phi coordinate wraps
    # padded sector
    # check to see if with padding, the phi coordinate wraps
    if phiend_padded >=  phistart_padded:
        # doesn't wrap
        in_padded_sector = ((rp >= radstart_padded) & (rp < radend_padded) &
                               (phip >= phistart_padded) & (phip < phiend_padded))
    else:
        # wraps
        in_padded_sector = ((rp >= radstart_padded) & (rp < radend_padded) &
                                            ((phip >= phistart_padded) | (phip < phiend_padded)))
    rot_sector_pix = np.where(in_padded_sector)

    # only padding
    # check to see if without padding, the phi coordinate wraps
    if phiend >=  phistart:
        # no wrap
        in_only_padding = np.where(((rp < radstart) | (rp >= radend) | (phip < phistart) | (phip >= phiend))
                                   & in_padded_sector)
    else:
        # wrap
        in_only_padding = np.where(((rp < radstart) | (rp >= radend) | ((phip < phistart) & (phip > phiend_padded))
                                    | (phip >= phiend & (phip < phistart_padded))) & in_padded_sector)
    rot_sector_pix_onlypadding = np.where(in_only_padding)

    blank_input = np.zeros(dims[1] * dims[0])
    blank_input[sector_ind] = sector
    blank_input.shape = [dims[0], dims[1]]

    # resample image based on new coordinates
    # scipy uses y,x convention when meshgrid uses x,y
    # stupid scipy functions can't work with masked arrays (NANs)
    # and trying to use interp2d with sparse arrays is way to slow
    # hack my way out of this by picking a really small value for NANs and try to detect them after the interpolation
    # then redo the transformation setting NaN to zero to reduce interpolation effects, but using the mask we derived
    minval = np.min([np.nanmin(blank_input), 0.0])
    nanpix = np.where(np.isnan(blank_input))
    medval = np.median(blank_input[np.where(~np.isnan(blank_input))])
    input_copy = np.copy(blank_input)
    input_copy[nanpix] = minval * 5.0
    rot_sector_mask = ndimage.map_coordinates(input_copy, [yp[rot_sector_pix], xp[rot_sector_pix]], cval=minval * 5.0)
    input_copy[nanpix] = medval
    rot_sector = ndimage.map_coordinates(input_copy, [yp[rot_sector_pix], xp[rot_sector_pix]], cval=np.nan)
    rot_sector[np.where(rot_sector_mask < minval)] = np.nan

    # save output sector. We need to reshape the array into 2d arrays to save it
    output_img.shape = [outputs_shape[1], outputs_shape[2]]
    output_img[rot_sector_pix] = np.nansum([output_img[rot_sector_pix], rot_sector], axis=0)
    output_img.shape = [outputs_shape[1] * outputs_shape[2]]

    # Increment the numstack counter if it is not None
    if output_img_numstacked is not None:
        output_img_numstacked.shape = [outputs_shape[1], outputs_shape[2]]
        output_img_numstacked[rot_sector_pix] += 1
        output_img_numstacked.shape = [outputs_shape[1] *  outputs_shape[2]]


def klip_parallelized(imgs, centers, parangs, wvs, IWA, fm_class, OWA=None, mode='ADI+SDI', annuli=5, subsections=4,
                      movement=3, numbasis=None, aligned_center=None, numthreads=None, minrot=0, maxrot=360,
                      spectrum=None, padding=3,
                      include_spec_in_model = False,
                      spec_from_model = False):
    #TODO MAKE THIS PAS FOR REALZZZ
    """
    multithreaded KLIP PSF Subtraction

    Args:
        imgs: array of 2D images for ADI. Shape of array (N,y,x)
        centers: N by 2 array of (x,y) coordinates of image centers
        parangs: N length array detailing parallactic angle of each image
        wvs: N length array of the wavelengths
        IWA: inner working angle (in pixels)
        fm_class: class that implements the the forward modelling functionality
        OWA: if defined, the outer working angle for pyklip. Otherwise, it will pick it as the cloest distance to a
            nan in the first frame
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
        padding: for each sector, how many extra pixels of padding should we have around the sides.


    Returns:
        sub_imgs: array of [array of 2D images (PSF subtracted)] using different number of KL basis vectors as
                    specified by numbasis. Shape of (b,N,y,x).
        fmout_np: output of forward modelling.
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


    dims = imgs.shape
    if isinstance(annuli, int):
        # use first image to figure out how to divide the annuli
        # TODO: what to do with OWA
        # need to make the next 10 lines or so much smarter

        x, y = np.meshgrid(np.arange(dims[2] * 1.0), np.arange(dims[1] * 1.0))
        nanpix = np.where(np.isnan(imgs[0]))
        if OWA is None:
            OWA = np.sqrt(np.min((x[nanpix] - centers[0][0]) ** 2 + (y[nanpix] - centers[0][1]) ** 2))
        dr = float(OWA - IWA) / (annuli)

        # calculate the annuli
        rad_bounds = [(dr * rad + IWA, dr * (rad + 1) + IWA) for rad in range(annuli)]
        # last annulus should mostly emcompass everything
        # rad_bounds[annuli - 1] = (rad_bounds[annuli - 1][0], imgs[0].shape[0])
    else:
        rad_bounds = annuli

    if isinstance(subsections, int):
        # divide annuli into subsections
        dphi = 2 * np.pi / subsections
        phi_bounds = [[dphi * phi_i, dphi * (phi_i + 1)] for phi_i in range(subsections)]
        phi_bounds[-1][1] = 2 * np.pi
    else:
        phi_bounds = subsections


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

    # make output array which also has an extra dimension for the number of KL modes to use
    output_imgs = mp.Array(ctypes.c_double, np.size(imgs)*np.size(numbasis))
    output_imgs_np = _arraytonumpy(output_imgs)
    output_imgs_np[:] = np.nan
    output_imgs_shape = imgs.shape + numbasis.shape
    # make an helper array to count how many frames overlap at each pixel
    output_imgs_numstacked = mp.Array(ctypes.c_int, np.size(imgs))

    # Create Custom Shared Memory array fmout to save output of forward modelling
    fmout_data, fmout_shape = fm_class.alloc_fmout(output_imgs_shape)


    # align and scale the images for each image. Use map to do this asynchronously]
    tpool = mp.Pool(processes=numthreads, initializer=_tpool_init,
                   initargs=(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs,
                             output_imgs_shape, output_imgs_numstacked, pa_imgs, wvs_imgs, centers_imgs, None, None,
                             fmout_data, fmout_shape), maxtasksperchild=50)

    # # SINGLE THREAD DEBUG PURPOSES ONLY
    if not parallel:
        _tpool_init(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs,
                                 output_imgs_shape, output_imgs_numstacked, pa_imgs, wvs_imgs, centers_imgs, None, None,
                                 fmout_data, fmout_shape)


    print("Begin align and scale images for each wavelength")
    aligned_outputs = []
    for threadnum in range(numthreads):
        #multitask this
        aligned_outputs += [tpool.apply_async(_align_and_scale_subset, args=(threadnum, aligned_center))]

        #save it to shared memory
    for aligned_output in aligned_outputs:
            aligned_output.wait()

    print("Align and scale finished")

    # list to store each threadpool task
    tpool_outputs = []
    sector_job_queued = np.zeros(tot_sectors) # count for jobs in the tpool queue for each sector

    # as each is finishing, queue up the aligned data to be processed with KLIP
    for sector_index, ((radstart, radend),(phistart,phiend)) in enumerate(itertools.product(rad_bounds, phi_bounds)):
        print("Starting KLIP for sector {0}".format(sector_index))
        # calculate sector size
        section_ind = _get_section_indicies(original_imgs_shape[1:], aligned_center, radstart, radend, phistart, phiend,
                                            padding, 0)

        sector_size = np.size(section_ind) #+ 2 * (radend- radstart) # some sectors are bigger than others due to boundary
        interm_data, interm_shape = fm_class.alloc_interm(sector_size, original_imgs_shape[0])

        for wv_index, wv_value in enumerate(unique_wvs):

            # pick out the science images that need PSF subtraction for this wavelength
            scidata_indicies = np.where(wvs == wv_value)[0]

            # perform KLIP asynchronously for each group of files of a specific wavelength and section of the image
            sector_job_queued[sector_index] += scidata_indicies.shape[0]
            if parallel:
                tpool_outputs += [tpool.apply_async(_klip_section_multifile_perfile,
                                                    args=(file_index, sector_index, radstart, radend, phistart, phiend,
                                                          parang, wv_value, wv_index, (radstart + radend) / 2., padding,
                                                          numbasis,
                                                          movement, aligned_center, minrot, maxrot, mode, spectrum,
                                                          fm_class,
                                                          include_spec_in_model,
                                                          spec_from_model))
                                    for file_index,parang in zip(scidata_indicies, pa_imgs_np[scidata_indicies])]

            # # SINGLE THREAD DEBUG PURPOSES ONLY
            if not parallel:
                tpool_outputs += [_klip_section_multifile_perfile(file_index, sector_index, radstart, radend, phistart, phiend,
                                                                  parang, wv_value, wv_index, (radstart + radend) / 2., padding,
                                                                  numbasis,
                                                                  movement, aligned_center, minrot, maxrot, mode, spectrum,
                                                                  fm_class,
                                                                  include_spec_in_model,
                                                                  spec_from_model)
                                    for file_index,parang in zip(scidata_indicies, pa_imgs_np[scidata_indicies])]

        # Run post processing on this sector here
        # Can be multithreaded code using the threadpool defined above
        # Check tpool job outputs. It there is stuff, go do things with it
        if parallel:
            while len(tpool_outputs) > 0:
                tpool_outputs.pop(0).wait()

            # if this is the last job finished for this sector,
            # do something here?

        # run custom function to handle end of sector post-processing analysis




    #close to pool now and make sure there's no processes still running (there shouldn't be or else that would be bad)
    print("Closing threadpool")
    tpool.close()
    tpool.join()

    # finished!
    # Let's take the mean based on number of images stacked at a location
    sub_imgs = _arraytonumpy(output_imgs, output_imgs_shape)
    sub_imgs_numstacked = _arraytonumpy(output_imgs_numstacked, original_imgs_shape, dtype=ctypes.c_int)
    sub_imgs = sub_imgs / sub_imgs_numstacked[:,:,:,None]

    # Let's reshape the output images
    # move number of KLIP modes as leading axis (i.e. move from shape (N,y,x,b) to (b,N,y,x)
    sub_imgs = np.rollaxis(sub_imgs.reshape((dims[0], dims[1], dims[2], numbasis.shape[0])), 3)

    #restore bad pixels
    sub_imgs[:, allnans[0], allnans[1], allnans[2]] = np.nan

    # put any finishing touches on the FM Output
    fmout_np = _arraytonumpy(fmout_data, fmout_shape)
    fmout_np = fm_class.cleanup_fmout(fmout_np)

    #all of the image centers are now at aligned_center
    centers[:,0] = aligned_center[0]
    centers[:,1] = aligned_center[1]

    # Output for the sole PSFs
    return sub_imgs, fmout_np



def _klip_section_multifile_perfile(img_num, sector_index, radstart, radend, phistart, phiend, parang, wavelength,
                                    wv_index, avg_rad, padding,
                                    numbasis, minmove, ref_center, minrot, maxrot, mode, spectrum,
                                    fm_class,
                                    include_spec_in_model = False,
                                    spec_from_model = False):
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
        padding: number of pixels to pad the sector by
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
    # x, y = np.meshgrid(np.arange(original_shape[2] * 1.0), np.arange(original_shape[1] * 1.0))
    # x.shape = (x.shape[0] * x.shape[1]) #Flatten
    # y.shape = (y.shape[0] * y.shape[1])
    # r = np.sqrt((x - ref_center[0])**2 + (y - ref_center[1])**2)
    # phi = np.arctan2(y - ref_center[1], x - ref_center[0])
    #
    # #grab the pixel location of the section we are going to anaylze based on the parallactic angle of the image
    # phi_rotate = ((phi + np.radians(parang)) % (2.0 * np.pi))
    # # in case of wrap around
    # if phistart < phiend:
    #     section_ind = np.where((r >= radstart) & (r < radend) & (phi_rotate >= phistart) & (phi_rotate < phiend))
    # else:
    #     section_ind = np.where((r >= radstart) & (r < radend) & ((phi_rotate >= phistart) | (phi_rotate < phiend)))
    section_ind = _get_section_indicies(original_shape[1:], ref_center, radstart, radend, phistart, phiend,
                                            padding, parang)
    section_ind_nopadding = _get_section_indicies(original_shape[1:], ref_center, radstart, radend, phistart, phiend,
                                            0, parang)
    if np.size(section_ind) <= 1:
        print("section is too small ({0} pixels), skipping...".format(np.size(section_ind)))
        return False
    #print(np.size(section_ind), np.min(phi_rotate), np.max(phi_rotate), phistart, phiend)

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
        ref_psfs_indicies = file_ind[0][closest_matched]

    else:
        # else just grab the reference PSFs for all the valid files
        ref_psfs_selected = ref_psfs[file_ind[0], :]
        ref_psfs_indicies = file_ind[0]

    # create a selection matrix for selecting elements
    unique_wvs = np.unique(wvs_imgs)
    numwv = np.size(unique_wvs)
    numcubes = np.size(wvs_imgs)/numwv
    numpix = np.shape(section_ind)[1]
    numref = np.shape(ref_psfs_indicies)[0]
    #print(numwv,numcubes,numpix,numref)
    #L = np.tile(np.identity(numwv), [1,numref])
    #Sel_wv = L[:, ref_psfs_indicies]


    aligned_imgs = _arraytonumpy(aligned, (aligned_shape[0], aligned_shape[1], aligned_shape[2] * aligned_shape[3]))[wv_index]
    output_imgs = _arraytonumpy(outputs, (outputs_shape[0], outputs_shape[1]*outputs_shape[2], outputs_shape[3]))
    output_imgs_numstacked = _arraytonumpy(outputs_numstacked, (outputs_shape[0], outputs_shape[1]*outputs_shape[2]), dtype=ctypes.c_int)
    fmout_np = _arraytonumpy(fmout, fmout_shape)

    # # generate models for the PSF of the science image
    # model_sci = fm_class.generate_models([original_shape[1], original_shape[2]], section_ind, [parang], [wavelength], radstart, radend, phistart, phiend, padding, ref_center, parang, wavelength)[0]
    # model_sci *= fm_class.flux_conversion[img_num] * fm_class.spectrallib[0][np.where(fm_class.input_psfs_wvs == wavelength)] * fm_class.dflux
    #
    # # generate models of the PSF for each reference segments. Output is of shape (N, pix_in_segment)
    # models_ref = fm_class.generate_models([original_shape[1], original_shape[2]], section_ind, pa_imgs[ref_psfs_indicies], wvs_imgs[ref_psfs_indicies], radstart, radend, phistart, phiend, padding, ref_center, parang, wavelength)
    # input_spectrum = fm_class.flux_conversion[:fm_class.spectrallib[0].shape[0]] * fm_class.spectrallib[0] * fm_class.dflux
    #
    # if include_spec_in_model:
    #     inputflux = np.ravel(np.tile(input_spectrum,(1,numcubes)))
    #     inputflux = inputflux[ref_psfs_indicies]
    #     models_ref = inputflux[:,None]*models_ref
    #     inputflux = None
    # elif spec_from_model:
    #     models_ref_wvSorted = np.zeros((numwv,numref,numpix))
    #     wvs_refs = wvs_imgs[ref_psfs_indicies]
    #     for wv_id,wv in enumerate(unique_wvs):
    #         where_ref_at_wv = np.where(wvs_refs == wv)[0]
    #         models_ref_wvSorted[wv_id,where_ref_at_wv,:] = models_ref[where_ref_at_wv,:]
    #     models_ref = models_ref_wvSorted
    #     inputflux = input_spectrum
    # else:
    #     inputflux = np.ravel(np.tile(input_spectrum,(1,numcubes)))
    #     inputflux = inputflux[ref_psfs_indicies]

    # JB: We can get rid of the input para model_sci. But there is commented code in klip_math that is using it.
    klip_math_return = klip_math(aligned_imgs[img_num, section_ind[0]], ref_psfs_selected, numbasis,
                                 covar_psfs=covar_files,)

    klipped, original_KL, evals, evecs = klip_math_return

    # try:
    #     klip_math_return = klip_math(aligned_imgs[img_num, section_ind[0]], ref_psfs_selected, numbasis, covar_psfs=covar_files, models_ref=models_ref, Sel_wv=Sel_wv, input_spectrum=input_spectrum, model_sci=model_sci)
    # except (RuntimeError) as err: #(ValueError, RuntimeError, TypeError) as err:
    #     print(err.message)
    #     return -1
    #
    # if models_ref is not None:
    #     # passed in models, so perturbed KL modes were partially calculated already
    #     klipped, original_KL, delta_KL_nospec = klip_math_return
    #     evals = None
    #     evecs = None
    # else:
    #     klipped, original_KL, evals, evecs = klip_math_return
    #     delta_KL_nospec = None
    #
    #
    # postklip_psf, oversubtraction, selfsubtraction = calculate_fm(delta_KL_nospec, original_KL, numbasis, aligned_imgs[img_num, section_ind[0]], model_sci, inputflux = inputflux)



    # write to output
    for thisnumbasisindex in range(klipped.shape[1]):
        if thisnumbasisindex == 0:
            #only increment the numstack counter for the first KL mode
            _save_rotated_section([original_shape[1], original_shape[2]], klipped[:, thisnumbasisindex], section_ind,
                             output_imgs[img_num,:,thisnumbasisindex], output_imgs_numstacked[img_num], parang,
                             radstart, radend, phistart, phiend, padding, ref_center, flipx=True)
        else:
            _save_rotated_section([original_shape[1], original_shape[2]], klipped[:, thisnumbasisindex], section_ind,
                             output_imgs[img_num,:,thisnumbasisindex], None, parang,
                             radstart, radend, phistart, phiend, padding, ref_center, flipx=True)
    #output_imgs[img_num, section_ind[0], :] = klipped

    # call FM Class to handle forward modelling if it wants to. Basiclaly we are passing in everything as a variable
    # and it can choose which variables it wants to deal with using **kwargs
    # result is stored in fmout
    fm_class.fm_from_eigen(klmodes=original_KL, evals=evals, evecs=evecs,
                          input_img_shape=[original_shape[1], original_shape[2]], input_img_num=img_num,
                          ref_psfs_indicies=ref_psfs_indicies, section_ind=section_ind, aligned_imgs=aligned_imgs,
                          pas=pa_imgs[ref_psfs_indicies], wvs=wvs_imgs[ref_psfs_indicies], radstart=radstart,
                          radend=radend, phistart=phistart, phiend=phiend, padding=padding, ref_center=ref_center,
                          parang=parang, ref_wv=wavelength, numbasis=numbasis, fmout=fmout_np)

    return sector_index
