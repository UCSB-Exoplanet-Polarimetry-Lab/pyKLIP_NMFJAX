"""
Utilties to subtract the background
"""
import numpy as np
import scipy.linalg as linalg
import scipy.optimize as optimize


def subtract_bkgd(input_frames, bkgd_frames, opt_maps, sub_maps, conserve_flux=False):
    """
    Performs background subtraction for each image using a cube of background frames
    Does subtraction region by region, where each region needs to specify an optimization
    map (where to optimize the linear combination of bkgd_frames to match the data) and a
    subtraction map (where to subtract the background after optimizing the background subtraction
    on the optimization region). `opt_maps` and `sub_maps` needs to be the same length. 

    Args:
        intput_frames (np.array): input frames to be subtracted (N_sci_frames, y, x)
        bkgd_frames (np.array): background frames of shape (N_bkgd_frames, y, x). 
                                Neds to be same dimensions as dataset.input!
        opt_maps (np.array): maps of optimization regions. Shape of (N_regions, y, x)
                             Pixel values = 1 for inside zone. 0 for outside zone.
        sub_maps (np.array): maps of subtraction regions. Shape of (N_regions, y, x)
                             Pixel values = 1 for inside zone. 0 for outside zone.
        conserve_flux (bool): if True, preseveres the flux of the background images (no scaling)

    Returns:
        sub_frames (np.array): background subtracted frames (N_sci_frames, y, x)

    """
    sub_frames = np.zeros(input_frames.shape) * np.nan

    sub_frames_ravel = sub_frames.reshape([sub_frames.shape[0], sub_frames.shape[1] * sub_frames.shape[2]])
    bkgd_frames_ravel = bkgd_frames.reshape([bkgd_frames.shape[0], bkgd_frames.shape[1] * bkgd_frames.shape[2]])
    
    for i, input_frame in enumerate(input_frames):
        input_ravel = input_frame.ravel()

        for opt_mask, sub_mask in zip(opt_maps, sub_maps):
            opt_zone = np.where(opt_mask.ravel() == 1)
            sub_zone = np.where(sub_mask.ravel() == 1)
            
            if not conserve_flux:
                # run LOCI
                result = linalg.lstsq(bkgd_frames_ravel.T[opt_zone[0],:], input_ravel[opt_zone])
                coeffs = result[0]
            else:
                def loci_noscale(params, sci_frame, ref_frames):
                    coeffs = params
                    last_val = 1 - np.sum(coeffs)
                    coeffs = np.append(coeffs, last_val)
                    
                    loci_psf = np.dot(ref_frames, coeffs)
                    sci_sub_loci = sci_frame - loci_psf
                    
                    return sci_sub_loci

            tot_frames = bkgd_frames_ravel.shape[0]
            guess = np.ones(tot_frames-1, dtype=float) * (1/tot_frames)
            result = optimize.leastsq(loci_noscale, guess, args=(input_ravel[opt_zone], bkgd_frames_ravel.T[opt_zone[0],:]))

            coeffs = result[0]
            coeffs = np.append(coeffs, 1-np.sum(coeffs))
            
            #loci_psf = np.dot(kl_basis.T, coeffs)
            loci_psf = np.dot(bkgd_frames_ravel.T, coeffs)

            sci_sub_loci = input_ravel - loci_psf

            sub_frames_ravel[i, sub_zone[0]] = sci_sub_loci[sub_zone]

    return sub_frames
            
    