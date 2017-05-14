        #wv_index_list = [self.input_psfs_wvs.index(wv) for wv in wvs]
__author__ = 'jruffio'
import multiprocessing as mp
import ctypes

import numpy as np
import pyklip.spectra_management as spec
import os
import itertools

from pyklip.fmlib.nofm import NoFM
import pyklip.fm as fm

from scipy import interpolate
from copy import copy

import astropy.io.fits as pyfits

from time import time
import matplotlib.pyplot as plt

debug = False


class MatchedFilter(NoFM):
    """
    Matched filter with forward modelling.
    """
    def __init__(self, inputs_shape,
                 numbasis,
                 input_psfs,
                 input_psfs_wvs,
                 spectrallib,
                 save_per_sector = None,
                 datatype="float",
                 fakes_sepPa_list = None,
                 disable_FM = None,
                 true_fakes_pos = None):
        '''
        Defining the forward model matched filter parameters

        Args:
            inputs_shape: shape of the inputs numpy array. Typically (N, y, x)
            numbasis: 1d numpy array consisting of the number of basis vectors to use
            input_psfs: the psf of the image. A numpy array with shape (wv, y, x)
            input_psfs_wvs: the wavelegnths that correspond to the input psfs
            spectrallib: if not None, a list of spectra in raw DN units. The spectra should:
                    - have the total flux of the star, ie correspond to a contrast of 1.
                    - represent the total flux of the PSF and not the simply peak value.
                    - be corrected for atmospheric and instrumental transmission.
                    - have the same size as the number of images in the dataset.
            save_per_sector: If not None, should be a filename where the fmout array will be saved after each sector.
                    (Caution: huge file!! easily tens of Gb.)
            datatype: datatype to be used for the numpy arrays: "double" or "float" (default).
            fakes_sepPa_list: [(sep_pix1,pa1),(sep_pix2,pa2),...].
                    List of separations and pas for the simulated planets in the data.
                    If not None, it will only calculated the matched filter at the position of the fakes and skip the rest.
            disable_FM: Disable the calculation of the forward model in the code.
                        The unchanged original PSF will be used instead. (Default False)
            true_fakes_pos: If True and fakes_only is True, calculate the forward model at the exact position of the
                    fakes and not at the center of the pixels. (Default False)

        '''
        # allocate super class
        super(MatchedFilter, self).__init__(inputs_shape, np.array(numbasis))

        if true_fakes_pos is None:
            self.true_fakes_pos = False
        else:
            self.true_fakes_pos = true_fakes_pos

        if datatype=="double":
            self.mp_data_type = ctypes.c_double
            self.np_data_type = float
        elif datatype=="float":
            self.mp_data_type = ctypes.c_float
            self.np_data_type = np.float32

        if save_per_sector is not None:
            self.fmout_dir = save_per_sector
            self.save_raw_fmout = True
        else:
            self.save_raw_fmout = False

        self.N_numbasis =  np.size(numbasis)
        self.ny = self.inputs_shape[1]
        self.nx = self.inputs_shape[2]
        self.N_frames = self.inputs_shape[0]

        self.fakes_sepPa_list = fakes_sepPa_list
        if disable_FM is None:
            self.disable_FM = False
        else:
            self.disable_FM = disable_FM

        self.inputs_shape = self.inputs_shape

        self.input_psfs_wvs = list(np.array(input_psfs_wvs,dtype=self.np_data_type))

        # Make sure the total flux of each PSF is unity for all wavelengths
        # So the peak value won't be unity.
        self.input_psfs = input_psfs/np.nansum(input_psfs,axis=(1,2))[:,None,None]
        numwv_psf,ny_psf,nx_psf =  self.input_psfs.shape

        self.spectrallib = spectrallib
        self.N_spectra = len(self.spectrallib)


        # create bounds for PSF stamp size
        self.row_m = int(np.floor(ny_psf/2.0))    # row_minus
        self.row_p = int(np.ceil(ny_psf/2.0))     # row_plus
        self.col_m = int(np.floor(nx_psf/2.0))    # col_minus
        self.col_p = int(np.ceil(nx_psf/2.0))     # col_plus

        self.psf_centx_notscaled = {}
        self.psf_centy_notscaled = {}
        self.curr_pa_fk = {}
        self.curr_sep_fk = {}

        x_psf_grid, y_psf_grid = np.meshgrid(np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2)
        psfs_func_list = []
        self.input_psfs[np.where(np.isnan(self.input_psfs))] = 0
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for wv_index in range(numwv_psf):
                model_psf = self.input_psfs[wv_index, :, :]
                psf_func = interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5)
                psfs_func_list.append(psf_func)

        self.psfs_func_list = psfs_func_list

        ny_PSF,nx_PSF = input_psfs.shape[1:]
        stamp_PSF_x_grid, stamp_PSF_y_grid = np.meshgrid(np.arange(0,nx_PSF,1)-nx_PSF//2,np.arange(0,ny_PSF,1)-ny_PSF//2)
        self.stamp_PSF_mask = np.ones((ny_PSF,nx_PSF))
        r_PSF_stamp = abs((stamp_PSF_x_grid) +(stamp_PSF_y_grid)*1j)
        self.stamp_PSF_mask[np.where(r_PSF_stamp < 7.)] = np.nan
        # self.stamp_PSF_mask[np.where(r_PSF_stamp < 4.)] = np.nan


    # def alloc_interm(self, max_sector_size, numsciframes):
    #     """Allocates shared memory array for intermediate step
    #
    #     Intermediate step is allocated for a sector by sector basis
    #
    #     Args:
    #         max_sector_size: number of pixels in this sector. Max because this can be variable. Stupid rotating sectors
    #
    #     Returns:
    #         interm: mp.array to store intermediate products from one sector in
    #         interm_shape:shape of interm array (used to convert to numpy arrays)
    #
    #     """
    #
    #     interm_size = max_sector_size * np.size(self.numbasis) * numsciframes * len(self.spectrallib)
    #
    #     interm = mp.Array(ctypes.c_double, interm_size)
    #     interm_shape = [numsciframes, len(self.spectrallib), max_sector_size, np.size(self.numbasis)]
    #
    #     return interm, interm_shape


    def alloc_fmout(self, output_img_shape):
        """
        Allocates shared memory for the output of the shared memory

        Args:
            output_img_shape: Not used

        Returns:
            fmout: mp.array to store auxilliary data in
            fmout_shape: shape of auxilliary array = (3,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)
                        The 3 is for saving the different term of the matched filter:
                            0: dot product
                            1: square of the norm of the model
                            2: Local estimated variance of the data
                            3: Number of pixels used in the matched filter

        """
        # fmout_size = 3*self.N_spectra*self.N_numbasis*self.N_frames*self.ny*self.nx
        # fmout = mp.Array(self.mp_data_type, fmout_size)
        # fmout_shape = (3,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)
        fmout_size = 4*self.N_spectra*self.N_numbasis*self.N_frames*self.ny*self.nx
        fmout = mp.Array(self.mp_data_type, fmout_size)
        fmout_shape = (4,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)

        return fmout, fmout_shape

    def skip_section(self, radstart, radend, phistart, phiend):
        """
        Returns a boolean indicating if the section defined by (radstart, radend, phistart, phiend) should be skipped.
        When True is returned the current section in the loop in klip_parallelized() is skipped.

        Args:
            radstart: minimum radial distance of sector [pixels]
            radend: maximum radial distance of sector [pixels]
            phistart: minimum azimuthal coordinate of sector [radians]
            phiend: maximum azimuthal coordinate of sector [radians]

        Returns:
            Boolean: False so by default it never skips.
        """

        margin_sep = np.sqrt(2)/2.
        margin_phi = np.sqrt(2)/(2*radstart)
        if self.fakes_sepPa_list is not None:
            skipSectionBool = True
            for sep_it,pa_it in self.fakes_sepPa_list:
                paend= ((2*np.pi-phistart +np.pi/2)% (2.0 * np.pi))
                pastart = ((2*np.pi-phiend +np.pi/2)% (2.0 * np.pi))
                # Normal case when there are no 2pi wrap
                if pastart < paend:
                    if (radstart-margin_sep<=sep_it<=radend+margin_sep) and ((pa_it%360)/180.*np.pi >= pastart-margin_phi) & ((pa_it%360)/180.*np.pi < paend+margin_phi):
                        skipSectionBool = False
                        break
                # 2 pi wrap case
                else:
                    if (radstart-margin_sep<=sep_it<=radend+margin_sep) and (((pa_it%360)/180.*np.pi >= pastart-margin_phi) | ((pa_it%360)/180.*np.pi < paend+margin_phi)):
                        skipSectionBool = False
                        break
        else:
            skipSectionBool = False

        return skipSectionBool


    def fm_from_eigen(self, klmodes=None, evals=None, evecs=None, input_img_shape=None, input_img_num=None,
                      ref_psfs_indicies=None, section_ind=None,section_ind_nopadding=None, aligned_imgs=None, pas=None,
                     wvs=None, radstart=None, radend=None, phistart=None, phiend=None, padding=None,IOWA = None, ref_center=None,
                     parang=None, ref_wv=None, numbasis=None, fmout=None, perturbmag=None,klipped=None, flipx=True, **kwargs):
        """
        Calculate and project the FM at every pixel of the sector. Store the result in fmout.

        Args:
            klmodes: unpertrubed KL modes
            evals: eigenvalues of the covariance matrix that generated the KL modes in ascending order
                   (lambda_0 is the 0 index) (shape of [nummaxKL])
            evecs: corresponding eigenvectors (shape of [p, nummaxKL])
            input_img_shape: 2-D shape of inpt images ([ysize, xsize])
            input_img_num: index of sciece frame
            ref_psfs_indicies: array of indicies for each reference PSF
            section_ind: array indicies into the 2-D x-y image that correspond to this section.
                         Note needs be called as section_ind[0]
            pas: array of N parallactic angles corresponding to N reference images [degrees]
            wvs: array of N wavelengths of those referebce images
            radstart: radius of start of segment
            radend: radius of end of segment
            phistart: azimuthal start of segment [radians]
            phiend: azimuthal end of segment [radians]
            padding: amount of padding on each side of sector
            IOWA: tuple (IWA,OWA) where IWA = Inner working angle and OWA = Outer working angle both in pixels.
                It defines the separation interva in which klip will be run.
            ref_center: center of image
            numbasis: array of KL basis cutoffs
            parang: parallactic angle of input image [DEGREES]
            ref_wv: wavelength of science image
            fmout: numpy output array for FM output. Shape is (N, y, x, b)
            klipped: array of shape (p,b) that is the PSF subtracted data for each of the b KLIP basis
                     cutoffs. If numbasis was an int, then sub_img_row_selected is just an array of length p
            kwargs: any other variables that we don't use but are part of the input
        """
        ref_wv = ref_wv.astype(self.np_data_type)

        sci = aligned_imgs[input_img_num, section_ind[0]]
        refs = aligned_imgs[ref_psfs_indicies, :]
        refs = refs[:, section_ind[0]]

        # Calculate the PA,sep 2D map
        x_grid, y_grid = np.meshgrid(np.arange(self.nx * 1.0)- ref_center[0], np.arange(self.ny * 1.0)- ref_center[1])
        x_grid=x_grid.astype(self.np_data_type)
        y_grid=y_grid.astype(self.np_data_type)
        r_grid = np.sqrt((x_grid)**2 + (y_grid)**2)
        pa_grid = np.arctan2( -x_grid,y_grid) % (2.0 * np.pi)
        if flipx:
            paend= ((-phistart + np.pi/2.)% (2.0 * np.pi))
            pastart = ((-phiend + np.pi/2.)% (2.0 * np.pi))
        else:
            pastart = ((phistart - np.pi/2.)% (2.0 * np.pi))
            paend= ((phiend - np.pi/2.)% (2.0 * np.pi))
        # Normal case when there are no 2pi wrap
        if pastart < paend:
            where_section = np.where((r_grid >= radstart) & (r_grid < radend) & (pa_grid >= pastart) & (pa_grid < paend))
        # 2 pi wrap case
        else:
            where_section = np.where((r_grid >= radstart) & (r_grid < radend) & ((pa_grid >= pastart) | (pa_grid < paend)))

        # Get a list of the PAs and sep of the PA,sep map falling in the current section
        r_list = r_grid[where_section]
        pa_list = pa_grid[where_section]
        x_list = x_grid[where_section]
        y_list = y_grid[where_section]
        row_id_list = where_section[0]
        col_id_list = where_section[1]
        # Only select pixel with fakes if needed
        if self.fakes_sepPa_list is not None:
            r_list_tmp = []
            pa_list_tmp = []
            row_id_list_tmp = []
            col_id_list_tmp = []
            for sep_it,pa_it in self.fakes_sepPa_list:
                x_it = sep_it*np.cos(np.radians(90+pa_it))
                y_it = sep_it*np.sin(np.radians(90+pa_it))
                dist_list = np.sqrt((x_list-x_it)**2+(y_list-y_it)**2)
                min_id = np.nanargmin(dist_list)
                min_dist = dist_list[min_id]
                if min_dist < np.sqrt(2)/2.:
                    if self.true_fakes_pos:
                        r_list_tmp.append(sep_it)
                        pa_list_tmp.append(np.radians(pa_it))
                    else:
                        r_list_tmp.append(r_list[min_id])
                        pa_list_tmp.append(pa_list[min_id])
                    row_id_list_tmp.append(row_id_list[min_id])
                    col_id_list_tmp.append(col_id_list[min_id])
            r_list = r_list_tmp
            pa_list = pa_list_tmp
            row_id_list = row_id_list_tmp
            col_id_list = col_id_list_tmp

        # Loop over the input template spectra and the number of KL modes in numbasis
        for spec_id,N_KL_id in itertools.product(range(self.N_spectra),range(self.N_numbasis)):
            # Calculate the projection of the FM and the klipped section for every pixel in the section.
            # 1/ Inject a fake at one pa and sep in the science image
            # 2/ Inject the corresponding planets at the same PA and sep in the reference images remembering that the
            # references rotate.
            # 3/ Calculate the perturbation of the KL modes
            # 4/ Calculate the FM
            # 5/ Calculate dot product (matched filter)
            for sep_fk,pa_fk,row_id,col_id in zip(r_list,np.rad2deg(pa_list),row_id_list,col_id_list):
                # 1/ Inject a fake at one pa and sep in the science image
                model_sci,mask = self.generate_model_sci(input_img_shape, section_ind, parang, ref_wv,
                                                         radstart, radend, phistart, phiend, padding, ref_center,
                                                         parang, ref_wv,sep_fk,pa_fk, flipx)
                # Normalize the science image according to the spectrum. the model is normalize to unit contrast,
                model_sci = model_sci*self.spectrallib[spec_id][input_img_num]
                where_fk = np.where(mask==2)[0]
                where_background = np.where(mask>=1)[0] # Caution: it includes where the fake is...
                where_background_strict = np.where(mask==1)[0]

                # 2/ Inject the corresponding planets at the same PA and sep in the reference images remembering that the
                # references rotate.
                if not self.disable_FM:
                    models_ref = self.generate_models(input_img_shape, section_ind, pas, wvs, radstart, radend,
                                                      phistart, phiend, padding, ref_center, parang, ref_wv,sep_fk,pa_fk, flipx)

                    # Normalize the models with the spectrum. the model is normalize to unit contrast,
                    input_spectrum = self.spectrallib[spec_id][ref_psfs_indicies]
                    models_ref = models_ref * input_spectrum[:, None]

                    # 3/ Calculate the perturbation of the KL modes
                    # using original Kl modes and reference models, compute the perturbed KL modes.
                    # Spectrum is already in the model, that's why we use perturb_specIncluded(). (Much faster)
                    delta_KL = fm.perturb_specIncluded(evals, evecs, klmodes, refs, models_ref)

                    # 4/ Calculate the FM: calculate postklip_psf using delta_KL
                    # postklip_psf has unit broadband contrast
                    postklip_psf, oversubtraction, selfsubtraction = fm.calculate_fm(delta_KL, klmodes, numbasis,
                                                                                     sci, model_sci, inputflux=None)
                else:
                    #if one doesn't want the FM
                    if np.size(numbasis) == 1:
                        postklip_psf = model_sci[None,:]
                    else:
                        postklip_psf = model_sci

                # 5/ Calculate dot product (matched filter)
                # fmout_shape = (3,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)
                #         The 3 is for saving the different term of the matched filter:
                #             0: dot product
                #             1: square of the norm of the model
                #             2: Local estimated variance of the data
                sky = np.nanmean(klipped[where_background_strict,N_KL_id])
                # postklip_psf[N_KL_id,where_fk] = postklip_psf[N_KL_id,where_fk]-np.mean(postklip_psf[N_KL_id,where_background])
                # Subtract local sky background to the klipped image
                klipped_sub = klipped[where_fk,N_KL_id]-sky
                klipped_sub_finite = np.where(np.isfinite(klipped_sub))
                # klipped_sub_nan = np.where(np.isnan(klipped_sub))
                postklip_psf[N_KL_id,np.where(np.isnan(klipped))[0]] = np.nan
                if float(np.sum(np.isfinite(klipped_sub)))/float(np.size(klipped_sub))<=0.75:
                    dot_prod = np.nan
                else:
                    dot_prod = np.nansum(klipped_sub*postklip_psf[N_KL_id,where_fk])
                model_norm = np.nansum(postklip_psf[N_KL_id,where_fk]*postklip_psf[N_KL_id,where_fk])
                klipped_rm_pl = klipped[:,N_KL_id]-sky-(dot_prod/model_norm)*postklip_psf[N_KL_id,:]
                if float(np.sum(np.isfinite(klipped_rm_pl[where_background])))/float(np.size(klipped_rm_pl[where_background]))<=0.75:
                    variance = np.nan
                    npix = np.nan
                else:
                    variance = np.nanvar(klipped_rm_pl[where_background])
                    npix = np.sum(np.isfinite(klipped_rm_pl[where_background]))

                fmout[0,spec_id,N_KL_id,input_img_num,row_id,col_id] = dot_prod
                fmout[1,spec_id,N_KL_id,input_img_num,row_id,col_id] = model_norm
                fmout[2,spec_id,N_KL_id,input_img_num,row_id,col_id] = variance
                fmout[3,spec_id,N_KL_id,input_img_num,row_id,col_id] = npix

                # Plot sector, klipped and FM model for debug only
                if 0 and row_id>=10:# and np.nansum(klipped[where_fk,N_KL_id]) != 0:
                    #if 0:
                    # print(klipped_sub)
                    # print(np.isfinite(klipped_sub))
                    # print(np.size(klipped_sub))
                    # print(float(np.sum(np.isfinite(klipped_sub)))/float(np.size(klipped_sub)))
                    # print(float(np.sum(np.isfinite(klipped[where_background,N_KL_id])))/float(np.size(klipped[where_background,N_KL_id])))
                    print(sep_fk,pa_fk,row_id,col_id)
                    print(dot_prod,model_norm,variance)
                    print(np.nanmean(klipped-sky),sky,dot_prod,model_norm,np.nanmean((dot_prod/model_norm)*postklip_psf[N_KL_id,:]))
                    print(klipped.shape,postklip_psf[N_KL_id,:].shape)
                    print(float(np.sum(np.isfinite(klipped_rm_pl[where_background]))),float(np.size(klipped_rm_pl[where_background])))
                    blackboard1 = np.zeros((self.ny,self.nx))
                    blackboard2 = np.zeros((self.ny,self.nx))
                    blackboard3 = np.zeros((self.ny,self.nx))
                    #print(section_ind)
                    plt.figure(1)
                    plt.subplot(1,3,1)
                    blackboard1.shape = [input_img_shape[0] * input_img_shape[1]]
                    blackboard1[section_ind] = mask
                    blackboard1[section_ind] = blackboard1[section_ind] + 1
                    blackboard1.shape = [input_img_shape[0],input_img_shape[1]]
                    plt.imshow(blackboard1)
                    plt.colorbar()
                    plt.subplot(1,3,2)
                    blackboard2.shape = [input_img_shape[0] * input_img_shape[1]]
                    # blackboard2[section_ind[0][where_fk]] = klipped[where_fk,N_KL_id]
                    blackboard2[section_ind[0]] = klipped_rm_pl
                    blackboard2.shape = [input_img_shape[0],input_img_shape[1]]
                    plt.imshow(blackboard2)
                    plt.colorbar()
                    plt.subplot(1,3,3)
                    blackboard3.shape = [input_img_shape[0] * input_img_shape[1]]
                    blackboard3[section_ind[0][where_fk]] = postklip_psf[N_KL_id,where_fk]
                    blackboard3.shape = [input_img_shape[0],input_img_shape[1]]
                    plt.imshow(blackboard3)
                    plt.colorbar()
                    #print(klipped[where_fk,N_KL_id])
                    #print(postklip_psf[N_KL_id,where_fk])
                    print(np.sum(klipped[where_fk,N_KL_id]*postklip_psf[N_KL_id,where_fk]))
                    print(np.sum(postklip_psf[N_KL_id,where_fk]*postklip_psf[N_KL_id,where_fk]))
                    print(np.sum(klipped[where_fk,N_KL_id]*klipped[where_fk,N_KL_id]))
                    plt.show()



    def fm_end_sector(self, interm_data=None, fmout=None, sector_index=None,
                               section_indicies=None):
        """
        Save the fmout object at the end of each sector if save_per_sector was defined when initializing the class.
        """
        #fmout_shape = (3,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)
        if self.save_raw_fmout:
            hdu = pyfits.PrimaryHDU(fmout)
            hdulist = pyfits.HDUList([hdu])
            hdulist.writeto(self.fmout_dir,clobber=True)
        return

    def save_fmout(self, dataset, fmout, outputdir, fileprefix, numbasis, klipparams=None, calibrate_flux=False,
                   spectrum=None):
        """
        Saves the fmout data to disk following the instrument's savedata function

        Args:
            dataset: Instruments.Data instance. Will use its dataset.savedata() function to save data
            fmout: the fmout data passed from fm.klip_parallelized which is passed as the output of cleanup_fmout
            outputdir: output directory
            fileprefix: the fileprefix to prepend the file name
            numbasis: KL mode cutoffs used
            klipparams: string with KLIP-FM parameters
            calibrate_flux: if True, flux calibrate the data (if applicable)
            spectrum: if not None, the spectrum to weight the data by. Length same as dataset.wvs
        """

        #fmout_shape = (3,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)
        fmout[np.where(fmout==0)] = np.nan

        # The mf.MatchedFilter class calculate the projection of the FM on the data for each pixel and images.
        # The final combination to form the cross  cross correlation, matched filter and contrast maps is done right
        # here.
        FMCC_map = np.nansum(fmout[0,:,:,:,:,:],axis=2) \
                        / np.sqrt(np.nansum(fmout[1,:,:,:,:,:],axis=2))
        FMCC_map[np.where(FMCC_map==0)]=np.nan
        self.FMCC_map = FMCC_map

        FMMF_map = np.nansum(fmout[0,:,:,:,:,:]/fmout[2,:,:,:,:,:],axis=2) \
                        / np.sqrt(np.nansum(fmout[1,:,:,:,:,:]/fmout[2,:,:,:,:,:],axis=2))
        FMMF_map[np.where(FMMF_map==0)]=np.nan
        self.FMMF_map = FMMF_map

        contrast_map = np.nansum(fmout[0,:,:,:,:,:]/fmout[2,:,:,:,:,:],axis=2) \
                        / np.nansum(fmout[1,:,:,:,:,:]/fmout[2,:,:,:,:,:],axis=2)
        contrast_map[np.where(contrast_map==0)]=np.nan
        self.contrast_map = contrast_map

        self.metricMap = [self.FMMF_map,self.FMCC_map,self.contrast_map]


        for k in range(np.size(self.numbasis)):
            # Save the outputs (matched filter, shape map and klipped image) as fits files
            suffix = "FMMF-KL{0}".format(self.numbasis[k])
            dataset.savedata(outputdir+os.path.sep+fileprefix+'-'+suffix+'.fits',
                             self.FMMF_map[0,k,:,:],
                             filetype=suffix)

            suffix = "FMCont-KL{0}".format(self.numbasis[k])
            dataset.savedata(outputdir+os.path.sep+fileprefix+'-'+suffix+'.fits',
                             self.contrast_map[0,k,:,:],
                             filetype=suffix)

            suffix = "FMCC-KL{0}".format(self.numbasis[k])
            dataset.savedata(outputdir+os.path.sep+fileprefix+'-'+suffix+'.fits',
                             self.FMCC_map[0,k,:,:],
                             filetype=suffix)

        return

    def generate_model_sci(self, input_img_shape, section_ind, pa, wv, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv,sep_fk,pa_fk, flipx):
        """
        Generate model PSFs at the correct location of this segment of the science image denotated by its wv and
        parallactic angle.

        Args:
            input_img_shape: 2-D shape of inpt images ([ysize, xsize])
            section_ind: array indicies into the 2-D x-y image that correspond to this section.
                         Note needs be called as section_ind[0]
            pa: parallactic angle of the science image [degrees]
            wv: wavelength of the science image
            radstart: radius of start of segment (not used)
            radend: radius of end of segment (not used)
            phistart: azimuthal start of segment [radians] (not used)
            phiend: azimuthal end of segment [radians] (not used)
            padding: amount of padding on each side of sector
            ref_center: center of image
            parang: parallactic angle of input image [DEGREES] (not used)
            ref_wv: wavelength of science image
            sep_fk: separation of the planet to be injected.
            pa_fk: position angle of the planet to be injected.
            flipx: if True, flip x coordinate in final image

        Return: (models, mask)
            models: vector of size p where p is the number of pixels in the segment
            mask: vector of size p where p is the number of pixels in the segment
                    if pixel == 1: arc shape where to calculate the standard deviation
                    if pixel == 2: 7 pixels disk around the position of the planet.
        """
        # create some parameters for a blank canvas to draw psfs on
        nx = input_img_shape[1]
        ny = input_img_shape[0]
        x_grid, y_grid = np.meshgrid(np.arange(nx * 1.)-ref_center[0], np.arange(ny * 1.)-ref_center[1])

        numwv, ny_psf, nx_psf =  self.input_psfs.shape

        # create bounds for PSF stamp size
        row_m = int(np.floor(ny_psf/2.0))    # row_minus
        row_p = int(np.ceil(ny_psf/2.0))     # row_plus
        col_m = int(np.floor(nx_psf/2.0))    # col_minus
        col_p = int(np.ceil(nx_psf/2.0))     # col_plus

        # a blank img array of write model PSFs into
        whiteboard = np.zeros((ny,nx))
        # grab PSF given wavelength
        wv_index = [spec.find_nearest(self.input_psfs_wvs,wv)[1]]

        sign = -1.
        if flipx:
            sign = 1.

        # The trigonometric calculation are save in a dictionary to avoid calculating them many times.
        recalculate_trig = False
        if pa not in self.psf_centx_notscaled:
            recalculate_trig = True
        else:
            if pa_fk != self.curr_pa_fk[pa] or sep_fk != self.curr_sep_fk[pa]:
                recalculate_trig = True
        if recalculate_trig: # we could actually store the values for the different pas too...
            # flipx requires the opposite rotation
            self.psf_centx_notscaled[pa] = sep_fk * np.cos(np.radians(90. - sign*pa_fk - pa))
            self.psf_centy_notscaled[pa] = sep_fk * np.sin(np.radians(90. - sign*pa_fk - pa))
            self.curr_pa_fk[pa] = pa_fk
            self.curr_sep_fk[pa] = sep_fk

        psf_centx = (ref_wv/wv) * self.psf_centx_notscaled[pa]
        psf_centy = (ref_wv/wv) * self.psf_centy_notscaled[pa]

        # create a coordinate system for the image that is with respect to the model PSF
        # round to nearest pixel and add offset for center
        l = int(round(psf_centx + ref_center[0]))
        k = int(round(psf_centy + ref_center[1]))
        # recenter coordinate system about the location of the planet
        # x_vec_stamp_centered = x_grid[0, (l-col_m):(l+col_p)]-psf_centx
        # y_vec_stamp_centered = y_grid[(k-row_m):(k+row_p), 0]-psf_centy
        x_vec_stamp_centered = x_grid[0, np.max([(l-col_m),0]):np.min([(l+col_p),nx])]-psf_centx
        y_vec_stamp_centered = y_grid[np.max([(k-row_m),0]):np.min([(k+row_p),ny]), 0]-psf_centy
        # rescale to account for the align and scaling of the refernce PSFs
        # e.g. for longer wvs, the PSF has shrunk, so we need to shrink the coordinate system
        x_vec_stamp_centered /= (ref_wv/wv)
        y_vec_stamp_centered /= (ref_wv/wv)

        # use intepolation spline to generate a model PSF and write to temp img
        # whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = \
        #         self.psfs_func_list[wv_index[0]](x_vec_stamp_centered,y_vec_stamp_centered).transpose()
        whiteboard[np.max([(k-row_m),0]):np.min([(k+row_p),ny]), np.max([(l-col_m),0]):np.min([(l+col_p),nx])] = \
                self.psfs_func_list[wv_index[0]](x_vec_stamp_centered,y_vec_stamp_centered).transpose()

        # write model img to output (segment is collapsed in x/y so need to reshape)
        whiteboard.shape = [input_img_shape[0] * input_img_shape[1]]
        segment_with_model = copy(whiteboard[section_ind])
        whiteboard.shape = [input_img_shape[0],input_img_shape[1]]

        # Define the masks for where the planet is and the background.
        r_grid = abs(x_grid +y_grid*1j)
        th_grid = (np.arctan2(sign*x_grid,y_grid)-sign*np.radians(pa))% (2.0 * np.pi)
        w = self.stamp_PSF_mask.shape[0]//2
        thstart = (np.radians(pa_fk)- float(w)/sep_fk) % (2.0 * np.pi) # -(2*np.pi-np.radians(pa))
        thend = (np.radians(pa_fk) + float(w)/sep_fk) % (2.0 * np.pi) # -(2*np.pi-np.radians(pa))
        if thstart < thend:
            where_mask = np.where((r_grid>=(sep_fk-w)) & (r_grid<(sep_fk+w)) & (th_grid >= thstart) & (th_grid < thend))
        else:
            where_mask = np.where((r_grid>=(sep_fk-w)) & (r_grid<(sep_fk+w)) & ((th_grid >= thstart) | (th_grid < thend)))
        whiteboard[where_mask] = 1
        #TODO check the modification I did to these lines
        whiteboard = np.pad(whiteboard,((row_m,row_p),(col_m,col_p)),mode="constant",constant_values=0)
        # whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)][np.where(np.isnan(self.stamp_PSF_mask))]=2
        whiteboard[(k):(k+row_m+row_p), (l):(l+col_m+col_p)][np.where(np.isnan(self.stamp_PSF_mask))]=2
        whiteboard = np.ascontiguousarray(whiteboard[row_m:row_m+input_img_shape[0],col_m:col_m+input_img_shape[1]])
        whiteboard.shape = [input_img_shape[0] * input_img_shape[1]]
        mask = whiteboard[section_ind]

        # create a canvas to place the new PSF in the sector on
        if 0:#np.size(np.where(mask==2)[0])==0: 296
            print(pa,pa_fk)
            print(thstart,thend)
            whiteboard.shape = (input_img_shape[0], input_img_shape[1])
            blackboard = np.zeros((ny,nx))
            blackboard.shape = [input_img_shape[0] * input_img_shape[1]]
            blackboard[section_ind] = segment_with_model
            blackboard.shape = [input_img_shape[0],input_img_shape[1]]
            plt.figure(1)
            plt.subplot(1,3,1)
            im = plt.imshow(whiteboard)
            plt.colorbar(im)
            plt.subplot(1,3,2)
            im = plt.imshow(blackboard+whiteboard)
            plt.colorbar(im)
            plt.subplot(1,3,3)
            im = plt.imshow(np.degrees(th_grid))
            plt.colorbar(im)
            plt.show()

        return segment_with_model,mask

    def generate_models(self, input_img_shape, section_ind, pas, wvs, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv,sep_fk,pa_fk,flipx):
        """
        Generate model PSFs at the correct location of this segment for each image denotated by its wv and parallactic
        angle.

        Args:
            input_img_shape: 2-D shape of inpt images ([ysize, xsize])
            section_ind: array indicies into the 2-D x-y image that correspond to this section.
                         Note needs be called as section_ind[0]
            pas: array of N parallactic angles corresponding to N images [degrees]
            wvs: array of N wavelengths of those images
            radstart: radius of start of segment (not used)
            radend: radius of end of segment (not used)
            phistart: azimuthal start of segment [radians] (not used)
            phiend: azimuthal end of segment [radians] (not used)
            padding: amount of padding on each side of sector
            ref_center: center of image
            parang: parallactic angle of input image [DEGREES] (not used)
            ref_wv: wavelength of science image
            sep_fk: separation of the planet to be injected.
            pa_fk: position angle of the planet to be injected.
            flipx: if True, flip x coordinate in final image

        Return:
            models: array of size (N, p) where p is the number of pixels in the segment
        """
        # create some parameters for a blank canvas to draw psfs on
        nx = input_img_shape[1]
        ny = input_img_shape[0]
        x_grid, y_grid = np.meshgrid(np.arange(nx * 1.)-ref_center[0], np.arange(ny * 1.)-ref_center[1])

        numwv, ny_psf, nx_psf =  self.input_psfs.shape

        # create bounds for PSF stamp size
        row_m = int(np.floor(ny_psf/2.0))    # row_minus
        row_p = int(np.ceil(ny_psf/2.0))     # row_plus
        col_m = int(np.floor(nx_psf/2.0))    # col_minus
        col_p = int(np.ceil(nx_psf/2.0))     # col_plus

        sign = -1.
        if flipx:
            sign = 1.

        # a blank img array of write model PSFs into
        whiteboard = np.zeros((ny,nx))
        models = []
        #print(self.input_psfs.shape)
        for pa, wv in zip(pas, wvs):
            # grab PSF given wavelength
            wv_index = [spec.find_nearest(self.input_psfs_wvs,wv)[1]]

            # find center of psf
            # to reduce calculation of sin and cos, see if it has already been calculated before
            recalculate_trig = False
            if pa not in self.psf_centx_notscaled:
                recalculate_trig = True
            else:
                #print(self.psf_centx_notscaled[pa],pa)
                if pa_fk != self.curr_pa_fk[pa] or sep_fk != self.curr_sep_fk[pa]:
                    recalculate_trig = True
            if recalculate_trig: # we could actually store the values for the different pas too...
                self.psf_centx_notscaled[pa] = sep_fk * np.cos(np.radians(90. - sign*pa_fk - pa))
                self.psf_centy_notscaled[pa] = sep_fk * np.sin(np.radians(90. - sign*pa_fk - pa))
                self.curr_pa_fk[pa] = pa_fk
                self.curr_sep_fk[pa] = sep_fk

            psf_centx = (ref_wv/wv) * self.psf_centx_notscaled[pa]
            psf_centy = (ref_wv/wv) * self.psf_centy_notscaled[pa]

            # create a coordinate system for the image that is with respect to the model PSF
            # round to nearest pixel and add offset for center
            l = int(round(psf_centx + ref_center[0]))
            k = int(round(psf_centy + ref_center[1]))
            # recenter coordinate system about the location of the planet
            x_vec_stamp_centered = x_grid[0, (l-col_m):(l+col_p)]-psf_centx
            y_vec_stamp_centered = y_grid[(k-row_m):(k+row_p), 0]-psf_centy
            # rescale to account for the align and scaling of the refernce PSFs
            # e.g. for longer wvs, the PSF has shrunk, so we need to shrink the coordinate system
            x_vec_stamp_centered /= (ref_wv/wv)
            y_vec_stamp_centered /= (ref_wv/wv)

            # use intepolation spline to generate a model PSF and write to temp img
            whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = \
                    self.psfs_func_list[wv_index[0]](x_vec_stamp_centered,y_vec_stamp_centered).transpose()

            # write model img to output (segment is collapsed in x/y so need to reshape)
            whiteboard.shape = [input_img_shape[0] * input_img_shape[1]]
            segment_with_model = copy(whiteboard[section_ind])
            whiteboard.shape = [input_img_shape[0],input_img_shape[1]]

            models.append(segment_with_model)

            # create a canvas to place the new PSF in the sector on
            if 0:
                blackboard = np.zeros((ny,nx))
                blackboard.shape = [input_img_shape[0] * input_img_shape[1]]
                blackboard[section_ind] = segment_with_model
                blackboard.shape = [input_img_shape[0],input_img_shape[1]]
                plt.figure(1)
                plt.subplot(1,2,1)
                im = plt.imshow(whiteboard)
                plt.colorbar(im)
                plt.subplot(1,2,2)
                im = plt.imshow(blackboard)
                plt.colorbar(im)
                plt.show()

            whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = 0.0

        return np.array(models)


