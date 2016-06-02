        #wv_index_list = [self.input_psfs_wvs.index(wv) for wv in wvs]
__author__ = 'jruffio'
import multiprocessing as mp
import ctypes

import numpy as np
import pyklip.spectra_management as specmanage
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
    def __init__(self, inputs_shape,numbasis, input_psfs,input_psfs_wvs,# flux_conversion,
                 spectrallib = None, # Input spectra should be in flux not in contrast. The transmission is accounted for.
                 mute = False,
                 star_type = None,
                 filter = None,
                 save_per_sector = None,
                 datatype="float",
                 fakes_sepPa_list = None,
                 disable_FM = None):
        '''

        :param inputs_shape:
        :param numbasis:
        :param input_psfs:
        :param input_psfs_wvs:
        :param spectrallib:
        :param mute:
        :param star_type: String containing the spectral type of the star. 'A5','F4',... Assume type V star.
                        If None, the spectrum is assumed to be in contrast units.
        :param filter:
        :param save_per_sector:
        :param datatype:
        :param fakes_sepPa_list:
        :return:
        '''
        # allocate super class
        super(MatchedFilter, self).__init__(inputs_shape, np.array(numbasis))

        if datatype=="double":
            self.mp_data_type = ctypes.c_double
            self.np_data_type = float
        elif datatype=="float":
            self.mp_data_type = ctypes.c_float
            self.np_data_type = np.float32

        if save_per_sector is not None:
            self.fmout_dir = save_per_sector
            self.save_fmout = True
        else:
            self.save_fmout = False

        self.N_numbasis =  np.size(numbasis)
        self.ny = self.inputs_shape[1]
        self.nx = self.inputs_shape[2]
        self.N_frames = self.inputs_shape[0]

        if filter is None:
            filter = "H"

        self.fakes_sepPa_list = fakes_sepPa_list
        if disable_FM is None:
            self.disable_FM = False
        else:
            self.disable_FM = disable_FM

        self.inputs_shape = self.inputs_shape
        if spectrallib is not None:
            self.spectrallib = spectrallib
        else:
            spectra_folder = os.path.dirname(os.path.abspath(specmanage.__file__)) + os.sep + "spectra" + os.sep
            spectra_files = [spectra_folder + "t650g18nc.flx"]
            self.spectrallib = [specmanage.get_planet_spectrum(filename, filter)[1] for filename in spectra_files]

        self.N_spectra = len(self.spectrallib)


        self.input_psfs_wvs = list(np.array(input_psfs_wvs,dtype=self.np_data_type))
        self.nl = np.size(input_psfs_wvs)
        #self.flux_conversion = flux_conversion
        self.input_psfs = input_psfs
        # Make sure the total flux of each PSF is unity for all wavelengths
        # So the peak value won't be unity.
        self.sat_spot_spec = np.nanmax(self.input_psfs,axis=(1,2))
        self.aper_over_peak_ratio = np.zeros(37)
        for l_id in range(self.input_psfs.shape[0]):
            self.aper_over_peak_ratio[l_id] = np.nansum(self.input_psfs[l_id,:,:])/self.sat_spot_spec[l_id]
            self.input_psfs[l_id,:,:] = self.input_psfs[l_id,:,:]/np.nansum(self.input_psfs[l_id,:,:])
            #self.input_psfs[l_id,:,:] /= self.sat_spot_spec[l_id]

        self.nl, self.ny_psf, self.nx_psf =  self.input_psfs.shape

        # create bounds for PSF stamp size
        self.row_m = np.floor(self.ny_psf/2.0)    # row_minus
        self.row_p = np.ceil(self.ny_psf/2.0)     # row_plus
        self.col_m = np.floor(self.nx_psf/2.0)    # col_minus
        self.col_p = np.ceil(self.nx_psf/2.0)     # col_plus

        # TODO: calibrate to contrast units
        # Correct spectra for transmission

        if star_type is None:
            # Spectrum is in contrast units
            self.spectrallib = [self.sat_spot_spec*self.aper_over_peak_ratio*spectrum for spectrum in self.spectrallib]
        else:
            self.spectrallib = [self.sat_spot_spec*self.aper_over_peak_ratio*spectrum/(specmanage.get_star_spectrum(filter, star_type=star_type)[1]) for spectrum in self.spectrallib]
        # Normalize the spectra to unit broadband flux
        self.spectrallib = [spectrum/np.sum(spectrum) for spectrum in self.spectrallib]
        # import matplotlib.pyplot as plt
        # plt.plot(self.spectrallib[0]/np.sum(self.spectrallib[0]))
        # plt.plot(specmanage.get_star_spectrum(filter, star_type=star_type)[1]/np.sum(specmanage.get_star_spectrum(filter, star_type=star_type)[1]))
        # #plt.plot(spectrum/np.sum(spectrum))
        # plt.plot(self.sat_spot_spec/np.sum(self.sat_spot_spec))
        # plt.show()
        self.fake_contrast = 1.0#10**-5 # ratio of flux of the planet/flux of the star (broad band flux)

        self.psf_centx_notscaled = {}
        self.psf_centy_notscaled = {}
        self.curr_pa_fk = {}
        self.curr_sep_fk = {}

        self.nearestNeigh_PSF_interp = 0
        # nearestNeigh_PSF_interp == 2 is actually not really accurate since it doesn't include the proper scaling of the PSF with wavelength.
        self.sub_sampling_coef = 10

        numwv,ny_psf,nx_psf =  self.input_psfs.shape
        x_psf_grid, y_psf_grid = np.meshgrid(np.arange(ny_psf* 1.)-ny_psf/2, np.arange(nx_psf * 1.)-nx_psf/2)
        if  self.nearestNeigh_PSF_interp==1:
            x_psf_HD_vec = np.linspace(0,nx_psf-1.,nx_psf*self.sub_sampling_coef)-nx_psf/2
            y_psf_HD_vec = np.linspace(0,ny_psf-1.,ny_psf*self.sub_sampling_coef)-ny_psf/2
            x_psf_HD_grid, y_psf_HD_grid = np.meshgrid(x_psf_HD_vec, y_psf_HD_vec)
        elif self.nearestNeigh_PSF_interp ==2:
            row_ind_vec = np.arange(self.sub_sampling_coef,(self.ny_psf+1)*self.sub_sampling_coef,self.sub_sampling_coef)
            col_ind_vec = np.arange(self.sub_sampling_coef,(self.nx_psf+1)*self.sub_sampling_coef,self.sub_sampling_coef)
            self.row_ind_grid ,self.col_ind_grid = np.meshgrid(row_ind_vec,col_ind_vec)
        psfs_func_list = []
        for wv_index in range(numwv):
            model_psf = self.input_psfs[wv_index, :, :]
            #psfs_func_list.append(interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5))
            psf_func = interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5)
            if  self.nearestNeigh_PSF_interp == 1:
                eval_psf_func = psf_func(x_psf_HD_vec,y_psf_HD_vec)
                psf_func_near = interpolate.RegularGridInterpolator((x_psf_HD_vec,y_psf_HD_vec),eval_psf_func,
                                                                          method="nearest",bounds_error=False,fill_value=0.0)
                psfs_func_list.append(psf_func_near)
            elif self.nearestNeigh_PSF_interp == 2:
                x_psf_HD_vec = np.linspace(-1,nx_psf+1.-1./self.sub_sampling_coef,(nx_psf+2)*self.sub_sampling_coef)-nx_psf/2
                y_psf_HD_vec = np.linspace(-1,ny_psf+1.-1./self.sub_sampling_coef,(ny_psf+2)*self.sub_sampling_coef)-ny_psf/2

                eval_psf_func = psf_func(x_psf_HD_vec,y_psf_HD_vec)
                psfs_func_list.append(eval_psf_func)
                #plt.figure(1)
                #plt.imshow(eval_psf_func,interpolation="nearest")
                #plt.show()
            else:
                psfs_func_list.append(psf_func)

            # plt.figure(1)
            # print(zip(x_psf_HD_grid.ravel(), y_psf_HD_grid.ravel())[0])
            # plt.subplot(1,3,1)
            # plt.imshow(np.squeeze(self.input_psfs[10, :, :]),interpolation="nearest")
            # plt.subplot(1,3,2)
            # plt.imshow(eval_psf_func,interpolation="nearest")
            # plt.subplot(1,3,3)
            # plt.imshow(psf_func_near(zip(x_psf_HD_grid.ravel(), y_psf_HD_grid.ravel())).reshape((ny_psf*10,nx_psf*10)),interpolation = "nearest")
            # plt.show()

        self.psfs_func_list = psfs_func_list

        ny_PSF,nx_PSF = input_psfs.shape[1:]
        stamp_PSF_x_grid, stamp_PSF_y_grid = np.meshgrid(np.arange(0,nx_PSF,1)-nx_PSF/2,np.arange(0,ny_PSF,1)-ny_PSF/2)
        self.stamp_PSF_mask = np.ones((ny_PSF,nx_PSF))
        r_PSF_stamp = abs((stamp_PSF_x_grid) +(stamp_PSF_y_grid)*1j)
        self.stamp_PSF_mask[np.where(r_PSF_stamp < 3.)] = np.nan

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
        """Allocates shared memory for the output of the shared memory


        Args:
            output_img_shape: shape of output image (usually N,y,x,b)

        Returns:
            fmout: mp.array to store auxilliary data in
            fmout_shape: shape of auxilliary array

        """

        # The 3 is for saving the different term of the matched filter
        # 0: dot product
        # 1: square of the norm of the model
        # 2: square of the norm of the image
        fmout_size = 3*self.N_spectra*self.N_numbasis*self.N_frames*self.ny*self.nx
        fmout = mp.Array(self.mp_data_type, fmout_size)
        fmout_shape = (3,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)

        return fmout, fmout_shape

    def skipSection(self,radstart, radend, phistart, phiend):
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

        # print(radstart, radend, phistart, phiend)
        margin_sep = np.sqrt(2)/2
        margin_phi = np.sqrt(2)/(2*radstart)
        # print(margin_sep,margin_phi)
        if self.fakes_sepPa_list is not None:
            skipSectionBool = True
            for sep_it,pa_it in self.fakes_sepPa_list:
                #print(sep_it,pa_it,(pa_it%360)/180.*np.pi)
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
                # if (radstart-margin_sep<=sep_it<=radend+margin_sep) and (phistart-margin_phi<=(pa_it%360)/180.*np.pi<=phiend+margin_phi):
                #     skipSectionBool = False
                #     print("coucou")
                #     #print(radstart, radend, phistart, phiend)
                #     print(sep_it,pa_it,(pa_it%360)/180.*np.pi)
                    break
        else:
            skipSectionBool = False

        return skipSectionBool


    def fm_from_eigen(self, klmodes=None, evals=None, evecs=None, input_img_shape=None, input_img_num=None, ref_psfs_indicies=None, section_ind=None, aligned_imgs=None, pas=None,
                     wvs=None, radstart=None, radend=None, phistart=None, phiend=None, padding=None,IOWA = None, ref_center=None,
                     parang=None, ref_wv=None, numbasis=None, fmout=None, perturbmag=None,klipped=None, **kwargs):
        """

        Args:
            klmodes: unpertrubed KL modes
            evals: eigenvalues of the covariance matrix that generated the KL modes in ascending order
                   (lambda_0 is the 0 index) (shape of [nummaxKL])
            evecs: corresponding eigenvectors (shape of [p, nummaxKL])
            input_image_shape: 2-D shape of inpt images ([ysize, xsize])
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
        # for k in range(aligned_imgs.shape[0]):
        #     blackboard1 = np.zeros((self.ny,self.nx)) + np.nan
        #     #print(section_ind)
        #     plt.figure(1)
        #     # plt.subplot(1,2,1)
        #     blackboard1.shape = [input_img_shape[0] * input_img_shape[1]]
        #     tmp = copy(refs[k,:])
        #     #tmp[np.where(np.isnan(tmp))] = 1000
        #     blackboard1[section_ind] = tmp
        #     blackboard1.shape = [input_img_shape[0],input_img_shape[1]]
        #     plt.imshow(blackboard1,interpolation="nearest")
        #     plt.colorbar()
        #     # plt.subplot(1,2,2)
        #     # coucou = copy(aligned_imgs[ref_psfs_indicies, :])
        #     # print(coucou.shape)
        #     # coucou.shape = [coucou.shape[0],input_img_shape[0],input_img_shape[1]]
        #     # plt.imshow(coucou[k,:,:],interpolation="nearest")
        #     plt.show()


        # Calculate the PA,sep 2D map
        x_grid, y_grid = np.meshgrid(np.arange(self.nx * 1.0)- ref_center[0], np.arange(self.ny * 1.0)- ref_center[1])
        x_grid=x_grid.astype(self.np_data_type)
        y_grid=y_grid.astype(self.np_data_type)
        r_grid = np.sqrt((x_grid)**2 + (y_grid)**2)
        pa_grid = np.arctan2( -x_grid,y_grid) % (2.0 * np.pi)
        paend= ((2*np.pi-phistart +np.pi/2)% (2.0 * np.pi))
        pastart = ((2*np.pi-phiend +np.pi/2)% (2.0 * np.pi))
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
                if min_dist < np.sqrt(2)/2:
                    r_list_tmp.append(r_list[min_id])
                    pa_list_tmp.append(pa_list[min_id])
                    row_id_list_tmp.append(row_id_list[min_id])
                    col_id_list_tmp.append(col_id_list[min_id])
            r_list = r_list_tmp
            pa_list = pa_list_tmp
            row_id_list = row_id_list_tmp
            col_id_list = col_id_list_tmp
            # print(r_list,np.rad2deg(pa_list),row_id_list,col_id_list)



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
                #print(sep_fk,pa_fk,r_grid[row_id,col_id],pa_grid[row_id,col_id]/np.pi*180)

                # 1/ Inject a fake at one pa and sep in the science image
                if self.nearestNeigh_PSF_interp != 0:
                    # This is a test with a nearest neighbor interpolation but it is not quicker.
                    # Therefore to be ignored.
                    model_sci,mask = self.generate_model_sci_nearestNeigh(input_img_shape, section_ind, parang, ref_wv, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv,sep_fk,pa_fk)#32.,170.)#sep_fk,pa_fk)
                else:
                    model_sci,mask = self.generate_model_sci(input_img_shape, section_ind, parang, ref_wv, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv,sep_fk,pa_fk)#32.,170.)#sep_fk,pa_fk)

                # self.spectrallib[spec_id] is one of the input template spectrum
                #   It is normalized to unit broad band flux (sum(self.spectrallib[spec_id])=1)  in DN space.
                # self.fake_contrast = 1.0 right now. It used to be 10^-5 for random reasons.
                model_sci = model_sci*self.spectrallib[spec_id][self.input_psfs_wvs.index(ref_wv)]*self.fake_contrast
                where_fk = np.where(mask>=1)[0]
                where_background = np.where(mask==2)[0]

                # 2/ Inject the corresponding planets at the same PA and sep in the reference images remembering that the
                # references rotate.
                if not self.disable_FM:
                    if self.nearestNeigh_PSF_interp != 0:
                        models_ref = self.generate_models_nearestNeigh(input_img_shape, section_ind, pas, wvs, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv,sep_fk,pa_fk)#32.,170.)#,sep_fk,pa_fk)
                    else:
                        models_ref = self.generate_models(input_img_shape, section_ind, pas, wvs, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv,sep_fk,pa_fk)#32.,170.)#,sep_fk,pa_fk)

                    # Calculate the spectra to determine the flux of each model reference PSF
                    # self.spectrallib[spec_id] is one of the input template spectrum
                    #   It is normalized to unit broad band flux (sum(self.spectrallib[spec_id])=1)
                    # self.fake_contrast = 1.0 right now. It used to be 10^-5 for random reasons.
                    input_spectrum =  self.spectrallib[spec_id]
                    input_spectrum = np.ravel(np.tile(input_spectrum,(1, self.N_frames/self.nl)))*self.fake_contrast
                    input_spectrum = input_spectrum[ref_psfs_indicies]
                    models_ref = models_ref * input_spectrum[:, None]

                    # 3/ Calculate the perturbation of the KL modes
                    # using original Kl modes and reference models, compute the perturbed KL modes.
                    # Spectrum is already in the model, that's why we use perturb_specIncluded(). (Much faster)
                    #print(np.sum(np.isnan(evals)),np.sum(np.isnan(evecs)),np.sum(np.isnan(klmodes)),np.sum(np.isnan(refs)),np.sum(np.isnan(models_ref)))
                    delta_KL = fm.perturb_specIncluded(evals, evecs, klmodes, refs, models_ref)
                    #print(np.sum(np.isnan(delta_KL)))

                    # 4/ Calculate the FM: calculate postklip_psf using delta_KL
                    # postklip_psf has unit broadband flux
                    postklip_psf, oversubtraction, selfsubtraction = fm.calculate_fm(delta_KL, klmodes, numbasis, sci, model_sci, inputflux=None)
                else:
                    #if one doesn't want the FM
                    if np.size(numbasis) == 1:
                        postklip_psf = model_sci[None,:]
                    else:
                        postklip_psf = model_sci

                # 5/ Calculate dot product (matched filter)
                # fmout_shape = (3,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)
                # First dimension details:
                # 0: dot product
                # 1: square of the norm of the model
                # 2: square of the norm of the image
                sky = np.mean(klipped[where_background,N_KL_id])
                #print(sky)
                # Subtract local sky background to the klipped image
                klipped_sub = klipped[where_fk,N_KL_id]-sky
                fmout[0,spec_id,N_KL_id,input_img_num,row_id,col_id] = np.sum(klipped_sub*postklip_psf[N_KL_id,where_fk])
                fmout[1,spec_id,N_KL_id,input_img_num,row_id,col_id] = np.sum(postklip_psf[N_KL_id,where_fk]*postklip_psf[N_KL_id,where_fk])
                fmout[2,spec_id,N_KL_id,input_img_num,row_id,col_id] = np.sum(klipped_sub*klipped_sub)
                #print(fmout[0,spec_id,N_KL_id,input_img_num,row_id,col_id],fmout[1,spec_id,N_KL_id,input_img_num,row_id,col_id],fmout[2,spec_id,N_KL_id,input_img_num,row_id,col_id])

                # Plot for debug only
                if 0:

                    #if 0:
                    blackboard1 = np.zeros((self.ny,self.nx))
                    blackboard2 = np.zeros((self.ny,self.nx))
                    blackboard3 = np.zeros((self.ny,self.nx))
                    #print(section_ind)
                    plt.figure(1)
                    plt.subplot(1,3,1)
                    blackboard1.shape = [input_img_shape[0] * input_img_shape[1]]
                    blackboard1[section_ind] = mask
                    blackboard1.shape = [input_img_shape[0],input_img_shape[1]]
                    plt.imshow(blackboard1)
                    plt.colorbar()
                    plt.subplot(1,3,2)
                    blackboard2.shape = [input_img_shape[0] * input_img_shape[1]]
                    blackboard2[section_ind[0][where_fk]] = klipped[where_fk,N_KL_id]
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
        Does some forward modelling at the end of a sector after all images have been klipped for that sector.

        """

        #fmout_shape = (3,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)

        if self.save_fmout:
            hdu = pyfits.PrimaryHDU(fmout)
            hdulist = pyfits.HDUList([hdu])
            hdulist.writeto(self.fmout_dir,clobber=True)

        if 0:
            matched_filter_maps = np.nansum(fmout[0,:,:,:,:,:],axis=2)
            model_square_norm_maps = np.nansum(fmout[1,:,:,:,:,:],axis=2)
            image_square_norm_maps = np.nansum(fmout[2,:,:,:,:,:],axis=2)
            metric = matched_filter_maps/np.sqrt(model_square_norm_maps*image_square_norm_maps)
            metric = np.squeeze(metric[0,0,:,:])
                    #fmout_shape = (3,self.N_spectra,self.N_numbasis,self.N_frames,self.ny,self.nx)

            print(np.nanargmax(metric),np.nanmax(metric))

            plt.figure(1)
            plt.imshow(metric,interpolation="nearest")
            plt.colorbar()
            plt.show()

        return



    def generate_model_sci(self, input_img_shape, section_ind, pa, wv, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv,sep_fk,pa_fk):
        """
        Generate model PSFs at the correct location of this segment for each image denoated by its wv and parallactic angle

        Args:
            pas: array of N parallactic angles corresponding to N images [degrees]
            wvs: array of N wavelengths of those images
            radstart: radius of start of segment
            radend: radius of end of segment
            phistart: azimuthal start of segment [radians]
            phiend: azimuthal end of segment [radians]
            padding: amount of padding on each side of sector
            ref_center: center of image
            parang: parallactic angle of input image [DEGREES]
            ref_wv: wavelength of science image

        Return:
            models: array of size (N, p) where p is the number of pixels in the segment
        """
        # create some parameters for a blank canvas to draw psfs on
        nx = input_img_shape[1]
        ny = input_img_shape[0]
        x_grid, y_grid = np.meshgrid(np.arange(nx * 1.)-ref_center[0], np.arange(ny * 1.)-ref_center[1])

        numwv, ny_psf, nx_psf =  self.input_psfs.shape

        # create bounds for PSF stamp size
        row_m = np.floor(ny_psf/2.0)    # row_minus
        row_p = np.ceil(ny_psf/2.0)     # row_plus
        col_m = np.floor(nx_psf/2.0)    # col_minus
        col_p = np.ceil(nx_psf/2.0)     # col_plus

        # a blank img array of write model PSFs into
        whiteboard = np.zeros((ny,nx))
        #print(self.input_psfs.shape)
        #print(self.pa,self.sep)
        #print(pa,wv)
        # grab PSF given wavelength
        #print(self.input_psfs_wvs,wv)
        wv_index = [self.input_psfs_wvs.index(wv)]#np.where(wv == self.input_psfs_wvs)[0]
        #print(self.input_psfs_wvs.index(wv))
        #print(np.where(wv == np.array(self.input_psfs_wvs))[0])
        #model_psf = self.input_psfs[wv_index[0], :, :] #* self.flux_conversion * self.spectrallib[0][wv_index] * self.dflux

        # find center of psf
        # to reduce calculation of sin and cos, see if it has already been calculated before

        recalculate_trig = False
        if pa not in self.psf_centx_notscaled:
            recalculate_trig = True
        else:
            if pa_fk != self.curr_pa_fk[pa] or sep_fk != self.curr_sep_fk[pa]:
                recalculate_trig = True
        if recalculate_trig: # we could actually store the values for the different pas too...
            self.psf_centx_notscaled[pa] = sep_fk * np.cos(np.radians(90. - pa_fk - pa))
            self.psf_centy_notscaled[pa] = sep_fk * np.sin(np.radians(90. - pa_fk - pa))
            self.curr_pa_fk[pa] = pa_fk
            self.curr_sep_fk[pa] = sep_fk

        psf_centx = (ref_wv/wv) * self.psf_centx_notscaled[pa]
        psf_centy = (ref_wv/wv) * self.psf_centy_notscaled[pa]

        # create a coordinate system for the image that is with respect to the model PSF
        # round to nearest pixel and add offset for center
        l = round(psf_centx + ref_center[0])
        k = round(psf_centy + ref_center[1])
        # recenter coordinate system about the location of the planet
        x_vec_stamp_centered = x_grid[0, (l-col_m):(l+col_p)]-psf_centx
        y_vec_stamp_centered = y_grid[(k-row_m):(k+row_p), 0]-psf_centy
        # rescale to account for the align and scaling of the refernce PSFs
        # e.g. for longer wvs, the PSF has shrunk, so we need to shrink the coordinate system
        x_vec_stamp_centered /= (ref_wv/wv)
        y_vec_stamp_centered /= (ref_wv/wv)

        # use intepolation spline to generate a model PSF and write to temp img
        whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = \
                self.psfs_func_list[wv_index[0]](y_vec_stamp_centered,x_vec_stamp_centered)

        # write model img to output (segment is collapsed in x/y so need to reshape)
        whiteboard.shape = [input_img_shape[0] * input_img_shape[1]]
        segment_with_model = copy(whiteboard[section_ind])
        whiteboard.shape = [input_img_shape[0],input_img_shape[1]]


        whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = 1
        whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)][np.where(np.isfinite(self.stamp_PSF_mask))]=2
        whiteboard.shape = [input_img_shape[0] * input_img_shape[1]]
        mask = whiteboard[section_ind]

        # create a canvas to place the new PSF in the sector on
        if 0:#np.size(np.where(mask==2)[0])==0:
            whiteboard.shape = (input_img_shape[0], input_img_shape[1])
            print(whiteboard.shape)
            print(section_ind)
            blackboard = np.zeros((ny,nx))
            blackboard.shape = [input_img_shape[0] * input_img_shape[1]]
            blackboard[section_ind] = 1#segment_with_model
            blackboard.shape = [input_img_shape[0],input_img_shape[1]]
            plt.figure(1)
            plt.subplot(1,2,1)
            im = plt.imshow(whiteboard)
            plt.colorbar(im)
            plt.subplot(1,2,2)
            im = plt.imshow(blackboard)
            plt.colorbar(im)
            plt.show()

        return segment_with_model,mask

    def generate_model_sci_nearestNeigh(self, input_img_shape, section_ind, pa, wv, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv,sep_fk,pa_fk):
        """
        Generate model PSFs at the correct location of this segment for each image denoated by its wv and parallactic angle

        Args:
            pas: array of N parallactic angles corresponding to N images [degrees]
            wvs: array of N wavelengths of those images
            radstart: radius of start of segment
            radend: radius of end of segment
            phistart: azimuthal start of segment [radians]
            phiend: azimuthal end of segment [radians]
            padding: amount of padding on each side of sector
            ref_center: center of image
            parang: parallactic angle of input image [DEGREES]
            ref_wv: wavelength of science image

        Return:
            models: array of size (N, p) where p is the number of pixels in the segment
        """
        # create some parameters for a blank canvas to draw psfs on
        nx = input_img_shape[1]
        ny = input_img_shape[0]
        x_grid, y_grid = np.meshgrid(np.arange(nx * 1.)-ref_center[0], np.arange(ny * 1.)-ref_center[1])

        numwv, ny_psf, nx_psf =  self.input_psfs.shape

        # create bounds for PSF stamp size
        row_m = np.floor(ny_psf/2.0)    # row_minus
        row_p = np.ceil(ny_psf/2.0)     # row_plus
        col_m = np.floor(nx_psf/2.0)    # col_minus
        col_p = np.ceil(nx_psf/2.0)     # col_plus

        # a blank img array of write model PSFs into
        whiteboard = np.zeros((ny,nx))
        #print(self.input_psfs.shape)
        #print(self.pa,self.sep)
        #print(pa,wv)
        #print(self.input_psfs_wvs.index(wv))
        #print(np.where(wv == np.array(self.input_psfs_wvs))[0])
        #model_psf = self.input_psfs[wv_index[0], :, :] #* self.flux_conversion * self.spectrallib[0][wv_index] * self.dflux

        # find center of psf
        # to reduce calculation of sin and cos, see if it has already been calculated before

        recalculate_trig = False
        if pa not in self.psf_centx_notscaled:
            recalculate_trig = True
        else:
            if pa_fk != self.curr_pa_fk[pa] or sep_fk != self.curr_sep_fk[pa]:
                recalculate_trig = True
        if recalculate_trig: # we could actually store the values for the different pas too...
            self.psf_centx_notscaled[pa] = sep_fk * np.cos(np.radians(90. - pa_fk - pa))
            self.psf_centy_notscaled[pa] = sep_fk * np.sin(np.radians(90. - pa_fk - pa))
            self.curr_pa_fk[pa] = pa_fk
            self.curr_sep_fk[pa] = sep_fk

        psf_centx = (ref_wv/wv) * self.psf_centx_notscaled[pa]
        psf_centy = (ref_wv/wv) * self.psf_centy_notscaled[pa]

        # create a coordinate system for the image that is with respect to the model PSF
        # round to nearest pixel and add offset for center
        l = round(psf_centx + ref_center[0])
        k = round(psf_centy + ref_center[1])
        # recenter coordinate system about the location of the planet
        x_stamp_centered = x_grid[(k-row_m):(k+row_p), (l-col_m):(l+col_p)]-psf_centx
        y_stamp_centered = y_grid[(k-row_m):(k+row_p), (l-col_m):(l+col_p)]-psf_centy
        # rescale to account for the align and scaling of the refernce PSFs
        # e.g. for longer wvs, the PSF has shrunk, so we need to shrink the coordinate system
        x_stamp_centered /= (ref_wv/wv)
        y_stamp_centered /= (ref_wv/wv)

        if self.nearestNeigh_PSF_interp == 1:
            # use intepolation spline to generate a model PSF and write to temp img
            tmp = self.psfs_func_list[self.input_psfs_wvs.index(wv)](zip(x_stamp_centered.ravel(),y_stamp_centered.ravel()))
            whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = tmp.reshape((ny_psf, nx_psf))
        elif self.nearestNeigh_PSF_interp == 2:
            high_res_psf = self.psfs_func_list[self.input_psfs_wvs.index(wv)]
            deltax = int(round(-(psf_centx + ref_center[0]-l)*self.sub_sampling_coef)) #/(ref_wv/wv)
            deltay = int(round(-(psf_centy + ref_center[1]-k)*self.sub_sampling_coef)) #/(ref_wv/wv)
            row_ind_vec = np.arange(self.sub_sampling_coef+deltay,(self.ny_psf+1)*self.sub_sampling_coef+deltay,self.sub_sampling_coef)
            col_ind_vec = np.arange(self.sub_sampling_coef+deltax,(self.nx_psf+1)*self.sub_sampling_coef+deltax,self.sub_sampling_coef)
            row_ind_grid ,col_ind_grid = np.meshgrid(row_ind_vec,col_ind_vec)
            whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = high_res_psf[row_ind_grid.ravel(),col_ind_grid.ravel()].reshape((self.ny_psf,self.nx_psf))
            # print(row_ind_grid)
            # print(col_ind_vec.shape)
            # plt.figure(1)
            # plt.imshow(high_res_psf[row_ind_grid.ravel(),col_ind_grid.ravel()].reshape((self.ny_psf,self.nx_psf)),interpolation = "nearest")
            # plt.show()


        # write model img to output (segment is collapsed in x/y so need to reshape)
        whiteboard.shape = [input_img_shape[0] * input_img_shape[1]]
        segment_with_model = copy(whiteboard[section_ind])
        whiteboard.shape = [input_img_shape[0],input_img_shape[1]]

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

        whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = 1
        whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)][np.where(np.isfinite(self.stamp_PSF_mask))]=2
        whiteboard.shape = [input_img_shape[0] * input_img_shape[1]]
        mask = whiteboard[section_ind]

        return segment_with_model,mask

    def generate_models(self, input_img_shape, section_ind, pas, wvs, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv,sep_fk,pa_fk):
        """
        Generate model PSFs at the correct location of this segment for each image denoated by its wv and parallactic angle

        Args:
            pas: array of N parallactic angles corresponding to N images [degrees]
            wvs: array of N wavelengths of those images
            radstart: radius of start of segment
            radend: radius of end of segment
            phistart: azimuthal start of segment [radians]
            phiend: azimuthal end of segment [radians]
            padding: amount of padding on each side of sector
            ref_center: center of image
            parang: parallactic angle of input image [DEGREES]
            ref_wv: wavelength of science image

        Return:
            models: array of size (N, p) where p is the number of pixels in the segment
        """
        # create some parameters for a blank canvas to draw psfs on
        nx = input_img_shape[1]
        ny = input_img_shape[0]
        x_grid, y_grid = np.meshgrid(np.arange(nx * 1.)-ref_center[0], np.arange(ny * 1.)-ref_center[1])

        numwv, ny_psf, nx_psf =  self.input_psfs.shape

        # create bounds for PSF stamp size
        row_m = np.floor(ny_psf/2.0)    # row_minus
        row_p = np.ceil(ny_psf/2.0)     # row_plus
        col_m = np.floor(nx_psf/2.0)    # col_minus
        col_p = np.ceil(nx_psf/2.0)     # col_plus

        # a blank img array of write model PSFs into
        whiteboard = np.zeros((ny,nx))
        models = []
        #print(self.input_psfs.shape)
        for pa, wv in zip(pas, wvs):
            #print(self.pa,self.sep)
            #print(pa,wv)
            # grab PSF given wavelength
            #print(self.input_psfs_wvs)
            #print(wv)
            wv_index = [self.input_psfs_wvs.index(wv)]#np.where(wv == self.input_psfs_wvs)[0]
            #print(wv_index)
            #model_psf = self.input_psfs[wv_index[0], :, :] #* self.flux_conversion * self.spectrallib[0][wv_index] * self.dflux

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
                self.psf_centx_notscaled[pa] = sep_fk * np.cos(np.radians(90. - pa_fk - pa))
                self.psf_centy_notscaled[pa] = sep_fk * np.sin(np.radians(90. - pa_fk - pa))
                self.curr_pa_fk[pa] = pa_fk
                self.curr_sep_fk[pa] = sep_fk

            psf_centx = (ref_wv/wv) * self.psf_centx_notscaled[pa]
            psf_centy = (ref_wv/wv) * self.psf_centy_notscaled[pa]

            # create a coordinate system for the image that is with respect to the model PSF
            # round to nearest pixel and add offset for center
            l = round(psf_centx + ref_center[0])
            k = round(psf_centy + ref_center[1])
            # recenter coordinate system about the location of the planet
            x_vec_stamp_centered = x_grid[0, (l-col_m):(l+col_p)]-psf_centx
            y_vec_stamp_centered = y_grid[(k-row_m):(k+row_p), 0]-psf_centy
            # rescale to account for the align and scaling of the refernce PSFs
            # e.g. for longer wvs, the PSF has shrunk, so we need to shrink the coordinate system
            x_vec_stamp_centered /= (ref_wv/wv)
            y_vec_stamp_centered /= (ref_wv/wv)

            # use intepolation spline to generate a model PSF and write to temp img
            whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = \
                    self.psfs_func_list[wv_index[0]](y_vec_stamp_centered,x_vec_stamp_centered)

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


    def generate_models_nearestNeigh(self, input_img_shape, section_ind, pas, wvs, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv,sep_fk,pa_fk):
        """
        Generate model PSFs at the correct location of this segment for each image denoated by its wv and parallactic angle

        Args:
            pas: array of N parallactic angles corresponding to N images [degrees]
            wvs: array of N wavelengths of those images
            radstart: radius of start of segment
            radend: radius of end of segment
            phistart: azimuthal start of segment [radians]
            phiend: azimuthal end of segment [radians]
            padding: amount of padding on each side of sector
            ref_center: center of image
            parang: parallactic angle of input image [DEGREES]
            ref_wv: wavelength of science image

        Return:
            models: array of size (N, p) where p is the number of pixels in the segment
        """
        # create some parameters for a blank canvas to draw psfs on
        nx = input_img_shape[1]
        ny = input_img_shape[0]
        x_grid, y_grid = np.meshgrid(np.arange(nx * 1.)-ref_center[0], np.arange(ny * 1.)-ref_center[1])

        numwv, ny_psf, nx_psf =  self.input_psfs.shape

        # create bounds for PSF stamp size
        row_m = np.floor(ny_psf/2.0)    # row_minus
        row_p = np.ceil(ny_psf/2.0)     # row_plus
        col_m = np.floor(nx_psf/2.0)    # col_minus
        col_p = np.ceil(nx_psf/2.0)     # col_plus


        # grab PSF given wavelength
        # todo store in the object
        #input_psfs_wvs_index_dict = {wv : wv_ind for wv_ind,wv in enumerate(self.input_psfs_wvs)}

        # todo store in the object
        #pas_unique = np.unique(pas)

        #psf_centx_dict = {pa : sep_fk*np.cos(np.radians(90. - pa_fk - pa)) for pa in pas_unique}
        #psf_centy_dict = {pa : sep_fk*np.sin(np.radians(90. - pa_fk - pa)) for pa in pas_unique}
        #wv_ratio_dict = {(ref_wv,wv) : ref_wv/wv for ref_wv,wv in itertools.product(self.input_psfs_wvs,self.input_psfs_wvs)}

        if 1:
            # a blank img array of write model PSFs into
            whiteboard = np.zeros((ny,nx))
            models = []
            #print(self.input_psfs.shape)
            for pa, wv in zip(pas, wvs):
                #psf_centx = wv_ratio_dict[ref_wv,wv] * psf_centx_dict[pa]
                #psf_centy = wv_ratio_dict[ref_wv,wv] * psf_centy_dict[pa]
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
                    self.psf_centx_notscaled[pa] = sep_fk * np.cos(np.radians(90. - pa_fk - pa))
                    self.psf_centy_notscaled[pa] = sep_fk * np.sin(np.radians(90. - pa_fk - pa))
                    self.curr_pa_fk[pa] = pa_fk
                    self.curr_sep_fk[pa] = sep_fk

                psf_centx = (ref_wv/wv) * self.psf_centx_notscaled[pa]
                psf_centy = (ref_wv/wv) * self.psf_centy_notscaled[pa]

                # create a coordinate system for the image that is with respect to the model PSF
                # round to nearest pixel and add offset for center
                l = round(psf_centx + ref_center[0])
                k = round(psf_centy + ref_center[1])
                # recenter coordinate system about the location of the planet
                x_stamp_centered = x_grid[(k-row_m):(k+row_p), (l-col_m):(l+col_p)]-psf_centx
                y_stamp_centered = y_grid[(k-row_m):(k+row_p), (l-col_m):(l+col_p)]-psf_centy
                # rescale to account for the align and scaling of the refernce PSFs
                # e.g. for longer wvs, the PSF has shrunk, so we need to shrink the coordinate system
                #x_stamp_centered /= wv_ratio_dict[ref_wv,wv]
                #y_stamp_centered /= wv_ratio_dict[ref_wv,wv]
                x_stamp_centered /= (ref_wv/wv)
                y_stamp_centered /= (ref_wv/wv)

                # use intepolation spline to generate a model PSF and write to temp img
                #tmp = self.psfs_func_list[input_psfs_wvs_index_dict[wv]](zip(x_stamp_centered.ravel(),y_stamp_centered.ravel()))
                #tmp = self.psfs_func_list[self.input_psfs_wvs.index(wv)](zip(x_stamp_centered.ravel(),y_stamp_centered.ravel()))
                # tmp = self.psfs_func_list[self.input_psfs_wvs.index(wv)](zip(x_stamp_centered.ravel(),y_stamp_centered.ravel()))
                # whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = tmp.reshape((ny_psf, nx_psf))

                if self.nearestNeigh_PSF_interp == 1:
                    tmp = self.psfs_func_list[self.input_psfs_wvs.index(wv)](zip(x_stamp_centered.ravel(),y_stamp_centered.ravel()))
                    whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = tmp.reshape((ny_psf, nx_psf))
                elif self.nearestNeigh_PSF_interp == 2:
                    high_res_psf = self.psfs_func_list[self.input_psfs_wvs.index(wv)]
                    deltax = int(round(-(psf_centx + ref_center[0]-l)*self.sub_sampling_coef)) #/(ref_wv/wv)
                    deltay = int(round(-(psf_centy + ref_center[1]-k)*self.sub_sampling_coef)) #/(ref_wv/wv)
                    whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = high_res_psf[(self.row_ind_grid+deltay).ravel(),(self.col_ind_grid+deltax).ravel()].reshape((self.ny_psf,self.nx_psf))
                    # print(row_ind_grid)
                    # print(col_ind_vec.shape)
                    # plt.figure(1)
                    # plt.imshow(high_res_psf[row_ind_grid.ravel(),col_ind_grid.ravel()].reshape((self.ny_psf,self.nx_psf)),interpolation = "nearest")
                    # plt.show()

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
        #plt.figure(1)
        #plt.show()
        #return 0
