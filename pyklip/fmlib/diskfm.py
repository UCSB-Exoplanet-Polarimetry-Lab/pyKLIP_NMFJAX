import multiprocessing as mp
import ctypes
import numpy as np
import os
import copy
import pickle
import glob

from pyklip.fmlib.nofm import NoFM
import pyklip.fm as fm
from pyklip.klip import rotate



class DiskFM(NoFM):
    def __init__(self, inputs_shape, numbasis, dataset, model_disk, basis_file_pattern = 'klip-basis-', load_from_basis = False, save_basis = False, annuli = None, subsections = None, OWA = None):
        '''
        Takes an input model and runs KLIP-FM. Can be used in MCMCs by saving the basis 
        vectors. When disk is updated, FM can be run on the new disk without computing new basis
        vectors. 

        For first time, instantiate DiskFM with no save_basis and nominal model disk.
        Specify number of annuli and subsections used to save basis vectors

        '''
        super(DiskFM, self).__init__(inputs_shape, numbasis)
        self.inputs_shape = inputs_shape
        self.numbasis = numbasis
        self.maxnumbasis = max(numbasis)


        self.dataset = dataset
        self.images = dataset.input
        self.pas = dataset.PAs
        self.centers = dataset.centers
        self.wvs = dataset.wvs


        self.update_disk(model_disk)

        self.save_basis = save_basis
        self.annuli = annuli
        self.subsections = subsections

        self.klmodes_all = None
        self.evals_all = None
        self.evecs_all = None
        self.ref_psfs_indices_all = None
        self.section_ind_all = None

        self.basis_file_pattern = basis_file_pattern
        self.load_from_basis = load_from_basis



        if self.save_basis == True or load_from_basis == True:
            assert annuli is not None, "need annuli keyword to save basis"
            assert subsections is not None, "need annuli keyword to save basis"
            x, y = np.meshgrid(np.arange(inputs_shape[2] * 1.0), np.arange(inputs_shape[1] * 1.0))
            nanpix = np.where(np.isnan(dataset.input[0]))
            if OWA is None:
                OWA = np.sqrt(np.min((x[nanpix] - self.centers[0][0]) ** 2 + (y[nanpix] - self.centers[0][1]) ** 2))
            self.dr = (OWA - dataset.IWA) / annuli
            self.dphi = 2 * np.pi / subsections

        if load_from_basis is True:
            self.load_basis_files(basis_file_pattern)


    def alloc_fmout(self, output_img_shape):
        '''
        Allocates shared memory for output image and, if desired, evals and evecs    
        '''
        fmout_size = np.prod(output_img_shape)
        fmout_shape = output_img_shape
        fmout = mp.Array(ctypes.c_double, fmout_size)        
        return fmout, fmout_shape

    def fm_from_eigen(self, klmodes=None, evals=None, evecs=None, input_img_shape=None, input_img_num=None, ref_psfs_indicies=None, section_ind=None, aligned_imgs=None, pas=None,
                     wvs=None, radstart=None, radend=None, phistart=None, phiend=None, padding=None,IOWA = None, ref_center=None,
                     parang=None, ref_wv=None, numbasis=None, fmout=None, perturbmag=None, klipped=None, covar_files=None, **kwargs):
        '''
        FIXME
        '''

        sci = aligned_imgs[input_img_num, section_ind[0]]
        refs = aligned_imgs[ref_psfs_indicies, :]
        refs = refs[:, section_ind[0]]

        model_sci = self.model_disks[input_img_num, section_ind[0]]
        model_ref = self.model_disks[ref_psfs_indicies, :]
        model_ref = model_ref[:, section_ind[0]]
        
        refs_mean_sub = model_ref - np.nanmean(model_ref, axis = 1)[:, None]
        refs_mean_sub[np.where(np.isnan(refs_mean_sub))] = 0
        models_mean_sub = model_ref - np.nanmean(model_ref, axis = 1)[:, None]
        models_mean_sub[np.where(np.isnan(models_mean_sub))] = 0
        delta_KL= fm.perturb_specIncluded(evals, evecs, klmodes, refs_mean_sub, models_mean_sub, return_perturb_covar = False)
        postklip_psf, oversubtraction, selfsubtraction = fm.calculate_fm(delta_KL, klmodes, numbasis, sci, model_sci, inputflux = None)

        for thisnumbasisindex in range(np.size(numbasis)):
            fm._save_rotated_section(input_img_shape, postklip_psf[thisnumbasisindex], section_ind,
                                     fmout[input_img_num, :, :,thisnumbasisindex], None, parang,
                                     radstart, radend, phistart, phiend, padding,IOWA, ref_center, flipx=True)


        if self.save_basis is True:
            curr_rad = str(int(np.round((radstart - self.dataset.IWA) / self.dr)))
            curr_sub = str(int(np.round(phistart / self.dphi)))
            curr_im = str(input_img_num)
            if len(curr_im) < 2:
                curr_im = '0' + curr_im

            f = open(self.basis_file_pattern + 'r' + curr_rad + 's' + curr_sub + 'i' + curr_im + '.p', 'wb')
            pickle.dump(klmodes, f)
            pickle.dump(evals, f)
            pickle.dump(evecs, f)
            pickle.dump(ref_psfs_indicies, f)
            pickle.dump(section_ind, f)
            
    def fm_parallelized(self):
        '''
        Functions like klip_parallelized, but doesn't find new 
        evals and evecs. 
        '''
        assert self.klmodes_all is not None, "No evals or evecs defined"
        assert self.evecs_all is not None, "No evals or evecs defined"
        assert self.evals_all is not None, "No evals or evecs defined"
        assert self.ref_psfs_indices_all is not None, "No evals or evecs defined"
        assert self.section_ind_all is not None, "No evals or evecs defined"

        # Define phi bounds and rad bounds
        rad_bounds = [(self.dr * rad + self.IWA, self.dr * (rad + 1) + self.IWA) for rad in annuli_list]
        phi_bounds = [[self.dphi * phi_i, self.dphi_i  * (phi_i + 1)] for phi_i in range(subsections)]
        phi_bounds[-1][1] = 2. * np.pi - 0.0001

        iterator_sectors = itertoos.product(rad_bounds, phi_bounds)
        tot_sectors = len(rad_bounds) * len(phi_bounds)

        
        

        # FIXME output_imgs_shape
        fmout_data, fmout_shape = self.alloc_fmout(None)

        # tpool to fm_from_eigen

        # fm_from_eigen not parallel

        if not parallel:
            self.fm_from_eigen(self, klmodes=None, evals=None, evecs=None, input_img_shape=None, input_img_num=None, ref_psfs_indicies=None, section_ind=None, aligned_imgs=None, pas=None,
                     wvs=None, radstart=None, radend=None, phistart=None, phiend=None, padding=None,IOWA = None, ref_center=None,
                     parang=None, ref_wv=None, numbasis=None, fmout=None, perturbmag=None, klipped=None, covar_files=None, **kwargs)

        # cleanup fmout

        # fmout to numpy array
        fmout_np = None # FIXME

        return fmout_np
            


    def load_basis_files(self, basis_file_pattern):
        filenames = glob.glob(basis_file_pattern + '*.p')

        nums = [f[len(basis_file_pattern):] for f in filenames]
        rads = [n[1] for n in nums]
        subs = [n[3] for n in nums]
        imnum = [n[5:7] for n in nums]
        
        self.annuli_list = rads
        self.subs_list = subs
        self.imnum_list = imnum
        
        self.klmodes_all = []
        self.evals_all = []
        self.evecs_all = []
        self.ref_psfs_indices_all = []
        self.section_ind_all = []
        for f in filenames:
            basis_file = open(f)
            self.klmodes_all.append(pickle.load(basis_file))
            self.evals_all.append(pickle.load(basis_file))
            self.evecs_all.append(pickle.load(basis_file))
            self.ref_psfs_indices_all.append(pickle.load(basis_file))
            self.section_ind_all.append(pickle.load(basis_file))
        
        # Make flattened images for running paralellized
        self.original_imgs = mp.Array(self.mp_data_type, np.size(self.images))
        self.original_imgs_shape = self.images.shape
        self.original_imgs_np = fm._arraytonumpy(self.original_imgs, self.original_imgs_shape,dtype=self.np_data_type)
        self.original_imgs_np[:] = self.images


        # make array for recentered/rescaled image for each wavelength                               
        unique_wvs = np.unique(self.wvs)
        self.recentered_imgs = mp.Array(self.mp_data_type, np.size(self.images)*np.size(unique_wvs))
        self.recentered_imgs_shape = (np.size(unique_wvs),) + self.images.shape

        # remake the PA, wv, and center arrays as shared arrays                                            
        self.pa_imgs = mp.Array(self.mp_data_type, np.size(self.pas))
        self.pa_imgs_np = fm._arraytonumpy(self.pa_imgs,dtype=self.np_data_type)
        self.pa_imgs_np[:] = self.pas
        self.wvs_imgs = mp.Array(self.mp_data_type, np.size(self.wvs))
        self.wvs_imgs_np = fm._arraytonumpy(self.wvs_imgs,dtype=self.np_data_type)
        self.wvs_imgs_np[:] = self.wvs
        self.centers_imgs = mp.Array(self.mp_data_type, np.size(self.centers))
        self.centers_imgs_np = fm._arraytonumpy(self.centers_imgs, self.centers.shape,dtype=self.np_data_type)
        self.centers_imgs_np[:] = self.centers
        output_imgs = None
        output_imgs_numstacked = None
        output_imgs_shape = self.images.shape + self.numbasis.shape


        perturbmag, perturbmag_shape = self.alloc_perturbmag(output_imgs_shape, self.numbasis)
        
        fmout_data = None
        fmout_shape = None

        tpool = mp.Pool(processes=numthreads, initializer=fm._tpool_init,initargs=(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs, output_imgs_shape, output_imgs_numstacked, pa_imgs, wvs_imgs, centers_imgs, None, None, fmout_data, fmout_shape,perturbmag,perturbmag_shape), maxtasksperchild=50)



        fm._tpool_init(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs,output_imgs_shape, output_imgs_numstacked, pa_imgs, wvs_imgs, centers_imgs, None, None,fmout_data, fmout_shape,perturbmag,perturbmag_shape)



        print("Begin align and scale images for each wavelength")
        aligned_outputs = []
        for threadnum in range(numthreads):
            #multitask this                                                                    
            aligned_outputs += [tpool.apply_async(_align_and_scale_subset, args=(threadnum, aligned_center,numthreads,fm_class.np_data_type))]

            #save it to shared memory                                                          
        for aligned_output in aligned_outputs:
            aligned_output.wait()

        print("Align and scale finished")

        print aligned



    def save_fmout(self, dataset, fmout, outputdir, fileprefix, numbasis, klipparams=None, calibrate_flux=False, spectrum=None):
        '''
        Uses self.dataset parameters to save fmout, the output of
        fm_paralellized or klip_dataset
        '''
        KLmode_cube = np.nanmean(fmout, axis = 1)
        dataset.savedata(outputdir + '/' + fileprefix + "-fmpsf-KLmodes-all.fits", KLmode_cube,
                         klipparams=klipparams.format(numbasis=str(numbasis)), filetype="KL Mode Cube",
                         zaxis=numbasis)

    def cleanup_fmout(self, fmout):
        # will need to fix later
        """
        After running KLIP-FM, we need to reshape fmout so that the numKL dimension is the first one and not the last

        Args:
            fmout: numpy array of ouput of FM

        Return:
            fmout: same but cleaned up if necessary
        """
        dims = fmout.shape
        fmout = np.rollaxis(fmout.reshape((dims[0], dims[1], dims[2], dims[3])), 3)
        return fmout

    def update_disk(self, model_disk):
        self.model_disk = model_disk
        self.model_disks = np.zeros(self.inputs_shape)
        for i, pa in enumerate(self.pas):
            model_copy = copy.deepcopy(model_disk)
            model_copy = rotate(model_copy, pa, self.centers[i], flipx = True)
            model_copy[np.where(np.isnan(model_copy))] = 0.
            self.model_disks[i] = model_copy
        self.model_disks = np.reshape(self.model_disks, (self.inputs_shape[0], self.inputs_shape[1] * self.inputs_shape[2])) 
