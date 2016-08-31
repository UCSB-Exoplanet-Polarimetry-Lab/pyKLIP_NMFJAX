import multiprocessing as mp
import ctypes
import numpy as np
import os
import copy

from pyklip.fmlib.nofm import NoFM
import pyklip.fm as fm




class DiskFM(NoFM):
    def __init__(self, inputs_shape, numbasis, dataset, model_disk, restore_from_fmout = None, save_basis = False, annuli = None, subsections = None):
        super(DiskFM, self).__init__(inputs_shape, numbasis)
        self.inputs_shape = inputs_shape
        self.numbasis = numbasis
        self.maxnumbasis = max(numbasis)

        self.dataset = dataset
        self.pas = dataset.pas
        self.centers = dataset.centers

        self.model_disk = model_disk
        self.model_disks = np.zeros(inputs_shape)
        for i, pa in enumerate(self.pas):
            model_copy = copy.deepcopy(model_disk)
            model_copy = kliprotate(model_copy, pa, self.centers[i], flipx = True)
            model_copy[np.where(np.isnan(model_copy))] = 0.
            self.model_disks[i] = model_copy

        self.save_basis = save_basis
        self.annuli = annuli
        self.subsections = subsections

        if restore_from_file != None:
            basis_file = open(restore_from_file)
            self.klmodes_all = pickle.load(basis_file)
            self.evals_all = pickle.load(basis_file)
            self.evecs_all = pickle.load(basis_file)
            self.ref_psfs_indices_all = pickle.load(basis_file)
            self.section_ind_all = pickle.load(basis_file)
        else:
            self.klmodes_arr = None
            self.evals_arr = None
            self.evecs_arr = None
            self.ref_psfs_indices_all = None
            self.section_ind_all = None

    def alloc_fmout(self, output_img_shape):
        '''
        Allocates shared memory for output image and, if desired, evals and evecs    
        '''
        if self.save_basis is False:
            fmout_size = np.prod(output_img_shape)
            fmout_shape = output_image_shape
        else:
            fmout_size = np.prod(output_img_shape) 
            fmout_shape = (fmout_size)
            # FIXME add in other sizes for evals and evecs
        fmout = mp.Array(ctypes.c_double, fmout_size)

        
        return fmout, fmout_shape


    def fm_from_eigen(self, klmodes=None, evals=None, evecs=None, input_img_shape=None, input_img_num=None, ref_psfs_indicies=None, section_ind=None, aligned_imgs=None, pas=None,
                     wvs=None, radstart=None, radend=None, phistart=None, phiend=None, padding=None,IOWA = None, ref_center=None,
                     parang=None, ref_wv=None, numbasis=None, fmout=None, perturbmag=None, klipped=None, covar_files=None, **kwargs):
        '''
        FIXME
        '''
        if klmodes == None or evals == None or evecs == None:
            assert self.klmodes_all != None and self.evals_all != None and self.evecs_all != None, "No evals or evecs defined"
            klmodes = self.klmodes_all[input_img_num]
            evals = self.evals_all[input_img_num]
            evecs = self.evecs_all[input_img_num]
            ref_psfs_indices = self.ref_psfs_indices_all[input_img_num]
            section_ind = self.section_ind_all[input_img_num]
        sci = aligned_imgs[input_img_num, section_ind[0]]
        refs = aligned_imgs[ref_psfs_indicies, :]
        refs = refs[ref_psfs_indices, :]
        refs = refs[:, section_ind[0]]

        model_sci = self.model_disks[input_img_num, section_ind[0]]
        model_ref = self.model_disks[ref_psfs_indices, :]
        model_ref = model_ref[:, section_ind[0]]
        
        refs_mean_sub = model_ref - np.nanmean(model_ref, axis = 1)[:, None]
        refs_mean_sub[np.where(np.isnan(refs_mean_sub))] = 0
        models_mean_sub = model_ref - np.nanmean(model_ref, axis = 1)[:, None]
        model_mean_sub[np.where(np.isnan(models_mean_sub))] = 0
        delta_KL= fm.perturb_specIncluded(evals, evecs, klmodes, refs_mean_sub, models_mean_sub, return_perturb_covar = False)
        postklip_psf, oversubtraction, selfsubtraction = fm.calculate_fm(delta_KL, klmodes, numbasis, sci, model_sci, inputflux = None)

        # write forward modelled PSF to fmout (as output)
        # need to derotate the image in this step
        for thisnumbasisindex in range(np.size(numbasis)):
            # FIXME add real shape of fmout
            fmout_interim = None
            fm._save_rotated_section(input_img_shape, postklip_psf[thisnumbasisindex], section_ind,
                                     fmout[input_img_num, :, :,thisnumbasisindex], None, parang,
                                     radstart, radend, phistart, phiend, padding,IOWA, ref_center, flipx=True)
            # Reshape fmout
            # save to fmout
        if self.save_basis is True:
            pass

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
    def alloc_fmout(self):
        pass
    def read_file(self, basis_file):
        # Read in parameters from file
        pass
    def write_klipped(self):
        pass
    def save_fmout(self, fmout):
        '''
        Uses self.dataset parameters to save fmout, the output of
        fm_paralellized or klip_dataset
        '''

        # save fm out
        pass
    def update_disk(self, new_disk):
        # Rotates model 
        pass
