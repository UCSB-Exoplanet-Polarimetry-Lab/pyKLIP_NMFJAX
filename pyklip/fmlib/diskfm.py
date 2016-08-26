import multiprocessing as mp
import ctypes
import numpy as np
import os

from pyklip.fmlib.nofm import NoFM
import pyklip.fm as fm




class DiskFM(NoFM):
    def __init__(self, inputs_shape, numbasis, input_disk, restore_from_fmout = None):
        super(DiskFM, self).__init__(inputs_shape, numbasis)
        self. inputs_shape = inputs_shape
        self.numbasis = numbasis
        self.input_disk = input_disk

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


    def alloc_perturbmag(self, output_img_shape, numbasis):
        """
        Allocates shared memory to store the fractional magnitude of the linear KLIP perturbation
        Stores a number for each frame = max(oversub + selfsub)/std(PCA(image))

        Args:
            output_img_shape: shape of output image (usually N,y,x,b)
            numbasis: array/list of number of KL basis cutoffs requested

        Returns:
            perturbmag: mp.array to store linaer perturbation magnitude
            perturbmag_shape: shape of linear perturbation magnitude

        """
        perturbmag_shape = (output_img_shape[0], np.size(numbasis))
        perturbmag = mp.Array(ctypes.c_double, np.prod(perturbmag_shape))

        return perturbmag, perturbmag_shape


    def fm_from_eigen(self, klmodes = None, evals = None, evecs = None, input_img_shape = None, input_img_num = None, ref_psfs_indices = None,
                      section_ind = None, aligned_ims = None, pas = None, numbasis = None, fmout = None):
        if klmodes == None or evals == None or evecs == None:
            assert self.klmodes_all != None and self.evals_all != None and self.evecs_all != None, "No evals or evecs defined"
            klmodes = self.klmodes_all[input_img_num]
            evals = self.evals_all[input_img_num]
            evecs = self.evecs_all[input_img_num]
            ref_psfs_indices = self.ref_psfs_indices_all[input_img_num]
            section_ind = self.section_ind[input_img_num]
            


        sci = aligned_imgs[input_img_num, section_ind[0]]
        refs = aligned_imgs[ref_psfs_indicies, :]
        refs = refs[ref_psfs_indices, :]
        refs = refs[:, section_ind[0]]

        # FIXME: rotate disk here
        
        model_sci = self.input_disk[input_img_num, section_ind[0]]
        model_ref = self.input_disk[ref_psfs_indices, :]
        model_ref = model_ref[:, section_ind[0]]
        
        refs_mean_sub = model_ref - np.nanmean(model_ref, axis = 1)[:, None]
        refs_mean_sub[np.where(np.isnan(refs_mean_sub))] = 0
        models_mean_sub = model_ref - np.nanmean(model_ref, axis = 1)[:, None]
        model_mean_sub[np.where(np.isnan(models_mean_sub))] = 0
        delta_KL= fm.perturb_specIncluded(evals, evecs, klmodes, refs_mean_sub, models_mean_sub, return_perturb_covar = False)
        fm_psf, oversubtraction, selfsubtraction = fm.calculate_fm(delta_KL, klmodes, numbasis, sci, model_sci, inputflux = None)

    def alloc_fmout(self):
        pass
    def fm_from_file(self, input_disk):
        # Call fm_from_eigen with current parameters
        pass
    def read_file(self, basis_file):
        # Read in parameters from file
        pass
    def save_fmout(self):
        # save fm out
        pass
