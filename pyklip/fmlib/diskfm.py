import multiprocessing as mp
import ctypes
import numpy as np
import os
import copy

from pyklip.fmlib.nofm import NoFM
import pyklip.fm as fm
from pyklip.klip import rotate



class DiskFM(NoFM):
    def __init__(self, inputs_shape, numbasis, dataset, model_disk, basis_file_pattern = None, save_basis = False, annuli = None, subsections = None):
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
        self.pas = dataset.PAs
        self.centers = dataset.centers

        self.update_disk(model_disk)

        self.save_basis = save_basis
        self.annuli = annuli
        self.subsections = subsections

        if basis_file_pattern is not None:
            self.read_file(save_basis)
        else:
            self.klmodes_arr = None
            self.evals_arr = None
            self.evecs_arr = None
            self.ref_psfs_indices_all = None
            self.section_ind_all = None
            if save_basis == True:
                assert annuli is not None, "need annuli keyword to save basis"
                assert subsections is not None, "need annuli keyword to save basis"
                self.total_annuli = annuli
                self.total_subsections = subsections
                self.current_section = 0 # counter for saving basis vectors
    def alloc_fmout(self, output_img_shape):
        '''
        Allocates shared memory for output image and, if desired, evals and evecs    
        '''
        if self.save_basis is False:
            fmout_size = np.prod(output_img_shape)
            fmout_shape = output_img_shape
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
            ref_psfs_indicies = self.ref_psfs_indicies_all[input_img_num]
            section_ind = self.section_ind_all[input_img_num]
        sci = aligned_imgs[input_img_num, section_ind[0]]
        refs = aligned_imgs[ref_psfs_indicies, :]
#        refs = refs[ref_psfs_indicies, :]
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

        # write forward modelled PSF to fmout (as output)
        # need to derotate the image in this step
        for thisnumbasisindex in range(np.size(numbasis)):
            # FIXME add real shape of fmout
            if self.save_basis is True:
                # Reshape fmout
                # save to fmout
                fmout_interim = None # FIXME add correct size
                fm._save_rotated_section(input_img_shape, postklip_psf[thisnumbasisindex], section_ind,
                                         fmout_interim, None, parang,
                                         radstart, radend, phistart, phiend, padding,IOWA, ref_center, flipx=True)
                # FIXME fmout[indices] = fmout_interim
            else:
                fm._save_rotated_section(input_img_shape, postklip_psf[thisnumbasisindex], section_ind,
                                         fmout[input_img_num, :, :,thisnumbasisindex], None, parang,
                                         radstart, radend, phistart, phiend, padding,IOWA, ref_center, flipx=True)

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

        # FIXME much code here

    def read_file(self, basis_file):
        basis_file = open(restore_from_file)
        self.klmodes_all = pickle.load(basis_file)
        self.evals_all = pickle.load(basis_file)
        self.evecs_all = pickle.load(basis_file)
        self.ref_psfs_indices_all = pickle.load(basis_file)
        self.section_ind_all = pickle.load(basis_file)

    def save_fmout(self, dataset, fmout, outputdir, fileprefix, numbasis, klipparams=None, calibrate_flux=False, spectrum=None):
        '''
        Uses self.dataset parameters to save fmout, the output of
        fm_paralellized or klip_dataset
        '''
        if self.save_basis == False:
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
        # Let's reshape the output images
        # move number of KLIP modes as leading axis (i.e. move from shape (N,y,x,b) to (b,N,y,x)
        if self.save_basis is True:
            # Separate fmout from basis vectors
            
            # pull out basis vectors 

            # reformat basis vectors
            
            # pack back
            pass
        else:
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
