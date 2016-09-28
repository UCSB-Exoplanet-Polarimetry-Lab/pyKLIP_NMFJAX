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
    def __init__(self, inputs_shape, numbasis, dataset, model_disk, basis_file_pattern = 'klip-basis-', load_from_basis = False, save_basis = False, annuli = None, subsections = None, OWA = None, numthreads = None):
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

        if numthreads == None:
            self.numthreads = mp.cpu_count()

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
        Allocates shared memory for output image 
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

            # FIXME save per wavelength

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
        rad_bounds = [(self.dr * rad + self.IWA, self.dr * (rad + 1) + self.IWA) for rad in self.annuli_list]
        phi_bounds = [[self.dphi * phi_i, self.dphi_i  * (phi_i + 1)] for phi_i in self.subs_list]
        phi_bounds[-1][1] = 2. * np.pi - 0.0001

        iterator_sectors = itertools.product(rad_bounds, phi_bounds)
        tot_sectors = len(rad_bounds) * len(phi_bounds)



        fmout_data, fmout_shape = self.alloc_fmout(self.output_imgs_shape)
        # FIXME make numpy arrays


        # FIXME 

        # fm_from_eigen 
        for sector_index, ((radstart,radend), (phistart, phiend)) in enumerate(iterator_sectors):
            t_start_sector = time()
            print("Starting KLIP for sector {0}/{1}".format(sector_index+1,tot_sectors))
            if len(time_spent_per_sector_list)==0:
                print("Time spent on last sector: {0:.0f}s".format(0))
                print("Time spent since beginning: {0:.0f}s".format(0))
                print("First sector: Can't predict remaining time")
            else:
                print("Time spent on last sector: {0:.0f}s".format(time_spent_last_sector))
                print("Time spent since beginning: {0:.0f}s".format(np.sum(time_spent_per_sector_list)))
                print("Estimated remaining time: {0:.0f}s".format((tot_sectors-sector_index)*np.mean(time_spent_per_sector_list)))
            # calculate sector size                                             
 
            section_ind = self.section_ind_all[sector_index]
            sector_size = np.size(section_ind)
            original_KL = self.klmodes_all[sector_index]
            evals = self.evals_all[sector_index]
            evecs = self.evecs_all[sector_index]

            ref_psfs_indicies = self.ref_psfs_indices_all[sector_index]


            # iterate over image number
            # global variables defined in tpool init:
            #original, original_shape, aligned, aligned_shape, outputs, outputs_shape, outputs_numstacked, img_pa, img_wv, img_center, interm, interm_shape, fmout, fmout_shape, perturbmag, perturbmag_shape
            # original_KL

            # FIXME iteratte over image number
            fm_class.fm_from_eigen(klmodes=original_KL, evals=evals, evecs=evecs,
                                   input_img_shape=[original_shape[1], original_shape[2]], input_img_num=img_num,
                                   ref_psfs_indicies=ref_psfs_indicies, section_ind=section_ind, aligned_imgs=aligned_imgs,
                                   pas=pa_imgs[ref_psfs_indicies], wvs=wvs_imgs[ref_psfs_indicies], radstart=radstart,
                                   radend=radend, phistart=phistart, phiend=phiend, padding=padding,IOWA = IOWA, ref_center=ref_center,
                                   parang=parang, ref_wv=wavelength, numbasis=self.numbasis,maxnumbasis=self.maxnumbasis,
                                   fmout=fmout_np,perturbmag = None, klipped=klipped, covar_files=covar_files)



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

        # FIXME wavelengths?
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
        original_imgs = mp.Array(self.mp_data_type, np.size(self.images))
        original_imgs_shape = self.images.shape
        original_imgs_np = fm._arraytonumpy(original_imgs, original_imgs_shape,dtype=self.np_data_type)
        original_imgs_np[:] = self.images


        # make array for recentered/rescaled image for each wavelength                               
        unique_wvs = np.unique(self.wvs)
        recentered_imgs = mp.Array(self.mp_data_type, np.size(self.images)*np.size(unique_wvs))
        recentered_imgs_shape = (np.size(unique_wvs),) + self.images.shape

        # remake the PA, wv, and center arrays as shared arrays                                            
        pa_imgs = mp.Array(self.mp_data_type, np.size(self.pas))
        pa_imgs_np = fm._arraytonumpy(pa_imgs,dtype=self.np_data_type)
        pa_imgs_np[:] = self.pas
        wvs_imgs = mp.Array(self.mp_data_type, np.size(self.wvs))
        wvs_imgs_np = fm._arraytonumpy(wvs_imgs,dtype=self.np_data_type)
        wvs_imgs_np[:] = self.wvs
        centers_imgs = mp.Array(self.mp_data_type, np.size(self.centers))
        centers_imgs_np = fm._arraytonumpy(centers_imgs, self.centers.shape,dtype=self.np_data_type)
        centers_imgs_np[:] = self.centers
        output_imgs = None
        output_imgs_numstacked = None
        self.output_imgs_shape = self.images.shape + self.numbasis.shape

        perturbmag, perturbmag_shape = self.alloc_perturbmag(self.output_imgs_shape, self.numbasis)
        
        fmout_data = None
        fmout_shape = None
        
        #FIXME
#        if aligned_center is None:
        aligned_center = [int(self.images.shape[2]//2),int(self.images.shape[1]//2)]




        tpool = mp.Pool(processes=self.numthreads, initializer=fm._tpool_init,initargs=(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs, self.output_imgs_shape, output_imgs_numstacked, pa_imgs, wvs_imgs, centers_imgs, None, None, fmout_data, fmout_shape,perturbmag,perturbmag_shape), maxtasksperchild=50)

        fm._tpool_init(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs,self.output_imgs_shape, output_imgs_numstacked, pa_imgs, wvs_imgs, centers_imgs, None, None,fmout_data, fmout_shape,perturbmag,perturbmag_shape)

        print("Begin align and scale images for each wavelength")
        aligned_outputs = []
        for threadnum in range(self.numthreads):
            #multitask this                                                                    
            aligned_outputs += [tpool.apply_async(fm._align_and_scale_subset, args=(threadnum, aligned_center,self.numthreads,self.np_data_type))]

            #save it to shared memory                                           
        for aligned_output in aligned_outputs:
            aligned_output.wait()



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

        # FIXME align and scale
        for i, pa in enumerate(self.pas):
            model_copy = copy.deepcopy(model_disk)
            model_copy = rotate(model_copy, pa, self.centers[i], flipx = True)
            model_copy[np.where(np.isnan(model_copy))] = 0.
            self.model_disks[i] = model_copy
        self.model_disks = np.reshape(self.model_disks, (self.inputs_shape[0], self.inputs_shape[1] * self.inputs_shape[2])) 
