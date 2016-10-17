import multiprocessing as mp
import ctypes
import numpy as np
import os
import copy
import pickle
import glob
import scipy.ndimage as ndimage

from pyklip.fmlib.nofm import NoFM
import pyklip.fm as fm
from pyklip.klip import rotate
import ctypes
import itertools



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
        self.IWA = dataset.IWA
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
            self.OWA = OWA
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

    def fm_from_eigen(self, klmodes=None, evals=None, evecs=None, input_img_shape=None, input_img_num=None, ref_psfs_indicies=None, section_ind=None,section_ind_nopadding=None, aligned_imgs=None, pas=None,
                     wvs=None, radstart=None, radend=None, phistart=None, phiend=None, padding=None,IOWA = None, ref_center=None,
                     parang=None, ref_wv=None, numbasis=None, fmout=None, perturbmag=None, klipped=None, covar_files=None, **kwargs):
        '''
        FIXME
        '''
        sci = aligned_imgs[input_img_num, section_ind[0]]

        refs = aligned_imgs[ref_psfs_indicies, :]
        refs = refs[:, section_ind[0]]
        refs[np.where(np.isnan(refs))] = 0


        model_sci = self.model_disks[input_img_num, section_ind[0]]

        model_ref = self.model_disks[ref_psfs_indicies, :]
        model_ref = model_ref[:, section_ind[0]]
        model_ref[np.where(np.isnan(model_ref))] = 0

        delta_KL= fm.perturb_specIncluded(evals, evecs, klmodes, refs, model_ref, return_perturb_covar = False)
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
            # FIXME make it so that it doesn't save one per
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

        # Define phi bounds and rad bounds. There is one rad and phi bound per sector
        rad_bounds = [(self.dr * rad + self.IWA, self.dr * (rad + 1) + self.IWA) for rad in self.annuli_list]
        phi_bounds = [[self.dphi * phi_i, self.dphi  * (phi_i + 1)] for phi_i in self.subs_list]
        phi_bounds[-1][1] = 2. * np.pi - 0.0001


        fmout_data, fmout_shape = self.alloc_fmout(self.output_imgs_shape)
        fmout_np = fm._arraytonumpy(fmout_data, fmout_shape, dtype = self.np_data_type)

#        tpool = mp.Pool(processes = self.numthreads)


        # FIXME 

        # fm_from_eigen 
#        for sector_index, ((radstart,radend), (phistart, phiend)) in enumerate(iterator_sectors):
        for sector_index in range(len(self.section_ind_all)):


            radstart, radend = rad_bounds[sector_index]
            phistart, phiend = phi_bounds[sector_index]
            # calculate sector size                                             
 
            # sector index chooses both the 

            section_ind = self.section_ind_all[sector_index]
            sector_size = np.size(section_ind)
            original_KL = self.klmodes_all[sector_index]
            evals = self.evals_all[sector_index]
            evecs = self.evecs_all[sector_index]
            ref_psfs_indicies = self.ref_psfs_indices_all[sector_index]
            img_num = self.imnum_list[sector_index]


# No wavelength dependence yet
#            for wv_index, wv_value in enumerate(self.unique_wvs):
#                scidata_indicies = np.where(self.wvs == wv_value)[0]
#                
#                sector_job_queued[sector_index] += scidata_indicies.shape[0]
                
            parallel = False 
                

            
            if not parallel:
                self.fm_from_eigen(klmodes=original_KL, evals=evals, evecs=evecs,
                                   input_img_shape=[original_shape[1], original_shape[2]], input_img_num=img_num,
                                   ref_psfs_indicies=ref_psfs_indicies, section_ind=section_ind, aligned_imgs=self.aligned_imgs_np,

                                   pas=self.pa_imgs_np[ref_psfs_indicies], wvs=self.wvs_imgs_np[ref_psfs_indicies], radstart=radstart,
                                   radend=radend, phistart=phistart, phiend=phiend, padding=0.,IOWA = (self.IWA, self.OWA), ref_center=self.aligned_center,
                                   parang=self.pa_imgs_np[img_num], ref_wv=None, numbasis=self.numbasis,maxnumbasis=self.maxnumbasis,
                                   fmout=fmout_np,perturbmag = None, klipped=None, covar_files=None)

            else:
                pass

        fmout_np = fm._arraytonumpy(fmout_data, fmout_shape, dtype = self.np_data_type)
        fmout_np = self.cleanup_fmout(fmout_np)

        return fmout_np
            


    def load_basis_files(self, basis_file_pattern):
        # Need dr and dphi def
        
        filenames = glob.glob(basis_file_pattern + '*.p')
        assert len(filenames) > 0, "No files found"

        nums = [f[len(basis_file_pattern):] for f in filenames]
        rads = [float(n[1]) for n in nums]
        subs = [float(n[3]) for n in nums]
        imnum = [int(n[5:7]) for n in nums]

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
        output_imgs_shape = self.images.shape + self.numbasis.shape
        self.output_imgs_shape = output_imgs_shape

        perturbmag, perturbmag_shape = self.alloc_perturbmag(self.output_imgs_shape, self.numbasis)


        # FIXME
        fmout_data = None
        fmout_shape = None
        
        #FIXME
#        if aligned_center is None:
        aligned_center = [int(self.images.shape[2]//2),int(self.images.shape[1]//2)]
        self.aligned_center = aligned_center

        tpool = mp.Pool(processes=self.numthreads, initializer=fm._tpool_init,initargs=(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs, self.output_imgs_shape, output_imgs_numstacked, pa_imgs, wvs_imgs, centers_imgs, None, None, fmout_data, fmout_shape,perturbmag,perturbmag_shape), maxtasksperchild=50)

        
        # probably okay if these are global variables right now, can make them local later
        self._tpool_init(original_imgs, original_imgs_shape, recentered_imgs, recentered_imgs_shape, output_imgs,self.output_imgs_shape, output_imgs_numstacked, pa_imgs, wvs_imgs, centers_imgs, None, None,fmout_data, fmout_shape,perturbmag,perturbmag_shape)



        print("Begin align and scale images for each wavelength")
        aligned_outputs = []
        for threadnum in range(self.numthreads):
            #multitask this                                                                

            # write own align and scale subset (?)
            aligned_outputs += [tpool.apply_async(fm._align_and_scale_subset, args=(threadnum, aligned_center,self.numthreads,self.np_data_type))]

            #save it to shared memory                                           
        for aligned_output in aligned_outputs:
            aligned_output.wait()


        self.original = original_imgs
        self.original_shape = original_imgs_shape
        # someone who actually does spec data should make sure that this is actually aligned and scale right
        self.aligned_imgs = aligned
        self.aligned_imgs_np = fm._arraytonumpy(aligned, shape = (original_imgs_shape[0], original_imgs_shape[1] * original_imgs_shape[2]))
#        self.aligned_shape = original_imgs_shape
#        self.aligned_shape = aligned_shape

        # this looks stupid and probably is
        self.outputs = output_imgs
        self.outputs_shape = output_imgs_shape
        self.outputs_numstacked = output_imgs_numstacked
        self.img_pa = pa_imgs
        self.img_wv = wvs_imgs
        self.img_center =  centers_imgs
        self.fmout = fmout
        self.fmout_shape = fmout_shape
        self.pa_imgs = pa_imgs
        self.wvs_imgs = wvs_imgs
        self.wvs_imgs_np = wvs_imgs_np
        self.pa_imgs_np = pa_imgs_np


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

    def _tpool_init(self, original_imgs, original_imgs_shape, aligned_imgs, aligned_imgs_shape, output_imgs, output_imgs_shape,
                    output_imgs_numstacked,
                    pa_imgs, wvs_imgs, centers_imgs, interm_imgs, interm_imgs_shape, fmout_imgs, fmout_imgs_shape,
                    perturbmag_imgs, perturbmag_imgs_shape):
        """
        Initializer function for the thread pool that initializes various shared variables. Main things to note that all
        except the shapes are shared arrays (mp.Array) - output_imgs does not need to be mp.Array and can be anything. Need another version of this for load_image because global variables made in fm.py won't work in here. 
        
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
        perturbmag_imgs: array for output of size of linear perturbation to assess validity
        perturbmag_imgs_shape: shape of perturbmag_imgs
        """
        global original, original_shape, aligned, aligned_shape, outputs, outputs_shape, outputs_numstacked, img_pa, img_wv, img_center, interm, interm_shape, fmout, fmout_shape, perturbmag, perturbmag_shape
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
        
        #intermediate and FM arrays
        interm = interm_imgs
        interm_shape = interm_imgs_shape
        fmout = fmout_imgs
        fmout_shape = fmout_imgs_shape
        perturbmag = perturbmag_imgs
        perturbmag_shape = perturbmag_imgs_shape

    def _save_rotated_section(self, input_shape, sector, sector_ind, output_img, output_img_numstacked, angle, radstart, radend, phistart, phiend, padding,IOWA, img_center, flipx=True,
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
        IOWA: tuple (IWA,OWA) where IWA = Inner working angle and OWA = Outer working angle both in pixels.
        It defines the separation interva in which klip will be run.
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
        IWA,OWA = IOWA
        radstart_padded = np.max([radstart-padding,IWA])
        if OWA is not None:
            radend_padded = np.min([radend+padding,OWA])
        else:
            radend_padded = radend+padding
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
                                    | ((phip >= phiend) & (phip < phistart_padded))) & in_padded_sector)
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
        output_img.shape = [self.outputs_shape[1], self.outputs_shape[2]]
        output_img[rot_sector_pix] = np.nansum([output_img[rot_sector_pix], rot_sector], axis=0)
        output_img.shape = [self.outputs_shape[1] * self.outputs_shape[2]]

        # Increment the numstack counter if it is not None
        if output_img_numstacked is not None:
            output_img_numstacked.shape = [self.outputs_shape[1], self.outputs_shape[2]]
            output_img_numstacked[rot_sector_pix] += 1
            output_img_numstacked.shape = [self.outputs_shape[1] *  self.outputs_shape[2]]
