import multiprocessing as mp
import ctypes

import numpy as np
import pyklip.spectra_management as specmanage
import os
import scipy.ndimage as ndimage
import sys
from pyklip.fmlib.nofm import NoFM
from scipy import interpolate
from copy import copy

#import matplotlib.pyplot as plt
debug = False


class PlanetChar(NoFM):
    """
    Planet Characterization class. Goal to characterize the astrometry and photometry of a planet
    """
    def __init__(self, inputs_shape, numbasis, sep, pa, dflux, input_psfs, input_psfs_wvs, flux_conversion, wavelengths='H', spectrallib=None, star_spt=None, refine_fit=False):
        """
        Defining the planet to characterizae

        Args:
            inputs_shape: shape of the inputs numpy array. Typically (N, y, x)
            numbasis: 1d numpy array consisting of the number of basis vectors to use
            sep: separation of the planet
            pa: position angle of the planet
            dflux: guess for delta flux of planet averaged across band w.r.t star
            input_psfs: the psf of the image. A numpy array with shape (wv, y, x)
            input_psfs_wvs: the wavelegnths that correspond to the input psfs
            flux_conversion: an array of length N to convert from contrast to DN for each frame. Units of DN/contrast
            wavelengths: wavelengths of data. Can just be a string like 'H' for H-band
            spectrallib: if not None, a list of spectra
            star_spt: star spectral type, if None default to some random one
            refine_fit: refine the separation and pa supplied
        """
        # allocate super class
        super(PlanetChar, self).__init__(inputs_shape, numbasis)

        self.inputs_shape = inputs_shape
        self.numbasis = numbasis
        self.sep = sep
        self.pa = pa
        self.dflux = dflux
        if spectrallib is not None:
            self.spectrallib = spectrallib
        else:
            spectra_folder = os.path.dirname(os.path.abspath(specmanage.__file__)) + os.sep + "spectra" + os.sep
            spectra_files = [spectra_folder + "t650g18nc.flx", spectra_folder + "t800g100nc.flx"]
            self.spectrallib = [specmanage.get_planet_spectrum(filename, wavelengths)[1] for filename in spectra_files]

        # TODO: calibrate to contrast units
        # calibrate spectra to DN
        self.spectrallib = [spectrum/(specmanage.get_star_spectrum(wavelengths, star_type=star_spt)[1]) for spectrum in self.spectrallib]
        self.spectrallib = [spectrum/np.mean(spectrum) for spectrum in self.spectrallib]

        self.input_psfs = input_psfs
        self.input_psfs_wvs = input_psfs_wvs
        self.flux_conversion = flux_conversion

        self.psf_centx_notscaled = {}
        self.psf_centy_notscaled = {}

        numwv,ny_psf,nx_psf =  self.input_psfs.shape
        x_psf_grid, y_psf_grid = np.meshgrid(np.arange(ny_psf* 1.)-ny_psf/2, np.arange(nx_psf * 1.)-nx_psf/2)
        psfs_func_list = []
        for wv_index in range(numwv):
            model_psf = self.input_psfs[wv_index, :, :] #* self.flux_conversion * self.spectrallib[0][wv_index] * self.dflux
            #psfs_interp_model_list.append(interpolate.bisplrep(x_psf_grid,y_psf_grid,model_psf))
            #psfs_interp_model_list.append(interpolate.SmoothBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel()))
            psfs_func_list.append(interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5))
            #psfs_interp_model_list.append(interpolate.interp2d(x_psf_grid,y_psf_grid,model_psf,kind="cubic",bounds_error=False,fill_value=0.0))
            #psfs_interp_model_list.append(interpolate.Rbf(x_psf_grid,y_psf_grid,model_psf,function="gaussian"))

            if 0:
                import matplotlib.pylab as plt
                #print(x_psf_grid.shape)
                #print(psfs_interp_model_list[wv_index](x_psf_grid.ravel(),y_psf_grid.ravel()).shape)
                plt.figure(1)
                plt.subplot(1,3,1)
                a = psfs_func_list[wv_index](x_psf_grid[0,:],y_psf_grid[:,0])
                plt.imshow(a,interpolation="nearest")
                plt.colorbar()
                ##plt.imshow(psfs_interp_model_list[wv_index](np.linspace(-10,10,500),np.linspace(-10,10,500)),interpolation="nearest")
                plt.subplot(1,3,2)
                plt.imshow(self.input_psfs[wv_index, :, :],interpolation="nearest")
                plt.colorbar()
                plt.subplot(1,3,3)
                plt.imshow(abs(self.input_psfs[wv_index, :, :]-a),interpolation="nearest")
                plt.colorbar()
                plt.show()

        self.psfs_func_list = psfs_func_list


    def alloc_interm(self, max_sector_size, numsciframes):
        """Allocates shared memory array for intermediate step

        Intermediate step is allocated for a sector by sector basis

        Args:
            max_sector_size: number of pixels in this sector. Max because this can be variable. Stupid rotating sectors

        Returns:
            interm: mp.array to store intermediate products from one sector in
            interm_shape:shape of interm array (used to convert to numpy arrays)

        """

        interm_size = max_sector_size * np.size(self.numbasis) * numsciframes * len(self.spectrallib)

        interm = mp.Array(ctypes.c_double, interm_size)
        interm_shape = [numsciframes, len(self.spectrallib), max_sector_size, np.size(self.numbasis)]

        return interm, interm_shape


    def alloc_aux(self):
        """Allocates shared memory of an auxilliary array used in the start

        Note: It might be useful to store the pointer to the aux array into the state of this class if you use it
        for easy access

        Args:

        Returns:
            aux: mp.array to store auxilliary data in
            aux_shape: shape of auxilliary array

        """

        return None, None


    def generate_models(self, input_img_shape, section_ind, pas, wvs, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv):
        # TODO: change this to **kwargsgi t
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
        if debug:
            canvases = []
        models = []
        #print(self.input_psfs.shape)
        for pa, wv in zip(pas, wvs):
            #print(self.pa,self.sep)
            #print(pa,wv)
            # grab PSF given wavelength
            wv_index = np.where(wv == self.input_psfs_wvs)[0]
            #model_psf = self.input_psfs[wv_index[0], :, :] #* self.flux_conversion * self.spectrallib[0][wv_index] * self.dflux

            # find center of psf
            # to reduce calculation of sin and cos, see if it has already been calculated before
            if pa not in self.psf_centx_notscaled:
                self.psf_centx_notscaled[pa] = self.sep * np.cos(np.radians(90. - self.pa - pa))
                self.psf_centy_notscaled[pa] = self.sep * np.sin(np.radians(90. - self.pa - pa))
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
                    self.psfs_func_list[wv_index[0]](x_vec_stamp_centered,y_vec_stamp_centered)

            # write model img to output (segment is collapsed in x/y so need to reshape)
            whiteboard.shape = [input_img_shape[0] * input_img_shape[1]]
            segment_with_model = copy(whiteboard[section_ind])
            whiteboard.shape = [input_img_shape[0],input_img_shape[1]]

            models.append(segment_with_model)

            # create a canvas to place the new PSF in the sector on
            if debug:
                print(x_grid_stamp_centered[0,:],y_grid_stamp_centered[:,0])
                canvas = np.zeros(input_img_shape)
                canvas.shape = [input_img_shape[0] * input_img_shape[1]]
                canvas[section_ind] = segment_with_model
                canvas.shape = [input_img_shape[0], input_img_shape[1]]
                canvases.append(canvas)
                import matplotlib.pyplot as plt
                plt.figure(1)
                plt.subplot(2,2,1)
                im = plt.imshow(canvas)
                plt.colorbar(im)
                plt.subplot(2,2,2)
                im = plt.imshow(whiteboard)
                plt.colorbar(im)
                plt.subplot(2,2,3)
                plt.imshow(psfs_interp_model_list[wv_index[0]](np.arange(-20,20,1.),np.arange(-20,20,1.)),interpolation="nearest")#x_psf_grid[0,:],y_psf_grid[:,0]
                plt.subplot(2,2,4)
                plt.imshow(psfs_interp_model_list[wv_index[0]](x_grid_stamp_centered[0,:],y_grid_stamp_centered[:,0]),interpolation="nearest")#x_psf_grid[0,:],y_psf_grid[:,0]
                plt.show()
            whiteboard[(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = 0.0

        if debug:
            #import matplotlib.pylab as plt
            for canvas in canvases:
                im = plt.imshow(canvas)
                plt.colorbar(im)
                plt.show()

        return np.array(models)








