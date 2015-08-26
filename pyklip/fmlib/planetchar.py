import multiprocessing as mp
import ctypes

import numpy as np
import pyklip.spectra_management as specmanage
import os
import scipy.ndimage as ndimage

class PlanetChar():
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
        self.spectrallib = [spectrum/specmanage.get_star_spectrum(wavelengths, star_type=star_spt) for spectrum in self.spectrallib]
        self.spectrallib = [spectrum/np.mean(spectrum) for spectrum in self.spectrallib]

        self.input_psfs = input_psfs
        self.input_psfs_wvs = input_psfs_wvs
        self.flux_conversion = flux_conversion


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


    def generate_models(self, input_img_shape, section_ind, pas, wvs, radstart, radend, phistart, phiend, padding, ref_center, parang):
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

        Return:
            models: array of size (N, p) where p is the number of pixels in the segment
        """
        # create some parameters for a blank canvas to draw psfs on
        xc, yc = np.meshgrid(np.arange(input_img_shape.shape[1] * 1.), np.arnage(input_img_shape.shape[0] * 1.))
        rc = np.sqrt((xc - ref_center[0])**2 + (yc - ref_center[1])**2)
        phic = np.arctan2(yc - ref_center[1], xc - ref_center[0])

        models = []
        for pa, wv in zip(pas, wvs):
            #create a canvas to place the new PSF in the sector on
            canvas = np.zeros(input_img_shape)

            #find center of psf
            psf_centx = self.sep * np.cos(np.radians(self.pa - pa)) + ref_center[0]#note self.pa is position angle, pa is parallactic angle
            psf_centy = self.sep * np.sin(np.radians(self.pa - pa)) + ref_center[1]
            xc_psf = xc - psf_centx
            yc_psf = yc - psf_centy

            wv_index = np.where(wv == self.input_psfs_wvs)[0]
            model_psf = self.input_psfs[wv_index, :, :] * self.flux_conversion * self.spectrallib[0][wv_index]

            segment_with_model = ndimage.map_coordinates(model_psf, [yc_psf[section_ind], xc_psf[section_ind]])

            models.append(segment_with_model)

        return np.array(models)








