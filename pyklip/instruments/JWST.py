from pyklip.instruments.Instrument import Data
import pyklip.rdi as rdi
import pyklip.klip
from pyklip.klip import nan_gaussian_filter

from astropy.io import fits
from astropy import wcs
from astroquery.svo_fps import SvoFps

from scipy.ndimage import fourier_shift, median_filter, shift, gaussian_filter
from scipy.optimize import leastsq, minimize

from skimage.registration import phase_cross_correlation

import numpy as np
import os, shutil, re
import matplotlib.pyplot as plt
import time
import copy
import warnings

from spaceKLIP.psf import JWST_PSF
from spaceKLIP.utils import fourier_imshift

import matplotlib
# matplotlib.rc('font', serif='DejaVu Sans')
# matplotlib.rcParams.update({'font.size': 14})
import matplotlib.patheffects as PathEffects

from webbpsf_ext.image_manip import frebin
from webbpsf_ext.utils import get_one_siaf

class JWSTData(Data):
    """
    Class to interpret JWST data using pyKLIP.
    """
    ############################
    ### Class Initialization ###
    ############################

    ####################
    ### Constructors ###
    ####################

    def __init__(self, filepaths=None, psflib_filepaths=None, centering='jwstpipe',
                     scishiftfile=False, refshiftfile=False,
                     fiducial_point_override=False, blur=False,spectral_type=None,
                     load_file0_center=False,save_center_file=False, mask=None):

        # Initialize the super class
        super(JWSTData, self).__init__()

        # Mean wavelengths of the JWST filters from the SVO Filter Profile
        # Service
        self.wave = {}
        filter_list = SvoFps.get_filter_list(facility='JWST', instrument='NIRCAM')
        for i in range(len(filter_list)):
            name = filter_list['filterID'][i]
            name = name[name.rfind('.')+1:]
            self.wave[name] = filter_list['WavelengthMean'][i]/1e4 # micron
        filter_list = SvoFps.get_filter_list(facility='JWST', instrument='MIRI')
        for i in range(len(filter_list)):
            name = filter_list['filterID'][i]
            name = name[name.rfind('.')+1:]
            self.wave[name] = filter_list['WavelengthMean'][i]/1e4 # micron
        del filter_list

        self.centering = centering
        self.fiducial_point_override = fiducial_point_override
        self.blur = blur

        self.spectral_type = spectral_type
        # Get the target dataset
        reference = self.readdata(filepaths, scishiftfile,
                            load_file0_center=load_file0_center,
                            save_center_file=save_center_file, mask=mask)

        # If necessary, get the PSF library dataset for RDI procedures
        if psflib_filepaths != None:
            self.readpsflib(psflib_filepaths, reference, refshiftfile, mask=mask)
        else:
            self._psflib = None

    ################################
    ### Instance Required Fields ###
    ################################
    @property
    def input(self):
        return self._input
    @input.setter
    def input(self, newval):
        self._input = newval

    @property
    def centers(self):
        return self._centers
    @centers.setter
    def centers(self, newval):
        self._centers = newval

    @property
    def filenums(self):
        return self._filenums
    @filenums.setter
    def filenums(self, newval):
        self._filenums = newval

    @property
    def filenames(self):
        return self._filenames
    @filenames.setter
    def filenames(self, newval):
        self._filenames = newval

    @property
    def PAs(self):
        return self._PAs
    @PAs.setter
    def PAs(self, newval):
        self._PAs = newval

    @property
    def wvs(self):
        return self._wvs
    @wvs.setter
    def wvs(self, newval):
        self._wvs = newval

    @property
    def wcs(self):
        return self._wcs
    @wcs.setter
    def wcs(self, newval):
        self._wcs = newval

    @property
    def IWA(self):
        return self._IWA
    @IWA.setter
    def IWA(self, newval):
        self._IWA = newval

    @property
    def output(self):
        return self._output
    @output.setter
    def output(self, newval):
        self._output = newval

    @property
    def psflib(self):
        return self._psflib
    @psflib.setter
    def psflib(self, newval):
        self._psflib = newval

    # mask centers default to star centers if not overridden
    @property
    def mask_centers(self):
        """Center of mask in pixel coordinates (0-indexed)"""
        return self._mask_centers

    ###############
    ### Methods ###
    ###############

    def readdata(self, filepaths, scishiftfile=False, verbose=False,
                load_file0_center=False, save_center_file=False, mask=None):
        """
        Method to open and read JWST data.

        Parameters
        ----------
        filepaths : list of str
            A list of file paths.
        centering : str
            String descriptor for method to estimate image centers.
        verbose : bool
            Boolean for terminal print statements.

        Returns
        -------
        reference : array
            A 2D image to be aligned to for readpsflib.
        """
        # Ensure we have a list of file paths
        if isinstance(filepaths, str):
            filepaths = [filepaths]
            if verbose:
                print('Only 1 file path was provided, are you sure you meant to do this?')

        # Check list is not empty
        if len(filepaths) == 0:
            raise ValueError('Empty filepath list provided to JWSTData!')

        # Intialize some arrays
        data = []
        pxdq = []
        centers = []
        filenames = []
        pas = []
        wvs = []
        wcs_hdrs = []
        mask_centers = []

        # Go through files one by one
        for index, file in enumerate(filepaths):
            with fits.open(file) as f:

                # Let's figure out what instrument this is
                inst = f[0].header['INSTRUME']

                # Grab number of integrations, as this is how many images are
                # in the file
                nints = f[0].header['NINTS']
                pixel_scale = np.sqrt(f['SCI'].header['PIXAR_A2']) # arcsec; need this for later to calculate IWA

                sci_data = f['SCI'].data
                dq_data = f['DQ'].data

                # NIRCam specifics
                if inst == 'NIRCAM':
                    apname = f[0].header['APERNAME']
                    nrc_siaf = get_one_siaf(instrument='NIRCam')
                    ap_siaf = nrc_siaf[apname]

                    # Offset values of mask relative to that are not yet updated in pysiaf
                    # TODO: Remove these offsets when SIAF is finally updated to latest values
                    if '335R' in apname:
                        mask_xoff, mask_yoff = (-0.2, +0.2)
                    elif '430R' in apname:
                        mask_xoff, mask_yoff = (0, 0)
                    elif 'LWB' in apname:
                        mask_xoff, mask_yoff = (0, +1.1)
                    elif '210R' in apname:
                        mask_xoff, mask_yoff = (0, 0)
                    elif 'SWB' in apname:
                        mask_xoff, mask_yoff = (0, 0)
                    else:
                        print(f'WARNING: {apname} does not appear to be a coronagraphic obs!')
                        mask_xoff, mask_yoff = (0, 0)

                    if mask_xoff!=0 or mask_yoff!=0:
                        print(f'WARNING: Modifying NIRCam centers by ({mask_xoff:+.2f},{mask_yoff:+.2f})')

                    # Get aperture reference point (mask positions) 
                    # and modify to image index (-1) along with known mask offsets
                    xref = ap_siaf.XSciRef - 1 + mask_xoff
                    yref = ap_siaf.YSciRef - 1 + mask_yoff
                    crp1, crp2 = (xref, yref)
                    
                    # Compare empirical PSF with WebbPSF simulation to centroid only
                    # the first image. Later images are then registered to this first.
                    if index == 0:
                        # Initial guess 
                        self.nircam_centers = [crp1,crp2]
                        if load_file0_center and os.path.exists(save_center_file+'.npz'):
                            print("Loading the star center in the first file from: {}".format(save_center_file+".npz"))
                            saved_centers = np.load(save_center_file+'.npz')['center']
                            crp1 = saved_centers[0]
                            crp2 = saved_centers[1]
                        else:
                            crp1, crp2 = self.update_nircam_centers(sci_data[0].copy(),
                                            filter_name = f[0].header['FILTER'],
                                            image_mask = f[0].header['CORONMSK'],
                                            apname = apname,
                                            date = f[0].header['DATE-BEG'],
                                            spectral_type = self.spectral_type,
                                            save_center_file=save_center_file)

                        # Save centers
                        self.nircam_centers = [crp1, crp2]

                    # Load centers estimated from first image
                    crp1, crp2 = self.nircam_centers
                    # Assign centers of stellar position
                    img_centers = [crp1, crp2]*nints
                    # Assign mask_centers from SIAF aperture ref pixel
                    these_mask_centers = [xref, yref] * nints

                    # Check for fiducial point override
                    if 'NARROW' in apname:
                        self.fiducial_point_override = True
                # MIRI specifics
                elif inst == 'MIRI':
                    filt = f[0].header['FILTER']
                    self.orig_xoff = f[0].header['XOFFSET']
                    self.orig_yoff = f[0].header['YOFFSET']

                    # Cut out the unilluminated pixels
                    all_data, trim = trim_miri_data([sci_data, dq_data], filt)
                    sci_data, dq_data = all_data[0], all_data[1]

                    if filt == 'F1065C':
                        img_centers = [float(120.184-trim[0]), float(112.116-trim[1])]*nints
                    elif filt == 'F1140C':
                        img_centers = [float(119.749-trim[0]), float(112.236-trim[1])]*nints
                    elif filt == 'F1550C':
                        img_centers = [float(119.746-trim[0]), float(113.289-trim[1])]*nints
                    else:
                        raise ValueError('pyKLIP only currently supports F1065C/F1140C/F1550C MIRI data')

                    # assign mask centers -- currently star centers are mask centers
                    these_mask_centers = img_centers.copy()
                    warnings.warn("Adopting hard-coded [{0}, {1}] as the MIRI mask center".format(these_mask_centers[0], these_mask_centers[1]))

                # Get the images
                data.append(sci_data)
                pxdq.append(dq_data)
                centers.append(img_centers) # header keywords are in 1-indexed coordinates
                mask_centers.append(these_mask_centers)

                # Assign filenames based on the file and the integration
                filenames += ['{}_INT{}'.format(file.split('/')[-1], i+1) for i in range(nints)]

                # Get PA for all frame withing the file
                roll_ref = f['SCI'].header['ROLL_REF']
                roll_ref += f['SCI'].header['V3I_YANG']
                pas.append([roll_ref]*nints)

                # Get the filter wavelength, should be the same for each file
                # though (for JWST)
                filt = f[0].header['FILTER']
                wave = self.wave[filt] # micron
                wvs.append([wave]*nints)

                # Get WCS information
                wcs_hdr = wcs.WCS(header=f['SCI'].header, naxis=f['SCI'].header['WCSAXES'])
                for i in range(nints):
                    wcs_hdrs.append(wcs_hdr.deepcopy())

        # Convert to numpy arrays and collapse integrations along a single axis
        data = np.concatenate(data)
        pxdq = np.concatenate(pxdq)

        centers = np.concatenate(centers).reshape(-1, 2)
        mask_centers = np.concatenate(mask_centers).reshape(-1, 2)
        filenames = np.array(filenames)
        filenums = np.array(range(len(filenames)))
        pas = np.array(pas).flatten()
        wvs = np.array(wvs).flatten()

        # Get image centers based on desired algorithm
        if self.centering == 'basic' or self.centering == 'zero':
            reference = data[0].copy() # align to first science image
        else:
            # Need to subtract residual background so that image registration
            # does not fit to itprint(s)
            data_medsub = data-np.nanmedian(data, axis=(1, 2), keepdims=True)

            # For MIRI, let's only look at the central region
            if inst == 'MIRI':
                _, nx, ny = data_medsub.shape
                data_medsub = data_medsub[:,int(nx/3):int(2*nx/3),int(ny/3):int(2*ny/3)]
            elif inst == 'NIRCAM':
                tr = 10
                if (mask is None):
                    data_medsub = data_medsub[:,int(crp2-tr):int(crp2+tr+1),int(crp1-tr):int(crp1+tr+1)]

            # from matplotlib.colors import LogNorm
            # plt.imshow(data_medsub[0], norm=LogNorm())
            # plt.show()
            # exit()

            reference = data_medsub[0].copy() # align to first science image

            tstart = time.time()
            if self.centering == 'jwstpipe':
                ref_shifts = reference
                msub_shifts = data_medsub
                shifts, res_before, res_after = self.align_jwstpipe(ref_shifts, msub_shifts[1:], mask=mask)
            elif self.centering == 'imageregis':
                shifts, res_before, res_after = self.align_imageregis(ref_shifts, msub_shifts[1:])
            elif self.centering == 'savefile':
                shift_data = np.load(scishiftfile+'.npz')
                shifts = shift_data['shifts']
                res_before = shift_data['res_before']
                res_after = shift_data['res_after']
            else:
                raise ValueError('Unknown centering algorithm')

            # Save shifts if requested
            if self.centering != 'savefile' and scishiftfile != False:
                np.savez(scishiftfile, shifts=shifts, res_before=res_before, res_after=res_after)

            tend = time.time()

            f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
            ax[0].plot(res_before, label='before align')
            ax[0].plot(res_after, label='after align')
            ax[0].grid(axis='y')
            ax[0].set_xlabel('Image index')
            ax[0].set_ylabel(r'$\Sigma$(residual${}^2$)')
            ax[0].legend(loc='upper center')
            ax[0].set_title('Residual reference-image', y=1., pad=10, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            temp = np.unique(pas[1:])
            medcols = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
            for i in range(len(temp)):
                ww = pas[1:] == temp[i]
                shifts_pa = shifts[ww]
                shifts_med = np.median(shifts_pa, axis=0)
                ax[1].scatter(shifts_med[0]*pixel_scale*1000., shifts_med[1]*pixel_scale*1000, marker='o', color=medcols[i], edgecolor='k', zorder=99)
                ax[1].scatter(shifts[ww, 0]*pixel_scale*1000., shifts[ww, 1]*pixel_scale*1000., label='PA = %.0f deg' % temp[i])
            ax[1].axis('square')
            xlim = ax[1].get_xlim()
            temp = xlim[1]-xlim[0]
            ax[1].set_xlim([xlim[0]-0.35*temp, xlim[1]+0.35*temp])
            ylim = ax[1].get_ylim()
            temp = ylim[1]-ylim[0]
            ax[1].set_ylim([ylim[0]-0.15*temp, ylim[1]+0.15*temp])
            ax[1].grid(axis='both')
            ax[1].set_xlabel('Image x-shift [mas]')
            ax[1].set_ylabel('Image y-shift [mas]')
            ax[1].legend(loc='center right')
            ax[1].set_title('Image shifts', y=1., pad=10, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            plt.suptitle('Science image alignment -- '+self.centering+' method (%.0f s runtime)' % (tend-tstart))
            plt.tight_layout()
            plt.savefig(scishiftfile+'.pdf')
            plt.clf()
            # plt.savefig(centering+'_sci_bad.pdf')
            #plt.show()

            # # Let's median the shifts
            # temp = np.unique(pas[1:]) # Get unique PAs
            # for i in range(len(temp)):
            #     ww = pas[1:] == temp[i] # Get locations of each PA
            #     shifts_pa = shifts[ww]
            #     shifts[ww] = np.median(shifts_pa, axis=0)

            # Shifts are calculated as the shift of the image to match
            # the reference, therefore we *subtract* to get the correct
            # center relative to the reference.
            centers[1:] -= shifts[:, :2]

        # Need to align the images so that they have the same centers
        if inst == 'MIRI':
            image_center = np.array([data[0].shape[1]-1, data[0].shape[0]-1])/2.
        elif inst == 'NIRCAM':
            image_center = np.array([data[0].shape[1]-1, data[0].shape[0]-1])/2.
        for i, image in enumerate(data):
            if self.blur != False:
                # Blur if requested *before* align_and_scale
                image = nan_gaussian_filter(image, self.blur)
            recentered_image = pyklip.klip.align_and_scale(image, new_center=image_center, old_center=centers[i])
            mask_centers[i] = mask_centers[i] + (image_center - centers[i]) # shift mask center as well
            centers[i] = image_center
            data[i] = recentered_image

        # Assume an inner working angle of 0.5 lambda/D
        if self.fiducial_point_override:
            IWA = 1. # pix
        else:
            lambda_d_arcsec = ((wvs[0]/1e6)/6.5)*(180./np.pi)*3600.
            IWA = 0.5*lambda_d_arcsec/pixel_scale # pix

        # Assign all necessary properties
        self._input = data
        self._centers = centers
        self._filenums = filenums
        self._filenames = filenames
        self._PAs = pas
        self._wvs = wvs
        self._wcs = wcs_hdrs
        self._IWA = IWA
        self._mask_centers = mask_centers
        return reference

    def readpsflib(self, psflib_filepaths, reference=None, refshiftfile=False,
     verbose=False, mask=None):
        """
        Method to open and read JWST data for use as part of a PSF library.

        Parameters
        ----------
        psflib_filepaths : list of str
            A list of file paths.
        centering : str
            String descriptor for method to estimate image centers.
        reference : array
            A 2D image to be aligned to.
        verbose : bool
            Boolean for terminal print statements.

        Returns
        -------
        None, data are saved to a JWSTData object.
        """
        # Ensure we have a list of file paths
        if isinstance(psflib_filepaths, str):
            psflib_filepaths = [psflib_filepaths]
            if verbose:
                print('Only 1 psflib filepath was provided, are you sure you meant to do this?')

        # Check list is not empty
        if len(psflib_filepaths) == 0:
            raise ValueError('Empty psflib filepath list provided to JWSTData!')

        # Intialize some arrays
        psflib_data = []
        psflib_pxdq = []
        psflib_offsets = []
        psflib_centers = []
        psflib_filenames = []

        # Prepare reference data for RDI subtractions
        for index, file in enumerate(psflib_filepaths):
            with fits.open(file) as f:

                inst = f[0].header['INSTRUME']
                filt = f[0].header['FILTER']

                # Grab number of integrations, as this is how many images are
                # in the file
                nints = f[0].header['NINTS']
                pixel_scale = np.sqrt(f['SCI'].header['PIXAR_A2']) # arcsec; need this for later to calculate IWA

                sci_data = f['SCI'].data
                dq_data = f['DQ'].data

                # NIRCam specifics
                if inst == 'NIRCAM':
                    try:
                        crp1, crp2 = self.nircam_centers
                    except:
                        crp1 = f['SCI'].header['CRPIX1']-1
                        crp2 = f['SCI'].header['CRPIX2']-1
                    center = [crp1, crp2]
                # MIRI specifics
                elif inst == 'MIRI':
                    # Cut out the unilluminated pixels
                    all_data, trim = trim_miri_data([sci_data, dq_data], filt)
                    sci_data, dq_data = all_data[0], all_data[1]

                    if filt == 'F1065C':
                        center = [float(120.184-trim[0]), float(112.116-trim[1])]
                    elif filt == 'F1140C':
                        center = [float(119.749-trim[0]), float(112.236-trim[1])]
                    elif filt == 'F1550C':
                        center = [float(119.746-trim[0]), float(113.289-trim[1])]
                    else:
                        raise ValueError('pyKLIP only currently supports F1065C/F1140C/F1550C MIRI data')

                # Get the images
                psflib_data.append(sci_data)
                psflib_pxdq.append(dq_data)

                # from matplotlib.colors import LogNorm
                # plt.imshow(sci_data[0], norm=LogNorm())
                # plt.scatter(center[0], center[1], c='r')
                # plt.show()
                # exit()

                # Get the known offset between images
                if inst == 'NIRCAM':
                    offset = [f[0].header['XOFFSET'], f[0].header['YOFFSET']]/pixel_scale # pix
                elif inst == 'MIRI':
                    offset = [f[0].header['XOFFSET']-self.orig_xoff, f[0].header['YOFFSET']-self.orig_yoff]/pixel_scale

                psflib_offsets.append(offset.tolist()*nints)
                if self.centering == 'basic':
                    psflib_centers.append([sum(x) for x in zip(center, offset)]*nints)
                elif self.centering == 'zero':
                    psflib_centers.append([sum(x) for x in zip(center, [0., 0.])]*nints) # Wont be shifted
                else:
                    psflib_centers.append([sum(x) for x in zip(center, [0., 0.])]*nints) # dither will be detected using image registration

                # Assign filenames based on the file and the integration
                psflib_filenames += ['{}_INT{}'.format(file.split('/')[-1], i+1) for i in range(nints)]

        # Convert to numpy arrays and collapse along integration axis
        psflib_data = np.concatenate(psflib_data)
        psflib_pxdq = np.concatenate(psflib_pxdq)
        psflib_offsets = np.concatenate(psflib_offsets).reshape(-1, 2)
        psflib_centers = np.concatenate(psflib_centers).reshape(-1, 2)
        psflib_filenames = np.array(psflib_filenames)

        # Get image centers based on desired algorithm
        if self.centering == 'basic' or self.centering =='zero':
            pass
        else:
            if reference is None:
                raise UserWarning('Need reference for this centering algorithm')
            # Need to subtract residual background so that image registration
            # does not fit to it
            data_medsub = psflib_data-np.nanmedian(psflib_data, axis=(1, 2), keepdims=True)
            if inst == 'MIRI':
                _, nx, ny = data_medsub.shape
                data_medsub = data_medsub[:,int(nx/3):int(2*nx/3),int(ny/3):int(2*ny/3)]
            elif inst == 'NIRCAM':
                _, nx, ny = data_medsub.shape
                print('''WARNING: Are you using the NIRCam 335R mask? If not you should look at
                    how image registration is being done in the JWST.py file of pyKLIP''')
                #data_medsub = data_medsub[:,int(nx/3):int(3*nx/4),int(ny/4):int(2*ny/3)]
                tr = 10
                if (mask is None):
                    data_medsub = data_medsub[:,int(crp2-tr):int(crp2+tr+1),int(crp1-tr):int(crp1+tr+1)]
            tstart = time.time()
            if self.centering == 'jwstpipe':
                # Blurring was messing up the iamge registration
                # if self.blur != False:
                #     # Need to blur the images before image registration
                #     ref_shifts = gaussian_filter(reference, self.blur)
                #     msub_shifts = np.array([gaussian_filter(img, self.blur) for img in data_medsub])
                # else:
                ref_shifts = reference
                msub_shifts = data_medsub
                shifts, res_before, res_after = self.align_jwstpipe(ref_shifts, msub_shifts, mask=mask)
            elif self.centering == 'imageregis':
                shifts, res_before, res_after = self.align_imageregis(reference, data_medsub)
            elif self.centering == 'savefile':
                shift_data = np.load(refshiftfile+'.npz')
                shifts = shift_data['shifts']
                res_before = shift_data['res_before']
                res_after = shift_data['res_after']
            else:
                raise ValueError('Unknown centering algorithm')

            # Save shifts if requested
            if self.centering != 'savefile' and refshiftfile != False:
                np.savez(refshiftfile, shifts=shifts, res_before=res_before, res_after=res_after)

            tend = time.time()

            f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            ax[0].plot(res_before, label='before align')
            ax[0].plot(res_after, label='after align')
            ax[0].grid(axis='y')
            ax[0].set_xlabel('Image index')
            ax[0].set_ylabel(r'$\Sigma$(residual${}^2$)')
            ax[0].legend(loc='upper center')
            ax[0].set_title('Residual reference-image', y=1., pad=10, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            temp = np.unique(psflib_offsets, axis=0)
            medcols = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
            for i in range(temp.shape[0]):
                ww = np.where((psflib_offsets == temp[i]).all(axis=1))[0]
                shifts_dp = shifts[ww]
                shifts_med = np.median(shifts_dp, axis=0)
                ax[1].scatter(shifts_med[0]*pixel_scale*1000., shifts_med[1]*pixel_scale*1000, marker='o', color=medcols[i], edgecolor='k', zorder=99)
                ax[1].scatter(shifts[ww, 0]*pixel_scale*1000., shifts[ww, 1]*pixel_scale*1000., c=colors[i], label='dpos %.0f' % (i+1))
                #ax[1].plot([-temp[i, 0]*pixel_scale*1000., -temp[i, 0]*pixel_scale*1000.], [-temp[i, 1]*pixel_scale*1000.-5., -temp[i, 1]*pixel_scale*1000.+5.], color=colors[i])
                #ax[1].plot([-temp[i, 0]*pixel_scale*1000.-5., -temp[i, 0]*pixel_scale*1000.+5.], [-temp[i, 1]*pixel_scale*1000., -temp[i, 1]*pixel_scale*1000.], color=colors[i])
            ax[1].axis('square')
            xlim = ax[1].get_xlim()
            temp = xlim[1]-xlim[0]
            ax[1].set_xlim([xlim[0]-0.35*temp, xlim[1]+0.35*temp])
            ylim = ax[1].get_ylim()
            temp = ylim[1]-ylim[0]
            ax[1].set_ylim([ylim[0]-0.15*temp, ylim[1]+0.15*temp])
            ax[1].grid(axis='both')
            ax[1].set_xlabel('Image x-shift [mas]')
            ax[1].set_ylabel('Image y-shift [mas]')
            ax[1].legend(loc='center right')
            ax[1].set_title('Image shifts', y=1., pad=10, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            plt.suptitle('Reference image alignment -- '+self.centering+' method (%.0f s runtime)' % (tend-tstart))
            plt.tight_layout()
            plt.savefig(refshiftfile+'.pdf')
            plt.clf()
            # plt.savefig(centering+'_ref_bad.pdf')
            #plt.show()

            # Let's median the shifts
            # temp = np.unique(psflib_offsets, axis=0)
            # for i in range(len(temp)):
            #     ww = np.where((psflib_offsets == temp[i]).all(axis=1))[0]
            #     shifts_dp = shifts[ww]
            #     shifts[ww] =  np.median(shifts_dp, axis=0)

            # Shifts are calculated as the shift of the image to match
            # the reference, therefore we *subtract* to get the correct
            # center relative to the reference.
            psflib_centers -= shifts[:, :2]

        # Need to align the images so that they have the same centers
        if inst == 'MIRI':
            image_center = np.array([psflib_data[0].shape[1]-1, psflib_data[0].shape[0]-1])/2.
        elif inst == 'NIRCAM':
            image_center = np.array([psflib_data[0].shape[1]-1, psflib_data[0].shape[0]-1])/2.
        #image_center = center
        for i, image in enumerate(psflib_data):
            if self.blur != False:
                # Blur if requested *before* align_and_scale
                image = nan_gaussian_filter(image, self.blur)
            recentered_image = pyklip.klip.align_and_scale(image, new_center=image_center, old_center=psflib_centers[i])
            psflib_data[i] = recentered_image

        # Append the target images as well
        psflib_data = np.append(psflib_data, self._input, axis=0)
        psflib_filenames = np.append(psflib_filenames, self._filenames, axis=0)
        psflib_centers = np.append(psflib_centers, self._centers, axis=0)

        # Create the PSF library
        psflib = rdi.PSFLibrary(psflib_data, image_center, psflib_filenames, compute_correlation=True)

        # Prepare the library with the target dataset
        psflib.prepare_library(self)

        self._psflib = psflib
        return

    def fix_bad_pixels(self,
                       data,
                       pxdq):
        """
        Fix bad pixels using a median filter.

        Parameters
        ----------
        data : array
            A 3D data cube of images to be fixed.
        pxdq : array
            A 3D data cube of bad pixel maps.

        Returns
        -------
        data_fixed : array
            Fixed images.
        """

        if data.ndim != 3:
            raise UserWarning('Requires 3D data cube')

        # Fix bad pixels using median filter
        data_fixed = data.copy()
        for i in range(data.shape[0]):
            bad = pxdq[i] != 0
            data_fixed[i][bad] = median_filter(data[i], size=5)[bad]

        return data_fixed

    def recenterlsq(self, shft, data):
        return 1./np.max(fourier_imshift(data, shft))

    def fourier_imshift(self, image, shift, pad=False, cval=0.0):
        """
        Fourier image shift. Adapted from JWST stage 3 pipeline.

        Parameters
        ----------
        image : array
            A 2D/3D image to be shifted.
        shift : array
            xshift, yshift.
        pad : bool
            Should we pad the array before shifting, then truncate?
            Otherwise, the image is wrapped.
        cval : sequence or float, optional
            The values to set the padded values for each axis. Default is 0.
            ((before_1, after_1), ... (before_N, after_N)) unique pad constants for each axis.
            ((before, after),) yields same before and after constants for each axis.
            (constant,) or int is a shortcut for before = after = constant for all axes.

        Returns
        -------
        offset : array
            Shifted image.

        """
        return fourier_imshift(image, shift, pad=pad, cval=cval)

    def shift_subtract(self,
                       pp,
                       reference,
                       target,
                       mask=None):
        """
        From JWST stage 3 pipeline.

        Parameters
        ----------
        pp : tuple
            xshift, yshift, beta.
        reference : array
            A 2D image to be aligned to.
        target : array
            A 2D image to align to reference.
        mask : array
            A 2D image indicating pixels to be considered during the fit.

        Returns
        -------
        res : array
            A 1D vector containing the difference between reference and
            target.
        """
        shift = pp[:2]
        beta = pp[2]

        offset = fourier_imshift(target, shift)

        if mask is not None:
            return ((reference-beta*offset)*mask).ravel()
        else:
            return (reference-beta*offset).ravel()

    def align_fourierLSQ(self,
                         reference,
                         target,
                         mask=None):
        """
        From JWST stage 3 pipeline.

        Parameters
        ----------
        reference : array
            A 2D image to be aligned to.
        target : array
            A 2D image to align to reference.
        mask : array
            A 2D image indicating pixels to be considered during the fit.

        Returns
        -------
        pp : tuple
            xshift, yshift, beta.
        """
        p0 = [0., 0., 1.]
        pp = leastsq(self.shift_subtract,
                     p0,
                     args=(reference, target, mask),
                     full_output=True)

        return pp

    def align_jwstpipe(self,
                       reference,
                       data,
                       mask=None):
        """
        Align a 3D data cube of images to a reference image using the same
        algorithm as the JWST stage 3 pipeline.

        Parameters
        ----------
        reference : array
            A 2D image to be aligned to.
        target : array
            A 3D data cube of images to align to reference.

        Returns
        -------
        shifts : array
            xshift, yshift, beta to align each image.
        res_before : array
            Sum of squares of residuals between the reference and each image
            before alignment.
        res_after : array
            Sum of squares of residuals between the reference and each image
            after alignment.
        """
        if data.ndim != 3:
            raise UserWarning('Requires 3D data cube')

        shifts = []
        res_before = []
        res_after = []
        for i in range(data.shape[0]):
            pp = self.align_fourierLSQ(reference, data[i].copy(), mask=mask)
            shifts += [pp[0]]
            res_before += [np.sum((reference-pp[0][2]*data[i])**2)]
            res_after += [np.sum(pp[2]['fvec']**2)]

            # f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
            # p0 = ax[0].imshow(reference-data[i], origin='lower')
            # plt.colorbar(p0, ax=ax[0])
            # p1 = ax[1].imshow(reference-pp[0][2]*self.fourier_imshift(data[i].copy(), pp[0][:2]), origin='lower')
            # plt.colorbar(p1, ax=ax[1])
            # plt.show()
            # import pdb; pdb.set_trace()

        return np.array(shifts), np.array(res_before), np.array(res_after)

    def scale_subtract(self,
                       pp,
                       reference,
                       target,
                       mask=None):
        """
        Scale and subtract the target from the reference while computing the
        best image shift using scikit image registration.

        Parameters
        ----------
        pp : tuple
            beta.
        reference : array
            A 2D image to be aligned to.
        target : array
            A 2D image to align to reference.
        mask : array
            A 2D image indicating pixels to be considered during the fit.

        Returns
        -------
        res : array
            A 1D vector containing the difference between reference and
            target.
        """
        shift, error, diffphase = phase_cross_correlation(reference,
                                                          pp[0]*target,
                                                          upsample_factor=100)
        shift = shift[::-1]

        offset = fourier_imshift(target, shift)

        if mask is not None:
            return ((reference-pp[0]*offset)*mask).ravel()
        else:
            return (reference-pp[0]*offset).ravel()

    def align_imageregisLSQ(self,
                            reference,
                            target,
                            mask=None):
        """
        Run a least squares optimization on the scaling parameter while
        computing the best image shift using scikit image registration.

        Parameters
        ----------
        reference : array
            A 2D image to be aligned to.
        target : array
            A 2D image to align to reference.
        mask : array
            A 2D image indicating pixels to be considered during the fit.

        Returns
        -------
        pp : tuple
            beta.
        """
        p0 = [1.]
        pp = leastsq(self.scale_subtract,
                     p0,
                     args=(reference, target, mask),
                     full_output=True)

        return pp

    def align_imageregis(self,
                         reference,
                         data):
        """
        Align a 3D data cube of images to a reference image using scikit image
        registration.

        Parameters
        ----------
        reference : array
            A 2D image to be aligned to.
        target : array
            A 3D data cube of images to align to reference.

        Returns
        -------
        shifts : array
            xshift, yshift, beta to align each image.
        res_before : array
            Sum of squares of residuals between the reference and each image
            before alignment.
        res_after : array
            Sum of squares of residuals between the reference and each image
            after alignment.
        """
        if data.ndim != 3:
            raise UserWarning('Requires 3D data cube')

        shifts = []
        res_before = []
        res_after = []
        for i in range(data.shape[0]):
            pp = self.align_imageregisLSQ(reference, data[i].copy())
            shift, error, diffphase = phase_cross_correlation(reference,
                                                              pp[0][0]*data[i].copy(),
                                                              upsample_factor=1000)
            shift = shift[::-1]
            shifts += [np.array([shift[0], shift[1], pp[0][0]])]
            res_before += [np.sum((reference-pp[0][0]*data[i])**2)]
            res_after += [np.sum((reference-pp[0][0]*fourier_imshift(data[i].copy(), shift))**2)]

            # f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
            # p0 = ax[0].imshow(reference-data[i], origin='lower')
            # plt.colorbar(p0, ax=ax[0])
            # p1 = ax[1].imshow(reference-self.fourier_imshift(data[i].copy(), shift), origin='lower')
            # plt.colorbar(p1, ax=ax[1])
            # plt.show()
            # import pdb; pdb.set_trace()

        return np.array(shifts), np.array(res_before), np.array(res_after)

    def update_nircam_centers(self, 
            data0, filter_name, image_mask, apname,
            osamp=2, date=None, use_coeff=False,
            spectral_type='G2V', mask_radius=None,
            save_center_file=False):
        """
        A function that updates the nircam_centers by generating a PSF 
        and getting its relative offset compared to the data. 

        By default, uses transmission mask to mask center pixels (leakage),
        but can set mask_radius to use a circular mask that is centered on
        the SIAF aperture reference point (nominal star position).
        """

        import webbpsf_ext
        from webbpsf_ext.webbpsf_ext_core import _transmission_map
        from webbpsf_ext.coords import dist_image

        def crop_image(image, xycen, npix, return_indices=False):
            """
            Function to crop an image with option to return
            original cropped indices corresponding to original image.
            """

            xc, yc = xycen
            x1 = int(xc - npix / 2 + 0.5)
            x2 = x1 + npix
            y1 = int(yc - npix / 2 + 0.5)
            y2 = y1 + npix

            imsub = image[y1:y2,x1:x2]
            if return_indices:
                xsub_indarr = np.arange(x1,x2).astype('int')
                ysub_indarr = np.arange(y1,y2).astype('int')
                return imsub, xsub_indarr, ysub_indarr
            else:
                return imsub
        
        #Clean the input data
        indz = np.isnan(data0) | (data0 != data0)
        data0[indz] = 0.

        spectrum = webbpsf_ext.stellar_spectrum(spectral_type)

        # Get the Current centers
        # These are 0-based indices corresponding to center of pixel
        crp1, crp2 = self.nircam_centers
        
        # Small subarray for PSF
        # Will crop input image around stellar location
        # Make this odd so simulated PSF is in centered in middle of pixel
        fov_pix = 65
        
        kwargs = {
            'oversample': osamp,
            'date': date,
            'sp':spectrum,
            'use_coeff':use_coeff,
        }

        if image_mask.startswith('MASKA') or image_mask.startswith('MASKB'):
            image_mask = image_mask[:4]+image_mask[5:]

        instrument = "NIRCam"

        print("Generating WebbPSF image for centering. This might take a minute or so...")
        #Generate the psfs
        psf = JWST_PSF(instrument, filter_name, image_mask, fov_pix, **kwargs)

        # Get modeled coronagraphic PSF offset to nominal sci pixel location
        # First convert to tel coords from input data SIAF aperture
        # in case psf.inst_on aperture is different
        apsiaf = psf.inst_on.siaf[apname]
        xsci_ref, ysci_ref = (apsiaf.XSciRef, apsiaf.YSciRef)
        # These offsets need to be applied to pass proper reference point 
        # in current version of pysiaf so simulations work correctly 
        xoff = (crp1 + 1) - xsci_ref
        yoff = (crp2 + 1) - ysci_ref
        crtel = apsiaf.sci_to_tel((crp1 + 1) - xoff, (crp2 + 1) - yoff)
        model_psf = psf.gen_psf_idl(crtel, coord_frame='tel', return_oversample=False)

        # Create transmission mask for masking center regions
        if mask_radius is None:
            yi, xi = np.indices(data0.shape)
            xtel, ytel = apsiaf.sci_to_tel(xi+1-xoff,yi+1-yoff)
            tmask, _, _ = _transmission_map(psf.inst_on, (xtel, ytel), 'tel')
            mask = tmask**2
        else:
            # Mask out the center bits
            rs = dist_image(data0, center=(crp1,crp2))
            mask = np.ones(data0.shape)
            mask[rs<mask_radius] = 0

        # Interate a few times to get final center position
        # via image registration with model PSF
        print('Finding stellar position...')
        xc, yc = (crp1, crp2)
        for i in range(3):
            res = crop_image(data0, (xc,yc), fov_pix, return_indices=True)
            data_sub, xsub_indarr, ysub_indarr = res
            mask_sub = crop_image(mask, (xc, yc), fov_pix)

            # Get x/y shift values of the model relative to the data.
            # shift is returned as [y,x]
            ysh, xsh = phase_cross_correlation(data_sub*mask_sub, model_psf*mask_sub, 
                                               upsample_factor=1000, normalization=None,
                                               return_error=False)

            xc = np.mean(xsub_indarr) + xsh
            yc = np.mean(ysub_indarr) + ysh

        xsh, ysh = (xc-crp1, yc-crp2)
        print(f"Stellar position relative to coron center, (dx,dy)=({xsh:.2f},{ysh:.2f} pixels)")

        # Save shifts if requested
        if save_center_file != False:
            np.savez(save_center_file, center=np.array([xc,yc]), pixshift=np.array([xsh,ysh]))

        return xc, yc

    def savedata(self, filepath, data, klipparams=None, filetype='', zaxis=None, more_keywords=None):
        """
        Saves data for this instrument.

        Parameters
        ----------
        filepath : str
            Filepath to save to.
        data : array
            Data to save.
        klipparams : str
            A string of KLIP parameters. Write it to the 'PSFPARAM' keyword.
        filtype : str
            Type of file (e.g. "KL Mode Cube", "PSF Subtracted Spectral Cube").
            Written to 'FILETYPE' keyword.
        zaxis : list of int
            A list of values for the zaxis of the datacube (for KL mode cubes
            currently).
        more_keywords : dict
            A dictionary {key: value, key:value} of header keywords and values
            which will be written into the primary header.
        """
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(data=data))

        # Save all the files we used in the reduction. We'll assume you used
        # all the input files. Remove duplicates from list.
        filenames = np.unique(self.filenames)
        nfiles = np.size(filenames)
        hdulist[0].header['DRPNFILE'] = (nfiles, 'Num raw files used in pyKLIP')
        for i, filename in enumerate(filenames):
            hdulist[0].header['FILE_{0}'.format(i)] = filename + '.fits'

        # Write out PSF subtraction parameters and get pyKLIP revision number
        pykliproot = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        # The universal_newline argument is just so Python 3 returns a string
        # instead of bytes. This will probably come to bite me later.
        try:
            pyklipver = pyklip.__version__
        except:
            pyklipver = 'unknown'
        hdulist[0].header['PSFSUB'] = ('pyKLIP', 'PSF Subtraction Algo')
        hdulist[0].header.add_history('Reduced with pyKLIP using commit {0}'.format(pyklipver))
        hdulist[0].header['CREATOR'] = 'pyKLIP-{0}'.format(pyklipver)

        # Store commit number for pyklip
        hdulist[0].header['pyklipv'] = (pyklipver, 'pyKLIP version that was used')

        if klipparams is not None:
            klipparams_clean = klipparams.replace('\n', '')# Get rid of Non-ascii characters that sometimes sneak in.
            hdulist[0].header['PSFPARAM'] = (klipparams_clean, 'KLIP parameters')
            hdulist[0].header.add_history('pyKLIP reduction with parameters {0}'.format(klipparams_clean))

        # Write z axis units if necessary
        if zaxis is not None:
            # Writing a KL mode cube
            if 'KL Mode' in filetype:
                hdulist[0].header['CTYPE3'] = 'KLMODES'
                # Write them individually
                for i, klmode in enumerate(zaxis):
                    hdulist[0].header['KLMODE{0}'.format(i)] = (klmode, 'KL Mode of slice {0}'.format(i))
                hdulist[0].header['CUNIT3'] = 'N/A'
                hdulist[0].header['CRVAL3'] = 1
                hdulist[0].header['CRPIX3'] = 1.
                hdulist[0].header['CD3_3'] = 1.

        # Store WCS information
        wcshdr = self.output_wcs[0].to_header()
        for key in wcshdr.keys():
            hdulist[0].header[key] = wcshdr[key]

        # Store extra keywords in header
        if more_keywords is not None:
            for hdr_key in more_keywords:
                hdulist[0].header[hdr_key] = more_keywords[hdr_key]

        # But update the image center
        center = self.output_centers[0]
        hdulist[0].header.update({'PSFCENTX': center[0], 'PSFCENTY': center[1]})
        hdulist[0].header.update({'CRPIX1': center[0], 'CRPIX2': center[1]})
        hdulist[0].header.add_history('Image recentered to {0}'.format(str(center)))

        try:
            hdulist.writeto(filepath, overwrite=True)
        except TypeError:
            hdulist.writeto(filepath, clobber=True)
        hdulist.close()


def trim_miri_data(data, filt):
    '''
    Trim the MIRI data to remove regions that receive no illumination.

    Parameters
    ----------
    data : list of datacubes
        List of 3D datacubes that will be trimmed
    filt : string
        Filter data was gathered in

    Returns
    -------
    data_trim : list of datacubes
        Trimmed 3D datacubes
    trim : list
        Number of pixels trimmed from the left (trim[0]) and bottom (trim[1]))
    '''

    # Pixel values to trim around based on filter/mask, these were determined using
    # the MIRI psfmask files found on CRDS.
    if filt.lower() == 'f1065c':
        l,r,b,t = 14, 227, 5, 217
    elif filt.lower() == 'f1140c':
        l,r,b,t = 13, 227, 7, 216
    elif filt.lower() == 'f1550c':
        l,r,b,t = 13, 226, 8, 215
    elif filt.lower() == 'f2300c':
        l,r,b,t = 9, 277, 32, 299
    else:
        raise ValueError(f'Filter {filt} does not match one of [F1065C, F1140C, F1550C, F2300C]')

    # Copy data and trim accordingly
    data_trim = copy.copy(data)
    for i, arr in enumerate(data):
        data_trim[i] = arr[:,b:t+1,l:r+1] #Want to include the final row/column

    # Trim is how many left and bottom pixels were cut off
    trim = [l , b]

    return data_trim, trim

def organise_files(filepaths, copy_dir='./ORGANISED/', hierarchy='TARGPROP/FILTER'):
    """
    Function to take a list of JWST files, and then copy and organise them
    into folders based on header keys.

    Parameters
    ----------
    file_list : list of str
        List of strings for each file.
    copy_dir : str
        Directory to copy files to.
    hierarchy : str
        Structure of the new directory organisation, using available header
        keywords.
    """
    # Check if directory we are copying to exists
    if not os.path.isdir(copy_dir):
        os.makedirs(copy_dir)

    # Get the keys we want to sort by
    divisions = hierarchy.split('/')

    # Loop over all of the files
    for file in filepaths:
        with fits.open(file) as f:
            # Loop over each of the keys we are interested in to create a
            # directory string
            working_dir = copy_dir
            for i in range(len(divisions)):
                key_val = f[0].header[divisions[i]]
                working_dir += key_val+'/'

            # Check if this new directory string exists
            if not os.path.isdir(working_dir):
                os.makedirs(working_dir)

            # Save file to the new directory
            file_suffix = file.split('/')[-1]
            shutil.copyfile(file, working_dir+file_suffix)

    return None
