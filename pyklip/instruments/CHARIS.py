import os
import copy
import re
import numpy as np
import scipy.ndimage as ndimage
import astropy.io.fits as fits
import datetime
from astropy import wcs
import astropy.time as time
import astropy.coordinates as coord
import astropy.units as u
import pyklip.klip as klip
from pyklip.instruments.Instrument import Data
import pyklip.fakes as fakes
import pyklip.instruments.utils.global_centroid as global_centroid

class CHARISData(Data):
    """
    A sequence of CHARIS Data. Each CHARISData object has the following fields and functions

    Args:
        filepaths: list of filepaths to files
        skipslices: a list of datacube slices to skip (supply index numbers e.g. [0,1,2,3])
        PSF_cube: 3D array (nl,ny,nx) with the PSF cube to be used in the flux calculation.
        update_hrs: if True, update input file headers by making sat spot measurements. If None, will only update if missing hdr info


    Attributes:
        input: Array of shape (N,y,x) for N images of shape (y,x)
        centers: Array of shape (N,2) for N centers in the format [x_cent, y_cent]
        filenums: Array of size N for the numerical index to map data to file that was passed in
        filenames: Array of size N for the actual filepath of the file that corresponds to the data
        PAs: Array of N for the parallactic angle rotation of the target (used for ADI) [in degrees]
        wvs: Array of N wavelengths of the images (used for SDI) [in microns]. For polarization data, defaults to "None"
        wcs: Array of N wcs astormetry headers for each image.
        IWA: a floating point scalar (not array). Specifies to inner working angle in pixels
        OWA: a floating point scalar (not array). Specifies to outer working angle in pixels
        output: Array of shape (b, len(files), len(uniq_wvs), y, x) where b is the number of different KL basis cutoffs
        wv_indices: Array of N indicies specifying the slice of datacube this frame comes frame (accounts of skipslices)
                You can use this to index into the header to grab info for the respective slice
        spot_flux: Array of N of average satellite spot flux for each frame
        dn_per_contrast: Flux calibration factor in units of DN/contrast (divide by image to "calibrate" flux)
                Can also be thought of as the DN of the unocculted star
        flux_units: units of output data [DN, contrast]
        prihdrs: Array of N primary headers
        exthdrs: Array of N extension headers
        bad_sat_spots: a list of up to 4 elements indicating if a sat spot is systematically bad. Indexing is based on
            sat spot x location. Possible values are 0,1,2,3. [0,3] would mark upper left and lower right sat spots bad

    Methods:
        readdata(): reread in the data
        savedata(): save a specified data in the GPI datacube format (in the 1st extension header)
        calibrate_output(): calibrates flux of self.output
    """
    ##########################
    ###Class Initilization ###
    ##########################
    # some static variables to define the CHARIS instrument
    centralwave = {}  # in microns
    fpm_diam = {}  # in pixels
    flux_zeropt = {}
    spot_ratio = {} #w.r.t. central star
    # the quoted value for CHARIS lenslet scale is 16.2 mas/pixel
    lenslet_scale_x = 0.01616 # lenslet scale, calibrated against HST images of M5
    lenslet_scale_y = 0.01603 # lenslet scale, calibrated against HST images of M5
    lenslet_scale_x_err = 0.00005
    lenslet_scale_y_err = 0.00007
    lenslet_scale = 0.01616 # CHARIS data will be re-scaled to this uniform lenslet scale
    ifs_rotation = 0.0  # degrees CCW from +x axis to zenith

    obs_latitude = 19 + 49./60 + 43./3600 # radians
    obs_longitude = -(155 + 28./60 + 50./3600) # radians

    ####################
    ### Constructors ###
    ####################

    def __init__(self, filepaths, guess_spot_index=0, guess_spot_locs=None, guess_center_loc=None, skipslices=None,
                 PSF_cube=None, update_hdrs=None, sat_fit_method='global', IWA=None, OWA=None):
        """
        Initialization code for CHARISData

        Note:
            Argument information is in the GPIData class definition docstring
        """
        super(CHARISData, self).__init__()
        self._output = None
        self.flipx = False
        self.readdata(filepaths, guess_spot_index=guess_spot_index, guess_spot_locs=guess_spot_locs,
                      guess_center_loc=guess_center_loc, skipslices=skipslices, PSF_cube=PSF_cube,
                      update_hdrs=update_hdrs, sat_fit_method=sat_fit_method, IWA=IWA, OWA=OWA)

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
    def ivars(self):
        return self._ivars
    @ivars.setter
    def ivars(self, newval):
        self._ivars = newval

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
    def OWA(self):
        return self._OWA
    @OWA.setter
    def OWA(self, newval):
        self._OWA = newval

    @property
    def output(self):
        return self._output
    @output.setter
    def output(self, newval):
        self._output = newval

    ###############
    ### Methods ###
    ###############

    def readdata(self, filepaths, guess_spot_index=0, guess_spot_locs=None, guess_center_loc=None, skipslices=None,
                 PSF_cube=None, update_hdrs=None, sat_fit_method='global', IWA=None, OWA=None):
        """
        Method to open and read a list of CHARIS data

        Args:
            filespaths: a list of filepaths
            guess_spot_index: the wavelength index for which the initial guess is given
            guess_spot_locs: initial guess of the satellite spot pixel indices.
                            If None, will default to rough guesses for the four satellite spots of a typical
                            CHARIS data cube at the first wavelength slice, in [[x, y],...] format
            guess_center_loc: initial guess of the primary star center in [x, y] format
            skipslices: a list of wavelenegth slices to skip for each datacube (supply index numbers e.g. [0,1,2,3])
            PSF_cube: 3D array (nl,ny,nx) with the PSF cube to be used in the flux calculation.
            update_hrs: if True, update input file headers by making sat spot measurements. If None, will only update if missing hdr info
            sat_fit_method: 'global' or 'local'
            IWA: a floating point scalar (not array). Specifies to inner working angle in pixels
            OWA: a floating point scalar (not array). Specifies to outer working angle in pixels

        Returns:
            Technically none. It saves things to fields of the CHARISData object. See object doc string
        """
        # check to see if user just inputted a single filename string
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        # check that the list of files actually contains something
        if len(filepaths) == 0:
            raise ValueError("An empty filelist was passed in")

        if guess_spot_locs is None:
            # default to typical locations for 4 satellite spots in a CHARIS data cube, each location in [x, y] format
            guess_spot_locs = [[129, 90], [109, 129], [71, 109], [91, 70]]
            # TODO: modify _measure_sat_spots() to handle alternating diffraction spots,
            #       and then modify the conditional default guess_spot_locs values correspondingly

        #make some lists for quick appending
        data = []
        ivars = []
        filenums = []
        filenames = []
        rot_angles = []
        wvs = []
        wv_indices = []
        centers = []
        wcs_hdrs = []
        spot_fluxes = []
        spot_locs = []
        inttimes = []
        prihdrs = []
        exthdrs = []
        nan_indices = []

        if PSF_cube is not None:
            self.psfs = PSF_cube

        # flag to see if we modified any headers or data cubes
        # new cubes will be saved separately only when distortion correction is carried out (modified data), and in this
        # case, modified headers will be updated in the new files not the original ones. However, if only headers are
        # modified but not the data, then headers will be updated in the original files and no new files will be saved.
        modified_hdrs = False
        modified_data = np.zeros(len(filepaths)) # zero == False

        # read and organize data
        for index, filepath in enumerate(filepaths):
            with fits.open(filepath, lazy_load_hdus=False) as hdulist:
                cube = copy.copy(hdulist[1].data)
                ivar = copy.copy(hdulist[2].data)
                prihdr = copy.copy(hdulist[0].header)
                exthdr = copy.copy(hdulist[1].header)
                w = wcs.WCS(header=prihdr, naxis=[1, 2])
                astr_hdrs = [w.deepcopy() for _ in range(cube.shape[0])] # repeat astrom header for each wavelength slice

            # mask pixels that receive no light as nans. Include masking a 1 pix boundary around NaNs
            # only record nan indices here, apply mask after distortion correction and global centroid
            input_minfilter = ndimage.minimum_filter(cube, (0, 1, 1))
            nan_indices.append(np.where(input_minfilter == 0))

            # recalculate parang if necessary
            try:
                parang = prihdr['PARANG'] + 113.5
            except:
                parang = 0.

            # compute weavelengths
            cube_wv_indices = np.arange(cube.shape[0])
            thiswvs = prihdr['LAM_MIN'] * np.exp(cube_wv_indices * prihdr['DLOGLAM'])  # nm
            thiswvs /= 1e3  # now in microns

            # remove undesirable slices of the datacube if necessary
            if skipslices is not None:
                cube = np.delete(cube, skipslices, axis=0)
                ivar = np.delete(ivar, skipslices, axis=0)
                thiswvs = np.delete(thiswvs, skipslices)
                cube_wv_indices = np.delete(cube_wv_indices, skipslices)
                astr_hdrs = np.delete(astr_hdrs, skipslices)

            data.append(cube)
            ivars.append(ivar)
            rot_angles.append(np.ones(cube.shape[0], dtype=int) * parang)
            wvs.append(thiswvs)
            wv_indices.append(cube_wv_indices)
            filenums.append(np.ones(cube.shape[0], dtype=int) * index)
            wcs_hdrs.append(astr_hdrs)
            try:
                exptime = prihdr['EXPTIME']
            except KeyError:
                exptime = 20
            inttimes.append(np.ones(cube.shape[0], dtype=int) * exptime)
            prihdrs.append(prihdr)
            exthdrs.append(exthdr)
            filenames.append([filepath for i in range(cube.shape[0])])

        # rescale all data to a uniform lenslet scale
        # TODO: distortion correction in parallel?
        for index, filepath in enumerate(filepaths):
            try:
                if exthdrs[index]['PLATECAL'].lower() == 'true':
                    continue
            except:
                pass
            filename, cube, ivar, exthdr = _distortion_correction(filepath, data[index], ivars[index], exthdrs[index],
                                                                  CHARISData.lenslet_scale, CHARISData.lenslet_scale_x,
                                                                  CHARISData.lenslet_scale_y)
            data[index] = cube
            ivars[index] = ivar
            exthdrs[index] = exthdr
            filenames[index] = [filename for i in range(cube.shape[0])]
            modified_data[index] = 1
        modified_data = modified_data.astype(bool)
        # measure and update headers in the next two segments

        # fit for satellite spots globally if enabled
        spot_fit_coeffs = None
        if update_hdrs == True and sat_fit_method.lower() == 'global':
            print('fitting satellite spots globally')
            spot_fit_coeffs, photcal = _measure_sat_spots_global(filenames, data, ivars, prihdrs, guess_center_loc=guess_center_loc)

        # fit for satellite spots locally if enabled
        # read satellite spots info or update measured info depending on user input
        for index in range(len(data)):

            # not using copy here to utilize mutable numpy arrays
            # because modifications on an individual cube are intended to be applied to data as well
            cube = data[index]
            thiswvs = wvs[index]
            cube_wv_indices = wv_indices[index]

            # mask pixels that receive no light as nans. Include masking a 1 pix boundary around NaNs
            cube[nan_indices[index]] = np.nan

            if 'SATS0_0' in exthdr and not update_hdrs == True:
                # read in sat spots from file
                spot_loc, spot_flux = _read_sat_spots_from_hdr(exthdr, cube_wv_indices)
            elif sat_fit_method.lower() == 'global' and update_hdrs == True:
                try:
                    astrogrid_status = prihdr['X_GRDST']
                except:
                    astrogrid_status = None

                # TODO: decide whether to smooth, and whether to photocalibrate images here
                # TODO: spot_flux use peak flux or aperture flux?
                spot_loc, spot_flux = global_centroid.get_sats_satf(spot_fit_coeffs[index], cube, thiswvs,
                                                                    astrogrid=astrogrid_status)

                for wv_ind in cube_wv_indices:
                    for spot_num in range(len(spot_loc[wv_ind])):
                        fitx = spot_loc[wv_ind, spot_num, 0]
                        fity = spot_loc[wv_ind, spot_num, 1]
                        fitflux = spot_flux[wv_ind, spot_num]
                        if np.isnan(fitflux):
                            fitflux = 'nan'
                        _add_satspot_to_hdr(exthdr, wv_ind, spot_num, [fitx, fity], fitflux)
                spot_flux = np.nanmean(spot_flux, axis=1)
                modified_hdrs = True
            else:
                print("Finding satellite spots for cube {0}".format(index))
                try:
                    spot_loc, spot_flux, spot_fwhm = _measure_sat_spots(cube, thiswvs, guess_spot_index, guess_spot_locs,
                                                                        hdr=exthdr, highpass=False)
                except ValueError:
                    print("Unable to locate sat spots for cube{0}. Skipping...".format(index))
                    continue
                modified_hdrs = True # we need to update input cube headers

            # simple mean for center for now
            center = np.mean(spot_loc, axis=1)
            centers.append(center)
            spot_fluxes.append(spot_flux)
            spot_locs.append(spot_loc)

        # convert everything into numpy arrays
        # reshape arrays so that we collapse all the files together (i.e. don't care about distinguishing files)
        # wv_indices are never referenced from this point on so they're commented out currently
        data = np.array(data)
        dims = data.shape
        data = data.reshape([dims[0] * dims[1], dims[2], dims[3]])
        ivars = np.array(ivars).reshape([dims[0] * dims[1], dims[2], dims[3]])
        filenums = np.array(filenums).reshape([dims[0] * dims[1]])
        filenames = np.array(filenames).reshape([dims[0] * dims[1]])
        rot_angles = (np.array(rot_angles).reshape([dims[0] * dims[1]])) 
        wvs = np.array(wvs).reshape([dims[0] * dims[1]]) # wvs now has shape (N*nwv)
        # wv_indices = np.array(wv_indices).reshape([dims[0] * dims[1]])
        wcs_hdrs = np.array(wcs_hdrs).reshape([dims[0] * dims[1]])
        centers = np.array(centers).reshape([dims[0] * dims[1], 2])
        spot_fluxes = np.array(spot_fluxes).reshape([dims[0] * dims[1]])
        spot_locs = np.array(spot_locs).reshape([dims[0] * dims[1], spot_locs[0].shape[1], 2])
        inttimes = np.array(inttimes).reshape([dims[0] * dims[1]])

        # if there is more than 1 integration time, normalize all data to the first integration time
        if np.size(np.unique(inttimes)) > 1:
            inttime0 = inttimes[0]
            # normalize integration times
            data = data * inttime0/inttimes[:, None, None]
            spot_fluxes *= inttime0/inttimes


        #set these as the fields for the CHARISData object
        self._input = data
        self._ivars = ivars
        self._centers = centers
        self._filenums = filenums
        self._filenames = filenames
        self._PAs = rot_angles
        self._wvs = wvs
        self._wcs = wcs_hdrs
        self._IWA = 5
        self.prihdrs = prihdrs
        self.exthdrs = exthdrs

        self.spot_fluxes = spot_fluxes
        self.spot_locs = spot_locs

        if IWA != None:
            self._IWA = IWA
        if OWA != None:
            self._OWA = OWA

        # Required for automatically querying Simbad for the spectral type of the star.
        try:
            self.object_name = self.prihdrs[0]["OBJECT"]
        except KeyError:
            self.object_name = "None"

        # if cube has been distortion corrected, save to new file
        if np.any(modified_data == True):
            self.save_plate_cal_cubes(modified_data, dims)

        if update_hdrs or (update_hdrs is None and modified_hdrs):
            print("Updating original headers with sat spot measurements.")
            self.update_input_cubes()


    def update_input_cubes(self):
        """
        Updates the input spectral data cubes with the current headers. This is useful to permanately save the
        measured sat spot locations to disk. 
        """
        for filename, hdr in zip(np.unique(self.filenames), self.exthdrs):
            with fits.open(filename, mode='update') as hdulist:
                hdulist[1].header = hdr
                hdulist.flush()


    def save_plate_cal_cubes(self, modified_data, dims):

        modified_indices = np.argwhere(modified_data == True)
        save_data = np.reshape(self._input, dims)
        save_ivars = self.ivars.reshape(dims)
        save_filenames = np.unique(self.filenames)
        for index in modified_indices:
            hdulist = fits.HDUList()
            hdulist.append(fits.PrimaryHDU(header=self.prihdrs[index[0]], data=None))
            hdulist.append(fits.PrimaryHDU(header=self.exthdrs[index[0]], data=save_data[index[0]]))
            hdulist.append(fits.PrimaryHDU(data=save_ivars[index[0]]))
            hdulist.writeto(save_filenames[index[0]], overwrite=True)

    # TODO: is this savedata function still needed for CHARIS.py?
    def savedata(self, filepath, data, klipparams = None, filetype = None, zaxis = None, more_keywords=None,
                 center=None, astr_hdr=None, fakePlparams = None,user_prihdr = None, user_exthdr = None,
                 extra_exthdr_keywords = None, extra_prihdr_keywords = None):
        """
        Save data and header in the first extension header

        Note: In principle, the function only works inside klip_dataset(). In order to use it outside of klip_dataset,
            you need to define the following attributes:
                dataset.output_wcs = np.array([w.deepcopy() if w is not None else None for w in dataset.wcs])
                dataset.output_centers = dataset.centers

        Args:
            filepath: path to file to output
            data: 2D or 3D data to save
            klipparams: a string of klip parameters
            filetype: filetype of the object (e.g. "KL Mode Cube", "PSF Subtracted Spectral Cube")
            zaxis: a list of values for the zaxis of the datacub (for KL mode cubes currently)
            more_keywords (dictionary) : a dictionary {key: value, key:value} of header keywords and values which will
                                         written into the primary header
            astr_hdr: wcs astrometry header
            center: center of the image to be saved in the header as the keywords PSFCENTX and PSFCENTY in pixels.
                The first pixel has coordinates (0,0)
            fakePlparams: fake planet params
            user_prihdr: User defined primary headers to be used instead
            user_exthdr: User defined extension headers to be used instead
            extra_exthdr_keywords: Fits keywords to be added to the extension header before saving the file
            extra_prihdr_keywords: Fits keywords to be added to the primary header before saving the file

        """
        hdulist = fits.HDUList()
        if user_prihdr is None:
            hdulist.append(fits.PrimaryHDU(header=self.prihdrs[0]))
        else:
            hdulist.append(fits.PrimaryHDU(header=user_prihdr))
        if user_exthdr is None:
            hdulist.append(fits.ImageHDU(header=self.exthdrs[0], data=data, name="Sci"))
        else:
            hdulist.append(fits.ImageHDU(header=user_exthdr, data=data, name="Sci"))

        # save all the files we used in the reduction
        # we'll assume you used all the input files
        # remove duplicates from list
        filenames = np.unique(self.filenames)
        nfiles = np.size(filenames)
        # The following paragraph is only valid when reading raw GPI cube.
        try:
            hdulist[0].header["DRPNFILE"] = (nfiles, "Num raw files used in pyKLIP")
            for i, thispath in enumerate(filenames):
                thispath = thispath.replace("\\", '/')
                splited = thispath.split("/")
                fname = splited[-1]
                matches = re.search('S20[0-9]{6}[SE][0-9]{4}(_fixed)?', fname)
                filename = matches.group(0)
                hdulist[0].header["FILE_{0}".format(i)] = filename + '.fits'
        except:
            pass

        # write out psf subtraction parameters
        # get pyKLIP revision number
        pykliproot = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        # the universal_newline argument is just so python3 returns a string instead of bytes
        # this will probably come to bite me later
        try:
            pyklipver = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=pykliproot, universal_newlines=True).strip()
        except:
            pyklipver = "unknown"
        hdulist[0].header['PSFSUB'] = ("pyKLIP", "PSF Subtraction Algo")
        if user_prihdr is None:
            hdulist[0].header.add_history("Reduced with pyKLIP using commit {0}".format(pyklipver))
        if self.creator is None:
            hdulist[0].header['CREATOR'] = "pyKLIP-{0}".format(pyklipver)
        else:
            hdulist[0].header['CREATOR'] = self.creator
            hdulist[0].header.add_history("Reduced by {0}".format(self.creator))

        # store commit number for pyklip
        hdulist[0].header['pyklipv'] = (pyklipver, "pyKLIP version that was used")

        if klipparams is not None:
            hdulist[0].header['PSFPARAM'] = (klipparams, "KLIP parameters")
            hdulist[0].header.add_history("pyKLIP reduction with parameters {0}".format(klipparams))

        if fakePlparams is not None:
            hdulist[0].header['FAKPLPAR'] = (fakePlparams, "Fake planet parameters")
            hdulist[0].header.add_history("pyKLIP reduction with fake planet injection parameters {0}".format(fakePlparams))
        # store file type
        if filetype is not None:
            hdulist[0].header['FILETYPE'] = (filetype, "CHARIS File type")

        # store extra keywords in header
        if more_keywords is not None:
            for hdr_key in more_keywords:
                hdulist[0].header[hdr_key] = more_keywords[hdr_key]

        # JB's code to store keywords
        if extra_prihdr_keywords is not None:
            for name,value in extra_prihdr_keywords:
                hdulist[0].header[name] = value
        if extra_exthdr_keywords is not None:
            for name,value in extra_exthdr_keywords:
                hdulist[1].header[name] = value

        # write z axis units if necessary
        if zaxis is not None:
            #Writing a KL mode Cube
            if "KL Mode" in filetype:
                hdulist[1].header['CTYPE3'] = 'KLMODES'
                #write them individually
                for i, klmode in enumerate(zaxis):
                    hdulist[1].header['KLMODE{0}'.format(i)] = (klmode, "KL Mode of slice {0}".format(i))
            elif "spec" in filetype.lower():
                hdulist[1].header['CTYPE3'] = 'WAVE'
            else:
                hdulist[1].header['CTYPE3'] = 'NONE'

        if np.ndim(data) == 2:
            if 'CTYPE3' in  hdulist[1].header.keys():
                hdulist[1].header['CTYPE3'] = 'NONE'

        if user_exthdr is None:
            #use the dataset astr hdr if none was passed in
            if astr_hdr is None:
                astr_hdr = self.output_wcs[0]
            if astr_hdr is not None:
                #update astro header
                #I don't have a better way doing this so we'll just inject all the values by hand
                astroheader = astr_hdr.to_header()
                exthdr = hdulist[1].header
                exthdr['PC1_1'] = astroheader['PC1_1']
                exthdr['PC2_2'] = astroheader['PC2_2']
                try:
                    exthdr['PC1_2'] = astroheader['PC1_2']
                    exthdr['PC2_1'] = astroheader['PC2_1']
                except KeyError:
                    exthdr['PC1_2'] = 0.0
                    exthdr['PC2_1'] = 0.0
                #remove CD values as those are confusing
                try:
                    exthdr.remove('CD1_1')
                    exthdr.remove('CD1_2')
                    exthdr.remove('CD2_1')
                    exthdr.remove('CD2_2')
                except:
                    pass # nothing to do if they were removed already
                exthdr['CDELT1'] = 1
                exthdr['CDELT2'] = 1

            #use the dataset center if none was passed in
            if center is None:
                center = self.output_centers[0]
            if center is not None:
                hdulist[1].header.update({'PSFCENTX':center[0],'PSFCENTY':center[1]})
                hdulist[1].header.update({'CRPIX1':center[0],'CRPIX2':center[1]})
                hdulist[0].header.add_history("Image recentered to {0}".format(str(center)))

        try:
            hdulist.writeto(filepath, overwrite=True)
        except TypeError:
            hdulist.writeto(filepath, clobber=True)
        hdulist.close()


    def calibrate_output(self, img, spectral=False, units="contrast"):
        """
        Calibrates the flux of an output image. Can either be a broadband image or a spectral cube depending
        on if the spectral flag is set.

        Assumes the broadband flux calibration is just multiplication by a single scalar number whereas spectral
        datacubes may have a separate calibration value for each wavelength

        Args:
            img: unclaibrated image.
                 If spectral is not set, this can either be a 2-D or 3-D broadband image
                 where the last two dimensions are [y,x]
                 If specetral is True, this is a 3-D spectral cube with shape [wv,y,x]
            spectral: if True, this is a spectral datacube. Otherwise, it is a broadband image.
            units: currently only support "contrast" w.r.t central star

        Returns:
            img: calibrated image of the same shape (this is the same object as the input!!!)
        """
        return img


    def generate_psfs(self, boxrad=7):
        """
        Generates PSF for each frame of input data. Only works on spectral mode data.

        Args:
            boxrad: the halflength of the size of the extracted PSF (in pixels)

        Returns:
            saves PSFs to self.psfs as an array of size(N,psfy,psfx) where psfy=psfx=2*boxrad + 1
        """
        self.psfs = []

        for i,frame in enumerate(self.input):
            this_spot_locs = self.spot_locs[i]

            # now grab the values from them by parsing the header
            spot0 = this_spot_locs[0]
            spot1 = this_spot_locs[1]
            spot2 = this_spot_locs[2]
            spot3 = this_spot_locs[3]

            # put all the sat spot info together
            spots = [spot0, spot1, spot2, spot3]

            #now make a psf
            spotpsf = generate_psf(frame, spots, boxrad=boxrad)
            self.psfs.append(spotpsf)

        self.psfs = np.array(self.psfs)
        self.uncollapsed_psfs = self.psfs

        # collapse in time dimension
        numwvs = np.size(np.unique(self.wvs))
        self.psfs = np.reshape(self.psfs, (self.psfs.shape[0]//numwvs, numwvs, self.psfs.shape[1], self.psfs.shape[2]))
        self.psfs = np.mean(self.psfs, axis=0)


def _distortion_correction(filename, cube, ivar, exthdr, lenslet_scale, xscale, yscale):
    # TODO: how to modify rotation matrix coefficient?
    # TODO: any other header info needs to be modified?
    '''
    Rescale cube and ivar to a uniform lenslet scale, update extension header indicating distortion correction status,
    return a new filename along with distortion corrected cube, ivar, and updated extension header.
    All proceeding operations will be done on the distortion corrected data, which will be saved to the new filenames

    Args:
        filename: filename for the cube
        cube: data cube
        ivar: inverse variance cube for the corresponding data cube
        exthdr: extension header for the data cube
        lenslet_scale: uniform scale the data will be correct to
        xscale: measured x-axis lenslet scale for uncorrected CHARIS data
        yscale: measured y-axis lenslet scale for uncorrected CHARIS data

    Returns:
        filename: an updated path to save the distortion corrected file
        cube: distortion corrected cube
        ivar: distortion corrected inverse variance
        exthdr: updated header with keyword "PLATE_CAL" indicating distortion correction status
    '''

    # rescaling data and inverse variance to a uniform lenslet scale
    x = 1. * np.arange(cube.shape[1]) - cube.shape[1] // 2
    x, y = np.meshgrid(x, x)
    x_scaling = lenslet_scale / xscale
    y_scaling = lenslet_scale / yscale
    x = x * x_scaling + cube.shape[1] // 2
    y = y * y_scaling + cube.shape[1] // 2
    for i in range(cube.shape[0]):
        cube[i] = ndimage.map_coordinates(cube[i], [y, x])
        ivar[i] = ndimage.map_coordinates(ivar[i], [y, x])

    plate_cal_key = 'PLATECAL'
    plate_cal_str = 'True'
    exthdr.set(plate_cal_key, value=plate_cal_str, comment='distortion correction')
    filename = re.sub('.fits', '_plate_cal.fits', filename)

    return filename, cube, ivar, exthdr


def _read_sat_spots_from_hdr(hdr, wv_indices):
    """
    Read in the sat spot locations and fluxes from a header.

    We are looking for SATS0_0 for spot 0 in first slice. x/y position stored as string 'x y'
    We are looking for SATF0_0 for spot 0 flux in first slice. Stored as float
    """
    spot_locs = []
    spot_fluxes = []
    if hdr['X_GRDST'] == 'Xdiag' or hdr['X_GRDST'] == 'Ydiag':
        number_of_spots = 2
    else:
        number_of_spots = 4

    for i in wv_indices:
        thiswv_spot_locs = []
        thiswv_spot_fluxes = []

        for spot_num in range(number_of_spots):
            loc_str = hdr['SATS{0}_{1}'.format(i, spot_num)]
            loc = loc_str.split()
            loc = [float(loc[0]), float(loc[1])]

            flux = hdr['SATF{0}_{1}'.format(i, spot_num)]

            thiswv_spot_fluxes.append(flux)
            thiswv_spot_locs.append(loc)

        spot_locs.append(thiswv_spot_locs)
        spot_fluxes.append(np.nanmean(thiswv_spot_fluxes))

    return np.array(spot_locs), np.array(spot_fluxes)

#TODO: modify _measure_sat_spots() to handle alternating diffraction spots
def _measure_sat_spots(cube, wvs, guess_spot_index, guess_spot_locs, highpass=True, hdr=None):
    """
    Find sat spots in a datacube. TODO: return sat spot psf cube also

    If a header is passed, it will write the sat spot fluxes and locations to the header

    Args:
        cube: a single data cube
        wvs: array of wavelength values for this data cube, different from "wvs" in CHARISData class, this wvs here has shape (nwv,)
    """
    # use dictionary to store list of locs/fluxes for each slice
    spot_locs = {}
    spot_fluxes = {}
    spot_fwhms = {}

    # start with guess center
    start_frame = cube[guess_spot_index]
    if highpass:
        start_frame = klip.high_pass_filter(start_frame, 10)

    start_spot_locs = []
    start_spot_fluxes = []
    start_spot_fwhms = []
    for spot_num, guess_spot_loc in enumerate(guess_spot_locs):
        xguess, yguess = guess_spot_loc
        fitargs = fakes.airyfit2d(start_frame, xguess, yguess, searchrad=7)
        #fitargs = fakes.gaussfit2d(start_frame, xguess, yguess, searchrad=4)
        fitflux, fitfwhm, fitx, fity = fitargs
        start_spot_locs.append([fitx, fity])
        start_spot_fluxes.append(fitflux)
        start_spot_fwhms.append(fitfwhm)
        _add_satspot_to_hdr(hdr, guess_spot_index, spot_num, [fitx, fity], fitflux)

    spot_locs[guess_spot_index] = start_spot_locs
    spot_fluxes[guess_spot_index] = np.nanmean(start_spot_fluxes)
    spot_fwhms[guess_spot_index] = np.nanmean(start_spot_fwhms)

    # set this reference center to use for finding the spots at other wavelengths
    ref_wv = wvs[guess_spot_index]
    ref_center = np.mean(start_spot_locs, axis=0)
    ref_spot_locs_deltas = np.array(start_spot_locs) - ref_center[None, :] # offset from center

    for i, (frame, wv) in enumerate(zip(cube, wvs)):
        # we already did the inital index
        if i == guess_spot_index:
            continue

        if highpass:
            frame = klip.high_pass_filter(frame, 10)

        # guess where the sat spots are based on the wavelength
        wv_scaling = wv/ref_wv # shorter wavelengths closer in
        thiswv_guess_spot_locs = ref_spot_locs_deltas * wv_scaling + ref_center

        # fit each sat spot now
        thiswv_spot_locs = []
        thiswv_spot_fluxes = []
        thiswv_spot_fwhms = []
        for spot_num, guess_spot_loc in enumerate(thiswv_guess_spot_locs):
            xguess, yguess = guess_spot_loc
            fitargs = fakes.airyfit2d(frame, xguess, yguess, searchrad=7)
            #fitargs = fakes.gaussfit2d(frame, xguess, yguess, searchrad=4)
            fitflux, fitfwhm, fitx, fity = fitargs
            thiswv_spot_locs.append([fitx, fity])
            thiswv_spot_fluxes.append(fitflux)
            thiswv_spot_fwhms.append(fitfwhm)
            _add_satspot_to_hdr(hdr, i, spot_num, [fitx, fity], fitflux)

        spot_locs[i] = thiswv_spot_locs
        spot_fluxes[i] = np.nanmean(thiswv_spot_fluxes)
        spot_fwhms[i] = np.nanmean(thiswv_spot_fwhms)

    # turn them into numpy arrays
    locs = []
    fluxes = []
    fwhms = []
    for i in range(cube.shape[0]):
        locs.append(spot_locs[i])
        fluxes.append(spot_fluxes[i])
        fwhms.append(spot_fwhms[i])

    return np.array(locs), np.array(fluxes), np.array(fwhms)


def _measure_sat_spots_global(filenames, cubes, ivars, prihdrs, photocal=False, guess_center_loc=None):
    '''
    Main function of this module to fit for the locations of the four satellite spots

    Args:
        TODO: check if fids in global_centroid.py can be replaced by simply np.arange(len(cubes))
        filenames: list of filenames for every frame of all cubes, shape (ncube, nwv)
        cubes: list of input data cubes to be recentered
        ivars: inverse variance frames corresponding to cubes
        photocal: boolean, whether to scale each wavelength to the same photometric scale
        guess_center_loc:

    Returns:
        p: fitted coefficients for all data cubes
        phot: photocalibration scale factor

    '''

    # TODO: spotflux use peak flux or aperture flux?

    filepaths = np.unique(filenames)
    centroid_params, x, y, mask = global_centroid.fitcen_parallel(filepaths, cubes, ivars, prihdrs, smooth_coef=True,
                                                                  guess_center_loc=guess_center_loc)
    xsol, ysol = global_centroid.fitallrelcen(cubes, ivars, r1=15, r2=45)

    p = centroid_params.copy()
    if mask is not None:
        mask = np.where(mask)

    p[:, 2] += xsol - x + np.median((x - xsol)[mask])
    p[:, 5] += ysol - y + np.median((y - ysol)[mask])

    if photocal:
        # TODO: implement a proper spectral photo-calibration somewhere in the reduction sequence
        phot = global_centroid.specphotcal(filepaths, cubes, prihdrs, p)
    else:
        phot = 1.

    return p, phot


def _add_satspot_to_hdr(hdr, wv_ind, spot_num, pos, flux):
    """
    Write a single meausred satellite spot to the header
    """
    # make sure indicies are integers
    wv_ind = int(wv_ind)
    spot_num = int(spot_num)
    # define keys
    pos_key = 'SATS{0}_{1}'.format(wv_ind, spot_num)
    flux_key = 'SATF{0}_{1}'.format(wv_ind, spot_num)

    # store x/y postion as a string
    pos_string = "{x:7.3f} {y:7.3f}".format(x=pos[0], y=pos[1])

    # write positions
    hdr.set(pos_key, value=pos_string, comment="Location of sat. spot {1} of slice {0}".format(wv_ind, spot_num))
    # write flux
    hdr.set(flux_key, value=flux, comment="Peak flux of sat. spot {1} of slice {0}".format(wv_ind, spot_num))


def generate_psf(frame, locations, boxrad=5):
    """
    Generates a GPI PSF for the frame based on the satellite spots

    Args:
        frame: 2d frame of data
        location: array of (N,2) containing [x,y] coordinates of all N satellite spots
        boxrad: half length of box to use to pull out PSF

    Returns:
        genpsf: 2d frame of size (2*boxrad+1, 2*boxrad+1) with average PSF of satellite spots
    """
    genpsf = []
    #mask nans
    cleaned = np.copy(frame)
    cleaned[np.where(np.isnan(cleaned))] = 0


    for loc in locations:
        #grab satellite spot positions
        spotx = loc[0]
        spoty = loc[1]

        #interpolate image to grab satellite psf with it centered
        #add .1 to make sure we get 2*boxrad+1 but can't add +1 due to floating point precision (might cause us to
        #create arrays of size 2*boxrad+2)
        x, y = np.meshgrid(np.arange(spotx - boxrad, spotx + boxrad + 0.1, 1), np.arange(spoty - boxrad, spoty + boxrad + 0.1, 1))
        spotpsf = ndimage.map_coordinates(cleaned, [y, x])

        # if applicable, do a background subtraction
        if boxrad >= 7:
            y_img, x_img = np.indices(frame.shape, dtype=float)
            r_img = np.sqrt((x_img - spotx)**2 + (y_img - spoty)**2)
            noise_annulus = np.where((r_img > 9) & (r_img <= 12))
            background_mean = np.nanmean(cleaned[noise_annulus])
            spotpsf -= background_mean

        genpsf.append(spotpsf)

    genpsf = np.array(genpsf)
    genpsf = np.mean(genpsf, axis=0) #average the different psfs together

    return genpsf
