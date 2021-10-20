from pyklip.instruments.Instrument import Data
import pyklip.rdi as rdi
import pyklip.klip

from astropy.io import fits
from astropy import wcs

import numpy as np
import re
import os

class JWSTData(Data):
    """
    In development class to interpret JWST data using pyKLIP
    """

    ############################
    ### Class Initialisation ###
    ############################

    ####################
    ### Constructors ###
    ####################
    def __init__(self, filepaths=None, psflib_filepaths=None, centering='basic'):
        # Initialize the super class
        super(JWSTData, self).__init__()

        # Get the target dataset
        self.readdata(filepaths, centering)

        # If necessary, get the PSF library dataset for RDI procedures
        if psflib_filepaths != None:
            self.readpsflib(psflib_filepaths)
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

    ###############
    ### Methods ###
    ###############

    def readdata(self, filepaths, centering='basic', verbose=False):
        """
        Method to open and read JWST data

        Args:
            filepaths : a list of file paths
            centering : string descriptor for method ot estimate image centers
            verbose : Boolean for terminal print statements

        Returns:
            None, data are saved to a JWSTData object
        """

        # Ensure we have a list of file paths
        if isinstance(filepaths, str):
            filepaths = [filepaths]
            if verbose: print('Only 1 file path was provided, are you sure you meant to do this?')

        # Check list is not empty
        if len(filepaths) == 0:
            raise ValueError('Empty filepath list provided to JWSTData!')


        # Intialise some arrays
        data = []
        centers = []
        filenames = []
        pas = []
        wvs = []
        wcs_hdrs = []

        # Go through files one by one
        for index, file in enumerate(filepaths):
            with fits.open(file) as f:
                # Grab number of integrations, as this is how many images are in the file
                nints = f[0].header['NINTS']
                pixel_scale = np.sqrt(f[1].header['PIXAR_A2']) # Need this for later to calculate IWA
                
                # Get the images
                data.append(f[1].data)

                # Get image centers based on desired algorithm
                if centering == 'basic':
                    # Use the mid point of the image
                    centers.append([i / 2.0  for i in f[1].data.shape[-2:]] * nints)
                else:
                    raise ValueError('No other centering algorithms developed yet.')

                # Assign filenames based on the file and the integration
                filenames += ['{}_INT{}'.format(file.split('/')[-1], i+1) for i in range(nints)]

                # Get PA for all frame withing the file
                pas.append([f[1].header['PA_V3']] * nints)

                # Get the filter wavelength, should be the same for each file though (for JWST)
                filt = f[0].header['FILTER']
                wave = int(re.findall("\d+", filt)[0]) / 1e2 #Get wavelength in micron
                wvs.append([wave]*nints)

                # Get WCS information
                wcs_hdr = wcs.WCS(header=f[1].header, naxis=f[1].header['WCSAXES']) 
                for i in range(nints):
                    wcs_hdrs.append(wcs_hdr.deepcopy())

        # Convert to numpy arrays and collapse integrations along a single axis
        data = np.array(data)
        data = data.reshape(-1, *data.shape[-2:]) 
        centers = np.array(centers).reshape(-1, 2)
        filenames = np.array(filenames)
        filenums = np.array(range(len(filenames)))
        pas = np.array(pas).flatten()
        wvs = np.array(wvs).flatten()

        # Assume an inner working angle of 1 lambda / D
        lambda_d_arcsec = ((wvs[0] / 1e6) / 6.5) * (180 / np.pi) * 3600
        IWA = lambda_d_arcsec / pixel_scale

        # Assign all necessary properties
        self._input = data
        self._centers = centers
        self._filenums = filenums
        self._filenames = filenames
        self._PAs = pas
        self._wvs = wvs
        self._wcs = wcs_hdrs
        self._IWA = IWA

    def readpsflib(self, psflib_filepaths, centering='basic', verbose=False):
        """
        Method to open and read JWST data for use as part of a PSF library

        Args:
            psflib_filepaths : a list of file paths
            centering : string descriptor for method ot estimate image centers
            verbose : Boolean for terminal print statements

        Returns:
            None, data are saved to the psflib property
        """

        # Ensure we have a list of file paths
        if isinstance(psflib_filepaths, str):
            psflib_filepaths = [psflib_filepaths]
            if verbose: print('Only 1 psflib filepath was provided, are you sure you meant to do this?')

        # Check list is not empty
        if len(psflib_filepaths) == 0:
            raise ValueError('Empty psflib filepath list provided to JWSTData!')

        psflib_data = []
        psflib_centers = []
        psflib_filenames = []

        # Prepare reference data for RDI subtractions
        for index, file in enumerate(psflib_filepaths):
            with fits.open(file) as f:
                #Grab some header parameters
                nints = f[0].header['NINTS']
                pixel_scale = np.sqrt(f[1].header['PIXAR_A2'])

                psflib_data.append(f[1].data)

                if centering == 'basic':
                    center = [dim / 2.0  for dim in f[1].data.shape[-2:]]
                    psflib_offset = [f[0].header['XOFFSET'], f[0].header['YOFFSET']] / pixel_scale
                    psflib_centers.append([sum(x) for x in zip(center, psflib_offset)] * nints)

                psflib_filenames += ['{}_INT{}'.format(file.split('/')[-1], i+1) for i in range(nints)]

        # Convert to numpy arrays andn collapse along integration axis
        psflib_data = np.array(psflib_data)
        psflib_data = psflib_data.reshape(-1, *psflib_data.shape[-2:]) 
        psflib_centers = np.array(psflib_centers).reshape(-1, 2)
        psflib_filenames = np.array(psflib_filenames)

        # Append the target images as well
        psflib_data = np.append(psflib_data, self._input, axis=0)
        psflib_filenames = np.append(psflib_filenames, self._filenames, axis=0)
        psflib_centers = np.append(psflib_centers, self._centers, axis=0)

        #Need to align the images so that they have the same centers. 
        image_center = np.array(psflib_data[0].shape) / 2.0
        for i, image in enumerate(psflib_data):
            recentered_image = pyklip.klip.align_and_scale(image, new_center=image_center, old_center=psflib_centers[i])
            psflib_data[i] = recentered_image

        # Create the PSF library
        psflib = rdi.PSFLibrary(psflib_data, image_center, psflib_filenames, compute_correlation=True)

        # Prepare the library with the target dataset
        psflib.prepare_library(self)

        self._psflib = psflib

    def savedata(self, filepath, data, klipparams=None, filetype="", zaxis=None):
        """
        Saves data for this instrument

        Args:
            filepath: filepath to save to
            data: data to save
            klipparams: a string of KLIP parameters. Write it to the 'PSFPARAM' keyword
            filtype: type of file (e.g. "KL Mode Cube", "PSF Subtracted Spectral Cube"). Wrriten to 'FILETYPE' keyword
            zaxis: a list of values for the zaxis of the datacub (for KL mode cubes currently)
            more_keywords (dictionary) : a dictionary {key: value, key:value} of header keywords and values which will
                                         written into the primary header
        """
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(data=data))

        # save all the files we used in the reduction
        # we'll assume you used all the input files
        # remove duplicates from list
        filenames = np.unique(self.filenames)
        nfiles = np.size(filenames)
        hdulist[0].header["DRPNFILE"] = (nfiles, "Num raw files used in pyKLIP")
        for i, filename in enumerate(filenames):
            hdulist[0].header["FILE_{0}".format(i)] = filename + '.fits'

        # write out psf subtraction parameters
        # get pyKLIP revision number
        pykliproot = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        # the universal_newline argument is just so python3 returns a string instead of bytes
        # this will probably come to bite me later
        try:
            pyklipver = pyklip.__version__
        except:
            pyklipver = "unknown"
        hdulist[0].header['PSFSUB'] = ("pyKLIP", "PSF Subtraction Algo")
        hdulist[0].header.add_history("Reduced with pyKLIP using commit {0}".format(pyklipver))
        hdulist[0].header['CREATOR'] = "pyKLIP-{0}".format(pyklipver)

        # store commit number for pyklip
        hdulist[0].header['pyklipv'] = (pyklipver, "pyKLIP version that was used")

        if klipparams is not None:
            hdulist[0].header['PSFPARAM'] = (klipparams, "KLIP parameters")
            hdulist[0].header.add_history("pyKLIP reduction with parameters {0}".format(klipparams))

        # write z axis units if necessary
        if zaxis is not None:
            # Writing a KL mode Cube
            if "KL Mode" in filetype:
                hdulist[0].header['CTYPE3'] = 'KLMODES'
                # write them individually
                for i, klmode in enumerate(zaxis):
                    hdulist[0].header['KLMODE{0}'.format(i)] = (klmode, "KL Mode of slice {0}".format(i))
                hdulist[0].header['CUNIT3'] = "N/A"
                hdulist[0].header['CRVAL3'] = 1
                hdulist[0].header['CRPIX3'] = 1.
                hdulist[0].header['CD3_3'] = 1.

        # store WCS information
        wcshdr = self.output_wcs[0].to_header()
        for key in wcshdr.keys():
            hdulist[0].header[key] = wcshdr[key]

        # but update the image center
        center = self.output_centers[0]
        hdulist[0].header.update({'PSFCENTX': center[0], 'PSFCENTY': center[1]})
        hdulist[0].header.update({'CRPIX1': center[0], 'CRPIX2': center[1]})
        hdulist[0].header.add_history("Image recentered to {0}".format(str(center)))

        try:
            hdulist.writeto(filepath, overwrite=True)
        except TypeError:
            hdulist.writeto(filepath, clobber=True)
        hdulist.close()

def organise_files(filepaths, copy_dir='./ORGANISED/', heirarchy='TARGPROP/FILTER'):
    '''
    Function to take a list of JWST files, and then copy and organise them into folders based on header keys. 

    Args:
        file_list : list of strings for each file
        copy_dir : directory to copy files to
        heirarchy : Structure of the new directory organisation, using available header keywords.
    Returns:
        NoneType
    '''
    
    # Check if directory we are copying to exists
    if not os.path.isdir(copy_dir):
        os.makedirs(copy_dir)

    # Get the keys we want to sort by
    divisions = heirarchy.split('/')

    # Loop over all of the files
    for file in filepaths:
        with fits.open(file) as f:
            # Loop over each of the keys we are interested in to create a directory string
            working_dir = copy_dir
            for i in range(len(divisions)):
                key_val = f[0].header[divisions[i]]
                working_dir += key_val + '/'

            # Check if this new directory string exists
            if not os.path.isdir(working_dir):
                os.makedirs(working_dir)

            # Save file to the new directory
            file_suffix = file.split('/')[-1]
            shutil.copyfile(file, working_dir+file_suffix)

    return None



