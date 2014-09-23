import astropy.io.fits as pyfits
from astropy import wcs
import numpy as np
import os
#different importants depending on if python2.7 or python3
import sys
if sys.version_info < (3,0):
    #python 2.7 behavior
    import ConfigParser
    from Instrument import Data
else:
    import configparser as ConfigParser
    from pyklip.instruments.Instrument import Data

class GPIData(Data):
    """
    A sequence of GPI Data. Each GPIData object has the following fields and functions

    Fields:
        input: Array of shape (N,y,x) for N images of shape (y,x)
        centers: Array of shape (N,2) for N centers in the format [x_cent, y_cent]
        filenums: Array of size N for the numerical index to map data to file that was passed in
        filenames: Array of size N for the actual filepath of the file that corresponds to the data
        PAs: Array of N for the parallactic angle rotation of the target (used for ADI) [in degrees]
        wvs: Array of N wavelengths of the images (used for SDI) [in microns]. For polarization data, defaults to "None"
        wcs: Array of N wcs astormetry headers for each image.
        IWA: a floating point scalar (not array). Specifies to inner working angle in pixels
        output: Array of shape (b, len(files), len(uniq_wvs), y, x) where b is the number of different KL basis cutoffs

    Functions:
        readdata(): reread in the dadta
        savedata(): save a specified data in the GPI datacube format (in the 1st extension header)
    """
    ##########################
    ###Class Initilization ###
    ##########################
    #some static variables to define the GPI instrument
    centralwave = {}  # in microns
    fpm_diam = {}  # in pixels
    flux_zeropt = {}
    spot_ratio = {} #w.r.t. central star
    lenslet_scale = 1.0 # arcseconds per pixel (pixel scale)
    ifs_rotation = 0.0  # degrees CCW from +x axis to zenith

    ## read in GPI configuration file and set these static variables
    package_directory = os.path.dirname(os.path.abspath(__file__))
    configfile = package_directory + "/" + "GPI.ini"
    config = ConfigParser.ConfigParser()
    try:
        config.read(configfile)
        #get pixel scale
        lenslet_scale = float(config.get("instrument", "ifs_lenslet_scale"))  # arcsecond/pix
        #get IFS rotation
        ifs_rotation = float(config.get("instrument", "ifs_rotation")) #degrees
        #get some information specific to each band
        bands = ['Y', 'J', 'H', 'K1', 'K2']
        for band in bands:
            centralwave[band] = float(config.get("instrument", "cen_wave_{0}".format(band)))
            fpm_diam[band] = float(config.get("instrument", "fpm_diam_{0}".format(band))) / lenslet_scale  # pixels
            flux_zeropt[band] = float(config.get("instrument", "zero_pt_flux_{0}".format(band)))
            spot_ratio[band] = float(config.get("instrument", "APOD_{0}".format(band)))

    except ConfigParser.Error as e:
        print("Error reading GPI configuration file: {0}".format(e.message))
        raise e


    ####################
    ### Constructors ###
    ####################
    def __init__(self, filepaths=None):
        self._output = None
        if filepaths is None:
            self._input = None
            self._centers = None
            self._filenums = None
            self._filenames = None
            self._PAs = None
            self._wvs = None
            self._wcs = None
            self._IWA = None
            self.spot_flux = None
            self.contrast_scaling = None
        else:
            self.readdata(filepaths)

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

    ###############
    ### Methods ###
    ###############
    def readdata(self, filepaths):
        """
        Method to open and read a list of GPI data

        Inputs:
            filespaths: a list of filepaths

        Outputs:
            Technically none. It saves things to fields of the GPIData object. See object doc string
        """
        #check to see if user just inputted a single filename string
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        #make some lists for quick appending
        data = []
        filenums = []
        filenames = []
        rot_angles = []
        wvs = []
        centers = []
        wcs_hdrs = []
        spot_fluxes = []
        #extract data from each file
        for index, filepath in enumerate(filepaths):
            cube, center, pa, wv, astr_hdrs, filt_band, fpm_band, ppm_band, spot_flux = _gpi_process_file(filepath)

            data.append(cube)
            centers.append(center)
            spot_fluxes.append(spot_flux)
            rot_angles.append(pa)
            wvs.append(wv)
            filenums.append(np.ones(pa.shape[0]) * index)
            wcs_hdrs.append(astr_hdrs)

            #filename = np.chararray(pa.shape[0])
            #filename[:] = filepath
            filenames.append([filepath for i in range(pa.shape[0])])

        #convert everything into numpy arrays
        #reshape arrays so that we collapse all the files together (i.e. don't care about distinguishing files)
        data = np.array(data)
        dims = data.shape
        data = data.reshape([dims[0] * dims[1], dims[2], dims[3]])
        filenums = np.array(filenums).reshape([dims[0] * dims[1]])
        filenames = np.array(filenames).reshape([dims[0] * dims[1]])
        rot_angles = -(np.array(rot_angles).reshape([dims[0] * dims[1]])) + (90 - self.ifs_rotation)  # want North Up
        wvs = np.array(wvs).reshape([dims[0] * dims[1]])
        wcs_hdrs = np.array(wcs_hdrs).reshape([dims[0] * dims[1]])
        centers = np.array(centers).reshape([dims[0] * dims[1], 2])
        spot_fluxes = np.array(spot_fluxes).reshape([dims[0] * dims[1]])

        #set these as the fields for the GPIData object
        self._input = data
        self._centers = centers
        self._filenums = filenums
        self._filenames = filenames
        self._PAs = rot_angles
        self._wvs = wvs
        self._wcs = wcs_hdrs
        self._IWA = GPIData.fpm_diam[fpm_band]/2.0
        self.spot_flux = spot_fluxes
        self.contrast_scaling = GPIData.spot_ratio[ppm_band]/spot_fluxes

    @staticmethod
    def savedata(filepath, data, astr_hdr=None):
        """
        Save data in a GPI-like fashion. Aka, data and header are in the first extension header

        Inputs:
            filepath: path to file to output
            data: 2D or 3D data to save
            astr_hdr: wcs astrometry header
        """
        if astr_hdr is None:
            pyfits.writeto(filepath, data, clobber=True)
        else:
            hdulist = astr_hdr.to_fits()
            hdulist.append(hdulist[0])
            hdulist[1].data = data
            hdulist.writeto(filepath, clobber=True)
            hdulist.close()

    def calibrate_output(self, units="contrast"):
        """
        Calibrates the flux of the output klipped data.
        Inputs:
            units: currently only support "contrast" w.r.t central star
        Output:
            stores calibrated data in self.output
        """
        if units == "contrast":
            self.output *= self.contrast_scaling[None, :, None, None]
        



def _gpi_process_file(filepath):
    """
    Method to open and parse a GPI file

    Inputs:
        filepath: the file to open

    Outputs: (using z as size of 3rd dimension, z=37 for spec, z=1 for pol (collapsed to total intensity))
        cube: 3D data cube from the file. Shape is (z,281,281)
        center: array of shape (z,2) giving each datacube slice a [xcenter,ycenter] in that order
        parang: array of z of the parallactic angle of the target (same value just repeated z times)
        wvs: array of z of the wavelength of each datacube slice. (For pol mode, wvs = [None])
        astr_hdrs: array of z of the WCS header for each datacube slice
        filt_band: the band (Y, J, H, K1, K2) used in the IFS Filter (string)
        fpm_band: which coronagrpah was used (string)
    """
    print("Reading File: {0}".format(filepath))
    hdulist = pyfits.open(filepath)
    try:

        #grab the data and headers
        cube = hdulist[1].data
        exthdr = hdulist[1].header
        prihdr = hdulist[0].header

        #get some instrument configuration from the primary header
        filt_band = prihdr['IFSFILT'].split('_')[1]
        fpm_band = prihdr['OCCULTER'].split('_')[1]
        ppm_band = prihdr['APODIZER'].split('_')[1] #to determine sat spot ratios

        #grab the astro header
        w = wcs.WCS(header=exthdr, naxis=[1,2])

        #for spectral mode we need to treat each wavelegnth slice separately
        if exthdr['CTYPE3'].strip() == 'WAVE':
            channels = exthdr['NAXIS3']
            wvs = exthdr['CRVAL3'] + exthdr['CD3_3'] * np.arange(channels) #get wavelength solution
            center = []
            spot_fluxes = []
            #calculate centers from satellite spots
            for i in range(channels):
                spot0 = exthdr['SATS{wave}_0'.format(wave=i)].split()
                spot1 = exthdr['SATS{wave}_1'.format(wave=i)].split()
                spot2 = exthdr['SATS{wave}_2'.format(wave=i)].split()
                spot3 = exthdr['SATS{wave}_3'.format(wave=i)].split()
                centx = np.mean([float(spot0[0]), float(spot1[0]), float(spot2[0]), float(spot3[0])])
                centy = np.mean([float(spot0[1]), float(spot1[1]), float(spot2[1]), float(spot3[1])])
                center.append([centx, centy])

                spot0flux = exthdr['SATF{wave}_0'.format(wave=i)]
                spot1flux = exthdr['SATF{wave}_1'.format(wave=i)]
                spot2flux = exthdr['SATF{wave}_2'.format(wave=i)]
                spot3flux = exthdr['SATF{wave}_3'.format(wave=i)]
                spot_fluxes.append(np.mean([spot0flux, spot1flux, spot2flux, spot3flux]))

            parang = np.repeat(exthdr['AVPARANG'], channels) #populate PA for each wavelength slice (the same)
            astr_hdrs = [w.deepcopy() for i in range(channels)] #repeat astrom header for each wavelength slice
        #for pol mode, we consider only total intensity but want to keep the same array shape to make processing easier
        elif exthdr['CTYPE3'].strip() == 'STOKES':
            wvs = [None]
            cube = np.sum(cube, axis=0)  #sum to total intensity
            cube = cube.reshape([1, cube.shape[0], cube.shape[1]])  #maintain 3d-ness
            center = [[exthdr['PSFCENTX'], exthdr['PSFCENTY']]]
            parang = exthdr['AVPARANG']*np.ones(1)
            astr_hdrs = np.repeat(w, 1)
            spot_fluxes = [[0]] #not suported currently
        else:
            raise AttributeError("Unrecognized GPI Mode: %{mode}".format(mode=exthdr['CTYPE3']))
    finally:
        hdulist.close()

    return cube, center, parang, wvs, astr_hdrs, filt_band, fpm_band, ppm_band, spot_fluxes

