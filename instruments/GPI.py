import astropy.io.fits as pyfits
from astropy import wcs
import numpy as np
import scipy.ndimage as ndimage
import scipy.stats
import os
import re
import subprocess

import matplotlib.pyplot as plt
#different imports depending on if python2.7 or python3
import sys
from copy import copy
if sys.version_info < (3,0):
    #python 2.7 behavior
    import ConfigParser
    from Instrument import Data
    from utils.nair import nMathar
else:
    import configparser as ConfigParser
    from pyklip.instruments.Instrument import Data
    from pyklip.instruments.utils.nair import nMathar

from scipy.interpolate import interp1d


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
        spot_flux: Array of N of average satellite spot flux for each frame
        contrast_scaling: Flux calibration factors (multiply by image to "calibrate" flux)
        prihdrs: Array of N primary GPI headers (these are written by Gemini Observatory + GPI DRP Pipeline)
        exthdrs: Array of N extension GPI headers (these are written by GPI DRP Pipeline)

    Functions:
        readdata(): reread in the data
        savedata(): save a specified data in the GPI datacube format (in the 1st extension header)
        calibrate_output(): calibrates flux of self.output
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

    observatory_latitude = 0.0

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
        observatory_latitude = float(config.get("observatory", "observatory_lat"))
    except ConfigParser.Error as e:
        print("Error reading GPI configuration file: {0}".format(e.message))
        raise e


    ####################
    ### Constructors ###
    ####################
    def __init__(self, filepaths=None, skipslices=None):
        """
        Initialization code for GPIData

        Inputs:
            filepaths: list of filepaths to files
            skipslices: a list of datacube slices to skip (supply index numbers e.g. [0,1,2,3])
        """
        super(GPIData, self).__init__()
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
            self.prihdrs = None
            self.exthdrs = None
        else:
            self.readdata(filepaths, skipslices=skipslices)

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
    def readdata(self, filepaths, skipslices=None):
        """
        Method to open and read a list of GPI data

        Inputs:
            filespaths: a list of filepaths
            skipslices: a list of wavelenegth slices to skip for each datacube (supply index numbers e.g. [0,1,2,3])

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
        prihdrs = []
        exthdrs = []

        #extract data from each file
        for index, filepath in enumerate(filepaths):
            cube, center, pa, wv, astr_hdrs, filt_band, fpm_band, ppm_band, spot_flux, prihdr, exthdr = _gpi_process_file(filepath, skipslices=skipslices)

            data.append(cube)
            centers.append(center)
            spot_fluxes.append(spot_flux)
            rot_angles.append(pa)
            wvs.append(wv)
            filenums.append(np.ones(pa.shape[0]) * index)
            wcs_hdrs.append(astr_hdrs)
            prihdrs.append(prihdr)
            exthdrs.append(exthdr)

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

        #only do the wavelength solution and center recalculation if it isn't broadband imaging
        if np.size(np.unique(wvs)) > 1:
            # recalculate wavelegnths from satellite spots
            wvs = rescale_wvs(exthdrs, wvs, skipslices=skipslices)
            # recaclulate centers from satellite spots and new wavelegnth solution
            wvs_bycube = wvs.reshape([dims[0], dims[1]])
            centers_bycube = centers.reshape([dims[0], dims[1], 2])
            for i, cubewvs in enumerate(wvs_bycube):
                try:
                    centers_bycube[i] = calc_center(prihdrs[i], exthdrs[i], cubewvs, skipslices=skipslices)
                except KeyError:
                    print("Unable to recenter the data using a least squraes fit due to not enough header info for file "
                          "{0}".format(filenames[i*dims[1]]))

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
        self.contrast_scaling = GPIData.spot_ratio[ppm_band]/np.mean(spot_fluxes)
        self.prihdrs = prihdrs
        self.exthdrs = exthdrs

    def savedata(self, filepath, data, klipparams = None, center=None, astr_hdr=None):
        """
        Save data in a GPI-like fashion. Aka, data and header are in the first extension header

        Inputs:
            filepath: path to file to output
            data: 2D or 3D data to save
            astr_hdr: wcs astrometry header
            center: center of the image to be saved in the header as the keywords PSFCENTX and PSFCENTY in pixels.
                The first pixel has coordinates (0,0)
            klipparams: a string of klip parameters
        """
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(header=self.prihdrs[0]))
        hdulist.append(pyfits.ImageHDU(header=self.exthdrs[0], data=data, name="Sci"))

        #save all the files we used in the reduction
        #we'll assume you used all the input files
        #remove duplicates from list
        filenames = np.unique(self.filenames)
        nfiles = np.size(filenames)
        hdulist[0].header["DRPNFILE"] = nfiles
        for i, thispath in enumerate(filenames):
            thispath = thispath.replace("\\", '/')
            splited = thispath.split("/")
            fname = splited[-1]
            matches = re.search('S20[0-9]{6}[SE][0-9]{4}', fname)
            filename = matches.group(0)
            hdulist[0].header["FILE_{0}".format(i)] = filename + '.fits'

        #write out psf subtraction parameters
        #get pyKLIP revision number
        pykliproot = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        pyklipver = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=pykliproot).strip()
        hdulist[0].header['PSFSUB'] = "pyKLIP"
        hdulist[0].header.add_history("Reduced with pyKLIP using commit {0}".format(pyklipver))
        if self.creator is None:
            hdulist[0].header['CREATOR'] = "pyKLIP-{0}".format(pyklipver)
        else:
            hdulist[0].header['CREATOR'] = self.creator
            hdulist[0].header.add_history("Reduced by {0}".self.creator)

        #store commit number for pyklip
        hdulist[0].header['pyklipv'] = pyklipver

        if klipparams is not None:
            hdulist[0].header['PSFPARAM'] = klipparams
            hdulist[0].header.add_history("pyKLIP reduction with parameters {0}".format(klipparams))

        #use the dataset astr hdr if none was passed in
        if astr_hdr is None:
            astr_hdr = self.wcs[0]
        if astr_hdr is not None:
            #update astro header
            #I don't have a better way doing this so we'll just inject all the values by hand
            astroheader = astr_hdr.to_header()
            exthdr = hdulist[1].header
            exthdr['PC1_1'] = astroheader['PC1_1']
            exthdr['PC1_2'] = astroheader['PC1_2']
            exthdr['PC2_1'] = astroheader['PC2_1']
            exthdr['PC2_2'] = astroheader['PC2_2']
            #remove CD values as those are confusing
            exthdr.remove('CD1_1')
            exthdr.remove('CD1_2')
            exthdr.remove('CD2_1')
            exthdr.remove('CD2_2')
            exthdr['CDELT1'] = 1
            exthdr['CDELT2'] = 1

        #use the dataset center if none was passed in
        if center is None:
            center = self.centers[0]
        if center is not None:
            hdulist[1].header.update({'PSFCENTX':center[0],'PSFCENTY':center[1]})
            hdulist[1].header.update({'CRPIX1':center[0],'CRPIX2':center[1]})
            hdulist[0].header.add_history("Image recentered to {0}".format(str(center)))

        hdulist.writeto(filepath, clobber=True)
        hdulist.close()

    def calibrate_output(self, units="contrast"):
        """
        Calibrates the flux of the output of PSF subtracted data.

        Assumes self.output exists and has shape (b,N,y,x) for N is the number of images and b is
        number of KL modes used.

        Inputs:
            units: currently only support "contrast" w.r.t central star
        Output:
            stores calibrated data in self.output
        """
        if units == "contrast":
            self.output *= self.contrast_scaling
        

    def generate_psfs(self, boxrad=7):
        """
        Generates PSF for each frame of input data. Only works on spectral mode data.
        Currently hard coded assuming 37 spectral channels!!!

        Inputs:
            boxrad: the halflength of the size of the extracted PSF (in pixels)

        Outputs:
            saves PSFs to self.psfs as an array of size(N,psfy,psfx) where psfy=psfx=2*boxrad + 1
        """
        self.psfs = []

        for i,frame in enumerate(self.input):
            #figure out which header and which wavelength slice
            numwaves = np.size(np.unique(self.wvs))
            hdrindex = int(i)/int(numwaves)
            slice = i % numwaves
            #now grab the values from them by parsing the header
            hdr = self.exthdrs[hdrindex]
            spot0 = hdr['SATS{wave}_0'.format(wave=slice)].split()
            spot1 = hdr['SATS{wave}_1'.format(wave=slice)].split()
            spot2 = hdr['SATS{wave}_2'.format(wave=slice)].split()
            spot3 = hdr['SATS{wave}_3'.format(wave=slice)].split()

            #put all the sat spot info together
            spots = [[float(spot0[0]), float(spot0[1])],[float(spot1[0]), float(spot1[1])],
                     [float(spot2[0]), float(spot2[1])],[float(spot3[0]), float(spot3[1])]]
            #now make a psf
            spotpsf = generate_psf(frame, spots, boxrad=boxrad)
            self.psfs.append(spotpsf)

        self.psfs = np.array(self.psfs)

    def generate_psf_cube(self, boxw=14):
        """
        Generates an average PSF from all frames of input data. Only works on spectral mode data.
        Overall cube normalized to unity with norm 2.
        Currently hard coded assuming 37 spectral channels!!!

        The center of the PSF is exactly on the central pixel of the PSF.
        (If even width of the array it is the middle pixel with the highest row and column index.)
        The center pixel index is always (nx/2,nx/2) assuming integer division.

        Inputs:
            boxw: the width the extracted PSF (in pixels). Should be bigger than 12 because there is an interpolation
                of the background by a plane which is then subtracted to remove linear biases.

        Outputs:
            A cube of shape 37*boxw*boxw. Each slice [k,:,:] is the PSF for a given wavelength.
        """

        n_frames,ny,nx = self.input.shape
        x_grid, y_grid = np.meshgrid(np.arange(ny), np.arange(nx))
        unique_wvs = np.unique(self.wvs)
        numwaves = np.size(np.unique(self.wvs))

        psfs = np.zeros((numwaves,boxw,boxw,n_frames,4))


        for lambda_ref_id, lambda_ref in enumerate(unique_wvs):
            for i,frame in enumerate(self.input):
                #figure out which header and which wavelength slice
                hdrindex = int(i)/int(numwaves)
                slice = i % numwaves
                lambda_curr = unique_wvs[slice]
                #now grab the values from them by parsing the header
                hdr = self.exthdrs[hdrindex]
                spot0 = hdr['SATS{wave}_0'.format(wave=slice)].split()
                spot1 = hdr['SATS{wave}_1'.format(wave=slice)].split()
                spot2 = hdr['SATS{wave}_2'.format(wave=slice)].split()
                spot3 = hdr['SATS{wave}_3'.format(wave=slice)].split()

                #put all the sat spot info together
                spots = [[float(spot0[0]), float(spot0[1])],[float(spot1[0]), float(spot1[1])],
                         [float(spot2[0]), float(spot2[1])],[float(spot3[0]), float(spot3[1])]]

                #mask nans
                cleaned = np.copy(frame)
                cleaned[np.where(np.isnan(cleaned))] = 0

                for loc_id, loc in enumerate(spots):
                    #grab satellite spot positions
                    spotx = loc[0]
                    spoty = loc[1]
                    xarr_spot = np.round(spotx)
                    yarr_spot = np.round(spoty)
                    stamp = cleaned[(yarr_spot-np.floor(boxw/2.0)):(yarr_spot+np.ceil(boxw/2.0)),(xarr_spot-np.floor(boxw/2.0)):(xarr_spot+np.ceil(boxw/2.0))]
                    #x_stamp = x_grid[(yarr_spot-boxw/2):(yarr_spot+boxw/2),(xarr_spot-boxw/2):(xarr_spot+boxw/2)]
                    #y_stamp = y_grid[(yarr_spot-boxw/2):(yarr_spot+boxw/2),(xarr_spot-boxw/2):(xarr_spot+boxw/2)]
                    #print(spotx,spoty)
                    #print(stamp_x+ spotx-xarr_spot,stamp_y+spoty-yarr_spot)
                    stamp_x, stamp_y = np.meshgrid(np.arange(boxw, dtype=np.float32), np.arange(boxw, dtype=np.float32))
                    dx = spotx-xarr_spot
                    dy = spoty-yarr_spot
                    #stamp_x += spotx-xarr_spot
                    #stamp_y += spoty-yarr_spot
                    #stamp_x -= spotx-xarr_spot
                    #stamp_y -= spoty-yarr_spot
                    #print(spotx-xarr_spot,spoty-yarr_spot)



                    #mask the central blob to calculate background median
                    stamp_r = np.sqrt((stamp_x-dx-boxw/2)**2+(stamp_y-dy-boxw/2)**2)
                    stamp_masked = copy(stamp)
                    stamp_x_masked = stamp_x-dx
                    stamp_y_masked = stamp_y-dy
                    stamp_center = np.where(stamp_r<4)
                    stamp_masked[stamp_center] = np.nan
                    stamp_x_masked[stamp_center] = np.nan
                    stamp_y_masked[stamp_center] = np.nan
                    background_med =  np.nanmedian(stamp_masked)
                    stamp_masked -= background_med
                    #Solve 2d linear fit to remove background
                    xx = np.nansum(stamp_x_masked**2)
                    yy = np.nansum(stamp_y_masked**2)
                    xy = np.nansum(stamp_y_masked*stamp_x_masked)
                    xz = np.nansum(stamp_masked*stamp_x_masked)
                    yz = np.nansum(stamp_y_masked*stamp_masked)
                    #Cramer's rule
                    a = (xz*yy-yz*xy)/(xx*yy-xy*xy)
                    b = (xx*yz-xy*xz)/(xx*yy-xy*xy)
                    stamp -= a*(stamp_x-dx)+b*(stamp_y-dy) + background_med
                    #stamp -= background_med

                    #rescale to take into account wavelength widening
                    if 1:
                        stamp_r = np.sqrt((stamp_x-dx-boxw/2)**2+(stamp_y-dy-boxw/2)**2)
                        stamp_th = np.arctan2(stamp_x-dx-boxw/2,stamp_y-dy-boxw/2)
                        stamp_r /= lambda_ref/lambda_curr
                        stamp_x = stamp_r*np.cos(stamp_th)+boxw/2
                        stamp_y = stamp_r*np.sin(stamp_th)+boxw/2
                        #print(stamp_x,stamp_y)

                    stamp = ndimage.map_coordinates(stamp, [stamp_y+dx, stamp_x+dy])
                    #print(stamp)
                    psfs[lambda_ref_id,:,:,i,loc_id] = stamp


        #PSF_cube = np.mean(psfs[:,:,:,:,0],axis=(3))
        PSF_cube = np.mean(psfs,axis=(3,4))

        #Build the spectrum of the sat spots
        # Number of cubes in dataset
        N_cubes = int(self.input.shape[0])/int(numwaves)
        all_sat_spot_spec = np.zeros((37,N_cubes))
        for k in range(N_cubes):
            all_sat_spot_spec[:,k] = self.spot_flux[37*k:37*(k+1)]
        sat_spot_spec = np.nanmean(all_sat_spot_spec,axis=1)

        #stamp_x, stamp_y = np.meshgrid(np.arange(boxw, dtype=np.float32), np.arange(boxw, dtype=np.float32))
        #stamp_r = np.sqrt((stamp_x-boxw/2)**2+(stamp_y-boxw/2)**2)
        #stamp_center = np.where(stamp_r<3)
        PSF_cube /= np.sqrt(np.nansum(PSF_cube**2))
        for l in range(numwaves):
            #PSF_cube[l,:,:] -= np.nanmedian(PSF_cube[l,:,:][stamp_center])
            PSF_cube[l,:,:] *= sat_spot_spec[l]/np.nanmax(PSF_cube[l,:,:])
            PSF_cube[l,:,:][np.where(abs(PSF_cube[l,:,:])/np.nanmax(abs(PSF_cube[l,:,:]))<0.05)] = 0.0

        if 0:
            plt.figure(1)
            plt.plot(sat_spot_spec,'or')
            plt.plot(np.nanmax(PSF_cube,axis=(1,2)),"--b")
            plt.show()
        if 0: # for debugging purposes
            plt.figure(1)
            plt.imshow(PSF_cube[0,:,:],interpolation = 'nearest')
            plt.figure(2)
            plt.imshow(PSF_cube[36,:,:],interpolation = 'nearest')
            plt.show()

        self.psfs = PSF_cube

    def get_radial_psf(self,save = ""):
        """
        Return a pure radial PSF by averaging the original psf. The new PSF is invariant by rotation.
        A call to generate_psf_cube() is required prior to calling this function.
        The center pixel index is always (nx/2,nx/2) assuming integer division.

        Inputs:
            save: Optionally automatically save the radial psf cube as a fits file with filename:
                    save+"-original_radial_PSF_cube.fits"

        Outputs:
            rad_psf_cube: a (37,nx,nx) cube with the radial psf.

        """
        if np.size(np.shape(self.psfs)) == 3 and np.shape(self.psfs)[0] == 37:
            nl,ny,nx = self.psfs.shape
            # We should have nx = ny

            sat_spot_spec = np.nanmax(self.psfs,axis=(1,2))

            k_hd=4 # should be even
            nx_hd = k_hd*(nx-1) + 1
            hd_psf = np.zeros((nl,nx_hd,nx_hd))

            rad_psf_cube = np.zeros((nl,nx,nx))
            #current_slice = np.zeros((nx,nx))

            stamp_x, stamp_y = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(nx, dtype=np.float32))
            stamp_r = np.sqrt((stamp_x - nx/2)**2+(stamp_y - nx/2)**2)
            stamp_x_hd, stamp_y_hd = np.meshgrid(np.arange(nx_hd, dtype=np.float32)/(nx_hd-1)*(nx-1), np.arange(nx_hd, dtype=np.float32)/(nx_hd-1)*(nx-1))
            for l in range(nl):
                hd_psf[l,:,:] = ndimage.map_coordinates(self.psfs[l,:,:], [stamp_y_hd, stamp_x_hd])
                #hd_psf[l,nx/2*k_hd,nx/2*k_hd] = 0. # center
            stamp_r_hd = np.sqrt((stamp_x_hd-stamp_x_hd[nx/2*k_hd,nx/2*k_hd])**2+(stamp_y_hd-stamp_y_hd[nx/2*k_hd,nx/2*k_hd])**2)

            dr = 1.0/k_hd
            Dr = 2.0/k_hd
            r_samp = np.arange(0,np.max(stamp_r_hd)+dr,dr)

            radial_val = np.zeros((nl,np.size(r_samp)))

            for r_id, r_it in enumerate(r_samp):
                selec_pix = np.where( ((r_it-Dr/2.0) < stamp_r_hd) * (stamp_r_hd < (r_it+Dr/2.0)) )
                selec_y, selec_x = selec_pix
                radial_val[:,r_id] = np.nanmean(hd_psf[:,selec_y, selec_x],1)

            for l_id in np.arange(nl):
                f = interp1d(r_samp, radial_val[l_id,:], kind='cubic',bounds_error=False, fill_value=np.nan)
                rad_psf_cube[l_id,:,:] = f(stamp_r.reshape(nx*nx)).reshape(nx,nx)
                rad_psf_cube[l_id,:,:] *= sat_spot_spec[l_id]/np.nanmax(rad_psf_cube[l_id,:,:])

                if 0:
                    print(rad_psf_cube[l_id,0,0])
                    plt.figure(1)
                    plt.imshow(rad_psf_cube[l_id,:,:],interpolation = 'nearest')
                    plt.figure(2)
                    plt.plot(np.nanmax(self.psfs,axis=(1,2)))
                    plt.show()



            if save != "":
                self.savedata(save+"-original_radial_PSF_cube.fits", rad_psf_cube)

            return rad_psf_cube

        else:
            print("Wrong size of the PSFs stored in gpi dataset structure when calling get_radial_psf. Return 0")
            return 0

######################
## Static Functions ##
######################

def _gpi_process_file(filepath, skipslices=None):
    """
    Method to open and parse a GPI file

    Inputs:
        filepath: the file to open
        skipslices: a list of datacube slices to skip (supply index numbers e.g. [0,1,2,3])

    Outputs: (using z as size of 3rd dimension, z=37 for spec, z=1 for pol (collapsed to total intensity))
        cube: 3D data cube from the file. Shape is (z,281,281)
        center: array of shape (z,2) giving each datacube slice a [xcenter,ycenter] in that order
        parang: array of z of the parallactic angle of the target (same value just repeated z times)
        wvs: array of z of the wavelength of each datacube slice. (For pol mode, wvs = [None])
        astr_hdrs: array of z of the WCS header for each datacube slice
        filt_band: the band (Y, J, H, K1, K2) used in the IFS Filter (string)
        fpm_band: which coronagrpah was used (string)
        ppm_band: which apodizer was used (string)
        spot_fluxes: array of z containing average satellite spot fluxes for each image
        prihdr: primary header of the FITS file
        exthdr: 1st extention header of the FITS file
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
        #turns out WCS data can be wrong. Let's recalculate it using avparang
        parang = exthdr['AVPARANG']
        vert_angle = -(360-parang) + GPIData.ifs_rotation - 90
        vert_angle = np.radians(vert_angle)
        pc = np.array([[np.cos(vert_angle), np.sin(vert_angle)],[-np.sin(vert_angle), np.cos(vert_angle)]])
        cdmatrix = pc * GPIData.lenslet_scale /3600.
        w.wcs.cd[0,0] = cdmatrix[0,0]
        w.wcs.cd[0,1] = cdmatrix[0,1]
        w.wcs.cd[1,0] = cdmatrix[1,0]
        w.wcs.cd[1,1] = cdmatrix[1,1]

        #for spectral mode we need to treat each wavelegnth slice separately
        if exthdr['CTYPE3'].strip() == 'WAVE':
            channels = exthdr['NAXIS3']
            wvs = exthdr['CRVAL3'] + exthdr['CD3_3'] * np.arange(channels) #get wavelength solution
            center = []
            spot_fluxes = []
            #calculate centers from satellite spots
            for i in range(channels):
                #grab satellite spot positions
                spot0 = exthdr['SATS{wave}_0'.format(wave=i)].split()
                spot1 = exthdr['SATS{wave}_1'.format(wave=i)].split()
                spot2 = exthdr['SATS{wave}_2'.format(wave=i)].split()
                spot3 = exthdr['SATS{wave}_3'.format(wave=i)].split()
                centx = np.nanmean([float(spot0[0]), float(spot1[0]), float(spot2[0]), float(spot3[0])])
                centy = np.nanmean([float(spot0[1]), float(spot1[1]), float(spot2[1]), float(spot3[1])])
                center.append([centx, centy])

                #grab sat spot fluxes if they're there
                try:
                    spot0flux = float(exthdr['SATF{wave}_0'.format(wave=i)])
                    spot1flux = float(exthdr['SATF{wave}_1'.format(wave=i)])
                    spot2flux = float(exthdr['SATF{wave}_2'.format(wave=i)])
                    spot3flux = float(exthdr['SATF{wave}_3'.format(wave=i)])
                except KeyError:
                    spot0flux = 1
                    spot1flux = 1
                    spot2flux = 1
                    spot3flux = 1
                spot_fluxes.append(np.nanmean([spot0flux, spot1flux, spot2flux, spot3flux]))

            parang = np.repeat(exthdr['AVPARANG'], channels) #populate PA for each wavelength slice (the same)
            astr_hdrs = [w.deepcopy() for i in range(channels)] #repeat astrom header for each wavelength slice
        #for pol mode, we consider only total intensity but want to keep the same array shape to make processing easier
        elif exthdr['CTYPE3'].strip() == 'STOKES':
            wvs = [1.0]
            cube = np.sum(cube, axis=0)  #sum to total intensity
            cube = cube.reshape([1, cube.shape[0], cube.shape[1]])  #maintain 3d-ness
            center = [[exthdr['PSFCENTX'], exthdr['PSFCENTY']]]
            parang = exthdr['AVPARANG']*np.ones(1)
            astr_hdrs = np.repeat(w, 1)
            spot_fluxes = [[1]] #not suported currently
        else:
            raise AttributeError("Unrecognized GPI Mode: %{mode}".format(mode=exthdr['CTYPE3']))
    finally:
        hdulist.close()

    #remove undesirable slices of the datacube if necessary
    if skipslices is not None:
        cube = np.delete(cube, skipslices, axis=0)
        center = np.delete(center, skipslices, axis=0)
        parang = np.delete(parang, skipslices)
        wvs = np.delete(wvs, skipslices)
        astr_hdrs = np.delete(astr_hdrs, skipslices)
        spot_fluxes = np.delete(spot_fluxes, skipslices)

    return cube, center, parang, wvs, astr_hdrs, filt_band, fpm_band, ppm_band, spot_fluxes, prihdr, exthdr

def generate_psf(frame, locations, boxrad=5, medianboxsize=30):
    """
    Generates a GPI PSF for the frame based on the satellite spots

    Inputs:
        frame: 2d frame of data
        location: array of (N,2) containing [x,y] coordinates of all N satellite spots
        boxrad: half length of box to use to pull out PSF
        medianboxsize: size in pixels of box for median filter

    Outputs:
        genpsf: 2d frame of size (2*boxrad+1, 2*boxrad+1) with average PSF of satellite spots
    """
    genpsf = []
    #mask nans
    cleaned = np.copy(frame)
    cleaned[np.where(np.isnan(cleaned))] = 0

    #highpass filter to remove background
    #mask source for median filter
    masked = np.copy(cleaned)
    for loc in locations:
        spotx = np.round(loc[0])
        spoty = np.round(loc[1])
        masked[spotx-boxrad:spotx+boxrad+1, spoty-boxrad:spoty+boxrad+1] = scipy.stats.nanmedian(
            masked.reshape(masked.shape[0]*masked.shape[1]))
    #subtract out median filtered image

    #cleaned -= ndimage.median_filter(masked, size=(medianboxsize,medianboxsize))

    for loc in locations:
        #grab satellite spot positions
        spotx = loc[0]
        spoty = loc[1]

        #interpolate image to grab satellite psf with it centered
        #add .1 to make sure we get 2*boxrad+1 but can't add +1 due to floating point precision (might cause us to
        #create arrays of size 2*boxrad+2)
        x,y = np.meshgrid(np.arange(spotx-boxrad, spotx+boxrad+0.1, 1), np.arange(spoty-boxrad, spoty+boxrad+0.1, 1))
        spotpsf = ndimage.map_coordinates(cleaned, [y,x])
        genpsf.append(spotpsf)

    genpsf = np.array(genpsf)
    genpsf = np.mean(genpsf, axis=0) #average the different psfs together    

    return genpsf


def rescale_wvs(exthdrs, wvs, refwv=18, skipslices=None):
    """
    Hack to try to fix wavelength scaling issue. This will calculate the scaling between channels,
    and adjust the wavelength solution such that the scaling comes out linear in scaling vs wavelength.
    Finicky - requires that all images in the dataset have the same number of wavelength channels
    Input:
        exthdrs: a list of extension headers, from a pyklip.instrument dataset
        wvs: a list of wvs (can repeat. This function will only look at the first cube's wavelenghts)
        refwv (optional): integer index of the channel to normalize the scaling
        skipslices: list of skipped wavelength slices (needs to be consistent with the ones skipped by wv)
    Output:
        scaled_wvs: Nlambda*Nexthdrs array of wavelengths that produce a linear plot of wavelength vs scaling
    """
    #wvs_mean = wvs.reshape(len(exthdrs), len(wvs)/len(exthdrs)).mean(axis=0)
    wv_indicies = range(0, exthdrs[0]['NAXIS3'])
    if skipslices is not None:
        wv_indicies = np.delete(wv_indicies, skipslices)
    sats = np.array([[[h['SATS{0}_{1}'.format(i,j)].split() for i in wv_indicies]
                          for j in range(0,4)] for h in exthdrs], dtype=np.float)
    sats = sats.mean(axis=0)
    pairs = [(0,3), (1,2)]
    separations = np.mean([0.5*np.sqrt(np.diff(sats[p,:,0], axis=0)[0]**2 + np.diff(sats[p,:,1], axis=0)[0]**2) 
                           for p in pairs], 
                          axis=0) # average over each pair, the first axis
    scaling_factors = separations/separations[refwv]
    scaled_wvs = scaling_factors*wvs[refwv]
    return np.tile(scaled_wvs, len(exthdrs))


def calc_center_least_squares(xpos, ypos, wvs, orderx, ordery, displacement):
    """
	calcualte the center position, linear least squares fit to 4 parameters

	Inputs: xpos: array of length n of x positions of satellite spots
			ypos: array of length n of y positions of satellite spots
			wvs: the wavelength of each pair of positoins
			orderx: the x order (can be -1 or 1 in this case. -1 is under the center, 1 is above the center)
			ordery: the y order (e.g. pos0 is at pox=-1, posy=1).
			displacment: the displacement from zenith
	Outputs: four fit parameters (xcenter, ycenter, adrx, adry). xcenters = xcenter + ardx * displacement
	"""

    pos_x = np.matrix(xpos).T
    pos_y = np.matrix(ypos).T

    #create the B matrix for the transform. See email from James on how to make this
    Bx = np.append(np.matrix(np.ones(np.size(pos_x))).T, np.matrix(orderx*wvs).T,1)
    Bx = np.append(Bx,np.matrix(-ordery*wvs).T, 1)
    Bx = np.append(Bx, np.matrix(displacement).T , 1)
    Bx = np.append(Bx, np.matrix(np.zeros(np.size(pos_x))).T, 1)
    Bx = np.append(Bx, np.matrix(np.zeros(np.size(pos_x))).T, 1)
    By = np.append(np.matrix(np.zeros(np.size(pos_y))).T, np.matrix(ordery*wvs).T, 1)
    By = np.append(By, np.matrix(orderx*wvs).T, 1)
    By = np.append(By, np.matrix(np.zeros(np.size(pos_y))).T, 1)
    By = np.append(By, np.matrix(displacement).T , 1)
    By = np.append(By, np.matrix(np.ones(np.size(pos_y))).T, 1)

    B = np.append(Bx,By,0)

    #the measured inputs
    X = np.append(pos_x, pos_y, 0)

    #fit outputs
    Q = (B.T*B).I * B.T* X

    xcenter = float(Q[0])
    ycenter = float(Q[5])
    shift1 = float(Q[1])
    shift2 = float(Q[2])
    adrx = float(Q[3])
    adry = float(Q[4])

    # xcalc = xcenter + orderx*wvs*shift1 - ordery*wvs*shift2 + (adrx)*np.array(displacement)
    # ycalc = ycenter + ordery*wvs*shift1 + orderx*wvs*shift2 + (adry)*np.array(displacement)
    #
    # xres = (xpos - xcalc)
    # yres = (ypos - ycalc)
    #
    #
    # xcenters = xcenter + (adrx)*np.array(displacement)
    # ycenters = ycenter + (adry)*np.array(displacement)

    #old code - preserved just in case
	#one_params1 = np.polyfit(pos_one[:, 0], pos_one[:, 1], 1)
	#one_params2 = np.polyfit(pos_one[:, 1], pos_oplt.show()ne[:, 0], 1)
	#
	#two_params1 = np.polyfit(pos_two[:, 0], pos_two[:, 1], 1)
	#two_params2 = np.polyfit(pos_two[:, 1], pos_two[:, 0], 1)
	#
	#xcenter1 = (two_params1[1] - one_params1[1]) / (one_params1[0] - two_params1[0])
	#ycenter1 = two_params1[1] + two_params1[0] * xcenter1
	#ycenter2 = (two_params2[1] - one_params2[1]) / (one_params2[0] - two_params2[0])
	#xcenter2 = two_params2[1] + two_params2[0] * ycenter2

    return xcenter, ycenter, adrx, adry


def calc_center(prihdr, exthdr, wvs, ignoreslices=None, skipslices=None):
    """
    calcualte the center position of a spectral data cube

    Inputs:
        prihdr: primary GPI header
        exthdr: extention GPI header
        wvs: wvs of the datacube
        ignoreslices: slices to ignore in the fit. A list of wavelength slice indicies to ignore
                        if none, ignores slices 0,1, len-2, len-1 (first and last two)
        skipslices: slices that were already skipped in processing
    Outputs:
        centx, centy: star center
    """
    maxwvs = exthdr['NAXIS3']
    if ignoreslices is None:
        ignoreslices = np.array([0,1,maxwvs-2,maxwvs-1])
    ignoreslices %= np.size(wvs)

    utstart = prihdr['UTSTART']
    utstart = float(utstart[0:2]) + float(utstart[3:5])/60.+float(utstart[6:])/3600. #covert to decimal

    #Grab info for ADR correction
    #try to get environment parameters but sometimes we need to default
    #Get HA
    HA = prihdr['HA']
    HA_sgn = np.sign(float(HA[0:3]))
    if HA_sgn == 0:
        HA_sgn = 1
    HA = float(HA[0:3]) + HA_sgn*float(HA[4:6])/60. + HA_sgn*float(HA[7:])/3600.
    HA *= 15*np.pi/180. # rad
    #Get Temp
    Temp = prihdr['TAMBIENT'] + 273.15 #Kelvin
    #Get pressure
    Pressure = prihdr['PRESSUR2'] #Pascal
    #Get declination from header and convert to radians
    dec = exthdr['CRVAL2'] * np.pi/ 180. #rad

    #Calculate angle from zenith, need this for ADR corrections
    zenith = np.arccos(np.sin(GPIData.observatory_latitude)*np.sin(dec)
                       + np.cos(GPIData.observatory_latitude)*np.cos(dec)*np.cos(HA))

    spots_posx = []
    spots_posy = []
    order_x = []
    order_y = []
    displacement = []
    spot_wvs = []
    spots_wvs_index = []

    #calculate reference wavelegnth
    refwv = np.mean(wvs)
    n0 = nMathar(refwv, Pressure, Temp) #reference index of refrraction

    #get centers from header values inputted by GPI pipeline
    #mask = bin(int(pcenthdr['SATSMASK'],16)) #assume all spot locations are placed in header
    #iterate over headers in cube
    i = 0
    for wv in wvs:
        thisfour = []
        n = nMathar(wv, Pressure, Temp) #index of refraction

        # increiment loop index if we need to skip
        if skipslices is not None:
            while i in skipslices:
                i += 1
                # sanity check in case we get stuck in an infinite loop (hopefully won't)
                if i >= maxwvs:
                    print("oops.. infinite loop in skipping wavelenghts")
                    break

        for j in range(4):
            hdr_str = "sats%i_%i" % (i, j)
            cents = exthdr[hdr_str]
            args = cents.split()

            #append this data to the list
            #calcuate deltaZ effect of ADR
            displacement.append( (n-n0)/n0 * np.tan(zenith)) #deltaZ calculation
            spots_posx.append(float(args[0]))
            spots_posy.append(float(args[1]))
            spot_wvs.append(wv)
            spots_wvs_index.append(i)

            #this better account for all cases or this for loop is messed up
            if j == 0:
                order_x.append(-1)
                order_y.append(1)
            elif j == 1:
                order_x.append(-1)
                order_y.append(-1)
            elif j == 2:
                order_x.append(1)
                order_y.append(1)
            elif j ==3:
                order_x.append(1)
                order_y.append(-1)
            else:
                print "LOGIC ERROR: j value in loop somehow got to %f" %(j)
                continue

        i += 1

    spots_posx = np.array(spots_posx)
    spots_posy = np.array(spots_posy)
    order_x = np.array(order_x)
    order_y = np.array(order_y)
    displacement = np.array(displacement)
    spot_wvs = np.array(spot_wvs)
    spots_wvs_index = np.array(spots_wvs_index)

    good = np.where(~np.in1d(spots_wvs_index, ignoreslices))

    x0, y0, adrx, adry = calc_center_least_squares(spots_posx[good], spots_posy[good], spot_wvs[good], order_x[good],
                                                   order_y[good], displacement[good])
    centers_x = x0 + adrx*displacement
    centers_y = y0 + adry*displacement
    centers = np.array([centers_x, centers_y])
    # centers are duplicated 4 times (for each sat spot) and the dimensions are flipped. need to remove this...
    centers = np.swapaxes(centers, 0, 1)
    centers = centers.reshape([centers.shape[0]/4, 4, 2])
    centers = centers[:,0,:]
    return centers
