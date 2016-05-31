__author__ = 'JB'
import os
import astropy.io.fits as pyfits
from glob import glob
import multiprocessing as mp
import numpy as np
from scipy.signal import convolve2d

from pyklip.kpp.utils.kppSuperClass import KPPSuperClass
from pyklip.kpp.stat.statPerPix_utils import *
from pyklip.kpp.utils.GOI import *
import pyklip.kpp.utils.mathfunc as kppmath

class StatPerPix(KPPSuperClass):
    """
    Class for SNR calculation.
    """
    def __init__(self,filename,
                 inputDir = None,
                 outputDir = None,
                 mute=None,
                 N_threads=None,
                 label = None,
                 mask_radius = None,
                 IOWA = None,
                 N = None,
                 Dr = None,
                 Dth = None,
                 type = None,
                 rm_edge = None,
                 GOI_list_folder = None,
                 overwrite = False,
                 kernel_type = None,
                 kernel_width = None,
                 filename_noPlanets = None):
        """


        :param filename: Filename of the file on which to calculate the metric. It should be the complete path unless
                        inputDir is defined.
                        It can include wild characters. The file will be selected using the first output of glob.glob().
        :param mute: If True prevent printed log outputs.
        :param N_threads: Number of threads to be used for the metrics and the probability calculations.
                        If None use mp.cpu_count().
                        If -1 do it sequentially.
                        Note that it is not used for this super class.
        :param label: Define the suffix to the output folder when it is not defined. cf outputDir. Default is "default".
        """
        # allocate super class
        super(StatPerPix, self).__init__(filename,
                                     inputDir = inputDir,
                                     outputDir = outputDir,
                                     folderName = None,
                                     mute=mute,
                                     N_threads=N_threads,
                                     label=label,
                                     overwrite = overwrite)

        if mask_radius is None:
            self.mask_radius = 7
        else:
            self.mask_radius = mask_radius

        if Dr is None:
            self.Dr = 2
        else:
            self.Dr = Dr

        if type is None:
            self.type = "SNR"
        else:
            self.type = type

        self.IOWA = IOWA
        self.N = N
        self.Dth = Dth
        self.rm_edge = rm_edge
        self.GOI_list_folder = GOI_list_folder
        self.filename_noPlanets = filename_noPlanets

        self.kernel_type = kernel_type
        # The default value is defined later
        self.kernel_width = kernel_width


    def initialize(self,inputDir = None,
                         outputDir = None,
                         folderName = None,
                         compact_date = None,
                         label = None):
        """
        Initialize the non general inputs that are needed for the metric calculation and load required files.

        For this super class it simply reads the input file including fits headers and store it in self.image.
        One can also overwrite inputDir, outputDir which is basically the point of this function.
        The file is assumed here to be a fits containing a 2D image or a GPI 3D cube (assumes 37 spectral slice).

        Example for inherited classes:
        It can read the PSF cube or define the hat function.
        It can also read the template spectrum in a 3D scenario.
        It could also overwrite this function in case it needs to read multiple files or non fits file.

        :param inputDir: If defined it allows filename to not include the whole path and just the filename.
                        Files will be read from inputDir.
                        Note tat inputDir might be redefined using initialize at any point.
                        If inputDir is None then filename is assumed to have the absolute path.
        :param outputDir: Directory where to create the folder containing the outputs.
                        Note tat inputDir might be redefined using initialize at any point.
                        If outputDir is None:
                            If inputDir is defined: outputDir = inputDir+os.path.sep+"planet_detec_"
        :param folderName: Name of the folder containing the outputs. It will be located in outputDir.
                        Default folder name is "default_out".
                        The convention is to have one folder per spectral template.
                        If the keyword METFOLDN is available in the fits file header then the keyword value is used no
                        matter the input.
        :param label: Define the suffix to the output folder when it is not defined. cf outputDir. Default is "default".

        :return: None
        """
        if not self.mute:
            print("~~ INITializing "+self.__class__.__name__+" ~~")
        # The super class already read the fits file
        init_out = super(StatPerPix, self).initialize(inputDir = inputDir,
                                         outputDir = outputDir,
                                         folderName = folderName,
                                         label=label)

        # Get center of the image (star position)
        try:
            # Retrieve the center of the image from the fits headers.
            self.center = [self.exthdr['PSFCENTX'], self.exthdr['PSFCENTY']]
        except:
            # If the keywords could not be found the center is defined as the middle of the image
            if not self.mute:
                print("Couldn't find PSFCENTX and PSFCENTY keywords.")
            self.center = [(self.nx-1)/2,(self.ny-1)/2]

        if self.label == "CADI":
            self.center = [140,140]

        try:
            self.folderName = self.exthdr["METFOLDN"]+os.path.sep
        except:
            pass

        file_ext_ind = os.path.basename(self.filename_path)[::-1].find(".")
        self.prefix = os.path.basename(self.filename_path)[:-(file_ext_ind+1)]
        #self.prefix = "".join(os.path.basename(self.filename_path).split(".")[0:-1])
        self.suffix = self.type+"PerPix"
        tmp_suffix = ""
        if self.Dr is not None:
            tmp_suffix = tmp_suffix+"Dr"+str(self.Dr)
        elif self.N is not None:
            tmp_suffix = tmp_suffix+"N"+str(self.N)
        if self.Dth is not None:
            tmp_suffix = tmp_suffix+"Dth"+str(self.Dth)
        if self.kernel_type is not None:
            tmp_suffix = tmp_suffix+self.kernel_type
            # if self.kernel_width is not None:
            #     tmp_suffix = tmp_suffix+str(self.kernel_width)
        self.suffix = self.suffix+tmp_suffix

        if self.filename_noPlanets is not None:# Check file existence and define filename_path
            if self.inputDir is None or os.path.isabs(self.filename_noPlanets):
                try:
                    self.filename_noPlanets_path = os.path.abspath(glob(self.filename_noPlanets)[0])
                except:
                    raise Exception("File "+self.filename_noPlanets+"doesn't exist.")
            else:
                try:
                    self.filename_noPlanets_path = os.path.abspath(glob(self.inputDir+os.path.sep+self.filename_noPlanets)[0])
                except:
                    raise Exception("File "+self.inputDir+os.path.sep+self.filename_noPlanets+" doesn't exist.")

            # Open the fits file on which the metric will be applied
            hdulist = pyfits.open(self.filename_noPlanets_path)
            if not self.mute:
                print("Opened: "+self.filename_noPlanets_path)

            # grab the data and headers
            try:
                self.image_noPlanets = hdulist[1].data
                self.exthdr_noPlanets = hdulist[1].header
                self.prihdr_noPlanets = hdulist[0].header
            except:
                # This except was used for datacube not following GPI headers convention.
                if not self.mute:
                    print("Couldn't read the fits file with GPI conventions. Try assuming data in primary and no headers.")
                try:
                    self.image_noPlanets = hdulist[0].data
                except:
                    raise Exception("Couldn't read "+self.filename_noPlanets_path+". Is it a fits?")


        if self.kernel_type is not None:
            self.ny_PSF = 20 # should be even
            self.nx_PSF = 20 # should be even
            # Define the PSF as a gaussian
            if self.kernel_type == "gaussian":
                if self.kernel_width == None:
                    self.kernel_width = 1.25
                    if not self.mute:
                        print("Default width sigma = {0} used for the gaussian".format(self.kernel_width))

                if not self.mute:
                    print("Generate gaussian PSF")
                # Build the grid for PSF stamp.
                x_PSF_grid, y_PSF_grid = np.meshgrid(np.arange(0,self.ny_PSF,1)-self.ny_PSF/2,
                                                     np.arange(0,self.nx_PSF,1)-self.nx_PSF/2)

                self.PSF = kppmath.gauss2d(x_PSF_grid, y_PSF_grid,1.0,0.0,0.0,self.kernel_width,self.kernel_width)

            # Define the PSF as an aperture or "hat" function
            if self.kernel_type == "hat":
                if self.kernel_width == None:
                    self.kernel_width = 1.5
                    if not self.mute:
                        print("Default radius = {0} used for the hat function".format(self.kernel_width))

                # Build the grid for PSF stamp.
                x_PSF_grid, y_PSF_grid = np.meshgrid(np.arange(0,self.ny_PSF,1)-self.ny_PSF/2,
                                                     np.arange(0,self.nx_PSF,1)-self.nx_PSF/2)
                # Use aperture for the cross correlation.
                # Calculate the corresponding hat function
                self.PSF = kppmath.hat(x_PSF_grid, y_PSF_grid, self.kernel_width)

            self.PSF = self.PSF / np.sqrt(np.nansum(self.PSF**2))

        return init_out

    def check_existence(self):
        """

        :return: False
        """

        file_exist = (len(glob(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')) >= 1)

        if file_exist and not self.mute:
            print("Output already exist: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')

        if self.overwrite and not self.mute:
            print("Overwriting is turned ON!")

        return file_exist and not self.overwrite


    def calculate(self):
        """

        :param N: Defines the width of the ring by the number of pixels it has to contain
        :return: self.image the imput fits file.
        """
        if not self.mute:
            print("~~ Calculating "+self.__class__.__name__+" with parameters " + self.suffix+" ~~")

        if self.rm_edge is not None:
            # Mask out a band of 10 pixels around the edges of the finite pixels of the image.
            IWA,OWA,inner_mask,outer_mask = get_occ(self.image, centroid = self.center)
            conv_kernel = np.ones((self.rm_edge,self.rm_edge))
            wider_mask = convolve2d(outer_mask,conv_kernel,mode="same")
            self.image[np.where(np.isnan(wider_mask))] = np.nan

        # If GOI_list_folder is not None. Mask the known objects from the image that will be used for calculating the
        # PDF. This masked image is given separately to the probability calculation function.
        if self.filename_noPlanets is not None:
            self.image_without_planet = mask_known_objects(self.image_noPlanets,self.prihdr_noPlanets,self.exthdr_noPlanets,self.GOI_list_folder, mask_radius = self.mask_radius)
        else:
            self.image_without_planet = mask_known_objects(self.image,self.prihdr,self.exthdr,self.GOI_list_folder, mask_radius = self.mask_radius)
        # if self.GOI_list_folder is not None:
        #     self.image_without_planet = mask_known_objects(self.image,self.prihdr,self.exthdr,self.GOI_list_folder, mask_radius = self.mask_radius)
        # else:
        #     self.image_without_planet = self.image

        if self.kernel_type is not None:
            # Check if the input file is 2D or 3D
            if hasattr(self, 'nl'): # If the file is a 3D cube
                for l_id in np.arange(self.nl):
                    self.image[l_id,:,:] = convolve2d(self.image[l_id,:,:],self.PSF,mode="same")
                    self.image_noSignal[l_id,:,:] = convolve2d(self.image_noSignal[l_id,:,:],self.PSF,mode="same")
            else: # image is 2D
                print(self.image_noSignal.shape)
                self.image = convolve2d(self.image,self.PSF,mode="same")
                self.image_noSignal = convolve2d(self.image_noSignal,self.PSF,mode="same")

        if np.size(self.image.shape) == 3:
            # Not tested
            self.stat_cube_map = np.zeros(self.image.shape)
            for k in range(self.nl):
                self.stat_cube_map[k,:,:] = get_image_stat_map_perPixMasking(self.image[k,:,:],
                                                                        self.image_without_planet[k,:,:],
                                                                        mask_radius = self.mask_radius,
                                                                        IOWA = self.IOWA,
                                                                        N = self.N,
                                                                        centroid = self.center,
                                                                        mute = self.mute,
                                                                        N_threads = self.N_threads,
                                                                        Dr= self.Dr,
                                                                        Dth = self.Dth,
                                                                        type = self.type)
        elif np.size(self.image.shape) == 2:
            self.stat_cube_map = get_image_stat_map_perPixMasking(self.image,
                                                             self.image_without_planet,
                                                             mask_radius = self.mask_radius,
                                                             IOWA = self.IOWA,
                                                             N = self.N,
                                                             centroid = self.center,
                                                             mute = self.mute,
                                                             N_threads = self.N_threads,
                                                             Dr= self.Dr,
                                                             Dth = self.Dth,
                                                             type = self.type)
        return self.stat_cube_map


    def save(self):
        """

        :return: None
        """

        if not os.path.exists(self.outputDir+os.path.sep+self.folderName):
            os.makedirs(self.outputDir+os.path.sep+self.folderName)

        if hasattr(self,"prihdr") and hasattr(self,"exthdr"):
            # Save the parameters as fits keywords
            # STA##### stands for STAtistic
            self.exthdr["STA_TYPE"] = self.type

            self.exthdr["STAFILEN"] = self.filename_path
            self.exthdr["STAINDIR"] = self.inputDir
            self.exthdr["STAOUTDI"] = self.outputDir
            self.exthdr["STAFOLDN"] = self.folderName

            self.exthdr["STAMASKR"] = self.mask_radius
            self.exthdr["STA_IOWA"] = str(self.IOWA)
            self.exthdr["STA_N"] = self.N
            self.exthdr["STA_DR"] = self.Dr
            self.exthdr["STA_DTH"] = self.Dth
            self.exthdr["STA_TYPE"] = self.type
            self.exthdr["STARMEDG"] = self.rm_edge
            self.exthdr["STAGOILF"] = self.GOI_list_folder
            self.exthdr["STAKERTY"] = str(self.kernel_type)
            self.exthdr["STAKERWI"] = str(self.kernel_width)

            # # This parameters are not always defined
            # if hasattr(self,"spectrum_name"):
            #     self.exthdr["STASPECN"] = self.spectrum_name

            if not self.mute:
                print("Saving: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(header=self.prihdr))
            hdulist.append(pyfits.ImageHDU(header=self.exthdr, data=self.stat_cube_map, name=self.suffix))
            hdulist.writeto(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits', clobber=True)
        else:
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.ImageHDU(data=self.stat_cube_map, name=self.suffix))
            hdulist.append(pyfits.ImageHDU(name=self.suffix))

            hdulist[1].header["STA_TYPE"] = self.type

            hdulist[1].header["STAFILEN"] = self.filename_path
            hdulist[1].header["STAINDIR"] = self.inputDir
            hdulist[1].header["STAOUTDI"] = self.outputDir
            hdulist[1].header["STAFOLDN"] = self.folderName

            hdulist[1].header["STAMASKR"] = self.mask_radius
            hdulist[1].header["STA_IOWA"] = self.IOWA
            hdulist[1].header["STA_N"] = self.N
            hdulist[1].header["STA_DR"] = self.Dr
            hdulist[1].header["STA_DTH"] = self.Dth
            hdulist[1].header["STA_TYPE"] = self.type
            hdulist[1].header["STARMEDG"] = self.rm_edge
            hdulist[1].header["STAGOILF"] = self.GOI_list_folder
            hdulist[1].header["STAKERTY"] = str(self.kernel_type)
            hdulist[1].header["STAKERWI"] = str(self.kernel_width)

            if not self.mute:
                print("Saving: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')
            hdulist.writeto(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits', clobber=True)

        return None

    def load(self):
        """

        :return: None
        """

        return None