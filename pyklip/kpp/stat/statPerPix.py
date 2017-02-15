__author__ = 'JB'
import os
import astropy.io.fits as pyfits
from glob import glob
import multiprocessing as mp
import numpy as np
from scipy.signal import correlate2d

from pyklip.kpp.utils.kppSuperClass import KPPSuperClass
from pyklip.kpp.stat.statPerPix_utils import *
from pyklip.kpp.utils.GOI import *
import pyklip.kpp.utils.mathfunc as kppmath

class StatPerPix(KPPSuperClass):
    """
    Class for SNR calculation.
    """
    def __init__(self,read_func,filename,
                 folderName = None,
                 mute=None,
                 N_threads=None,
                 label = None,
                 overwrite = False,
                 mask_radius = None,
                 IOWA = None,
                 N = None,
                 Dr = None,
                 Dth = None,
                 type = None,
                 rm_edge = None,
                 OI_list_folder = None,
                 filename_noPlanets = None,
                 resolution = None):
        """
        Define the general parameters of the SNR calculation.

        Args:
            read_func: lambda function treturning a instrument object where the only input should be a list of filenames
                    to read.
                    For e.g.:
                    read_func = lambda filenames:GPI.GPIData(filenames,recalc_centers=False,recalc_wvs=False,highpass=False)
            filename: Filename of the file to process.
                        It should be the complete path unless inputDir is used in initialize().
                        It can include wild characters. The files will be reduced as given by glob.glob().
            folderName: foldername used in the definition of self.outputDir (where files shoudl be saved) in initialize().
                        folderName could be the name of the spectrum used for the reduction for e.g.
                        Default folder name is "default_out".
                        Convention is self.outputDir = #outputDir#/kpop_#labe#/#folderName#/
            mute: If True prevent printed log outputs.
            N_threads: Number of threads to be used for the metrics and the probability calculations.
                        If None use mp.cpu_count().
                        If -1 do it sequentially.
                        Note that it is not used for this super class.
            label: label used in the definition of self.outputDir (where files shoudl be saved) in initialize().
            .      Default is "default".
                   Convention is self.outputDir = #outputDir#/kpop_#labe#/#folderName#/
            overwrite: Boolean indicating whether or not files should be overwritten if they exist.
                       See check_existence().
            mask_radius: Radius of the mask used around the current pixel when use_mask_per_pixel = True.
                        Defautl value is 7 pixels.
            IOWA: (IWA,OWA) inner working angle, outer working angle. It defines boundary to the zones in which the
                        statistic is calculated.
                        If None, kpp.utils.GPIimage.get_IOWA() is used.
            N: Defines the width of the ring by the number of pixels it has to include.
                    The width of the annuli will therefore vary with sepration.
            Dr: If not None defines the width of the ring as Dr. N is then ignored if Dth is defined.
                Default value is 2 pixels.
            Dth: Define the angular size of a sector in degree (will apply for either Dr or N)
            type: Indicate the type of statistic to be calculated.
                        If "SNR" (default value) simple stddev calculation and returns SNR.
                        If "stddev" returns the pure standard deviation map.
                        If "proba" triggers proba calculation with pdf fitting.
            rm_edge: Remove the edge of the image based on the outermask of kpp.utils.GPIimage.get_occ().
                    (Not the edge of the array but the edge of the finite values in the image when there is some nan
                    padding)
            OI_list_folder: List of Object of Interest (OI) that should be masked from any standard deviation
                            calculation. See the online documentation for instructions on how to define it.
            filename_noPlanets: Filename pointing to the planet free version of filename.
                                The planet free images are used to estimate the standard deviation.
                                If filename_noPlanets has only one matching file from the function glob.glob(),
                                then it will be used for all matching filename.
                                If it has as many matching files as filename, then they will be used with a
                                one to one correspondance. Any othercase is ill-defined.
            resolution: Diameter of the resolution elements (in pix) used to do do the small sample statistic.
                    For e.g., FWHM of the PSF.
                    /!\ I am not sure the implementation is correct. We should probably do better.

        """
        # allocate super class
        super(StatPerPix, self).__init__(read_func,filename,
                                     folderName = folderName,
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
        self.resolution=resolution
        self.OI_list_folder = OI_list_folder
        self.filename_noPlanets = filename_noPlanets


    def initialize(self,inputDir = None,
                         outputDir = None,
                         folderName = None,
                         label = None):
        """
        Read the files using read_func (see the class  __init__ function).

        Can be called several time to process all the files matching the filename.

        Also define the output filename (if it were to be saved) such that check_existence() can be used.

        Args:
            inputDir: If defined it allows filename to not include the whole path and just the filename.
                            Files will be read from inputDir.
                            If inputDir is None then filename is assumed to have the absolute path.
            outputDir: Directory where to create the folder containing the outputs.
                    A kpop folder will be created to save the data. Convention is:
                    self.outputDir = outputDir+os.path.sep+"kpop_"+label+os.path.sep+folderName
            folderName: Name of the folder containing the outputs. It will be located in outputDir+os.path.sep+"kpop_"+label
                        Default folder name is "default_out".
                        A nice convention is to have one folder per spectral template.
                        If the file read has been created with KPOP, folderName is automatically defined from that
                        file.
            label: Define the suffix of the kpop output folder when it is not defined. cf outputDir. Default is "default".
            read: If true (default) read the fits file according to inputDir and filename otherwise only define self.outputDir.

        Return: True if all the files matching the filename (with wildcards) have been processed. False otherwise.
        """
        if not self.mute:
            print("~~ INITializing "+self.__class__.__name__+" ~~")
        # The super class already read the fits file
        init_out = super(StatPerPix, self).initialize(inputDir = inputDir,
                                         outputDir = outputDir,
                                         folderName = folderName,
                                         label=label)

        try:
            self.folderName = self.exthdr["KPPFOLDN"]+os.path.sep
        except:
            try:
                self.folderName = self.exthdr["METFOLDN"]+os.path.sep
                print("/!\ CAUTION: Reading deprecated data.")
            except:
                try:
                    self.folderName = self.exthdr["STAFOLDN"]+os.path.sep
                    print("/!\ CAUTION: Reading deprecated data.")
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
        self.suffix = self.suffix+tmp_suffix

        if self.filename_noPlanets is not None:# Check file existence and define filename_path
            if self.inputDir is None or os.path.isabs(self.filename_noPlanets):
                try:
                    if len(glob(self.filename_noPlanets)) == self.N_matching_files:
                        self.filename_noPlanets_path = os.path.abspath(glob(self.filename_noPlanets)[self.id_matching_file-1])
                    else:
                        self.filename_noPlanets_path = os.path.abspath(glob(self.filename_noPlanets)[0])
                except:
                    raise Exception("File "+self.filename_noPlanets+"doesn't exist.")
            else:
                try:
                    if len(glob(self.inputDir+os.path.sep+self.filename_noPlanets)) == self.N_matching_files:
                        self.filename_noPlanets_path = os.path.abspath(glob(self.inputDir+os.path.sep+self.filename_noPlanets)[self.id_matching_file-1])
                    else:
                        self.filename_noPlanets_path = os.path.abspath(glob(self.inputDir+os.path.sep+self.filename_noPlanets)[0])
                except:
                    raise Exception("File "+self.inputDir+os.path.sep+self.filename_noPlanets+" doesn't exist.")

            # Open the fits file on which the metric will be applied
            hdulist = pyfits.open(self.filename_noPlanets_path)
            if not self.mute:
                print("Opened: "+self.filename_noPlanets_path)

            # Read the image using the user defined reading function
            self.image_noPlanets_obj = self.read_func([self.filename_noPlanets_path])
            self.image_noPlanets = self.image_noPlanets_obj.input

            try:
                self.prihdr_noPlanets = self.image_obj.prihdrs[0]
            except:
                pass
            try:
                self.exthdr_noPlanets = self.image_obj.exthdrs[0]
            except:
                pass


        return init_out

    def check_existence(self):
        """
        Return whether or not a filename of the processed data can be found.

        If overwrite is True, the output is always false.

        Return: boolean
        """

        print(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')
        file_exist = (len(glob(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')) >= 1)

        if file_exist and not self.mute:
            print("Output already exist: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')

        if self.overwrite and not self.mute:
            print("Overwriting is turned ON!")

        return file_exist and not self.overwrite


    def calculate(self):
        """
        Calculate SNR map of the current image/cube.

        :return: Processed image.
        """
        if not self.mute:
            print("~~ Calculating "+self.__class__.__name__+" with parameters " + self.suffix+" ~~")

        if self.rm_edge is not None:
            # Mask out a band of 10 pixels around the edges of the finite pixels of the image.
            IWA,OWA,inner_mask,outer_mask = get_occ(self.image, centroid = self.center[0])
            conv_kernel = np.ones((self.rm_edge,self.rm_edge))
            wider_mask = correlate2d(outer_mask,conv_kernel,mode="same")
            self.image[np.where(np.isnan(wider_mask))] = np.nan

        # If OI_list_folder is not None. Mask the known objects from the image that will be used for calculating the
        # PDF. This masked image is given separately to the probability calculation function.
        if self.filename_noPlanets is not None:
            self.image_without_planet = mask_known_objects(self.image_noPlanets,self.prihdr_noPlanets,self.exthdr_noPlanets,self.OI_list_folder, mask_radius = self.mask_radius)
        else:
            self.image_without_planet = mask_known_objects(self.image,self.prihdr,self.exthdr,self.OI_list_folder, mask_radius = self.mask_radius)

        if np.size(self.image.shape) == 3:
            # Not tested
            self.stat_cube_map = np.zeros(self.image.shape)
            for k in range(self.nl):
                self.stat_cube_map[k,:,:] = get_image_stat_map_perPixMasking(self.image[k,:,:],
                                                                        self.image_without_planet[k,:,:],
                                                                        mask_radius = self.mask_radius,
                                                                        IOWA = self.IOWA,
                                                                        N = self.N,
                                                                        centroid = self.center[0],
                                                                        mute = self.mute,
                                                                        N_threads = self.N_threads,
                                                                        Dr= self.Dr,
                                                                        Dth = self.Dth,
                                                                        type = self.type,
                                                                        resolution = self.resolution)

        elif np.size(self.image.shape) == 2:
            self.stat_cube_map = get_image_stat_map_perPixMasking(self.image,
                                                             self.image_without_planet,
                                                             mask_radius = self.mask_radius,
                                                             IOWA = self.IOWA,
                                                             N = self.N,
                                                             centroid = self.center[0],
                                                             mute = self.mute,
                                                             N_threads = self.N_threads,
                                                             Dr= self.Dr,
                                                             Dth = self.Dth,
                                                             type = self.type,
                                                             resolution = self.resolution)

        return self.stat_cube_map


    def save(self):
        """
        Save the processed files as:
        #user_outputDir#+os.path.sep+"kpop_"+self.label+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits'

        :return: None
        """

        if not os.path.exists(self.outputDir+os.path.sep+self.folderName):
            os.makedirs(self.outputDir+os.path.sep+self.folderName)

        if not self.mute:
            print("Saving: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')
        hdulist = pyfits.HDUList()

        if hasattr(self,"prihdr"):
            hdulist.append(pyfits.PrimaryHDU(header=self.prihdr))
        else:
            hdulist.append(pyfits.ImageHDU(data=self.stat_cube_map, name=self.suffix))

        if hasattr(self,"exthdr"):
            # Save the parameters as fits keywords
            self.exthdr["KPPFILEN"] = os.path.basename(self.filename_path)
            self.exthdr["KPPFOLDN"] = self.folderName
            self.exthdr["KPPLABEL"] = self.label

            self.exthdr["KPPMASKR"] = self.mask_radius
            self.exthdr["KPP_IOWA"] = str(self.IOWA)
            self.exthdr["KPP_N"] = self.N
            self.exthdr["KPP_DR"] = self.Dr
            self.exthdr["KPP_DTH"] = self.Dth
            self.exthdr["KPP_TYPE"] = self.type
            self.exthdr["KPPRMEDG"] = self.rm_edge
            self.exthdr["KPPGOILF"] = self.OI_list_folder

            # This parameters are not always defined
            if hasattr(self,"filename_noSignal_path"):
                self.exthdr["KPPFILNS"] = self.filename_noSignal_path

            hdulist.append(pyfits.ImageHDU(header=self.exthdr, data=self.stat_cube_map, name=self.suffix))
        else:
            hdulist.append(pyfits.ImageHDU(name=self.suffix))

            hdulist[1].header["KPPFILEN"] = os.path.basename(self.filename_path)
            hdulist[1].header["KPPFOLDN"] = self.folderName
            hdulist[1].header["KPPLABEL"] = self.label

            hdulist[1].header["KPPMASKR"] = self.mask_radius
            hdulist[1].header["KPP_IOWA"] = str(self.IOWA)
            hdulist[1].header["KPP_N"] = self.N
            hdulist[1].header["KPP_DR"] = self.Dr
            hdulist[1].header["KPP_DTH"] = self.Dth
            hdulist[1].header["KPP_TYPE"] = self.type
            hdulist[1].header["KPPRMEDG"] = self.rm_edge
            hdulist[1].header["KPPGOILF"] = self.OI_list_folder


        hdulist.writeto(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits', clobber=True)

        return None

    def load(self):
        """

        :return: None
        """

        return None