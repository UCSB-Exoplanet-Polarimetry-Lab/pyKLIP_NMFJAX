__author__ = 'jruffio'
import os
import astropy.io.fits as pyfits
from glob import glob
import multiprocessing as mp
import numpy as np
from scipy.signal import convolve2d

from pyklip.kpp.utils.kppSuperClass import KPPSuperClass
from pyklip.kpp.stat.statPerPix_utils import *
from pyklip.kpp.utils.GOI import *
from pyklip.kpp.utils.GPIimage import *
import pyklip.instruments.GPI as GPI
from pyklip.kpp.metrics.FMMF import FMMF
import pyklip.parallelized as parallelized
from pyklip.kpp.stat.stat import Stat
import pyklip.kpp.utils.mathfunc as kppmath
from pyklip.kpp.metrics.shapeOrMF import ShapeOrMF
from pyklip.kpp.kppPerDir import *

class Contrast(KPPSuperClass):
    """
    Class for SNR calculation.
    """
    def __init__(self,filename,dir_fakes,PSF_cube_filename = None,
                 inputDir = None,
                 outputDir = None,
                 mute=None,
                 N_threads=None,
                 label = None,
                 mask_radius = None,
                 IOWA = None,
                 GOI_list_folder = None,
                 GPI_TSpT_csv = None,
                 overwrite = False,
                 contrast_filename = None,
                 fakes_SNR = None,
                 fakes_spectrum = None,
                 spectrum_name=None):
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
        super(Contrast, self).__init__(filename,
                                     inputDir = inputDir,
                                     outputDir = outputDir,
                                     folderName = None,
                                     mute=mute,
                                     N_threads=N_threads,
                                     label=label,
                                     overwrite = overwrite)

        if mask_radius is None:
            self.mask_radius = 3
        else:
            self.mask_radius = mask_radius

        if fakes_SNR is None:
            self.fakes_SNR = 10
        else:
            self.fakes_SNR = fakes_SNR

        if fakes_spectrum is None:
            self.fakes_spectrum = "t900g100nc"
        else:
            self.fakes_spectrum = fakes_spectrum

        if spectrum_name is None:
            self.spectrum_name = "t600g100nc"
        else:
            self.spectrum_name = spectrum_name


        self.IOWA = IOWA
        self.N = 400
        self.Dr = 4
        self.type = "stddev"
        self.suffix = "2Dcontrast"
        self.GOI_list_folder = GOI_list_folder
        self.GPI_TSpT_csv = GPI_TSpT_csv
        self.contrast_filename = contrast_filename
        self.PSF_cube_filename = PSF_cube_filename
        self.dir_fakes = dir_fakes


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
        init_out = super(Contrast, self).initialize(inputDir = inputDir,
                                         outputDir = outputDir,
                                         folderName = folderName,
                                         label=label)

        if self.contrast_filename is not None:
            # Check file existence and define filename_path
            if self.inputDir is None or os.path.isabs(self.contrast_filename):
                try:
                    if len(glob(self.contrast_filename)) == self.N_matching_files:
                        self.contrast_filename_path = os.path.abspath(glob(self.contrast_filename)[self.id_matching_file-1])
                    else:
                        self.contrast_filename_path = os.path.abspath(glob(self.contrast_filename)[0])
                except:
                    raise Exception("File "+self.contrast_filename+"doesn't exist.")
            else:
                try:
                    if len(glob(self.inputDir+os.path.sep+self.contrast_filename)) == self.N_matching_files:
                        self.contrast_filename_path = os.path.abspath(glob(self.inputDir+os.path.sep+self.contrast_filename)[self.id_matching_file-1])
                    else:
                        self.contrast_filename_path = os.path.abspath(glob(self.inputDir+os.path.sep+self.contrast_filename)[0])
                except:
                    raise Exception("File "+self.inputDir+os.path.sep+self.contrast_filename+" doesn't exist.")

        if self.PSF_cube_filename is not None:
            # Check file existence and define filename_path
            if self.inputDir is None or os.path.isabs(self.PSF_cube_filename):
                try:
                    self.PSF_cube_path = os.path.abspath(glob(self.PSF_cube_filename)[self.id_matching_file])
                except:
                    raise Exception("File "+self.PSF_cube_filename+"doesn't exist.")
            else:
                try:
                    self.PSF_cube_path = os.path.abspath(glob(self.inputDir+os.path.sep+self.PSF_cube_filename)[self.id_matching_file])
                except:
                    raise Exception("File "+self.inputDir+os.path.sep+self.PSF_cube_filename+" doesn't exist.")

            # Open the fits file on which the metric will be applied
            hdulist = pyfits.open(self.PSF_cube_path)
            if not self.mute:
                print("Opened: "+self.PSF_cube_path)

            # grab the data and headers
            try:
                self.PSF_cube = hdulist[1].data
            except:
                raise Exception("Couldn't read "+self.PSF_cube_path+". Is it a .fits file with GPI extensions convention?")

        # Get center of the image (star position)
        try:
            # Retrieve the center of the image from the fits headers.
            self.center = [self.exthdr['PSFCENTX'], self.exthdr['PSFCENTY']]
        except:
            # If the keywords could not be found the center is defined as the middle of the image
            if not self.mute:
                print("Couldn't find PSFCENTX and PSFCENTY keywords.")
            self.center = [(self.nx-1)/2,(self.ny-1)/2]


        try:
            self.folderName = self.exthdr["METFOLDN"]+os.path.sep
        except:
            pass

        file_ext_ind = os.path.basename(self.filename_path)[::-1].find(".")
        self.prefix = os.path.basename(self.filename_path)[:-(file_ext_ind+1)]

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

        # Inject fakes
        print(self.inputDir)


        if self.dir_fakes is None:
            self.dir_fakes = os.join.path(self.outputDir,self.folderName)

        if not os.path.exists(self.dir_fakes):
            os.makedirs(self.dir_fakes)

        if 0:
            if not self.mute:
                print("~~ Reducing pyklip no fakes ~~")
            spdc_glob = glob(self.inputDir+os.path.sep+"S*_spdc_distorcorr.fits")
            dataset = GPI.GPIData(spdc_glob,highpass=True,meas_satspot_flux=True,numthreads=self.N_threads,PSF_cube = self.PSF_cube)
            parallelized.klip_dataset(dataset,
                                      outputdir=self.dir_fakes,
                                      mode="ADI+SDI",
                                      annuli=9,
                                      subsections=4,
                                      movement=1,
                                      numbasis=[20,50,150],
                                      spectrum="methane",
                                      fileprefix="nofakes_k150a9s4m1methane",
                                      numthreads=self.N_threads,
                                      calibrate_flux=True)


        with open(self.contrast_filename_path, 'rt') as cvs_contrast:
            cvs_contrast_reader = csv.reader(filter(lambda row: row[0]!="#",cvs_contrast),delimiter=' ')
            list_contrast = list(cvs_contrast_reader)
            contrast_str_arr = np.array(list_contrast, dtype='string')
            col_names = contrast_str_arr[0]
            contrast_arr = contrast_str_arr[1::].astype(np.float)
            sep_samples = contrast_arr[:,0]
            Ttype_contrast = np.squeeze(contrast_arr[:,np.where("T-Type"==col_names)])
            Ltype_contrast = np.squeeze(contrast_arr[:,np.where("L-Type"==col_names)])


        if not self.mute:
            print("~~ Injecting fakes ~~")
        pa_shift_list = [0,30,60]
        for pa_shift in pa_shift_list:
            fake_flux_dict = dict(mode = "SNR",SNR=self.fakes_SNR,sep_arr = sep_samples, contrast_arr=Ttype_contrast)
            fake_position_dict = dict(mode = "spirals",pa_shift=pa_shift)

            # Inject the fakes
            if 0:
                spdc_glob = glob(self.inputDir+os.path.sep+"S*_spdc_distorcorr.fits")
                if not self.mute:
                    print("~~ Reading dataset ~~")
                dataset = GPI.GPIData(spdc_glob,highpass=True,meas_satspot_flux=True,numthreads=self.N_threads,PSF_cube = self.PSF_cube)
                GPI.generate_spdc_with_fakes(dataset,
                                         fake_position_dict,
                                         fake_flux_dict,
                                         outputdir = self.dir_fakes,
                                         planet_spectrum = self.fakes_spectrum,
                                         PSF_cube = self.PSF_cube_path,
                                         star_type = None,
                                         GOI_list_folder = self.GOI_list_folder,
                                         mute = False,
                                         suffix = self.fakes_spectrum+"_PA{0:02d}".format(pa_shift),
                                         SpT_file_csv = self.GPI_TSpT_csv)

            # Run pyklip on the fakes
            if 0:
                # spdc_glob = glob(self.dir_fakes+os.path.sep+"S*_spdc_distorcorr*_PA{0:02d}.fits".format(pa_shift))
                # dataset = GPI.GPIData(spdc_glob,highpass=True,meas_satspot_flux=True,numthreads=self.N_threads,PSF_cube = self.PSF_cube)
                parallelized.klip_dataset(dataset,
                                          outputdir=self.dir_fakes,
                                          mode="ADI+SDI",
                                          annuli=9,
                                          subsections=4,
                                          movement=1,
                                          numbasis=[20,50,150],
                                          spectrum="methane",
                                          fileprefix="fakes_PA{0:02d}_k150a9s4m1methane".format(pa_shift),
                                          numthreads=self.N_threads,
                                          calibrate_flux=True)


        #############################
        ###### PYKLIP without sky sub MF
        self.ny_PSF = 20 # should be even
        self.nx_PSF = 20 # should be even
        # Define the cross correlation kernel
        pykliproot = os.path.dirname(os.path.realpath(parallelized.__file__))
        planet_spectrum_dir = glob(os.path.join(pykliproot,"spectra","*",self.spectrum_name+".flx"))[0]
        import pyklip.spectra_management as spec
        spectrum = spec.get_planet_spectrum(planet_spectrum_dir,"H")[1]

        hdulist = pyfits.open(glob(os.path.join(self.dir_fakes,"nofakes_k150a9s4m1methane-KL50-speccube.fits"))[0])
        self.pyklip_noSky_image = hdulist[1].data
        self.nl = self.pyklip_noSky_image.shape[0]

        image_collapsed = np.zeros((self.ny,self.nx))
        for k in range(self.nl):
            image_collapsed = image_collapsed + spectrum[k]*self.pyklip_noSky_image[k,:,:]
        self.pyklip_noSky_image = image_collapsed/np.sum(spectrum)

        kernel_width = 1.5
        # Build the grid for PSF stamp.
        x_PSF_grid, y_PSF_grid = np.meshgrid(np.arange(0,self.ny_PSF,1)-self.ny_PSF/2,
                                             np.arange(0,self.nx_PSF,1)-self.nx_PSF/2)
        gauss_PSF = kppmath.gauss2d(x_PSF_grid, y_PSF_grid,1.0,0.0,0.0,kernel_width,kernel_width)
        self.pyklip_noSky_image = convolve2d(self.pyklip_noSky_image,gauss_PSF,mode="same")

        self.real_contrast_list = []
        self.pyklip_noSky_fakes_val = []
        self.sep_list = []
        for pa_shift in pa_shift_list:
            # get throughput for pyklip images
            hdulist = pyfits.open(os.path.join(self.dir_fakes,"fakes_PA{0:02d}_k150a9s4m1methane-KL50-speccube.fits".format(pa_shift)))
            self.pyklip_noSky_image_fakes = hdulist[1].data
            self.exthdr_fakes = hdulist[1].header
            self.prihdr_fakes = hdulist[0].header

            # Collapse pyklip cube according to spectrum
            image_collapsed = np.zeros((self.ny,self.nx))
            for k in range(self.nl):
                image_collapsed = image_collapsed + spectrum[k]*self.pyklip_noSky_image_fakes[k,:,:]
            self.pyklip_noSky_image_fakes = image_collapsed/np.sum(spectrum)

            self.pyklip_noSky_image_fakes = convolve2d(self.pyklip_noSky_image_fakes,gauss_PSF,mode="same")

            row_real_object_list,col_real_object_list = get_pos_known_objects(self.prihdr_fakes,self.exthdr_fakes,fakes_only=True)
            sep,pa_real_object_list = get_pos_known_objects(self.prihdr_fakes,self.exthdr_fakes,pa_sep=True,fakes_only=True)
            self.sep_list.extend(sep)
            for fake_id in range(100):
                try:
                    self.real_contrast_list.append(self.exthdr_fakes["FKCONT{0:02d}".format(fake_id)])
                except:
                    continue
            self.pyklip_noSky_fakes_val.extend([self.pyklip_noSky_image_fakes[np.round(row_real_object),np.round(col_real_object)] \
                                         for row_real_object,col_real_object in zip(row_real_object_list,col_real_object_list)])

        whereNoNans = np.where(np.isfinite(self.pyklip_noSky_fakes_val))
        self.pyklip_noSky_fakes_val = np.array(self.pyklip_noSky_fakes_val)[whereNoNans]
        self.sep_list =  np.array(self.sep_list)[whereNoNans]
        self.real_contrast_list =  np.array(self.real_contrast_list)[whereNoNans]

        self.sep_list,self.pyklip_noSky_fakes_val,self.real_contrast_list = zip(*sorted(zip(self.sep_list,self.pyklip_noSky_fakes_val,self.real_contrast_list)))
        whereInRange = np.where((np.array(self.sep_list)>0.2)*(np.array(self.sep_list)<1.1))
        z = np.polyfit(np.array(self.sep_list)[whereInRange],np.array(self.pyklip_noSky_fakes_val)[whereInRange]/np.array(self.real_contrast_list)[whereInRange],1)
        pyklip_noSky_throughput_func = np.poly1d(z)


        # import matplotlib.pyplot as plt
        # plt.figure(2)
        # plt.title("pyklip no sky sub")
        # plt.plot(self.sep_list,np.array(self.pyklip_noSky_fakes_val)/np.array(self.real_contrast_list),"*")
        # plt.plot(self.sep_list,pyklip_noSky_throughput_func(self.sep_list),"-")
        # plt.xlabel("Separation (arcsec)", fontsize=20)
        # plt.ylabel("Throughput (arbritrary units)", fontsize=20)
        # ax= plt.gca()
        # ax.tick_params(axis='x', labelsize=20)
        # ax.tick_params(axis='y', labelsize=20)
        # plt.show()


        if self.GOI_list_folder is not None:
            self.pyklip_noSky_image_without_planet = mask_known_objects(self.pyklip_noSky_image,self.prihdr,self.exthdr,self.GOI_list_folder, mask_radius = self.mask_radius)
        else:
            self.pyklip_noSky_image_without_planet = self.pyklip_noSky_image

        self.pyklip_noSky_1Dstddev,self.pyklip_noSky_stddev_rSamp = get_image_stddev(self.pyklip_noSky_image_without_planet,
                                                                     self.IOWA,
                                                                     N = None,
                                                                     centroid = self.center,
                                                                     r_step = self.Dr/2,
                                                                     Dr=self.Dr)
        self.pyklip_noSky_stddev_rSamp = np.array([r_tuple[0] for r_tuple in self.pyklip_noSky_stddev_rSamp])
        self.pyklip_noSky_1Dstddev = np.array(self.pyklip_noSky_1Dstddev)


        #############################
        ###### PYKLIP with sky sub MF
        filename = "*_k150a9s4m1methane-KL50-speccube.fits"
        pyklip_MFgauss = ShapeOrMF(filename,"MF","gaussian",N_threads=self.N_threads,overwrite=False,
                                   label="k150a9s4m1methane-KL50",mute=self.mute,keepPrefix=True,kernel_width=1.5,
                                   GPI_TSpT_csv=self.GPI_TSpT_csv)
        err_list = kppPerDir(self.dir_fakes,
                              [pyklip_MFgauss],
                              spec_path_list=[self.spectrum_name],
                              mute_error = False)
        for err in err_list:
            print(err)

        nofakes_filename = os.path.join(self.dir_fakes,"planet_detec_k150a9s4m1methane-KL50",self.spectrum_name,
                                           "nofakes_k150a9s4m1methane-KL50-speccube-MF3Dgaussian.fits")
        fakes_filename_list = [os.path.join(self.dir_fakes,"planet_detec_k150a9s4m1methane-KL50",self.spectrum_name,
                               "fakes_PA{0:02d}_k150a9s4m1methane-KL50-speccube-MF3Dgaussian.fits".format(pa_shift)) for pa_shift in pa_shift_list]
        separation1,contrast_curve1,throughput_tuple1 = calculate_constrat(nofakes_filename,
                           fakes_filename_list,
                           GOI_list_folder=self.GOI_list_folder,
                           mask_radius=self.mask_radius,IOWA=(0.2,1.1),Dr=self.Dr,
                           save_dir = self.inputDir,
                           suffix="pyklip_MFgauss",spec_type="T-type")



        #############################
        ###### PYKLIP with sky sub shape
        filename = "*_k150a9s4m1methane-KL50-speccube.fits"
        pyklip_SHgauss = ShapeOrMF(filename,"shape","gaussian",N_threads=self.N_threads,overwrite=False,
                                   label="k150a9s4m1methane-KL50",mute=self.mute,keepPrefix=True,kernel_width=1.5,
                                   GPI_TSpT_csv=self.GPI_TSpT_csv)
        err_list = kppPerDir(self.dir_fakes,
                              [pyklip_SHgauss],
                              spec_path_list=[self.spectrum_name],
                              mute_error = False)
        for err in err_list:
            print(err)

        nofakes_filename = os.path.join(self.dir_fakes,"planet_detec_k150a9s4m1methane-KL50",self.spectrum_name,
                                           "nofakes_k150a9s4m1methane-KL50-speccube-shape3Dgaussian.fits")
        fakes_filename_list = [os.path.join(self.dir_fakes,"planet_detec_k150a9s4m1methane-KL50",self.spectrum_name,
                               "fakes_PA{0:02d}_k150a9s4m1methane-KL50-speccube-shape3Dgaussian.fits".format(pa_shift)) for pa_shift in pa_shift_list]
        separation2,contrast_curve2,throughput_tuple2 = calculate_constrat(nofakes_filename,
                           fakes_filename_list,
                           GOI_list_folder=self.GOI_list_folder,
                           mask_radius=self.mask_radius,IOWA=(0.2,1.1),Dr=self.Dr,
                           save_dir = self.inputDir,
                           suffix="pyklip_SHgauss",spec_type="T-type")


        #############################
        ###### FMMF
        if 0:
            for pa_shift in pa_shift_list:
                FMMFObj = FMMF(filename = "S*_spdc_distorcorr*_PA{0:02d}.fits".format(pa_shift),
                                outputDir=None,
                                N_threads=self.N_threads,
                                predefined_sectors = "oneAc",#"oneAc",#"HR_4597",#"smallSep",
                                label = "FMMF_PA{0:02d}".format(pa_shift),
                                quickTest=False,
                                overwrite=False,
                                mute_progression = True,
                                numbasis=[30],
                                mvt=0.5,
                                mvt_noTemplate=False,
                                SpT_file_csv = self.GPI_TSpT_csv,
                                fakes_only=True)
                inputDir = self.dir_fakes
                kppPerDir(inputDir,[FMMFObj],spec_path_list=[self.spectrum_name],mute_error=False)


        nofakes_filename = os.path.join(os.path.join(self.inputDir,"planet_detec_FMMF",self.spectrum_name,
                                           "*_0.50-FMSH.fits".format(pa_shift)))
        fakes_filename_list = [os.path.join(self.dir_fakes,"planet_detec_FMMF_PA{0:02d}".format(pa_shift),self.spectrum_name,
                                               "*_0.50-FMSH.fits".format(pa_shift)) for pa_shift in pa_shift_list]
        separation3,contrast_curve3,throughput_tuple3 = calculate_constrat(nofakes_filename,
                           fakes_filename_list,
                           GOI_list_folder=self.GOI_list_folder,
                           mask_radius=self.mask_radius,IOWA=(0.2,1.1),Dr=self.Dr,
                           save_dir = self.inputDir,
                           suffix="FMSH",spec_type="T-type")

        nofakes_filename = os.path.join(os.path.join(self.inputDir,"planet_detec_FMMF",self.spectrum_name,
                                           "*_0.50-FMMF.fits".format(pa_shift)))
        fakes_filename_list = [os.path.join(self.dir_fakes,"planet_detec_FMMF_PA{0:02d}".format(pa_shift),self.spectrum_name,
                                               "*_0.50-FMMF.fits".format(pa_shift)) for pa_shift in pa_shift_list]
        separation4,contrast_curve4,throughput_tuple4 = calculate_constrat(nofakes_filename,
                           fakes_filename_list,
                           GOI_list_folder=self.GOI_list_folder,
                           mask_radius=self.mask_radius,IOWA=(0.2,1.1),Dr=self.Dr,
                           save_dir = self.inputDir,
                           suffix="FMMF",spec_type="T-type")

        nofakes_filename = os.path.join(os.path.join(self.inputDir,"planet_detec_FMMF",self.spectrum_name,
                                           "*_0.50-FMpF.fits".format(pa_shift)))
        fakes_filename_list = [os.path.join(self.dir_fakes,"planet_detec_FMMF_PA{0:02d}".format(pa_shift),self.spectrum_name,
                                               "*_0.50-FMpF.fits".format(pa_shift)) for pa_shift in pa_shift_list]
        separation5,contrast_curve5,throughput_tuple5 = calculate_constrat(nofakes_filename,
                           fakes_filename_list,
                           GOI_list_folder=self.GOI_list_folder,
                           mask_radius=self.mask_radius,IOWA=(0.2,1.1),Dr=self.Dr,
                           save_dir = self.inputDir,
                           suffix="FMpF",spec_type="T-type")



        import matplotlib.pyplot as plt
        #############################
        ###### FINAL CONTRAST PLOT
        legend_str_list = []
        plt.figure(4,figsize=(8,6))
        plt.plot(sep_samples,Ttype_contrast,"--", color='b', linewidth=3.0)
        legend_str_list.append("Jason T-type pyklip")
        plt.plot(sep_samples,Ltype_contrast,"--", color='r', linewidth=3.0)
        legend_str_list.append("Jason L-type pyklip")
        plt.plot(self.pyklip_noSky_stddev_rSamp*0.01413,5*self.pyklip_noSky_1Dstddev/pyklip_noSky_throughput_func(self.pyklip_noSky_stddev_rSamp*0.01413),":", color='purple', linewidth=3.0)
        legend_str_list.append("JB's T-type pyklip no sky sub")
        plt.plot(separation1,contrast_curve1,".-", color='purple', linewidth=3.0)
        legend_str_list.append("JB's T-type pyklip MF")
        plt.plot(separation2,contrast_curve2,"--", color='purple', linewidth=3.0)
        legend_str_list.append("JB's T-type pyklip shape")
        plt.plot(separation3,contrast_curve3, color='yellow', linewidth=3.0)
        legend_str_list.append("JB's T-type FMSH")
        plt.plot(separation4,contrast_curve4, color='orange', linewidth=3.0)
        legend_str_list.append("JB's T-type FMMF")
        plt.plot(separation5,contrast_curve5, color='red', linewidth=3.0)
        legend_str_list.append("JB's T-type FMpF")
        plt.xlabel("Separation (arcsec)", fontsize=20)
        plt.ylabel("Contrast (log10)", fontsize=20)
        plt.legend(legend_str_list)
        ax= plt.gca()
        ax.set_yscale('log')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.show()

        exit()

        # self.flux_1Dstddev_map = get_image_stat_map(self.image,
        #                                             self.image_without_planet,
        #                                             IOWA = self.IOWA,
        #                                             N = None,
        #                                             centroid = self.center,
        #                                             r_step = self.Dr/2,
        #                                             Dr = self.Dr,
        #                                             type = "stddev",
        #                                             image_wide = None)
        #
        #
        # self.fluxMap_stddev = get_image_stat_map_perPixMasking(self.image,
        #                                                  self.image_without_planet,
        #                                                  mask_radius = self.mask_radius,
        #                                                  IOWA = self.IOWA,
        #                                                  N = self.N,
        #                                                  centroid = self.center,
        #                                                  mute = self.mute,
        #                                                  N_threads = self.N_threads,
        #                                                  Dr= self.Dr,
        #                                                  Dth = None,
        #                                                  type = self.type)

        return None

    def save(self):
        """

        :return: None
        """

        if not os.path.exists(self.outputDir+os.path.sep+self.folderName):
            os.makedirs(self.outputDir+os.path.sep+self.folderName)

        self.suffix = "1Dcontrast"
        if not self.mute:
            print("Saving: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.csv')
        with open(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.csv', 'w+') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ')
            csvwriter.writerows([["Seps","T-Type"]])
            csvwriter.writerows(zip(self.flux_stddev_rSamp*0.01413,5*self.flux_1Dstddev*self.throughput))

        if not self.mute:
            print("Saving: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.png')
        plt.savefig(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+".png", bbox_inches='tight')

        # if hasattr(self,"prihdr") and hasattr(self,"exthdr"):
        #     # Save the parameters as fits keywords
        #     # STA##### stands for STAtistic
        #     self.exthdr["STA_TYPE"] = self.type
        #
        #     self.exthdr["STAFILEN"] = self.filename_path
        #     self.exthdr["STAINDIR"] = self.inputDir
        #     self.exthdr["STAOUTDI"] = self.outputDir
        #     self.exthdr["STAFOLDN"] = self.folderName
        #
        #     self.exthdr["STAMASKR"] = self.mask_radius
        #     self.exthdr["STA_IOWA"] = str(self.IOWA)
        #     self.exthdr["STA_N"] = self.N
        #     self.exthdr["STA_DR"] = self.Dr
        #     self.exthdr["STA_TYPE"] = self.type
        #     self.exthdr["STAGOILF"] = self.GOI_list_folder
        #
        #     # # This parameters are not always defined
        #     # if hasattr(self,"spectrum_name"):
        #     #     self.exthdr["STASPECN"] = self.spectrum_name
        #
        #     self.suffix = "2Dcontrast"
        #     if not self.mute:
        #         print("Saving: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')
        #     hdulist = pyfits.HDUList()
        #     hdulist.append(pyfits.PrimaryHDU(header=self.prihdr))
        #     hdulist.append(pyfits.ImageHDU(header=self.exthdr, data=self.fluxMap_stddev, name=self.suffix))
        #     hdulist.writeto(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits', clobber=True)
        # else:
        #     hdulist = pyfits.HDUList()
        #     hdulist.append(pyfits.ImageHDU(data=self.fluxMap_stddev, name=self.suffix))
        #     hdulist.append(pyfits.ImageHDU(name=self.suffix))
        #
        #     hdulist[1].header["STA_TYPE"] = self.type
        #
        #     hdulist[1].header["STAFILEN"] = self.filename_path
        #     hdulist[1].header["STAINDIR"] = self.inputDir
        #     hdulist[1].header["STAOUTDI"] = self.outputDir
        #     hdulist[1].header["STAFOLDN"] = self.folderName
        #
        #     hdulist[1].header["STAMASKR"] = self.mask_radius
        #     hdulist[1].header["STA_IOWA"] = self.IOWA
        #     hdulist[1].header["STA_N"] = self.N
        #     hdulist[1].header["STA_DR"] = self.Dr
        #     hdulist[1].header["STA_TYPE"] = self.type
        #     hdulist[1].header["STAGOILF"] = self.GOI_list_folder
        #
        #     self.suffix = "2Dcontrast"
        #     if not self.mute:
        #         print("Saving: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits')
        #     hdulist.writeto(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.fits', clobber=True)

        # plt.close(1)
        plt.show()
        return None

    def load(self):
        """

        :return: None
        """

        return None



def gather_contrasts(base_dir,filename_filter_list,mute = False,epoch_suffix=None):
    """
    Build the multiple combined ROC curve from individual frame ROC curve while making sure they have the same inputs.
    If the folders are organized following the convention below then it will make sure there is a ROC file for each
    filename_filter in each epoch. Otherwise it skips the epoch.

    The folders need to be organized as:
     base_dir/TARGET/autoreduced/EPOCH_Spec/filename_filter

    In the function TARGET and EPOCH are wild characters.

    It looks for all the file matching filename_filter using glob.glob and then add each individual ROC to build the
    master ROC.

    Plot master_N_false_pos vs master_N_true_detec to get a ROC curve.

    :param base_dir: Base directory from which the file search go.
    :param filename_filter: Filename filter with wild characters indicating which files to pick.
    :param mute: If True, mute prints. Default is False.
    :return: threshold_sampling,master_N_false_pos,master_N_true_detec:
        threshold_sampling: The metric sampling. It is the curve parametrization.
        master_N_false_pos: Number of false positives as a function of threshold_sampling
        master_N_true_detec: Number of true positives as a function of threshold_sampling
    """

    N_CONT = len(filename_filter_list)
    sep_samp_list = [[]]*N_CONT
    cont_list = [[]]*N_CONT

    if epoch_suffix is None:
        epoch_suffix = ""

    dirs_to_reduce = os.listdir(base_dir)
    N=0
    for object in dirs_to_reduce:
        if not object.startswith('.'):
            #print(object)

            epochDir_glob = glob(base_dir+object+os.path.sep+"autoreduced"+os.path.sep+"*_*_Spec"+epoch_suffix+os.path.sep)

            for epochDir in epochDir_glob:
                inputDir = os.path.abspath(epochDir)

                file_list = []
                for filename_filter in filename_filter_list:
                    try:
                        print(inputDir+os.path.sep+filename_filter)
                        print(glob(inputDir+os.path.sep+filename_filter)[0])
                        file_list.append(glob(inputDir+os.path.sep+filename_filter)[0])
                        # if not mute:
                        #     print("ROC: {0} in {1}. Adding.".format(filename_filter,inputDir))
                    except:
                        pass
                        # if not mute:
                        #     print("ROC: {0} unvailable in {1}. Skipping".format(filename_filter,inputDir))

                if len(file_list) == N_CONT:
                    print(file_list)
                    N=N+1
                    for index,filename in enumerate(file_list):
                        with open(filename, 'rb') as csvfile:
                            # reader = csv.reader(csvfile, delimiter=' ')
                            # csv_as_list = list(reader)
                            # detec_table_labels = csv_as_list[0]
                            # detec_table = np.array(csv_as_list[1::], dtype='string').astype(np.float)

                            cvs_contrast_reader = csv.reader(filter(lambda row: row[0]!="#",csvfile),delimiter=' ')
                            list_contrast = list(cvs_contrast_reader)
                            # print(list_contrast)
                            contrast_str_arr = np.array(list_contrast, dtype='string')
                            # print(contrast_str_arr)
                            col_names = contrast_str_arr[0]
                            contrast_arr = contrast_str_arr[1::].astype(np.float)
                            sep_samples = contrast_arr[:,0]

                            methane_idx = np.where("T-Type"==col_names)[0]

                            methane_contrast = np.squeeze(contrast_arr[:,methane_idx])

                        try:
                            cont_list[index] = cont_list[index]+methane_contrast
                        except:
                            sep_samp_list[index] = sep_samples
                            cont_list[index] = methane_contrast

    print("N files = {0}".format(N))

    return sep_samp_list,np.array(cont_list)/N


def calculate_constrat(nofakes_filename,fakes_filename_list,GOI_list_folder=None,mask_radius=None,IOWA=None,Dr=None,save_dir = None,suffix=None,spec_type=None):
    '''

    :param nofakes_filename:
    :param fakes_filename_list:
    :param GOI_list_folder:
    :param mask_radius:
    :param IOWA:
    :param Dr:
    :return:
    '''


    real_contrast_list = []
    sep_list = []
    metric_fakes_val = []

    for fakes_filename in fakes_filename_list:
        # get throughput for pyklip images
        hdulist = pyfits.open(glob(fakes_filename)[0])
        metric_image_fakes = hdulist[1].data
        exthdr_fakes = hdulist[1].header
        prihdr_fakes = hdulist[0].header

        row_real_object_list,col_real_object_list = get_pos_known_objects(prihdr_fakes,exthdr_fakes,fakes_only=True)
        sep,pa_real_object_list = get_pos_known_objects(prihdr_fakes,exthdr_fakes,pa_sep=True,fakes_only=True)
        sep_list.extend(sep)
        for fake_id in range(100):
            try:
                real_contrast_list.append(exthdr_fakes["FKCONT{0:02d}".format(fake_id)])
            except:
                continue
        metric_fakes_val.extend([metric_image_fakes[np.round(row_real_object),np.round(col_real_object)] \
                                     for row_real_object,col_real_object in zip(row_real_object_list,col_real_object_list)])

    whereNoNans = np.where(np.isfinite(metric_fakes_val))
    metric_fakes_val = np.array(metric_fakes_val)[whereNoNans]
    sep_list =  np.array(sep_list)[whereNoNans]
    real_contrast_list =  np.array(real_contrast_list)[whereNoNans]

    sep_list,metric_fakes_val,real_contrast_list = zip(*sorted(zip(sep_list,metric_fakes_val,real_contrast_list)))
    whereInRange = np.where((np.array(sep_list)>IOWA[0])*(np.array(sep_list)<IOWA[1]))
    z = np.polyfit(np.array(sep_list)[whereInRange],np.array(metric_fakes_val)[whereInRange]/np.array(real_contrast_list)[whereInRange],1)
    metric_throughput_func = np.poly1d(z)

    if 0:
        #/home/sda/Dropbox (GPI)/GPIDATA-Fakes/c_Eri/autoreduced/20141218_H_Spec_Cont/planet_detec_FMMF_PA00/t600g100nc/*_0.50-FMSH.fits'
        #/home/sda/Dropbox (GPI)/GPIDATA-Fakes/c_Eri/autoreduced/20141218_H_Spec_Cont/planet_detec_FMMF_PA30/t600g100nc/ _0.50-FMSH.fits
        import matplotlib.pyplot as plt
        plt.figure(2)
        plt.title(suffix)
        plt.plot(sep_list,np.array(metric_fakes_val)/np.array(real_contrast_list),"*")
        plt.plot(sep_list,metric_throughput_func(sep_list),"-")
        plt.xlabel("Separation (arcsec)", fontsize=20)
        plt.ylabel("Throughput (arbritrary units)", fontsize=20)
        ax= plt.gca()
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.show()

    hdulist = pyfits.open(glob(nofakes_filename)[0])
    metric_image = hdulist[1].data
    exthdr = hdulist[1].header
    prihdr = hdulist[0].header
    center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    if GOI_list_folder is not None:
        metric_image_without_planet = mask_known_objects(metric_image,prihdr,exthdr,GOI_list_folder, mask_radius = mask_radius)
    else:
        metric_image_without_planet = metric_image

    metric_1Dstddev,metric_stddev_rSamp = get_image_stddev(metric_image_without_planet,
                                                                 IOWA,
                                                                 N = None,
                                                                 centroid = center,
                                                                 r_step = Dr/2,
                                                                 Dr=Dr)
    metric_stddev_rSamp = np.array([r_tuple[0] for r_tuple in metric_stddev_rSamp])
    metric_1Dstddev = np.array(metric_1Dstddev)

    contrast_curve = 5*metric_1Dstddev/metric_throughput_func(metric_stddev_rSamp*0.01413)

    if save_dir is not None:
        if suffix is None:
            suffix = "default"

        with open(os.path.join(save_dir,"contrast-"+suffix+'.csv'), 'w+') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ')
            csvwriter.writerows([["Seps",spec_type]])
            csvwriter.writerows(zip(metric_stddev_rSamp*0.01413,contrast_curve))

        with open(os.path.join(save_dir,"throughput-"+suffix+'.csv'), 'w+') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ')
            csvwriter.writerows([["Seps","throughput","metric","contrast"]])
            csvwriter.writerows(zip(sep_list,
                                    np.array(metric_fakes_val)/np.array(real_contrast_list),
                                    np.array(metric_fakes_val),
                                    np.array(real_contrast_list)))

    throughput_tuple = (sep_list,np.array(metric_fakes_val)/np.array(real_contrast_list),np.array(metric_fakes_val),np.array(real_contrast_list))

    return metric_stddev_rSamp*0.01413,contrast_curve,throughput_tuple

