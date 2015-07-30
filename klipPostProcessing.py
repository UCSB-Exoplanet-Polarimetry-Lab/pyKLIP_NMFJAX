
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from scipy.signal import convolve
from scipy.optimize import curve_fit
import astropy.io.fits as pyfits
from astropy.modeling import models, fitting
from copy import copy
import warnings
from scipy.stats import nanmedian
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import multiprocessing as mp
import itertools
import glob, os
from sys import stdout
import platform
import xml.etree.cElementTree as ET

import spectra_management as spec
from kpp_utils import *
from kpp_pdf import *
from kpp_std import *
from kpp_metrics import *
from kpp_detections import *
import instruments.GPI as GPI

def planet_detection_in_dir_per_file_per_spectrum_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call planet_detection_in_dir_per_file_per_spectrum() with a tuple of parameters.
    """
    return planet_detection_in_dir_per_file_per_spectrum(*params)

def planet_detection_in_dir_per_file_per_spectrum(spectrum_name_it,
                                                  filename,
                                                  filter,
                                                  sat_spot_spec = None,
                                                  metrics = None,
                                                  outputDir = '',
                                                  star_type = "",
                                                  star_temperature = None,
                                                  PSF_cube = None,
                                                  metrics_only = False,
                                                  planet_detection_only = False,
                                                  mute = False,
                                                  GOI_list = None,
                                                  overwrite_metric = False,
                                                  overwrite_stat = False,
                                                  proba_using_mask_per_pixel = False,
                                                  SNR = True,
                                                  probability = True,
                                                  N_threads_metric = None,
                                                  detection_metric = None):

        # Define the output Foldername
        if spectrum_name_it != "":
            spectrum_name = spectrum_name_it.split(os.path.sep)
            spectrum_name = spectrum_name[len(spectrum_name)-1].split(".")[0]
        else:
            spectrum_name = "satSpotSpec"

        folderName = spectrum_name+os.path.sep



        if spectrum_name_it != "":
            if not mute:
                print("spectrum model: "+spectrum_name_it)
            # Interpolate the spectrum of the planet based on the given filename
            wv,planet_sp = spec.get_planet_spectrum(spectrum_name_it,filter)

            if sat_spot_spec is not None and (star_type !=  "" or star_temperature is not None):
                # Interpolate a spectrum of the star based on its spectral type/temperature
                wv,star_sp = spec.get_star_spectrum(filter,star_type,star_temperature)
                spectrum = (sat_spot_spec/star_sp)*planet_sp
            else:
                if not mute:
                    print("No star spec or sat spot spec so using sole planet spectrum.")
                spectrum = copy(planet_sp)
        else:
            if sat_spot_spec is not None:
                if not mute:
                    print("Default sat spot spectrum will be used.")
                spectrum = copy(sat_spot_spec)
            else:
                if not mute:
                    print("Using gpi filter "+filter+" spectrum. Could find neither sat spot spectrum nor planet spectrum.")
                wv,spectrum = spec.get_gpi_filter(filter)

        if 0:
            plt.plot(sat_spot_spec)
            plt.plot(spectrum)
            plt.show()


        if not planet_detection_only:
            if not mute:
                print("Calling calculate_metrics() on "+filename)
            calculate_metrics(filename,
                              metrics,
                                PSF_cube = PSF_cube,
                                outputDir = outputDir,
                                folderName = folderName,
                                spectrum=spectrum,
                                mute = mute,
                                SNR = SNR,
                                probability = probability,
                                GOI_list = GOI_list,
                                overwrite_metric = overwrite_metric,
                                overwrite_stat = overwrite_stat,
                                proba_using_mask_per_pixel = proba_using_mask_per_pixel,
                                N_threads = N_threads_metric )


        if not metrics_only:
            if not mute:
                print("Calling candidate_detection() on "+outputDir+folderName)
            candidate_detection(outputDir+folderName,
                                mute = mute,
                                metric = detection_metric)



def planet_detection_in_dir_per_file_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call planet_detection_in_dir_per_file() with a tuple of parameters.
    """
    return planet_detection_in_dir_per_file(*params)

def planet_detection_in_dir_per_file(filename,
                                      metrics = None,
                                      directory = "."+os.path.sep,
                                      outputDir = '',
                                      spectrum_model = "",
                                      star_type = "",
                                      star_temperature = None,
                                      user_defined_PSF_cube = None,
                                      metrics_only = False,
                                      planet_detection_only = False,
                                      mute = False,
                                      threads = False,
                                      GOI_list = None,
                                      overwrite_metric = False,
                                      overwrite_stat = False,
                                      proba_using_mask_per_pixel = False,
                                      SNR = True,
                                      probability = True,
                                      detection_metric = None):
    # Get the number of KL_modes for this file based on the filename *-KL#-speccube*
    splitted_name = filename.split("-KL")
    splitted_after_KL = splitted_name[1].split("-speccube.")
    N_KL_modes = int(splitted_after_KL[0])

    # Get the prefix of the filename
    splitted_before_KL = splitted_name[0].split(os.path.sep)
    prefix = splitted_before_KL[np.size(splitted_before_KL)-1]

    compact_date = prefix.split("-")[1]



    #grab the headers
    hdulist = pyfits.open(filename)
    prihdr = hdulist[0].header
    try:
        filter = prihdr['IFSFILT'].split('_')[1]
    except:
        # If the keywords could not be found.
        if not mute:
            print("Couldn't find IFSFILT keyword. Assuming H.")
        filter = "H"

    if user_defined_PSF_cube is not None:
        if not mute:
            print("User defined PSF cube: "+user_defined_PSF_cube)
        filelist_ori_PSFs_cube = glob.glob(user_defined_PSF_cube)
    else:
        filelist_ori_PSFs_cube = glob.glob(directory+os.path.sep+"*"+compact_date+"*-original_radial_PSF_cube.fits")

    if np.size(filelist_ori_PSFs_cube) == 1:
        if not mute:
            print("I found a radially averaged PSF. I'll take it.")
        hdulist = pyfits.open(filelist_ori_PSFs_cube[0])
        #print(hdulist[1].data.shape)
        PSF_cube = hdulist[1].data[:,::-1,:]
        #PSF_cube = np.transpose(hdulist[1].data,(1,2,0))[:,::-1,:] #np.swapaxes(np.rollaxis(hdulist[1].data,2,1),0,2)
        sat_spot_spec = np.nanmax(PSF_cube,axis=(1,2))
        # Remove the spectral shape from the psf cube because it is dealt with independently
        for l_id in range(PSF_cube.shape[0]):
            PSF_cube[l_id,:,:] /= sat_spot_spec[l_id]
    elif np.size(filelist_ori_PSFs_cube) == 0:
        if not mute:
            print("I didn't find any PSFs file so trying to generate one.")

        print(directory+os.path.sep+compact_date+"*_spdc_distorcorr.fits")
        cube_filename_list = glob.glob(directory+os.path.sep+compact_date+"*_spdc_distorcorr.fits")
        if len(cube_filename_list) == 0:
            cube_filename_list = glob.glob(directory+os.path.sep+compact_date+"*_spdc.fits")
            if len(cube_filename_list) == 0:
                if not mute:
                    print("I can't find the cubes to calculate the PSF from so I will use a default gaussian.")
                sat_spot_spec = None
                PSF_cube = None

        dataset = GPI.GPIData(cube_filename_list)
        if not mute:
            print("Calculating the planet PSF from the satellite spots...")
        dataset.generate_psf_cube(20)
        # Save the original PSF calculated from combining the sat spots
        dataset.savedata(directory + os.path.sep + prefix+"-original_PSF_cube.fits", dataset.psfs,
                                  astr_hdr=dataset.wcs[0], filetype="PSF Spec Cube")
        # Calculate and save the rotationally invariant psf (ie smeared out/averaged).
        PSF_cube = dataset.get_radial_psf(save = directory + os.path.sep + prefix)
        sat_spot_spec = np.nanmax(PSF_cube,axis=(1,2))
        # Remove the spectral shape from the psf cube because it is dealt with independently
        for l_id in range(PSF_cube.shape[0]):
            PSF_cube[l_id,:,:] /= sat_spot_spec[l_id]
    else:
        if not mute:
            print("I found several files for the PSF cube matching the given filter so I don't know what to do and I quit")
        return 0


    prefix = prefix+"-KL"+str(N_KL_modes)

    if len(spectrum_model) == 1 and not isinstance(spectrum_model,list):
        spectrum_model =[spectrum_model]
    else:
        if not mute:
            print("Iterating over several model spectra")

    if outputDir == '':
        outputDir = directory
    outputDir += os.path.sep+"planet_detec_"+prefix+os.path.sep

    if threads:
        N_threads = np.size(spectrum_model)
        pool = NoDaemonPool(processes=N_threads)
        #pool = mp.Pool(processes=N_threads)

        N_threads_metric = mp.cpu_count()/N_threads
        if N_threads_metric <= 1:
            N_threads_metric = None

        pool.map(planet_detection_in_dir_per_file_per_spectrum_star, itertools.izip(spectrum_model,
                                                                           itertools.repeat(filename),
                                                                           itertools.repeat(filter),
                                                                           itertools.repeat(sat_spot_spec),
                                                                           itertools.repeat(metrics),
                                                                           itertools.repeat(outputDir),
                                                                           itertools.repeat(star_type),
                                                                           itertools.repeat(star_temperature),
                                                                           itertools.repeat(PSF_cube),
                                                                           itertools.repeat(metrics_only),
                                                                           itertools.repeat(planet_detection_only),
                                                                           itertools.repeat(mute),
                                                                           itertools.repeat(GOI_list),
                                                                           itertools.repeat(overwrite_metric),
                                                                           itertools.repeat(overwrite_stat),
                                                                           itertools.repeat(proba_using_mask_per_pixel),
                                                                           itertools.repeat(SNR),
                                                                           itertools.repeat(probability),
                                                                           itertools.repeat(N_threads_metric),
                                                                           itertools.repeat(detection_metric)))
        pool.close()
    else:
        for spectrum_name_it in spectrum_model:
            planet_detection_in_dir_per_file_per_spectrum(spectrum_name_it,
                                                          filename,
                                                          filter,
                                                          sat_spot_spec = sat_spot_spec,
                                                          metrics = metrics,
                                                          outputDir = outputDir,
                                                          star_type = star_type,
                                                          star_temperature = star_temperature,
                                                          PSF_cube = PSF_cube,
                                                          metrics_only = metrics_only,
                                                          planet_detection_only = planet_detection_only,
                                                          mute = mute,
                                                          GOI_list = GOI_list,
                                                          overwrite_metric = overwrite_metric,
                                                          overwrite_stat = overwrite_stat,
                                                          proba_using_mask_per_pixel = proba_using_mask_per_pixel,
                                                          SNR = SNR,
                                                          probability = probability,
                                                          N_threads_metric = None,
                                                          detection_metric = detection_metric)


    if not metrics_only:
        if not mute:
            print("Calling gather_detections() on "+outputDir)
        gather_detections(outputDir,PSF_cube, mute = mute,which_metric = detection_metric)

def planet_detection_in_dir_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call planet_detection_in_dir() with a tuple of parameters.
    """
    return planet_detection_in_dir(*params)

def planet_detection_in_dir(directory = "."+os.path.sep,
                            filename_prefix_is = '',
                            numbasis = None,
                            outputDir = '',
                            spectrum_model = "",
                            star_type = "",
                            star_temperature = None,
                            user_defined_PSF_cube = None,
                            metrics = None,
                            threads = False,
                            metrics_only = False,
                            planet_detection_only = False,
                            mute = True,
                            GOI_list = None,
                            overwrite_metric = False,
                            overwrite_stat = False,
                            proba_using_mask_per_pixel = False,
                            SNR = True,
                            probability = True,
                            detection_metric = None):
    '''
    Apply the planet detection algorithm for all pyklip reduced cube respecting a filter in a given folder.
    By default the filename filter used is pyklip-*-KL*-speccube.fits.
    It will look for a PSF file like pyklip-*-KL*-speccube-solePSFs.fits to extract a klip reduced PSF.
        Note: If you want to include a spectrum it can be already included in the PSF when reducing it with klip.
            The other option if there is no psf available for example is to give a spectrum as an input.
    If no pyklip-*-KL*-speccube-solePSFs.fits is found it will for a pyklip-*-original_radial_PSF_cube.fits which is a PSF
    built from the sat spots but not klipped.

    /!\ The klipped PSF was not well understood by JB so anything related to it in pyklip will not work. It needs to be
    entirely "rethought" and implemented from scratch.

    Inputs:
        directory: directory in which the function will look for suitable fits files and run the planet detection algorithm.
        filename_prefix_is: Look for file containing of the form "/"+filename_filter+"-KL*-speccube.fits"
        numbasis: Integer. Apply algorithm only to klipped images that used numbasis KL modes.
        outputDir: Directory to where to save the output folder. By default it is saved in directory.
        spectrum_model: Mark Marley's spectrum filename. E.g. "/Users/jruffio/gpi/pyklip/t800g100nc.flx"
        star_type: Spectral type of the star (works only for type V). E.g. "G5" for G5V type.
        star_temperature: Temperature of the star. (replace star_type)
        threads: If true, parallel computation of several files (no prints in the console).
                Otherwise sequential with bunch of prints (for debugging).
        metrics_only: If True, compute only the metrics (Matched filter SNR, shape SNR, ...) but the planet detection
                algorithm is not applied.
        planet_detection_only: ????????
        mute: ???????????????

    Outputs:
        For each file an output folder is created:
            outputDir = directory + "/planet_detec_"+prefix+"_KL"+str(N_KL_modes)+"/"
        The outputs of the detection can be found there.

    '''

    if numbasis is not None:
        numbasis = str(numbasis)
    else:
        numbasis = '*'

    if filename_prefix_is == '':
        filelist_klipped_cube = glob.glob(directory+os.path.sep+"pyklip-*-KL"+numbasis+"-speccube.fits")
    else:
        filelist_klipped_cube = glob.glob(directory+os.path.sep+filename_prefix_is+"-KL"+numbasis+"-speccube.fits")
        #print(directory+"/"+filename_prefix_is+"-KL"+numbasis+"-speccube.fits")

    if len(filelist_klipped_cube) == 0:
        if not mute:
            print("No suitable files found in: "+directory)
    else:
        if not mute:
            print(directory+"contains suitable file for planet detection:")
            for f_name in filelist_klipped_cube:
                print(f_name)

        if 0 and threads:
            N_threads = np.size(filelist_klipped_cube)
            pool = NoDaemonPool(processes=N_threads)
            #pool = mp.Pool(processes=N_threads)
            pool.map(planet_detection_in_dir_per_file_star, itertools.izip(filelist_klipped_cube,
                                                                           itertools.repeat(metrics),
                                                                           itertools.repeat(directory),
                                                                           itertools.repeat(outputDir),
                                                                           itertools.repeat(spectrum_model),
                                                                           itertools.repeat(star_type),
                                                                           itertools.repeat(star_temperature),
                                                                           itertools.repeat(user_defined_PSF_cube),
                                                                           itertools.repeat(metrics_only),
                                                                           itertools.repeat(planet_detection_only),
                                                                           itertools.repeat(mute),
                                                                           itertools.repeat(threads),
                                                                           itertools.repeat(GOI_list),
                                                                           itertools.repeat(overwrite_metric),
                                                                           itertools.repeat(overwrite_stat),
                                                                           itertools.repeat(proba_using_mask_per_pixel),
                                                                           itertools.repeat(SNR),
                                                                           itertools.repeat(probability),
                                                                           itertools.repeat(detection_metric)))
            pool.close()
        else:
            for filename in filelist_klipped_cube:
                planet_detection_in_dir_per_file(filename,
                                                 metrics = metrics,
                                                 directory = directory,
                                                 outputDir = outputDir,
                                                 spectrum_model = spectrum_model,
                                                 star_type = star_type,
                                                 star_temperature = star_temperature,
                                                 user_defined_PSF_cube = user_defined_PSF_cube,
                                                 metrics_only = metrics_only,
                                                 planet_detection_only = planet_detection_only,
                                                 mute = mute,
                                                 threads = threads,
                                                 GOI_list = GOI_list,
                                                 overwrite_metric = overwrite_metric,
                                                 overwrite_stat = overwrite_stat,
                                                 proba_using_mask_per_pixel = proba_using_mask_per_pixel,
                                                 SNR = SNR,
                                                 probability = probability,
                                                 detection_metric = detection_metric)





def planet_detection_campaign(campaign_dir = "."+os.path.sep):
    '''
    Run the candidate finder (or "planet detection" but I feel like it is too optimistic...) on every single stars in
    GPI campaign directory using default parameters. It will go through all the folders in that directory (a folder is
    assumed to correspond to a star) and look for suitable klipped images in the autoreduced folder.
    For each klipped images it will create a corresponding "planet_detec_<filePrefix>" folder where the candidates will
    be displayed. You will also find the metric maps as well as probability maps corresponding to the different spectrum
    template.

    /!\ Still partially hardcoded: GOI_list.xml path. Works only for JB.

    /!\ Function needs cleaning and comments.

    - Suitable files:
    Only the files reduced by the data cruncher with parameters k100a7s4m3 and 20KL modes will be used. The other files
    will be ignored.

    - Spectrum templates:
    Four spectra will be used for pattern recognition in the datacubes: t1500g100nc,t950g32nc,t600g32nc and default
    satellite spots spectrum. This is using Mark Marley's model and filename convention.

    - Calculate matched filter and shape metric

    - Use shape only for candidate detection purposes.

    - Calculate PSF cube from satellite spots of the dataset (The data cruncher reduced cube that should be in
    campaign_dir/<object>/autoreduced/ )

    :param campaign_dir: Directory of the campaign. Where the observed star folders should be.

    :return: Apply candidates finding algorithm to any suitable files of any observed stars in campaign_dir.
            See planet_detection_in_dir_per_file() for the outputs details.
    '''


    outputDir = ''
    star_type = ''
    metrics = None

    filename_filter = "pyklip-*-k100a7s4m3"
    numbasis = 20
    spectrum_model = ["."+os.path.sep+"spectra"+os.path.sep+"g100ncflx"+os.path.sep+"t1500g100nc.flx",
                          "."+os.path.sep+"spectra"+os.path.sep+"g32ncflx"+os.path.sep+"t950g32nc.flx",
                          "."+os.path.sep+"spectra"+os.path.sep+"g32ncflx"+os.path.sep+"t600g32nc.flx",
                          ""]
    star_type = "G4"
    metrics = ["matchedFilter","shape"]
    detection_metric = "shape"

    if 0:
        if platform.system() == "Windows":
            user_defined_PSF_cube = "C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\code\\pyklipH-S20141218-k100a7s4m3-original_radial_PSF_cube.fits"
        else:
            user_defined_PSF_cube = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB/code/pyklipH-S20141218-k100a7s4m3-original_radial_PSF_cube.fits"
    else:
        user_defined_PSF_cube = None

    if platform.system() == "Windows":
        GOI_list = "C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\GOI_list.xml"
    elif platform.system() == "Linux":
        GOI_list = "/home/sda/Dropbox (GPI)/SCRATCH/Scratch/JB/GOI_list.xml"
    else:
        GOI_list = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB/GOI_list.xml"

    inputDirs = []
    for inputDir in os.listdir(campaign_dir):
        if not inputDir.startswith('.'):

            inputDirs.append(campaign_dir+inputDir+os.path.sep+"autoreduced"+os.path.sep)

            inputDir = campaign_dir+inputDir+os.path.sep+"autoreduced"+os.path.sep
            print(inputDir)
            if 1:
                planet_detection_in_dir(inputDir,
                                        outputDir= outputDir,
                                        filename_prefix_is=filename_filter,
                                        spectrum_model=spectrum_model,
                                        star_type=star_type,
                                        metrics = metrics,
                                        numbasis=numbasis,
                                        user_defined_PSF_cube=user_defined_PSF_cube,
                                        metrics_only = False,
                                        planet_detection_only = False,
                                        threads = True,
                                        mute = False,
                                        GOI_list = GOI_list,
                                        overwrite_metric=True,
                                        overwrite_stat=True,
                                        proba_using_mask_per_pixel = True,
                                        SNR = False,
                                        probability = True,
                                        detection_metric = detection_metric)

    if 0 and 0:
        N_threads = len(inputDirs)
        print(N_threads)
        pool = mp.Pool(processes=N_threads)
        #Check parameters
        #pool.map(planet_detection_in_dir_star, itertools.izip(inputDirs,
        #                                                               itertools.repeat(filename_filter),
        #                                                               itertools.repeat(numbasis),
        #                                                               itertools.repeat(outputDir),
        #                                                               itertools.repeat(spectrum_model),
        #                                                               itertools.repeat(star_type),
        #                                                               itertools.repeat(None),
        #                                                               itertools.repeat(user_defined_PSF_cube),
        #                                                               itertools.repeat(metrics),
        #                                                               itertools.repeat(False),
        #                                                               itertools.repeat(True),
        #                                                               itertools.repeat(False),
        #                                                               itertools.repeat(False)))
        pool.close()

