from pyklip.kpp.kpp_metrics import *
from pyklip.kpp.kpp_detections import *
from pyklip.instruments import GPI


def planet_detection_in_dir_per_file_per_spectrum_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call planet_detection_in_dir_per_file_per_spectrum() with a tuple of parameters.
    """
    return planet_detection_in_dir_per_file_per_spectrum(*params)

def planet_detection_in_dir_per_file_per_spectrum(spectrum_filename,
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
                                                  GOI_list_folder = None,
                                                  overwrite_metric = False,
                                                  overwrite_stat = False,
                                                  proba_using_mask_per_pixel = False,
                                                  SNR = True,
                                                  probability = True,
                                                  N_threads_metric = None,
                                                  detection_metric = None,
                                                  noPlots = False):
    '''
    Calculate the metrics and probabilities of a given datacube and/or run the candidate finder algorithm for a given 
    spectral template.

    Combine the spectral template, the satellite spot spectrum and the spectral type (or temperature) of the star to
    create a corrected spectrum as it would appear on GPI's detector using:
        spectrum_on_the_detector = (sat_spot_spec/star_sp)*planet_sp
    If either one these is missing it will try to do the best it can with what it has.
    
                
    :param spectrum_filename: Path to one Mark Marley's spectrum file.
                            If the satellite spots spectrum as well as the spectral type (or temperature) of the star is
                            not given then it won't correct the spectrum for atmospheric and instrumental absorption.
                            If the string is empty ("") it will by default take the satellite spots spectrum (if given)
                            as spectral template. If sat_spot_spec is not given then it will take GPI's filter spectrum
                            for the spectral band defined by filter.
    :param filename: filename of the fits file containing the data cube to be analyzed. The cube should be a GPI cube
                    with 37 slices.
    :param filter: String with the name of the filter corresponding to filename fits file. It should have value in 'Z', 
                'Y', 'J', 'H', 'K1' and 'K2'. 
                The filter can be extracted from the header doing the following:
                    hdulist = pyfits.open(filename)
                    prihdr = hdulist[0].header
                    filter = prihdr['IFSFILT'].split('_')[1]
    :param sat_spot_spec: Vector with 37 elements containing the spectrum of the satellite spots.
    :param metrics: List of strings giving the metrics to be calculated. E.g. ["shape", "matchedFilter"]
                    The metrics available are "flatCube", "weightedFlatCube", "shape", "matchedFilter".
                    Note by default the flatCube is always saved however its statistic is not computed if "flatcube" is
                    not part of metrics. So If metrics is None basically the function creates a flat cube only.
    :param outputDir: Directory where to create the folder containing the outputs. default directory is "./"
    :param star_type: String containing the spectral type of the star. 'A5','F4',... Assume type V star. It is ignored 
                    of temperature is defined.
    :param star_temperature: Temperature of the star. Overwrite star_type if defined.
    :param PSF_cube: The PSF cube used for the matched filter and the shape metric. The spectrum in PSF_cube matters.
                    If spectrum is not None the spectrum of the PSF is multiplied by spectrum. If spectrum is None the
                    spectrum of PSF_cube is taken as spectral template. In order to remove confusion better giving a
                    flat spectrum to PSF_cube.
                    If nl,ny_PSF,nx_PSF = PSF_cube.shape, nl is the number of wavelength samples and should be 37,
                    ny_PSF and nx_PSF are the spatial dimensions of the PSF_cube.
                    If PSF_cube is None then a default 2d gaussian is used with a width of 1.5 pix.
    :param metrics_only: Boolean. If True the function call will only calculate the metrics and probabilities with 
                        kpp_metrics.calculate_metrics() and will not run the candidate finder with 
                        kpp_detections.candidate_detection().
    :param planet_detection_only: Boolean. If True the function call will only run the candidate finder with 
                        kpp_detections.candidate_detection() and will not calculate the metrics and probabilities with 
                        kpp_metrics.calculate_metrics().
    :param mute: If True prevent printed log outputs.
    :param GOI_list_folder: Folder where are stored the table with the known objects.
    :param overwrite_metric: If True force recalculate and overwrite existing metric fits file. If False check if metric
                            has been calculated and load the file if possible.
    :param overwrite_stat: If True force recalculate and overwrite existing SNR or probability map fits file.
                            If False check if fits file exists before recalculating them.
    :param proba_using_mask_per_pixel: Trigger a per pixel probability calculation. For each pixel it masks the small
                                disk around it before calculating the statistic of the annulus at the same separation.
    :param SNR: If True trigger SNR calculation.
    :param probability: If True trigger probability calculation.
    :param N_threads: Number of threads to be used for the metrics and the probability calculations.
                    If None do it sequentially.
    :param detection_metric: String matching either of the following: "shape", "matchedFilter", "maxShapeMF". It tells 
                            which metric should be used for the detection. The default value is "shape".
    :param noPlots: Prevent the use of matplotlib. No png will be produced.
    :return: 1
            Return the outputs as defined in kpp_metrics.calculate_metrics() and kpp_detections.candidate_detection().
    '''

    # Define the output Foldername
    if spectrum_filename != "":
        spectrum_name = spectrum_filename.split(os.path.sep)
        spectrum_name = spectrum_name[len(spectrum_name)-1].split(".")[0]
    else:
        spectrum_name = "satSpotSpec"

    folderName = spectrum_name+os.path.sep


    # Do the best it can with the spectral information given in inputs.
    if spectrum_filename != "":
        # spectrum_filename is not empty it is assumed to be a valid path.
        if not mute:
            print("Spectrum model: "+spectrum_filename)
        # Interpolate the spectrum of the planet based on the given filename
        wv,planet_sp = spec.get_planet_spectrum(spectrum_filename,filter)

        if sat_spot_spec is not None and (star_type !=  "" or star_temperature is not None):
            # Interpolate a spectrum of the star based on its spectral type/temperature
            wv,star_sp = spec.get_star_spectrum(filter,star_type,star_temperature)
            # Correct the ideal spectrum given in spectrum_filename for atmospheric and instrumental absorption.
            spectrum = (sat_spot_spec/star_sp)*planet_sp
        else:
            # If the sat spot spectrum or the spectral type of the star is missing it won't correct anything and keep
            # the ideal spectrum as it is.
            if not mute:
                print("No star spec or sat spot spec so using sole planet spectrum.")
            spectrum = copy(planet_sp)
    else:
        # If spectrum_filename is an empty string the function takes the sat spot spectrum by default.
        if sat_spot_spec is not None:
            if not mute:
                print("Default sat spot spectrum will be used.")
            spectrum = copy(sat_spot_spec)
        else:
            # If the sat spot spectrum is also not given then it just take the band filter spectrum.
            if not mute:
                print("Using gpi filter "+filter+" spectrum. Could find neither sat spot spectrum nor planet spectrum.")
            wv,spectrum = spec.get_gpi_filter(filter)

    if 0:
        plt.plot(sat_spot_spec)
        plt.plot(spectrum)
        plt.show()

    # Compute the metrics and probability maps
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
                            GOI_list_folder = GOI_list_folder,
                            overwrite_metric = overwrite_metric,
                            overwrite_stat = overwrite_stat,
                            proba_using_mask_per_pixel = proba_using_mask_per_pixel,
                            N_threads = N_threads_metric,
                            noPlots = noPlots)

    # Run the candidate finder on the previously computed metrics
    if not metrics_only:
        if not mute:
            print("Calling candidate_detection() on "+outputDir+folderName)
        candidate_detection(outputDir+folderName,
                            mute = mute,
                            metric = detection_metric,
                            noPlots = noPlots)

    return 1



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
                                      GOI_list_folder = None,
                                      overwrite_metric = False,
                                      overwrite_stat = False,
                                      proba_using_mask_per_pixel = False,
                                      SNR = True,
                                      probability = True,
                                      detection_metric = None,
                                      N_cores = None):
    '''
    Calculate the metrics and probabilities of a given datacube and/or run the candidate finder algorithm for a given
    cube and several spectral templates.


    :param filename: filename of the fits file containing the data cube to be analyzed. The cube should be a GPI cube
                    with 37 slices.
    :param metrics: List of strings giving the metrics to be calculated. E.g. ["shape", "matchedFilter"]
                    The metrics available are "flatCube", "weightedFlatCube", "shape", "matchedFilter".
                    Note by default the flatCube is always saved however its statistic is not computed if "flatcube" is
                    not part of metrics. So If metrics is None basically the function creates a flat cube only.
    :param directory: Directory of the folder containing the klipped cube and the data cubes reduced by the GPI
                    pipeline. It is also where the planet detection output folder will be created by default.
    :param outputDir: Directory in which to save the output folder of the planet detection. Value by default is
                    directory.
    :param spectrum_model: List containing the path to the different spectrum models to be used. The spectrum model has
                        to be generated by Mark Marley.
    :param star_type: String containing the spectral type of the star. 'A5','F4',... Assume type V star. It is ignored
                    of temperature is defined.
    :param star_temperature: Temperature of the star. Overwrite star_type if defined.
    :param user_defined_PSF_cube: Filename of a PSF_cube to be used for the matched filtering. By default the function
                                will try to look for a fits file named like
                                <prefix><data><suffix>-original_radial_PSF_cube.fits". If it doesn't find any it will
                                try to generate it by loading the entire data cube sequence. It is meant to work with
                                the GPI dropbox. If it can't do that it will use a 2d gaussian.
    :param metrics_only: Boolean. If True the function call will only calculate the metrics and probabilities with
                        kpp_metrics.calculate_metrics() and will not run the candidate finder with
                        kpp_detections.candidate_detection().
    :param planet_detection_only: Boolean. If True the function call will only run the candidate finder with
                        kpp_detections.candidate_detection() and will not calculate the metrics and probabilities with
                        kpp_metrics.calculate_metrics().
    :param mute: If True prevent printed log outputs.
    :param threads: Boolean. If true a different process will be created for every single spectrum template for
                    parallelization.
    :param N_cores: Number of cores to be used. If None use all existing cores.
    :param GOI_list_folder: Folder where are stored the table with the known objects.
    :param overwrite_metric: If True force recalculate and overwrite existing metric fits file. If False check if metric
                            has been calculated and load the file if possible.
    :param overwrite_stat: If True force recalculate and overwrite existing SNR or probability map fits file.
                            If False check if fits file exists before recalculating them.
    :param proba_using_mask_per_pixel: Trigger a per pixel probability calculation. For each pixel it masks the small
                                disk around it before calculating the statistic of the annulus at the same separation.
    :param SNR: If True trigger SNR calculation.
    :param probability: If True trigger probability calculation.
    :param detection_metric: String matching either of the following: "shape", "matchedFilter", "maxShapeMF". It tells
                            which metric should be used for the detection. The default value is "shape".
    :return: 1 if successful, None otherwise.
            An output folder is created for filename as:
                planetDetecFolder = outputDir + "/planet_detec_"+prefix+"_KL"+str(N_KL_modes)+"/"
            Create a subfolder per spectrum in planetDetecFolder where the outputs of kpp_metrics.calculate_metrics()
            and kpp_detections.candidate_detection() are saved.
            Save also the outputs of kpp_detections.gather_detections() in planetDetecFolder.
    '''

    # Get the number of KL_modes for this file based on the filename *-KL#-speccube*
    splitted_name = filename.split("-KL")
    splitted_after_KL = splitted_name[1].split("-speccube.")
    N_KL_modes = int(splitted_after_KL[0])

    # Get the prefix of the filename
    splitted_before_KL = splitted_name[0].split(os.path.sep)
    prefix = splitted_before_KL[np.size(splitted_before_KL)-1]

    #grab the headers of the fits file
    hdulist = pyfits.open(filename)
    prihdr = hdulist[0].header

    # Get the date from the filename SYYYYMMDD
    date = prihdr['DATE']
    compact_date = "S"+date.replace("-","")
    #compact_date = prefix.split("-")[1]

    #grab the headers of the fits file
    hdulist = pyfits.open(filename)
    prihdr = hdulist[0].header

    # Retrieve the filter used for the current data cube
    try:
        filter = prihdr['IFSFILT'].split('_')[1]
    except:
        # If the keywords could not be found.
        if not mute:
            print("Couldn't find IFSFILT keyword. Assuming H.")
        filter = "H"

    # Look for the PSF cube dits file
    if user_defined_PSF_cube is not None:
        # If the user gave his own PSF cube
        if not mute:
            print("User defined PSF cube: "+user_defined_PSF_cube)
        filelist_ori_PSFs_cube = glob.glob(user_defined_PSF_cube)
    else:
        # If nothing was specified it tries to look for an existing PSF cube in directory
        #print(directory+os.path.sep+prefix+"-original_radial_PSF_cube.fits")
        filelist_ori_PSFs_cube = glob.glob(directory+os.path.sep+prefix+"-original_radial_PSF_cube.fits")

    if np.size(filelist_ori_PSFs_cube) == 1:
        # Load the PSF cube if a file has been found
        if not mute:
            print("I found a radially averaged PSF. I'll take it.")
        hdulist = pyfits.open(filelist_ori_PSFs_cube[0])
        #print(hdulist[1].data.shape)
        PSF_cube = hdulist[1].data
        sat_spot_spec = np.nanmax(PSF_cube,axis=(1,2))
        # Remove the spectral shape from the psf cube because it is dealt with independently
        for l_id in range(PSF_cube.shape[0]):
            PSF_cube[l_id,:,:] /= sat_spot_spec[l_id]
    elif np.size(filelist_ori_PSFs_cube) == 0:
        # If no PSF cube file has been try let's try to generate one.
        # Indeed if the input directory is an autoreduced folder from the dropbox it also has all the data cube of the
        # observation sequence.
        if not mute:
            print("I didn't find any PSFs file so trying to generate one.")

        ##print(directory+os.path.sep+compact_date+"*_spdc_distorcorr.fits")
        #cube_filename_list = glob.glob(directory+os.path.sep+compact_date+"*_spdc_distorcorr.fits")
        cube_filename_list = []
        file_index = 0
        EOList = False
        while not EOList:
            try:
                curr_spdc_filename = prihdr["FILE_{0}".format(file_index)]
                cube_filename_list.append(directory+os.path.sep+curr_spdc_filename.split(".")[0]+"_spdc_distorcorr.fits")
            except:
                EOList = True
            file_index = file_index +1

        if len(cube_filename_list) == 0:
            # In this case the function couldn't find the data cube of the observation sequence and will therefore
            # ask the next functions to use a default gaussian PSF.
            if not mute:
                print("I can't find the cubes to calculate the PSF from so I will use a default gaussian.")
            sat_spot_spec = None
            PSF_cube = None
        else:
            print(cube_filename_list)
            # Load all the data cube into a dataset object
            dataset = GPI.GPIData(cube_filename_list)
            if not mute:
                print("Calculating the planet PSF from the satellite spots...")
            # generate the PSF cube from the satellite spots
            dataset.generate_psf_cube(20)
            # Save the original PSF calculated from combining the sat spots
            dataset.savedata(directory + os.path.sep + prefix+"-original_PSF_cube.fits", dataset.psfs,
                                      astr_hdr=dataset.wcs[0], filetype="PSF Spec Cube")
            # Calculate and save the rotationally invariant psf (ie smeared out/averaged).
            # Generate a radially averaged PSF cube and save it in directory
            PSF_cube = dataset.get_radial_psf(save = directory + os.path.sep + prefix)
            # Extract the satellite spot spectrum for the PSF cube.
            sat_spot_spec = np.nanmax(PSF_cube,axis=(1,2))
            # Remove the spectral shape from the psf cube because it is dealt with independently
            for l_id in range(PSF_cube.shape[0]):
                PSF_cube[l_id,:,:] /= sat_spot_spec[l_id]
    else:
        #if not mute:
        #    print("I found several files for the PSF cube matching the given filter so I don't know what to do and I quit")
        #return None
        raise Exception("I found several files for the PSF cube matching the given filter so I don't know what to do and I quit")


    if len(spectrum_model) == 1 and not isinstance(spectrum_model,list):
        spectrum_model =[spectrum_model]
    else:
        if not mute:
            print("Iterating over several model spectra")

    # Define the prefix used for the output folder name
    prefix = prefix+"-KL"+str(N_KL_modes)
    if outputDir == '':
        outputDir = directory
    outputDir += os.path.sep+"planet_detec_"+prefix+os.path.sep

    if threads:
        # If the parallelization is on then create as many processes as there are spectrum templates
        N_threads = np.size(spectrum_model)
        # Create non deamonic processes such that they can themselves create child processes
        pool = NoDaemonPool(processes=N_threads)
        #pool = mp.Pool(processes=N_threads)

        # If there are many more cores than there are spectra then it will also parallelize the metric and probability
        # calculation.
        if N_cores is not None:
            N_threads_metric = N_cores/N_threads
        else:
            N_threads_metric = mp.cpu_count()/N_threads
        if N_threads_metric <= 1:
            N_threads_metric = None

        # Run planet dection on the given file with a given spectrum template in a parallelized fashion
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
                                                                           itertools.repeat(GOI_list_folder),
                                                                           itertools.repeat(overwrite_metric),
                                                                           itertools.repeat(overwrite_stat),
                                                                           itertools.repeat(proba_using_mask_per_pixel),
                                                                           itertools.repeat(SNR),
                                                                           itertools.repeat(probability),
                                                                           itertools.repeat(N_threads_metric),
                                                                           itertools.repeat(detection_metric),
                                                                           itertools.repeat(True)))
        pool.close()
    else:
        # Run planet dection on the given file with a given spectrum template sequentially (will be really slow...)
        for spectrum_filename_it in spectrum_model:
            planet_detection_in_dir_per_file_per_spectrum(spectrum_filename_it,
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
                                                          GOI_list_folder = GOI_list_folder,
                                                          overwrite_metric = overwrite_metric,
                                                          overwrite_stat = overwrite_stat,
                                                          proba_using_mask_per_pixel = proba_using_mask_per_pixel,
                                                          SNR = SNR,
                                                          probability = probability,
                                                          N_threads_metric = None,
                                                          detection_metric = detection_metric,
                                                          noPlots = True)

    # If the user didn't ask for the metrics only the next function gather the detection results with the different
    # spectra. It creates outputs in the input directory.
    if not metrics_only:
        if not mute:
            print("Calling gather_detections() on "+outputDir)
        gather_detections(outputDir,filename,PSF_cube, mute = mute,which_metric = detection_metric, GOI_list_folder = GOI_list_folder, noPlots = False)

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
                            threads = 0,
                            metrics_only = False,
                            planet_detection_only = False,
                            mute = True,
                            GOI_list_folder = None,
                            overwrite_metric = False,
                            overwrite_stat = False,
                            proba_using_mask_per_pixel = False,
                            SNR = True,
                            probability = True,
                            detection_metric = None,
                            N_cores = None):
    '''
    Apply the planet detection algorithm for all pyklip reduced cube respecting a filter in a given folder.
    By default the filename filter used is pyklip-*-KL*-speccube.fits.

    :param directory: Directory of the folder containing the klipped cube and the data cubes reduced by the GPI
                    pipeline. It is also where the planet detection output folder will be created by default.
    :param filename_prefix_is: Filename filter to selectively apply the planet detection algorithm.
                            The filter is defined as <filename_prefix_is>-KL<numbasis>-speccube.fits
    :param numbasis: Integer. The number of KL modes used as filename filter. See the input filename_prefix_is.
                    numbasis can be defined without filename_prefix_is.
    :param outputDir: Directory in which to save the output folder of the planet detection. Value by default is
                    directory.
    :param spectrum_model: List containing the path to the different spectrum models to be used. The spectrum model has
                        to be generated by Mark Marley.
    :param star_type: String containing the spectral type of the star. 'A5','F4',... Assume type V star. It is ignored
                    of temperature is defined.
    :param star_temperature: Temperature of the star. Overwrite star_type if defined.
    :param user_defined_PSF_cube: Filename of a PSF_cube to be used for the matched filtering. By default the function
                                will try to look for a fits file named like
                                <prefix><data><suffix>-original_radial_PSF_cube.fits". If it doesn't find any it will
                                try to generate it by loading the entire data cube sequence. It is meant to work with
                                the GPI dropbox. If it can't do that it will use a 2d gaussian.
    :param metrics: List of strings giving the metrics to be calculated. E.g. ["shape", "matchedFilter"]
                    The metrics available are "flatCube", "weightedFlatCube", "shape", "matchedFilter".
                    Note by default the flatCube is always saved however its statistic is not computed if "flatcube" is
                    not part of metrics. So If metrics is None basically the function creates a flat cube only.
    :param threads: Boolean. If true a different process will be created for every single spectrum template for
                    parallelization.
    :param N_cores: Number of cores to be used. If None use all existing cores.
    :param metrics_only: Boolean. If True the function call will only calculate the metrics and probabilities with
                        kpp_metrics.calculate_metrics() and will not run the candidate finder with
                        kpp_detections.candidate_detection().
    :param planet_detection_only: Boolean. If True the function call will only run the candidate finder with
                        kpp_detections.candidate_detection() and will not calculate the metrics and probabilities with
                        kpp_metrics.calculate_metrics().
    :param mute: If True prevent printed log outputs.
    :param GOI_list_folder: Folder where are stored the table with the known objects.
    :param overwrite_metric: If True force recalculate and overwrite existing metric fits file. If False check if metric
                            has been calculated and load the file if possible.
    :param overwrite_stat: If True force recalculate and overwrite existing SNR or probability map fits file.
                            If False check if fits file exists before recalculating them.
    :param proba_using_mask_per_pixel: Trigger a per pixel probability calculation. For each pixel it masks the small
                                disk around it before calculating the statistic of the annulus at the same separation.
    :param SNR: If True trigger SNR calculation.
    :param probability: If True trigger probability calculation.
    :param detection_metric: String matching either of the following: "shape", "matchedFilter", "maxShapeMF". It tells
                            which metric should be used for the detection. The default value is "shape".
    :return: 1 if successful, None otherwise.
        For each file an output folder is created:
            outputDir = directory + "/planet_detec_"+prefix+"_KL"+str(N_KL_modes)+"/"
        For each spectrum template a subfolder is created in the planet detection one.
        The outputs of the detection can be found there.

    '''

    # If a no given number of KL modes has been defined then take all of them.
    if numbasis is not None:
        numbasis = str(numbasis)
    else:
        numbasis = '*'

    # Look for files following the filename filter
    if filename_prefix_is == '':
        # Default filename filter. The pyklip- prefix is the default name of the data cruncher.
        filelist_klipped_cube = glob.glob(directory+os.path.sep+"pyklip-*-KL"+numbasis+"-speccube.fits")
    else:
        # User defined filename filter.
        filelist_klipped_cube = glob.glob(directory+os.path.sep+filename_prefix_is+"-KL"+numbasis+"-speccube.fits")
        #print(directory+"/"+filename_prefix_is+"-KL"+numbasis+"-speccube.fits")


    if len(filelist_klipped_cube) == 0:
        #No file following the filename filter could be found.
        if not mute:
            print("No suitable files found in: "+directory)
        return None
        #raise Exception("No suitable files found in: "+directory)
    else:
        if not mute:
            print(directory+"contains suitable file for planet detection:")
            for f_name in filelist_klipped_cube:
                print(f_name)

        # Run the planet detection for every single file sastisfying the filename filter.
        err_list = []
        for filename in filelist_klipped_cube:
            #if 1:
            try:
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
                                                 GOI_list_folder = GOI_list_folder,
                                                 overwrite_metric = overwrite_metric,
                                                 overwrite_stat = overwrite_stat,
                                                 proba_using_mask_per_pixel = proba_using_mask_per_pixel,
                                                 SNR = SNR,
                                                 probability = probability,
                                                 detection_metric = detection_metric,
                                                 N_cores = N_cores)
            except Exception as myErr:
                err_list.append(myErr)
                if not mute:
                    print("//!\\\\ "+filename+" raised an Error. Candidate finder did not run.")
        return err_list

def planet_detection_campaign(campaign_dir = "."+os.path.sep):
    '''
    Run the candidate finder (or "planet detection" but I feel like it is too optimistic...) on every single stars in
    GPI campaign directory using default parameters. It will go through all the folders in that directory (a folder is
    assumed to correspond to a star) and look for suitable klipped images in the autoreduced folder.
    For each klipped images it will create a corresponding "planet_detec_<filePrefix>" folder where the candidates will
    be displayed. You will also find the metric maps as well as probability maps corresponding to the different spectrum
    template.

    /!\ Still partially hardcoded: GOI_list_folder.xml path. Works only for JB.

    /!\ Function needs cleaning and comments.

    - Suitable files:
    Only the files reduced by the data cruncher with parameters k100a7s4m3 and 20KL modes will be used. The other files
    will be ignored.

    - Spectral templates:
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
        GOI_list_folder = "C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\GOI_list\\"
    elif platform.system() == "Linux":
        GOI_list_folder = "/home/sda/Dropbox (GPI)/SCRATCH/Scratch/JB/GOI_list/"
    else:
        GOI_list_folder = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB/GOI_list/"

    inputDirs = []
    #dirs_to_reduce = os.listdir(campaign_dir) #all
    dirs_to_reduce = ["AU_Mic","bet_Cir","c_Eri","d_Sco","HD_984","HD_28287","HD_88117","HD_95086","HD_100491","HD_118991_A","HD_123058",
                      "HD_133803","HD_155114","HD_157587","HD_161719","HD_164249_A","HR_4669","HR_5663","V343_Nor","V371_Nor","V1358_Ori"]
    for inputDir in ["c_Eri"]:
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
                                        GOI_list_folder = GOI_list_folder,
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

