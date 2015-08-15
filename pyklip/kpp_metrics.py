__author__ = 'JB'

from scipy.signal import convolve2d

import spectra_management as spec
from pyklip.kpp_pdf import *
from kpp_std import *


def calculate_shape_metric_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call calculate_shape_metric() with a tuple of parameters.
    """
    return calculate_shape_metric(*params)

def calculate_shape_metric(row_indices,col_indices,cube,PSF_cube,stamp_PSF_mask, mute = True):
    '''
    Calculate the shape metric on the given datacube for the pixels targeted by row_indices and col_indices.
    These lists of indices can basically be given from the numpy.where function following the example:
        import numpy as np
        row_indices,col_indices = np.where(np.finite(np.mean(cube,axis=0)))
    By truncating the given lists in small pieces it is then easy to parallelized.

    The shape metric is a normalized matched filter from which the flux component has been removed.
    It is a sort of pattern recognition.
    The shape value is a dot product normalized by the norm of the vectors

    :param row_indices: Row indices list of the pixels where to calculate the metric in cube.
                        Indices should be given from a 2d image.
    :param col_indices: Column indices list of the pixels where to calculate the metric in cube.
                        Indices should be given from a 2d image.
    :param cube: Cube from which one wants the metric map. PSF_cube should be norm-2 normalized.
                PSF_cube /= np.sqrt(np.sum(PSF_cube**2))
    :param PSF_cube: PSF_cube template used for calculated the metric. If nl,ny_PSF,nx_PSF = PSF_cube.shape, nl is the
                     number of wavelength samples, ny_PSF and nx_PSF are the spatial dimensions of the PSF_cube.
    :param stamp_PSF_mask: 2d mask of size (ny_PSF,nx_PSF) used to mask the central part of a stamp slice. It is used as
                        a type of a high pass filter. Before calculating the metric value of a stamp cube around a given
                        pixel the average value of the surroundings of each slice of that stamp cube will be removed.
                        The pixel used for calculating the average are the one equal to one in the mask.
    :param mute: If True prevent printed log outputs.
    :return: Vector of length row_indices.size with the value of the metric for the corresponding pixels.
    '''

    # Shape of the PSF cube
    nl,ny_PSF,nx_PSF = PSF_cube.shape

    # Number of rows and columns to add around a given pixel in order to extract a stamp.
    row_m = np.floor(ny_PSF/2.0)    # row_minus
    row_p = np.ceil(ny_PSF/2.0)     # row_plus
    col_m = np.floor(nx_PSF/2.0)    # col_minus
    col_p = np.ceil(nx_PSF/2.0)     # col_plus

    # Number of pixels on which the metric has to be computed
    N_it = row_indices.size
    # Define an shape vector full of nans
    shape_map = np.zeros((N_it,)) + np.nan
    # Loop over all pixels (row_indices[id],col_indices[id])
    for id,k,l in zip(range(N_it),row_indices,col_indices):
        if not mute:
            # Print the progress of the function
            stdout.write("\r{0}/{1}".format(id,N_it))
            stdout.flush()

        # Extract stamp cube around the current pixel from the whoel cube
        stamp_cube = copy(cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)])
        # Remove average value of the surrounding pixels in each slice of the stamp cube
        for slice_id in range(nl):
            stamp_cube[slice_id,:,:] -= np.nanmean(stamp_cube[slice_id,:,:]*stamp_PSF_mask)
        # Dot product of the PSF with stamp cube.
        ampl = np.nansum(PSF_cube*stamp_cube)
        # Normalize the dot product square by the squared norm-2 of the stamp cube.
        # Because we keep the sign shape value is then found in [-1.,1.]
        try:
            shape_map[id] = np.sign(ampl)*ampl**2/np.nansum(stamp_cube**2)
        except:
            # In case ones divide by zero...
            shape_map[id] =  np.nan

    # The shape value here can be seen as a cosine square as it is a normalized squared dot product.
    # Taking the square root to make it a simple cosine.
    return np.sign(shape_map)*np.sqrt(abs(shape_map))


def calculate_MF_metric_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call calculate_MF_metric() with a tuple of parameters.
    """
    return calculate_MF_metric(*params)

def calculate_MF_metric(row_indices,col_indices,cube,PSF_cube,stamp_PSF_mask, mute = True):
    '''
    Calculate the matched filter metric on the given datacube for the pixels targeted by row_indices and col_indices.
    These lists of indices can basically be given from the numpy.where function following the example:
        import numpy as np
        row_indices,col_indices = np.where(np.finite(np.mean(cube,axis=0)))
    By truncating the given lists in small pieces it is then easy to parallelized.

    The matched filter is a metric allowing one to pull out a known signal from noisy (not correlated) data.
    It is basically a dot product (meaning projection) of the template PSF cube with a stamp cube.

    :param row_indices: Row indices list of the pixels where to calculate the metric in cube.
                        Indices should be given from a 2d image.
    :param col_indices: Column indices list of the pixels where to calculate the metric in cube.
                        Indices should be given from a 2d image.
    :param cube: Cube from which one wants the metric map. PSF_cube should be norm-2 normalized.
                PSF_cube /= np.sqrt(np.sum(PSF_cube**2))
    :param PSF_cube: PSF_cube template used for calculated the metric. If nl,ny_PSF,nx_PSF = PSF_cube.shape, nl is the
                     number of wavelength samples, ny_PSF and nx_PSF are the spatial dimensions of the PSF_cube.
    :param stamp_PSF_mask: 2d mask of size (ny_PSF,nx_PSF) used to mask the central part of a stamp slice. It is used as
                        a type of a high pass filter. Before calculating the metric value of a stamp cube around a given
                        pixel the average value of the surroundings of each slice of that stamp cube will be removed.
                        The pixel used for calculating the average are the one equal to one in the mask.
    :param mute: If True prevent printed log outputs.
    :return: Vector of length row_indices.size with the value of the metric for the corresponding pixels.
    '''

    # Shape of the PSF cube
    nl,ny_PSF,nx_PSF = PSF_cube.shape

    # Number of rows and columns to add around a given pixel in order to extract a stamp.
    row_m = np.floor(ny_PSF/2.0)    # row_minus
    row_p = np.ceil(ny_PSF/2.0)     # row_plus
    col_m = np.floor(nx_PSF/2.0)    # col_minus
    col_p = np.ceil(nx_PSF/2.0)     # col_plus

    # Number of pixels on which the metric has to be computed
    N_it = row_indices.size
    # Define an matched filter vector full of nans
    matchedFilter_map = np.zeros((N_it)) + np.nan
    # Loop over all pixels (row_indices[id],col_indices[id])
    for id,k,l in zip(range(N_it),row_indices,col_indices):
        if not mute:
            # Print the progress of the function
            stdout.write("\r{0}/{1}".format(id,N_it))
            stdout.flush()

        # Extract stamp cube around the current pixel from the whoel cube
        stamp_cube = copy(cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)])
        # Remove average value of the surrounding pixels in each slice of the stamp cube
        for slice_id in range(nl):
            stamp_cube[slice_id,:,:] -= np.nanmean(stamp_cube[slice_id,:,:]*stamp_PSF_mask)
        # Dot product of the PSF with stamp cube which is the matched filter value.
        matchedFilter_map[id] = np.nansum(PSF_cube*stamp_cube)

    return matchedFilter_map

def calculate_shapeAndMF_metric_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call calculate_MF_metric() with a tuple of parameters.
    """
    return calculate_shapeAndMF_metric(*params)

def calculate_shapeAndMF_metric(row_indices,col_indices,cube,PSF_cube,stamp_PSF_mask, mute = True):
    '''
    Calculate the shape and the matched filter metrics on the given datacube for the pixels targeted by row_indices and
    col_indices.
    These lists of indices can basically be given from the numpy.where function following the example:
        import numpy as np
        row_indices,col_indices = np.where(np.finite(np.mean(cube,axis=0)))
    By truncating the given lists in small pieces it is then easy to parallelized.
    Both metrics are calculated in one single loop which should make it slightly faster than running
    calculate_MF_metric() and calculate_shape_metric() seperatly.

    The matched filter is a metric allowing one to pull out a known signal from noisy (not correlated) data.
    It is basically a dot product (meaning projection) of the template PSF cube with a stamp cube.

    The shape metric is a normalized matched filter from which the flux component has been removed.
    It is a sort of pattern recognition.
    The shape value is a dot product normalized by the norm of the vectors

    :param row_indices: Row indices list of the pixels where to calculate the metric in cube.
                        Indices should be given from a 2d image.
    :param col_indices: Column indices list of the pixels where to calculate the metric in cube.
                        Indices should be given from a 2d image.
    :param cube: Cube from which one wants the metric map. PSF_cube should be norm-2 normalized.
                PSF_cube /= np.sqrt(np.sum(PSF_cube**2))
    :param PSF_cube: PSF_cube template used for calculated the metric. If nl,ny_PSF,nx_PSF = PSF_cube.shape, nl is the
                     number of wavelength samples, ny_PSF and nx_PSF are the spatial dimensions of the PSF_cube.
    :param stamp_PSF_mask: 2d mask of size (ny_PSF,nx_PSF) used to mask the central part of a stamp slice. It is used as
                        a type of a high pass filter. Before calculating the metric value of a stamp cube around a given
                        pixel the average value of the surroundings of each slice of that stamp cube will be removed.
                        The pixel used for calculating the average are the one equal to one in the mask.
    :param mute: If True prevent printed log outputs.
    :return: Vector of length row_indices.size with the value of the metric for the corresponding pixels.
    '''

    # Shape of the PSF cube
    nl,ny_PSF,nx_PSF = PSF_cube.shape

    # Number of rows and columns to add around a given pixel in order to extract a stamp.
    row_m = np.floor(ny_PSF/2.0)    # row_minus
    row_p = np.ceil(ny_PSF/2.0)     # row_plus
    col_m = np.floor(nx_PSF/2.0)    # col_minus
    col_p = np.ceil(nx_PSF/2.0)     # col_plus

    # Number of pixels on which the metric has to be computed
    N_it = row_indices.size
    # Define a matched filter vector and a shape vector both full of nans
    matchedFilter_map = np.zeros((N_it)) + np.nan
    shape_map = np.zeros((N_it)) + np.nan
    # Loop over all pixels (row_indices[id],col_indices[id])
    for id,k,l in zip(range(N_it),row_indices,col_indices):
        if not mute:
            # Print the progress of the function
            stdout.write("\r{0}/{1}".format(id,N_it))
            stdout.flush()

        # Extract stamp cube around the current pixel from the whoel cube
        stamp_cube = copy(cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)])
        # Remove average value of the surrounding pixels in each slice of the stamp cube
        for slice_id in range(nl):
            stamp_cube[slice_id,:,:] -= np.nanmean(stamp_cube[slice_id,:,:]*stamp_PSF_mask)
        # Dot product of the PSF with stamp cube.
        ampl = np.nansum(PSF_cube*stamp_cube)
        # The dot product is actually also the matched filter
        matchedFilter_map[id] = ampl
        # Normalize the dot product square by the squared norm-2 of the stamp cube.
        # Because we keep the sign shape value is then found in [-1.,1.]
        try:
            shape_map[id] = np.sign(ampl)*ampl**2/np.nansum(stamp_cube**2)
        except:
            # In case ones divide by zero...
            shape_map[id] =  np.nan

    # The shape value here can be seen as a cosine square as it is a normalized squared dot product.
    shape_map = np.sign(shape_map)*np.sqrt(abs(shape_map))

    # criterion is here a cosine squared so we take the square root to get something similar to a cosine.
    return shape_map,matchedFilter_map

def calculate_metrics(filename,
                        metrics = None,
                        PSF_cube = None,
                        outputDir = None,
                        folderName = None,
                        spectrum = None,
                        mute = False,
                        SNR = True,
                        probability = True,
                        GOI_list = None,
                        overwrite_metric = False,
                        overwrite_stat = False,
                        proba_using_mask_per_pixel = False,
                        N_threads = None,
                        noPlots = False):
    '''
    Calculate some metrics on a given data cube as well as the corresponding SNR and probability maps.

    Metrics:
        - flatCube
            flat_cube = np.mean(cube,0)
        - weightedFlatCube
            Collapsed cube with slices being weighted according to spectrum as defined by the keyword OR the spectrum of
            the PSF_cube. If PSF_cube has a spectrum then the spectrum used for weighted flat cube and for shape or
            matched filter will be different.
        - shape
            The shape metric is a normalized matched filter from which the flux component has been removed.
            It is a sort of pattern recognition.
            The shape value is a dot product normalized by the norm of the vectors
            Calculated using calculate_shape_metric().
        - matchedFilter
            The matched filter is a metric allowing one to pull out a known signal from noisy (not correlated) data.
            It is basically a dot product (meaning projection) of the template PSF cube with a stamp cube.
            Calculated using calculate_MF_metric().

    Statistic:
        statistic maps are calculated for every metric in metrics.
        - SNR
            Calculated using radialStdMap(). Should be changed.
        - probability
            Calculated using get_image_probability_map() with a per pixel mask if proba_using_mask_per_pixel is True.

    :param filename: filename of the fits file containing the data cube to be analyzed. The cube should be a GPI cube
                    with 37 slices.
    :param metrics: List of strings giving the metrics to be calculated. E.g. ["shape", "matchedFilter"]
                    The metrics available are "flatCube", "weightedFlatCube", "shape", "matchedFilter".
                    Note by default the flatCube is always saved however its statistic is not computed if "flatcube" is
                    not part of metrics. So If metrics is None basically the function creates a flat cube only.
    :param PSF_cube: The PSF cube used for the matched filter and the shape metric. The spectrum in PSF_cube matters.
                    If spectrum is not None the spectrum of the PSF is multiplied by spectrum. If spectrum is None the
                    spectrum of PSF_cube is taken as spectral template. In order to remove confusion better giving a
                    flat spectrum to PSF_cube.
                    If nl,ny_PSF,nx_PSF = PSF_cube.shape, nl is the number of wavelength samples and should be 37,
                    ny_PSF and nx_PSF are the spatial dimensions of the PSF_cube.
                    If PSF_cube is None then a default 2d gaussian is used with a width of 1.5 pix.
    :param outputDir: Directory where to create the folder containing the outputs. default directory is "./"
    :param folderName: Name of the folder containing the outputs. It will be located in outputDir. Default folder name 
                    is "default_out".
                    The convention is to have one folder for a given data cube with a given spectral template.
                    Therefore don't save the outputs of different calls to calculate_metrics() in the same folder.
    :param spectrum: Spectrum used for the metrics using a spectrum: weightedFlatCube, shape, matchedFilter.
                    It is combined with PSF_cube if PSF_cube is not None.
    :param mute: If True prevent printed log outputs.
    :param SNR: If True trigger SNR calculation.
    :param probability: If True trigger probability calculation.
    :param GOI_list: XML file with the list of known object in the campaign. These object will be mask from the images
                    before calculating the probability.
    :param overwrite_metric: If True force recalculate and overwrite existing metric fits file. If False check if metric
                            has been calculated and load the file if possible.
    :param overwrite_stat: If True force recalculate and overwrite existing SNR or probability map fits file.
                            If False check if fits file exists before recalculating them.
    :param proba_using_mask_per_pixel: Trigger a per pixel probability calculation. For each pixel it masks the small
                                disk around it before calculating the statistic of the annulus at the same separation.
    :param N_threads: Number of threads to be used for the metrics and the probability calculations.
                    If None do it sequentially.
    :return: A fits file for each of the things calculated.
            Metrics are saved as outputDir+folderName+prefix+'-<myMetric>.fits'
            Statistic maps are saved as outputDir+folderName+prefix+'-<myMetric>_<stat>.fits'
            Beside spectrum is saved as fits and png for information. The fits file contains the wavelength sampling
            (first row) as well as the spectrum (second row).
            The function itself returns 1 if it successfully executed and None otherwise.
    '''
    # Load filename. It contains the data cube.
    hdulist = pyfits.open(filename)

    # grab the data and headers
    try:
        cube = hdulist[1].data
        exthdr = hdulist[1].header
        prihdr = hdulist[0].header
    except:
        # This except was used for datacube not following GPI headers convention.
        # /!\ This is however not supported at the moment
        print("Couldn't read the fits file normally. Try another way.")
        cube = hdulist[0].data
        prihdr = hdulist[0].header
        return None


    # Normalization to have reasonable values of the pixel.
    # Indeed some image are in contrast units and the code below doesn't like slices with values around 10**-7.
    # /!\ It was some standard deviation function which failed because of the small values.
    # /!\ This line should be removed and the problem solved but it hasn't been done yet.
    # Besides one should check if it creates problem for the probability (JB doesn't thing so)
    cube /= np.nanstd(cube[10,:,:])
    #cube *= 10.0**7


    # Get input cube dimensions
    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
        # Checking that the cube has the 37 spectral slices of a normal GPI cube.
        if nl != 37:
            if not mute:
                print("Returning None. Spectral dimension of "+filename+" is not correct...")
            return None
    else:
        print("Returning None. fits file was not a cube...")
        return None

    # Collapse the cube using a simple mean.
    flat_cube = np.mean(cube,0)

    # Build the PSF cube with the right spectrum
    # make a copy of the PSF_cube because we will modify it with spectrum.
    PSF_cube_cpy = copy(PSF_cube)
    # If PSF_cube is actually defined
    if PSF_cube is not None:
        # Check PSF_cube shape
        if np.size(np.shape(PSF_cube_cpy)) != 3:
            if not mute:
                print("Returning None. Wrong PSF dimensions. Image Should be 3D.")
            return None
        # The PSF is user-defined.
        nl, ny_PSF, nx_PSF = PSF_cube_cpy.shape
    else:
        # If PSF_cube is not defined then a default 2d gaussian is used with a width of 1.5pixel.
        if not mute:
            print("No specified PSF cube so one is built with simple gaussian PSF. (no spectral widening)")
        # Build the grid for PSF stamp.
        ny_PSF = 8 # should be even
        nx_PSF = 8 # should be even
        x_PSF_grid, y_PSF_grid = np.meshgrid(np.arange(0,ny_PSF,1)-ny_PSF/2,np.arange(0,nx_PSF,1)-nx_PSF/2)
        # Use a simple 2d gaussian PSF for now. The width is probably not even the right one.
        # I just set it so that "it looks" right.
        PSF = gauss2d(x_PSF_grid, y_PSF_grid,1.0,0.0,0.0,1.5,1.5)
        # Normalize the PSF with a norm 2
        PSF /= np.sqrt(np.sum(PSF**2))

        # Duplicate the PSF to get a PSF cube.
        PSF_cube_cpy = np.tile(PSF,(nl,1,1))
    # Extract the spectrum of the PSF cube.
    tmp_spectrum = np.nanmax(PSF_cube_cpy,axis=(1,2))

    # Apply the template spectrum on the PSF cube before norm-2 renormalization.
    spectrum_cpy = spectrum
    # If spectrum is not None. Multiply it to the PSF cube
    if spectrum_cpy is not None:
        for k in range(nl):
            PSF_cube_cpy[k,:,:] *= spectrum_cpy[k]
    else:
        # If spectrum is none then use the spectrum of the PSF cube. Might be a flat spectrum...
        spectrum_cpy = tmp_spectrum

    # normalize spectrum with norm 2.
    spectrum_cpy /= np.sqrt(np.nansum(spectrum_cpy**2))
    # normalize PSF with norm 2.
    PSF_cube_cpy /= np.sqrt(np.sum(PSF_cube_cpy**2))

    # Get center of the image (star position)
    try:
        # Retrieve the center of the image from the fits headers.
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    except:
        # If the keywords could not be found the center is defined as the middle of the image
        if not mute:
            print("Couldn't find PSFCENTX and PSFCENTY keywords.")
        center = [(nx-1)/2,(ny-1)/2]

    # Get the filter of the image
    try:
        # Retrieve the filter used from the fits headers.
        filter = prihdr['IFSFILT'].split('_')[1]
    except:
        # If the keywords could not be found assume that the filter is H...
        if not mute:
            print("Couldn't find IFSFILT keyword. Assuming H.")
        filter = "H"

    # If outputDir is None define it as the project directory.
    if outputDir is None:
        outputDir = "."+os.path.sep
    else:
        outputDir = outputDir+os.path.sep

    # Define a default folderName is the one given is None.
    if folderName is None:
        folderName = os.path.sep+"default_out" +os.path.sep
    else:
        folderName = folderName+os.path.sep

    if not os.path.exists(outputDir+folderName): os.makedirs(outputDir+folderName)

    # Get current star name
    try:
        # OBJECT: keyword in the primary header with the name of the star.
        prefix = prihdr['OBJECT'].strip().replace (" ", "_")
    except:
        # If the object name could nto be found cal lit unknown_object
        prefix = "UNKNOWN_OBJECT"


    # Get the nans pixels of the flat_cube. We won't bother trying to calculate metrics for those.
    flat_cube_nans = np.where(np.isnan(flat_cube))

    # Remove the very edges of the image. We can't calculate a proper projection of an image stamp onto the PSF if we 
    # are too close from the edges of the array.
    flat_cube_mask = np.ones((ny,nx))
    flat_cube_mask[flat_cube_nans] = np.nan
    flat_cube_noEdges_mask = copy(flat_cube_mask)
    # remove the edges if not already nans
    flat_cube_noEdges_mask[0:ny_PSF/2,:] = np.nan
    flat_cube_noEdges_mask[:,0:nx_PSF/2] = np.nan
    flat_cube_noEdges_mask[(ny-ny_PSF/2):ny,:] = np.nan
    flat_cube_noEdges_mask[:,(nx-nx_PSF/2):nx] = np.nan
    # Get the pixel coordinates corresponding to non nan pixel and not too close from the edges of the array.
    flat_cube_noNans_noEdges = np.where(np.isnan(flat_cube_noEdges_mask) == 0)

    # Define the function used to calculate the standard deviation
    # It probably shouldn't be hard coded but JB is not using the SNR maps anymore anyway...
    std_function = radialStdMap
    #std_function = ringSection_based_StdMap

    # Define booleans triggering a metric calculation.
    # Their values will depend on overwrite_metric and if the fits file already exist.
    activate_metric_calc_WFC = True
    activate_metric_calc_MF = True
    activate_metric_calc_shape = True

    # Define booleans triggering a probability map calculation.
    # Their values will depend on overwrite_stat and if the fits file already exist.
    activate_proba_calc_FC = True
    activate_proba_calc_WFC = True
    activate_proba_calc_MF = True
    activate_proba_calc_shape = True

    # Define booleans triggering a SNR map calculation.
    # Their values will depend on overwrite_stat and if the fits file already exist.
    activate_SNR_calc_FC = True
    activate_SNR_calc_WFC = True
    activate_SNR_calc_MF = True
    activate_SNR_calc_shape = True

    if metrics is not None:
        if len(metrics) == 1 and not isinstance(metrics,list):
            metrics = [metrics]

        # Calculate a flat cube stat only if the user asked for it through "metrics".
        if "flatCube" in metrics:
            # If overwrite_stat is False check if the fits files already exists. If not make sure it is calculated.
            if not overwrite_stat:
                if SNR:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-flatCube_SNR.fits')
                    if len(doesFileExist) != 0:
                        activate_SNR_calc_FC = False
                if probability:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-flatCube_proba.fits')
                    if len(doesFileExist) != 0:
                        activate_proba_calc_FC = False

            # Calculate the SNR of the flat cube only if required.
            if SNR and activate_SNR_calc_FC:
                if not mute:
                    print("Calculating SNR of flatCube for "+filename)
                # Calculate the standard deviation map.
                flat_cube_std = std_function(flat_cube, centroid=center)
                # Divide the flat cube by the standard deviation map to get the SNR.
                flat_cube_SNR = flat_cube/flat_cube_std
            # Calculate the probability map of the flat cube only if required.
            if probability and activate_proba_calc_FC:
                if not mute:
                    print("Calculating proba of flatCube for "+filename)
                image = flat_cube
                # If GOI_list is not None. Mask the known objects from the image that will be used for calculating the
                # PDF. This masked image is given separately to the probability calculation function.
                if GOI_list is not None:
                    image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                else:
                    image_without_planet = image
                #image_without_planet = copy(image)
                # Get the probability map
                flat_cube_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)

        # Calculate a weighted flat cube only if the user asked for it through "metrics".
        if "weightedFlatCube" in metrics:
            # If overwrite_metric is False check if the fits file already exists. If not make sure it is calculated.
            if not overwrite_metric:
                activate_metric_calc_WFC = False
                try:
                    hdulist = pyfits.open(outputDir+folderName+prefix+'-weightedFlatCube.fits')
                    weightedFlatCube = hdulist[1].data
                    hdulist.close()
                except:
                    activate_metric_calc_WFC = True

            # Build the weighted flat cube if required.
            if activate_metric_calc_WFC:
                if not mute:
                    print("Calculating weightedFlatCube for "+filename)
                weightedFlatCube = np.average(cube,axis=0,weights=spectrum_cpy)
                #weightedFlatCube_SNR = weightedFlatCube/radialStdMap(weightedFlatCube,dr,Dr, centroid=center)

            # If overwrite_stat is False check if the fits files already exists. If not make sure it is calculated.
            if not overwrite_stat:
                if SNR:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-weightedFlatCube_SNR.fits')
                    if len(doesFileExist) != 0:
                        activate_SNR_calc_WFC = False
                if probability:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-weightedFlatCube_proba.fits')
                    if len(doesFileExist) != 0:
                        activate_proba_calc_WFC = False

            # Calculate the SNR of the weighted flat cube only if required.
            if SNR and activate_SNR_calc_WFC:
                if not mute:
                    print("Calculating SNR of weightedFlatCube for "+filename)
                weightedFlatCube_SNR = weightedFlatCube/std_function(weightedFlatCube,centroid=center)
            # Calculate the probability map of the weighted flat cube only if required.
            if probability and activate_proba_calc_WFC:
                if not mute:
                    print("Calculating proba of weightedFlatCube for "+filename)
                image = weightedFlatCube
                # If GOI_list is not None. Mask the known objects from the image that will be used for calculating the
                # PDF. This masked image is given separately to the probability calculation function.
                if GOI_list is not None:
                    image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                else:
                    image_without_planet = image
                #image_without_planet = copy(image)
                # Get the probability map
                weightedFlatCube_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)

        # Calculate the matched filter metric only if the user asked for it through "metrics".
        if "matchedFilter" in metrics and "shape" not in metrics:
            # If overwrite_metric is False check if the fits file already exists. If not make sure it is calculated.
            if not overwrite_metric:
                activate_metric_calc_MF = False
                try:
                    hdulist = pyfits.open(outputDir+folderName+prefix+'-matchedFilter.fits')
                    matchedFilter_map = hdulist[1].data
                    hdulist.close()
                except:
                    activate_metric_calc_MF = True

            # Build the matched filter metric if required.
            if activate_metric_calc_MF:
                if not mute:
                    print("Calculating matchedFilter (no shape) for "+filename)
                matchedFilter_map = np.ones((ny,nx)) + np.nan

                # Calculate the criterion map.
                # For each pixel calculate the dot product of a stamp around it with the PSF.
                # We use the PSF cube to consider also the spectrum of the planet we are looking for.
                if not mute:
                    print("Calculate the criterion map. It is done pixel per pixel so it might take a while...")
                stamp_PSF_x_grid, stamp_PSF_y_grid = np.meshgrid(np.arange(0,nx_PSF,1)-nx_PSF/2,np.arange(0,ny_PSF,1)-ny_PSF/2)
                stamp_PSF_mask = np.ones((nl,ny_PSF,nx_PSF))
                r_PSF_stamp = abs((stamp_PSF_x_grid) +(stamp_PSF_y_grid)*1j)
                #r_PSF_stamp = np.tile(r_PSF_stamp,(nl,1,1))
                stamp_PSF_mask[np.where(r_PSF_stamp < 2.5)] = np.nan

                if N_threads is not None:
                    #pool = NoDaemonPool(processes=N_threads)
                    pool = mp.Pool(processes=N_threads)

                    ## cut images in N_threads part
                    # get the first and last index of each chunck
                    N_pix = flat_cube_noNans_noEdges[0].size
                    chunk_size = N_pix/N_threads
                    N_chunks = N_pix/chunk_size

                    # Get the chunks
                    chunks_row_indices = []
                    chunks_col_indices = []
                    for k in range(N_chunks-1):
                        chunks_row_indices.append(flat_cube_noNans_noEdges[0][(k*chunk_size):((k+1)*chunk_size)])
                        chunks_col_indices.append(flat_cube_noNans_noEdges[1][(k*chunk_size):((k+1)*chunk_size)])
                    chunks_row_indices.append(flat_cube_noNans_noEdges[0][((N_chunks-1)*chunk_size):N_pix])
                    chunks_col_indices.append(flat_cube_noNans_noEdges[1][((N_chunks-1)*chunk_size):N_pix])

                    outputs_list = pool.map(calculate_MF_metric_star, itertools.izip(chunks_row_indices,
                                                                                       chunks_col_indices,
                                                                                       itertools.repeat(cube),
                                                                                       itertools.repeat(PSF_cube_cpy),
                                                                                       itertools.repeat(stamp_PSF_mask)))

                    for row_indices,col_indices,out in zip(chunks_row_indices,chunks_col_indices,outputs_list):
                        matchedFilter_map[(row_indices,col_indices)] = out
                    pool.close()
                else:
                    matchedFilter_map[flat_cube_noNans_noEdges] = calculate_MF_metric(flat_cube_noNans_noEdges[0],flat_cube_noNans_noEdges[1],cube,PSF_cube_cpy,stamp_PSF_mask)


                # Mask out a band of 10 pixels around the edges of the finite pixels of the image.
                IWA,OWA,inner_mask,outer_mask = get_occ(matchedFilter_map, centroid = center)
                conv_kernel = np.ones((10,10))
                wider_mask = convolve2d(outer_mask,conv_kernel,mode="same")
                matchedFilter_map[np.where(np.isnan(wider_mask))] = np.nan

            # If overwrite_stat is False check if the fits files already exists. If not make sure it is calculated.
            if not overwrite_stat:
                if SNR:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-matchedFilter_SNR.fits')
                    if len(doesFileExist) != 0:
                        activate_SNR_calc_MF = False
                if probability:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-matchedFilter_proba.fits')
                    if len(doesFileExist) != 0:
                        activate_proba_calc_MF = False

            # Calculate the SNR of the matched filter only if required.
            if SNR and activate_SNR_calc_MF:
                if not mute:
                    print("Calculating SNR of matchedFilter (no shape) for "+filename)
                matchedFilter_SNR_map = matchedFilter_map/std_function(matchedFilter_map, centroid=center)
            # Calculate the probability map of the matched filter only if required.
            if probability and activate_proba_calc_MF:
                if not mute:
                    print("Calculating proba of matchedFilter (no shape) for "+filename)
                image = matchedFilter_map
                # If GOI_list is not None. Mask the known objects from the image that will be used for calculating the
                # PDF. This masked image is given separately to the probability calculation function.
                if GOI_list is not None:
                    image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                else:
                    image_without_planet = image
                #image_without_planet = copy(image)
                # Get the probability map
                matchedFilter_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)

        # Calculate the shape metric only if the user asked for it through "metrics".
        if "shape" in metrics and "matchedFilter" not in metrics:
            # If overwrite_metric is False check if the fits file already exists. If not make sure it is calculated.
            if not overwrite_metric:
                activate_metric_calc_shape = False
                try:
                    hdulist = pyfits.open(outputDir+folderName+prefix+'-shape.fits')
                    shape_map = hdulist[1].data
                    hdulist.close()
                except:
                    activate_metric_calc_shape = True

            # Build the shape metric if required.
            if activate_metric_calc_shape:
                if not mute:
                    print("Calculating shape (no matchedFilter) for "+filename)
                shape_map = -np.ones((ny,nx)) + np.nan

                # Calculate the criterion map.
                # For each pixel calculate the dot product of a stamp around it with the PSF.
                # We use the PSF cube to consider also the spectrum of the planet we are looking for.
                if not mute:
                    print("Calculate the criterion map. It is done pixel per pixel so it might take a while...")
                stamp_PSF_x_grid, stamp_PSF_y_grid = np.meshgrid(np.arange(0,nx_PSF,1)-nx_PSF/2,np.arange(0,ny_PSF,1)-ny_PSF/2)
                stamp_PSF_mask = np.ones((nl,ny_PSF,nx_PSF))
                r_PSF_stamp = abs((stamp_PSF_x_grid) +(stamp_PSF_y_grid)*1j)
                #r_PSF_stamp = np.tile(r_PSF_stamp,(nl,1,1))
                stamp_PSF_mask[np.where(r_PSF_stamp < 2.5)] = np.nan

                if N_threads is not None:
                    #pool = NoDaemonPool(processes=N_threads)
                    pool = mp.Pool(processes=N_threads)

                    ## cut images in N_threads part
                    # get the first and last index of each chunck
                    N_pix = flat_cube_noNans_noEdges[0].size
                    chunk_size = N_pix/N_threads
                    N_chunks = N_pix/chunk_size

                    # Get the chunks
                    chunks_row_indices = []
                    chunks_col_indices = []
                    for k in range(N_chunks-1):
                        chunks_row_indices.append(flat_cube_noNans_noEdges[0][(k*chunk_size):((k+1)*chunk_size)])
                        chunks_col_indices.append(flat_cube_noNans_noEdges[1][(k*chunk_size):((k+1)*chunk_size)])
                    chunks_row_indices.append(flat_cube_noNans_noEdges[0][((N_chunks-1)*chunk_size):N_pix])
                    chunks_col_indices.append(flat_cube_noNans_noEdges[1][((N_chunks-1)*chunk_size):N_pix])

                    outputs_list = pool.map(calculate_shape_metric_star, itertools.izip(chunks_row_indices,
                                                                                       chunks_col_indices,
                                                                                       itertools.repeat(cube),
                                                                                       itertools.repeat(PSF_cube_cpy),
                                                                                       itertools.repeat(stamp_PSF_mask)))

                    for row_indices,col_indices,out in zip(chunks_row_indices,chunks_col_indices,outputs_list):
                        shape_map[(row_indices,col_indices)] = out
                    pool.close()
                else:
                    shape_map[flat_cube_noNans_noEdges] = calculate_shape_metric(flat_cube_noNans_noEdges[0],flat_cube_noNans_noEdges[1],cube,PSF_cube_cpy,stamp_PSF_mask)


                # Mask out a band of 10 pixels around the edges of the finite pixels of the image.
                IWA,OWA,inner_mask,outer_mask = get_occ(shape_map, centroid = center)
                conv_kernel = np.ones((10,10))
                wider_mask = convolve2d(outer_mask,conv_kernel,mode="same")
                shape_map[np.where(np.isnan(wider_mask))] = np.nan

            # If overwrite_stat is False check if the fits files already exists. If not make sure it is calculated.
            if not overwrite_stat:
                if SNR:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-shape_SNR.fits')
                    if len(doesFileExist) != 0:
                        activate_SNR_calc_shape = False
                if probability:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-shape_proba.fits')
                    if len(doesFileExist) != 0:
                        activate_proba_calc_shape = False

            # Calculate the SNR of the shape metric only if required.
            if SNR and activate_SNR_calc_shape:
                if not mute:
                    print("Calculating SNR of shape (no matchedFilter) for "+filename)
                shape_SNR_map = shape_map/std_function(shape_map, centroid=center)
            # Calculate the probability map of the shape metric only if required.
            if probability and activate_proba_calc_shape:
                if not mute:
                    print("Calculating proba of shape (no matchedFilter) for "+filename)
                image = shape_map
                # If GOI_list is not None. Mask the known objects from the image that will be used for calculating the
                # PDF. This masked image is given separately to the probability calculation function.
                if GOI_list is not None:
                    image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                else:
                    image_without_planet = image
                #image_without_planet = copy(image)
                # Get the probability map
                shape_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)


        # Calculate the shape metric and the matched filter metric only if the user asked for it through "metrics".
        # Calculate them in a single loop hopefully to speed up the computation.
        if "matchedFilter" in metrics and "shape" in metrics:
            # If overwrite_metric is False check if the fits file already exists. If not make sure it is calculated.
            # Note that the algorithm is not smart enough to realize if only one of the two metrics actually exists for
            # this case.
            if not overwrite_metric:
                activate_metric_calc_MF = False
                activate_metric_calc_shape = False
                try:
                    hdulist = pyfits.open(outputDir+folderName+prefix+'-matchedFilter.fits')
                    matchedFilter_map = hdulist[1].data
                    hdulist.close()
                    hdulist = pyfits.open(outputDir+folderName+prefix+'-shape.fits')
                    shape_map = hdulist[1].data
                    hdulist.close()
                except:
                    activate_metric_calc_MF = True
                    activate_metric_calc_shape = True

            # Build the the shape and the matched filter if required.
            if activate_metric_calc_MF and activate_metric_calc_shape:
                if not mute:
                    print("Calculating shape and matchedFilter for "+filename)
                matchedFilter_map = np.ones((ny,nx)) + np.nan
                shape_map = -np.ones((ny,nx)) + np.nan

                # Calculate the criterion map.
                # For each pixel calculate the dot product of a stamp around it with the PSF.
                # We use the PSF cube to consider also the spectrum of the planet we are looking for.
                if not mute:
                    print("Calculate the criterion map. It is done pixel per pixel so it might take a while...")
                stamp_PSF_x_grid, stamp_PSF_y_grid = np.meshgrid(np.arange(0,nx_PSF,1)-nx_PSF/2,np.arange(0,ny_PSF,1)-ny_PSF/2)
                stamp_PSF_mask = np.ones((nl,ny_PSF,nx_PSF))
                r_PSF_stamp = abs((stamp_PSF_x_grid) +(stamp_PSF_y_grid)*1j)
                #r_PSF_stamp = np.tile(r_PSF_stamp,(nl,1,1))
                stamp_PSF_mask[np.where(r_PSF_stamp < 2.5)] = np.nan

                if N_threads is not None:
                    #pool = NoDaemonPool(processes=N_threads)
                    pool = mp.Pool(processes=N_threads)

                    ## cut images in N_threads part
                    # get the first and last index of each chunck
                    N_pix = flat_cube_noNans_noEdges[0].size
                    chunk_size = N_pix/N_threads
                    N_chunks = N_pix/chunk_size

                    # Get the chunks
                    chunks_row_indices = []
                    chunks_col_indices = []
                    for k in range(N_chunks-1):
                        chunks_row_indices.append(flat_cube_noNans_noEdges[0][(k*chunk_size):((k+1)*chunk_size)])
                        chunks_col_indices.append(flat_cube_noNans_noEdges[1][(k*chunk_size):((k+1)*chunk_size)])
                    chunks_row_indices.append(flat_cube_noNans_noEdges[0][((N_chunks-1)*chunk_size):N_pix])
                    chunks_col_indices.append(flat_cube_noNans_noEdges[1][((N_chunks-1)*chunk_size):N_pix])

                    outputs_list = pool.map(calculate_shapeAndMF_metric_star, itertools.izip(chunks_row_indices,
                                                                                       chunks_col_indices,
                                                                                       itertools.repeat(cube),
                                                                                       itertools.repeat(PSF_cube_cpy),
                                                                                       itertools.repeat(stamp_PSF_mask)))

                    for row_indices,col_indices,out in zip(chunks_row_indices,chunks_col_indices,outputs_list):
                        shape_map[(row_indices,col_indices)] = out[0]
                        matchedFilter_map[(row_indices,col_indices)] = out[1]
                    pool.close()
                else:
                    shape_map[flat_cube_noNans_noEdges],matchedFilter_map[flat_cube_noNans_noEdges] = \
                        calculate_shapeAndMF_metric(flat_cube_noNans_noEdges[0],flat_cube_noNans_noEdges[1],cube,PSF_cube_cpy,stamp_PSF_mask)


                # Mask out a band of 10 pixels around the edges of the finite pixels of the image.
                IWA,OWA,inner_mask,outer_mask = get_occ(shape_map, centroid = center)
                conv_kernel = np.ones((10,10))
                wider_mask = convolve2d(outer_mask,conv_kernel,mode="same")
                shape_map[np.where(np.isnan(wider_mask))] = np.nan
                matchedFilter_map[np.where(np.isnan(wider_mask))] = np.nan

            # If overwrite_stat is False check if the fits files already exists. If not make sure it is calculated.
            if not overwrite_stat:
                if SNR:
                    doesFileExist1 = glob.glob(outputDir+folderName+prefix+'-shape_SNR.fits')
                    doesFileExist2 = glob.glob(outputDir+folderName+prefix+'-matchedFilter_SNR.fits')
                    if len(doesFileExist1) != 0 and len(doesFileExist2) != 0:
                        activate_SNR_calc_MF = False
                        activate_SNR_calc_shape = False
                if probability:
                    doesFileExist1 = glob.glob(outputDir+folderName+prefix+'-shape_proba.fits')
                    doesFileExist2 = glob.glob(outputDir+folderName+prefix+'-matchedFilter_proba.fits')
                    if len(doesFileExist1) != 0 and len(doesFileExist2) != 0:
                        activate_proba_calc_MF = False
                        activate_proba_calc_shape = False

            # Calculate the SNR of the shape and matched filter metrics only if required.
            if SNR and activate_SNR_calc_MF and activate_SNR_calc_shape:
                if not mute:
                    print("Calculating SNR of shape and matchedFilter for "+filename)
                shape_SNR_map = shape_map/std_function(shape_map,centroid=center)
                matchedFilter_SNR_map = matchedFilter_map/std_function(matchedFilter_map,centroid=center)
            # Calculate the probability map of the shape and matched filter metrics  only if required.
            if probability and activate_proba_calc_MF and activate_proba_calc_shape:
                if not mute:
                    print("Calculating proba of shape and matchedFilter for "+filename)
                image = shape_map
                # If GOI_list is not None. Mask the known objects from the image that will be used for calculating the
                # PDF. This masked image is given separately to the probability calculation function.
                if GOI_list is not None:
                    image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                else:
                    image_without_planet = image
                #image_without_planet = copy(image)
                # Get the probability map
                shape_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)
                image = matchedFilter_map
                # If GOI_list is not None. Mask the known objects from the image that will be used for calculating the
                # PDF. This masked image is given separately to the probability calculation function.
                if GOI_list is not None:
                    image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                else:
                    image_without_planet = image
                #image_without_planet = copy(image)
                # Get the probability map
                matchedFilter_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)


    # The core of the function is here over.
    # Now saving the results in fits files...

    if not mute:
        print("Saving metrics maps as: "+outputDir+folderName+prefix+'-<myMetric>.fits')
        print("Saving stat maps as: "+outputDir+folderName+prefix+'-<myMetric>_<stat>.fits')

    hdulist2 = pyfits.HDUList()
    #try:
    hdulist2.append(pyfits.PrimaryHDU(header=prihdr))
    hdulist2.append(pyfits.ImageHDU(header=exthdr, data=flat_cube, name="Sci"))
    #hdulist2[1].data = flat_cube
    hdulist2.writeto(outputDir+folderName+prefix+'-flatCube.fits', clobber=True)
    if metrics is not None:
        if "flatCube" in metrics:
            if SNR and activate_SNR_calc_FC:
                hdulist2[1].data = flat_cube_SNR
                hdulist2.writeto(outputDir+folderName+prefix+'-flatCube_SNR.fits', clobber=True)
            if probability and activate_proba_calc_FC:
                hdulist2[1].data = flat_cube_proba_map
                hdulist2.writeto(outputDir+folderName+prefix+'-flatCube_proba.fits', clobber=True)
        if "weightedFlatCube" in metrics:
            if activate_metric_calc_WFC:
                hdulist2[1].data = weightedFlatCube
                hdulist2.writeto(outputDir+folderName+prefix+'-weightedFlatCube.fits', clobber=True)
            if SNR and activate_SNR_calc_WFC:
                hdulist2[1].data = weightedFlatCube_SNR
                hdulist2.writeto(outputDir+folderName+prefix+'-weightedFlatCube_SNR.fits', clobber=True)
            if probability and activate_proba_calc_WFC:
                hdulist2[1].data = weightedFlatCube_proba_map
                hdulist2.writeto(outputDir+folderName+prefix+'-weightedFlatCube_proba.fits', clobber=True)
        if "matchedFilter" in metrics:
            if activate_metric_calc_MF:
                hdulist2[1].data = matchedFilter_map
                hdulist2.writeto(outputDir+folderName+prefix+'-matchedFilter.fits', clobber=True)
            if SNR and activate_SNR_calc_MF:
                hdulist2[1].data = matchedFilter_SNR_map
                hdulist2.writeto(outputDir+folderName+prefix+'-matchedFilter_SNR.fits', clobber=True)
            if probability and activate_proba_calc_MF:
                hdulist2[1].data = matchedFilter_proba_map
                hdulist2.writeto(outputDir+folderName+prefix+'-matchedFilter_proba.fits', clobber=True)
        if "shape" in metrics:
            if activate_metric_calc_shape:
                hdulist2[1].data = shape_map
                hdulist2.writeto(outputDir+folderName+prefix+'-shape.fits', clobber=True)
            if SNR and activate_SNR_calc_shape:
                hdulist2[1].data = shape_SNR_map
                hdulist2.writeto(outputDir+folderName+prefix+'-shape_SNR.fits', clobber=True)
            if probability and activate_proba_calc_shape:
                hdulist2[1].data = shape_proba_map
                hdulist2.writeto(outputDir+folderName+prefix+'-shape_proba.fits', clobber=True)
    # The following commented code was used when analyzing fits files that didn't follow GPI headers conventions.
    """
    except:
        print("No exthdr so only using primary to save data...")
        hdulist2.append(pyfits.PrimaryHDU(header=prihdr))
        hdulist2[0].data = flat_cube
        hdulist2.writeto(outputDir+folderName+prefix+'-flatCube.fits', clobber=True)
        hdulist2[0].data = flat_cube_SNR
        hdulist2.writeto(outputDir+folderName+prefix+'-flatCubeSNR.fits', clobber=True)
        if metrics is not None:
            if "weightedFlatCube" in metrics:
                hdulist2[0].data = weightedFlatCube
                hdulist2.writeto(outputDir+folderName+prefix+'-weightedFlatCube.fits', clobber=True)
                hdulist2[0].data = weightedFlatCube_SNR
                hdulist2.writeto(outputDir+folderName+prefix+'-weightedFlatCube_SNR.fits', clobber=True)
            if "matchedFilter" in metrics:
                hdulist2[0].data = matchedFilter_map
                hdulist2.writeto(outputDir+folderName+prefix+'-matchedFilter.fits', clobber=True)
                hdulist2[0].data = matchedFilter_SNR_map
                hdulist2.writeto(outputDir+folderName+prefix+'-matchedFilter_SNR.fits', clobber=True)
            if "shape" in metrics:
                hdulist2[0].data = shape_map
                hdulist2.writeto(outputDir+folderName+prefix+'-shape.fits', clobber=True)
                hdulist2[0].data = shape_SNR_map
                hdulist2.writeto(outputDir+folderName+prefix+'-shape_SNR.fits', clobber=True)
    """
    hdulist2.close()

    # Save a plot of the spectral template used as PNG
    spec_sampling = spec.get_gpi_wavelength_sampling(filter)
    if not noPlots:
        plt.close(1)
        plt.figure(1)
        plt.plot(spec_sampling,spectrum_cpy,"rx-",markersize = 7, linewidth = 2)
        plt.title("Template spectrum use in this folder")
        plt.xlabel("Wavelength (mum)")
        plt.ylabel("Norm-2 Normalized")
        plt.savefig(outputDir+folderName+'template_spectrum.png', bbox_inches='tight')
        plt.close(1)

    # Save a fits file of the spectral template used including the wavelengths.
    hdulist2[1].data = [spec_sampling,spectrum_cpy]
    hdulist2.writeto(outputDir+folderName+'template_spectrum.fits', clobber=True)

    # Successful return
    return 1
# END calculate_metrics() DEFINITION
