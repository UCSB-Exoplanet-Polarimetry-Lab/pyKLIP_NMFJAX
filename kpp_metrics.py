__author__ = 'JB'

from scipy.signal import convolve2d

import spectra_management as spec
from kpp_pdf import *
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
    :param cube: Cube from which one wants the shape metric map. PSF_cube should be norm-2 normalized.
                PSF_cube /= np.sqrt(np.sum(PSF_cube**2))
    :param PSF_cube: PSF_cube template used for calculated the shape. If nl,ny_PSF,nx_PSF = PSF_cube.shape, nl is the
                     number of wavelength samples, ny_PSF and nx_PSF are the spatial dimensions of the PSF_cube.
    :param stamp_PSF_mask: 2d mask of size (ny_PSF,nx_PSF) used to mask the central part of a stamp slice. It is used as
                        a type of a high pass filter. Before calculating the shape value of a stamp cube around a given
                        pixel the average value of the surroundings of each slice of that stamp cube will be removed.
                        The pixel used for calculating the average are the one equal to one in the mask.
    :return:
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

def calculate_MF_metric(row_indices,col_indices,cube,PSF_cube,stamp_PSF_mask):
    '''
    Calculate the matched filter metric on the given datacube for the pixels targeted by row_indices and col_indices.
    These lists of indices can basically be given from the numpy.where function following the example:
        import numpy as np
        row_indices,col_indices = np.where(np.finite(np.mean(cube,axis=0)))
    By truncating the given lists in small pieces it is then easy to parallelized.

    The matched filter is a metric allowing one to pull out a known signal from noisy (not correlated) data.
    It is basically a squared dot product (meaning projection) of the template PSF cube with a stamp cube.

    :param row_indices: Row indices list of the pixels where to calculate the metric in cube.
                        Indices should be given from a 2d image.
    :param col_indices: Column indices list of the pixels where to calculate the metric in cube.
                        Indices should be given from a 2d image.
    :param cube: Cube from which one wants the shape metric map. PSF_cube should be norm-2 normalized.
                PSF_cube /= np.sqrt(np.sum(PSF_cube**2))
    :param PSF_cube: PSF_cube template used for calculated the shape. If nl,ny_PSF,nx_PSF = PSF_cube.shape, nl is the
                     number of wavelength samples, ny_PSF and nx_PSF are the spatial dimensions of the PSF_cube.
    :param stamp_PSF_mask: 2d mask of size (ny_PSF,nx_PSF) used to mask the central part of a stamp slice. It is used as
                        a type of a high pass filter. Before calculating the shape value of a stamp cube around a given
                        pixel the average value of the surroundings of each slice of that stamp cube will be removed.
                        The pixel used for calculating the average are the one equal to one in the mask.
    :return:
    '''

    nl,ny_PSF,nx_PSF = PSF_cube.shape

    row_m = np.floor(ny_PSF/2.0)
    row_p = np.ceil(ny_PSF/2.0)
    col_m = np.floor(nx_PSF/2.0)
    col_p = np.ceil(nx_PSF/2.0)

    N_it = row_indices.size
    matchedFilter_map = np.zeros((N_it)) + np.nan
    #stdout.write("\r%d" % 0)
    for id,k,l in zip(range(N_it),row_indices,col_indices):
        #stdout.flush()
        #stdout.write("\r%d" % k)

        stamp_cube = copy(cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)])
        for slice_id in range(nl):
            stamp_cube[slice_id,:,:] -= np.nanmean(stamp_cube[slice_id,:,:]*stamp_PSF_mask)
        ampl = np.nansum(PSF_cube*stamp_cube)
        #matchedFilter_map[id] = np.sign(ampl)*ampl**2
        matchedFilter_map[id] = ampl

    # criterion is here a cosine squared so we take the square root to get something similar to a cosine.
    return matchedFilter_map

def calculate_shapeAndMF_metric_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call calculate_MF_metric() with a tuple of parameters.
    """
    return calculate_shapeAndMF_metric(*params)

def calculate_shapeAndMF_metric(row_indices,col_indices,cube,PSF_cube,stamp_PSF_mask, mute = True):

    nl,ny_PSF,nx_PSF = PSF_cube.shape

    row_m = np.floor(ny_PSF/2.0)
    row_p = np.ceil(ny_PSF/2.0)
    col_m = np.floor(nx_PSF/2.0)
    col_p = np.ceil(nx_PSF/2.0)

    N_it = row_indices.size
    matchedFilter_map = np.zeros((N_it)) + np.nan
    shape_map = np.zeros((N_it)) + np.nan

    #stdout.write("\r%d" % 0)
    for id,k,l in zip(range(N_it),row_indices,col_indices):
        #stdout.flush()
        #stdout.write("\r%d" % k)

        stamp_cube = copy(cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)])
        for slice_id in range(nl):
            stamp_cube[slice_id,:,:] -= np.nanmean(stamp_cube[slice_id,:,:]*stamp_PSF_mask)
        ampl = np.nansum(PSF_cube*stamp_cube)
        matchedFilter_map[id] = np.sign(ampl)*ampl**2
        try:
            shape_map[id] = np.sign(ampl)*ampl**2/np.nansum(stamp_cube**2)
        except:
            shape_map[id] =  np.nan

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
                        filename_no_planets = None,
                        SNR = True,
                        probability = True,
                        GOI_list = None,
                        overwrite_metric = False,
                        overwrite_stat = False,
                        proba_using_mask_per_pixel = False,
                        N_threads = None):
    '''
    Calculate the metrics for future planet detection. The SNR map is a metric for example but JB thinks it's not the best one.

    Inputs:
        filename: Path and name of the fits file to be analyzed.
        metrics: flatCube is calculated by default
            - "weightedFlatCube"
            - "matchedFilter"
            - "shape"
        PSF_cube: User-defined cube PSF. PSF_cube should not have any spectral


        outputDir: Directory where to save the outputs

    Outputs:

    '''
    hdulist = pyfits.open(filename)

    #grab the data and headers
    try:
        cube = hdulist[1].data
        exthdr = hdulist[1].header
        prihdr = hdulist[0].header
    except:
        print("Couldn't read the fits file normally. Try another way.")
        cube = hdulist[0].data
        prihdr = hdulist[0].header


    # Normalization to have reasonable values of the pixel.
    # Indeed some image are in contrast units and the code below doesn't like slices with values around 10**-7.
    cube /= np.nanstd(cube[10,:,:])
    #cube *= 10.0**7


    # Get input cube dimensions
    # Transform a 2D image into a cube with one slice because the code below works for cubes
    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
        #print(cube.shape)
        if nl != 37:
            if not mute:
                print("Spectral dimension of "+filename+" is not correct... quitting")
            return
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        cube_cpy = cube[None,:]
        nl = 1

    if nl != 1:
        # If data cube we can use the spectrum of the planet to more accurately detect planets.
        flat_cube = np.mean(cube,0)

        # Build the PSF.
        PSF_cube_cpy = copy(PSF_cube)
        if PSF_cube_cpy is not None:
            if np.size(np.shape(PSF_cube_cpy)) != 3:
                if not mute:
                    print("Wrong PSF dimensions. Image is 3D.")
                return 0
            # The PSF is user-defined.
            nl, ny_PSF, nx_PSF = PSF_cube_cpy.shape
            tmp_spectrum = np.nanmax(PSF_cube_cpy,axis=(1,2))
        else:
            if not mute:
                print("No specified PSF cube so one is built with simple gaussian PSF. (no spectral widening)")
            # Gaussian PSF with 1.5pixel sigma as nothing was specified by the user.
            # Build the grid for PSF stamp.
            ny_PSF = 8 # should be even
            nx_PSF = 8
            x_PSF_grid, y_PSF_grid = np.meshgrid(np.arange(0,ny_PSF,1)-ny_PSF/2,np.arange(0,nx_PSF,1)-nx_PSF/2)
            # Use a simple 2d gaussian PSF for now. The width is probably not even the right one.
            # I just set it so that "it looks" right.
            PSF = gauss2d(x_PSF_grid, y_PSF_grid,1.0,0.0,0.0,1.5,1.5)
            # Normalize the PSF with a norm 2
            PSF /= np.sqrt(np.sum(PSF**2))

            # Duplicate the PSF to get a PSF cube.
            # Besides apply the spectrum of the planet on that cube before renormalization.
            PSF_cube_cpy = np.tile(PSF,(nl,1,1))

        spectrum_cpy = spectrum
        if spectrum_cpy is not None:
            for k in range(nl):
                PSF_cube_cpy[k,:,:] *= spectrum_cpy[k]
        else:
            if PSF_cube_cpy is not None:
                spectrum_cpy = tmp_spectrum
            else:
                spectrum_cpy = np.ones(nl)

        spectrum_cpy /= np.sqrt(np.nansum(spectrum_cpy**2))
        # normalize PSF with norm 2.
        PSF_cube_cpy /= np.sqrt(np.sum(PSF_cube_cpy**2))

    else: # Assuming 2D image
        flat_cube = cube
        print("Metric calculation Not ready for 2D images to be corrected first")
        return

    try:
        # Retrieve the center of the image from the fits keyword.
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    except:
        # If the keywords could not be found.
        if not mute:
            print("Couldn't find PSFCENTX and PSFCENTY keywords.")
        center = [(nx-1)/2,(ny-1)/2]


    try:
        filter = prihdr['IFSFILT'].split('_')[1]
    except:
        # If the keywords could not be found.
        if not mute:
            print("Couldn't find IFSFILT keyword. Assuming H.")
        filter = "H"

    ##
    # Preliminaries and some sanity checks before saving the metrics maps fits file.
    if outputDir is None:
        outputDir = "."+os.path.sep
    else:
        outputDir = outputDir+os.path.sep

    if folderName is None:
        folderName = os.path.sep+"default_out" +os.path.sep
    else:
        folderName = folderName+os.path.sep

    if not os.path.exists(outputDir+folderName): os.makedirs(outputDir+folderName)

    try:
        # OBJECT: keyword in the primary header with the name of the star.
        prefix = prihdr['OBJECT'].strip().replace (" ", "_")
    except:
        prefix = "UNKNOWN_OBJECT"

    # Smoothing of the image. Remove the median of an arc centered on each pixel.
    # Actually don't do pixel per pixel but consider small boxes.
    # This function has to be cleaned.
    #flat_cube = subtract_radialMed(flat_cube,2,20,center)
    flat_cube_nans = np.where(np.isnan(flat_cube))

    # Calculate metrics only if the pixel is 5 pixel away from a nan
    flat_cube_mask = np.ones((ny,nx))
    flat_cube_mask[flat_cube_nans] = np.nan
    #widen the nans region (disabled)
    #conv_kernel = np.ones((5,5))
    #flat_cube_wider_mask = convolve2d(flat_cube_mask,conv_kernel,mode="same")
    flat_cube_wider_mask = copy(flat_cube_mask)
    # remove the edges if not already nans
    flat_cube_wider_mask[0:ny_PSF/2,:] = np.nan
    flat_cube_wider_mask[:,0:nx_PSF/2] = np.nan
    flat_cube_wider_mask[(ny-ny_PSF/2):ny,:] = np.nan
    flat_cube_wider_mask[:,(nx-nx_PSF/2):nx] = np.nan
    # Exctract the finite pixels from the mask
    flat_cube_wider_notNans = np.where(np.isnan(flat_cube_wider_mask) == 0)

    std_function = radialStdMap
    #std_function = ringSection_based_StdMap

    activate_metric_calc_WFC = True
    activate_metric_calc_MF = True
    activate_metric_calc_shape = True

    activate_proba_calc_FC = True
    activate_proba_calc_WFC = True
    activate_proba_calc_MF = True
    activate_proba_calc_shape = True

    activate_SNR_calc_FC = True
    activate_SNR_calc_WFC = True
    activate_SNR_calc_MF = True
    activate_SNR_calc_shape = True

    if metrics is not None:
        if len(metrics) == 1 and not isinstance(metrics,list):
            metrics = [metrics]

        if "flatCube" in metrics:
            if not overwrite_stat:
                if SNR:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-flatCube_SNR.fits')
                    if len(doesFileExist) != 0:
                        activate_SNR_calc_FC = False
                if probability:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-flatCube_proba.fits')
                    if len(doesFileExist) != 0:
                        activate_proba_calc_FC = False

            if SNR and activate_SNR_calc_FC:
                if not mute:
                    print("Calculating SNR of flatCube for "+filename)
                # Calculate the standard deviation map.
                # the standard deviation is calculated on annuli of width Dr. There is an annulus centered every dr.
                #flat_cube_std = radialStdMap(flat_cube, centroid=center)
                flat_cube_std = std_function(flat_cube, centroid=center)

                # Divide the convolved flat cube by the standard deviation map to get the SNR.
                flat_cube_SNR = flat_cube/flat_cube_std

            if probability and activate_proba_calc_FC:
                if not mute:
                    print("Calculating proba of flatCube for "+filename)
                image = flat_cube
                if GOI_list is not None:
                    image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                else:
                    image_without_planet = image
                #image_without_planet = copy(image)
                center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
                flat_cube_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)


        if "weightedFlatCube" in metrics:
            if not overwrite_metric:
                activate_metric_calc_WFC = False
                try:
                    hdulist = pyfits.open(outputDir+folderName+prefix+'-weightedFlatCube.fits')
                    weightedFlatCube = hdulist[1].data
                    hdulist.close()
                except:
                    activate_metric_calc_WFC = True

            if activate_metric_calc_WFC:
                if not mute:
                    print("Calculating weightedFlatCube for "+filename)
                weightedFlatCube = np.average(cube,axis=0,weights=spectrum_cpy)
                #weightedFlatCube_SNR = weightedFlatCube/radialStdMap(weightedFlatCube,dr,Dr, centroid=center)

            if not overwrite_stat:
                if SNR:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-weightedFlatCube_SNR.fits')
                    if len(doesFileExist) != 0:
                        activate_SNR_calc_WFC = False
                if probability:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-weightedFlatCube_proba.fits')
                    if len(doesFileExist) != 0:
                        activate_proba_calc_WFC = False

            if SNR and activate_SNR_calc_WFC:
                if not mute:
                    print("Calculating SNR of weightedFlatCube for "+filename)
                weightedFlatCube_SNR = weightedFlatCube/std_function(weightedFlatCube,centroid=center)
            if probability and activate_proba_calc_WFC:
                if not mute:
                    print("Calculating proba of weightedFlatCube for "+filename)
                image = weightedFlatCube
                if GOI_list is not None:
                    image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                else:
                    image_without_planet = image
                #image_without_planet = copy(image)
                center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
                weightedFlatCube_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)

        if "matchedFilter" in metrics and "shape" not in metrics:
            if not overwrite_metric:
                activate_metric_calc_MF = False
                try:
                    hdulist = pyfits.open(outputDir+folderName+prefix+'-matchedFilter.fits')
                    matchedFilter_map = hdulist[1].data
                    hdulist.close()
                except:
                    activate_metric_calc_MF = True

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
                    pool = NoDaemonPool(processes=N_threads)
                    #pool = mp.Pool(processes=N_threads)

                    ## cut images in N_threads part
                    # get the first and last index of each chunck
                    N_pix = flat_cube_wider_notNans[0].size
                    chunk_size = N_pix/N_threads
                    N_chunks = N_pix/chunk_size

                    # Get the chunks
                    chunks_row_indices = []
                    chunks_col_indices = []
                    for k in range(N_chunks-1):
                        chunks_row_indices.append(flat_cube_wider_notNans[0][(k*chunk_size):((k+1)*chunk_size)])
                        chunks_col_indices.append(flat_cube_wider_notNans[1][(k*chunk_size):((k+1)*chunk_size)])
                    chunks_row_indices.append(flat_cube_wider_notNans[0][((N_chunks-1)*chunk_size):N_pix])
                    chunks_col_indices.append(flat_cube_wider_notNans[1][((N_chunks-1)*chunk_size):N_pix])

                    outputs_list = pool.map(calculate_MF_metric_star, itertools.izip(chunks_row_indices,
                                                                                       chunks_col_indices,
                                                                                       itertools.repeat(cube),
                                                                                       itertools.repeat(PSF_cube_cpy),
                                                                                       itertools.repeat(stamp_PSF_mask)))

                    for row_indices,col_indices,out in zip(chunks_row_indices,chunks_col_indices,outputs_list):
                        matchedFilter_map[(row_indices,col_indices)] = out
                    pool.close()
                else:
                    matchedFilter_map[flat_cube_wider_notNans] = calculate_MF_metric(flat_cube_wider_notNans[0],flat_cube_wider_notNans[1],cube,PSF_cube_cpy,stamp_PSF_mask)


                IWA,OWA,inner_mask,outer_mask = get_occ(matchedFilter_map, centroid = center)
                conv_kernel = np.ones((10,10))
                wider_mask = convolve2d(outer_mask,conv_kernel,mode="same")
                matchedFilter_map[np.where(np.isnan(wider_mask))] = np.nan

            if not overwrite_stat:
                if SNR:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-matchedFilter_SNR.fits')
                    if len(doesFileExist) != 0:
                        activate_SNR_calc_MF = False
                if probability:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-matchedFilter_proba.fits')
                    if len(doesFileExist) != 0:
                        activate_proba_calc_MF = False

            if SNR and activate_SNR_calc_MF:
                if not mute:
                    print("Calculating SNR of matchedFilter (no shape) for "+filename)
                matchedFilter_SNR_map = matchedFilter_map/std_function(matchedFilter_map, centroid=center)
            if probability and activate_proba_calc_MF:
                if not mute:
                    print("Calculating proba of matchedFilter (no shape) for "+filename)
                image = matchedFilter_map
                if GOI_list is not None:
                    image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                else:
                    image_without_planet = image
                #image_without_planet = copy(image)
                center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
                matchedFilter_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)

        if "shape" in metrics and "matchedFilter" not in metrics:
            if not overwrite_metric:
                activate_metric_calc_shape = False
                try:
                    hdulist = pyfits.open(outputDir+folderName+prefix+'-shape.fits')
                    shape_map = hdulist[1].data
                    hdulist.close()
                except:
                    activate_metric_calc_shape = True

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
                    pool = NoDaemonPool(processes=N_threads)
                    #pool = mp.Pool(processes=N_threads)

                    ## cut images in N_threads part
                    # get the first and last index of each chunck
                    N_pix = flat_cube_wider_notNans[0].size
                    chunk_size = N_pix/N_threads
                    N_chunks = N_pix/chunk_size

                    # Get the chunks
                    chunks_row_indices = []
                    chunks_col_indices = []
                    for k in range(N_chunks-1):
                        chunks_row_indices.append(flat_cube_wider_notNans[0][(k*chunk_size):((k+1)*chunk_size)])
                        chunks_col_indices.append(flat_cube_wider_notNans[1][(k*chunk_size):((k+1)*chunk_size)])
                    chunks_row_indices.append(flat_cube_wider_notNans[0][((N_chunks-1)*chunk_size):N_pix])
                    chunks_col_indices.append(flat_cube_wider_notNans[1][((N_chunks-1)*chunk_size):N_pix])

                    outputs_list = pool.map(calculate_shape_metric_star, itertools.izip(chunks_row_indices,
                                                                                       chunks_col_indices,
                                                                                       itertools.repeat(cube),
                                                                                       itertools.repeat(PSF_cube_cpy),
                                                                                       itertools.repeat(stamp_PSF_mask)))

                    for row_indices,col_indices,out in zip(chunks_row_indices,chunks_col_indices,outputs_list):
                        shape_map[(row_indices,col_indices)] = out
                    pool.close()
                else:
                    shape_map[flat_cube_wider_notNans] = calculate_shape_metric(flat_cube_wider_notNans[0],flat_cube_wider_notNans[1],cube,PSF_cube_cpy,stamp_PSF_mask)


                IWA,OWA,inner_mask,outer_mask = get_occ(shape_map, centroid = center)
                conv_kernel = np.ones((10,10))
                wider_mask = convolve2d(outer_mask,conv_kernel,mode="same")
                shape_map[np.where(np.isnan(wider_mask))] = np.nan

            if not overwrite_stat:
                if SNR:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-shape_SNR.fits')
                    if len(doesFileExist) != 0:
                        activate_SNR_calc_shape = False
                if probability:
                    doesFileExist = glob.glob(outputDir+folderName+prefix+'-shape_proba.fits')
                    if len(doesFileExist) != 0:
                        activate_proba_calc_shape = False

            if SNR and activate_SNR_calc_shape:
                if not mute:
                    print("Calculating SNR of shape (no matchedFilter) for "+filename)
                shape_SNR_map = shape_map/std_function(shape_map, centroid=center)
            if probability and activate_proba_calc_shape:
                if not mute:
                    print("Calculating proba of shape (no matchedFilter) for "+filename)
                image = shape_map
                if GOI_list is not None:
                    image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                else:
                    image_without_planet = image
                #image_without_planet = copy(image)
                center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
                shape_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)


        if "matchedFilter" in metrics and "shape" in metrics:
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
                    pool = NoDaemonPool(processes=N_threads)
                    #pool = mp.Pool(processes=N_threads)

                    ## cut images in N_threads part
                    # get the first and last index of each chunck
                    N_pix = flat_cube_wider_notNans[0].size
                    chunk_size = N_pix/N_threads
                    N_chunks = N_pix/chunk_size

                    # Get the chunks
                    chunks_row_indices = []
                    chunks_col_indices = []
                    for k in range(N_chunks-1):
                        chunks_row_indices.append(flat_cube_wider_notNans[0][(k*chunk_size):((k+1)*chunk_size)])
                        chunks_col_indices.append(flat_cube_wider_notNans[1][(k*chunk_size):((k+1)*chunk_size)])
                    chunks_row_indices.append(flat_cube_wider_notNans[0][((N_chunks-1)*chunk_size):N_pix])
                    chunks_col_indices.append(flat_cube_wider_notNans[1][((N_chunks-1)*chunk_size):N_pix])

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
                    shape_map[flat_cube_wider_notNans],matchedFilter_map[flat_cube_wider_notNans] = \
                        calculate_shapeAndMF_metric(flat_cube_wider_notNans[0],flat_cube_wider_notNans[1],cube,PSF_cube_cpy,stamp_PSF_mask)


                IWA,OWA,inner_mask,outer_mask = get_occ(shape_map, centroid = center)
                conv_kernel = np.ones((10,10))
                wider_mask = convolve2d(outer_mask,conv_kernel,mode="same")
                shape_map[np.where(np.isnan(wider_mask))] = np.nan
                matchedFilter_map[np.where(np.isnan(wider_mask))] = np.nan

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

            if SNR and activate_SNR_calc_MF and activate_SNR_calc_shape:
                if not mute:
                    print("Calculating SNR of shape and matchedFilter for "+filename)
                #shape_SNR_map = shape_map/radialStdMap(shape_map,dr,Dr, centroid=center)
                shape_SNR_map = shape_map/std_function(shape_map,centroid=center)
                #matchedFilter_SNR_map = matchedFilter_map/radialStdMap(matchedFilter_map,dr,Dr, centroid=center)
                matchedFilter_SNR_map = matchedFilter_map/std_function(matchedFilter_map,centroid=center)
            if probability and activate_proba_calc_MF and activate_proba_calc_shape:
                if not mute:
                    print("Calculating proba of shape and matchedFilter for "+filename)
                center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
                image = shape_map
                if GOI_list is not None:
                    image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                else:
                    image_without_planet = image
                #image_without_planet = copy(image)
                shape_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)
                image = matchedFilter_map
                if GOI_list is not None:
                    image_without_planet = mask_known_objects(image,prihdr,GOI_list, mask_radius = 7)
                else:
                    image_without_planet = image
                #image_without_planet = copy(image)
                matchedFilter_proba_map = get_image_probability_map(image,image_without_planet,use_mask_per_pixel = proba_using_mask_per_pixel,centroid = center,N_threads = N_threads)



    ## ortho_criterion is actually the sine squared between the two vectors
    ## ortho_criterion_map = 1 - criterion_map
    ## ratio_shape_SNR = 10
    ## criterion_map = np.minimum(ratio_shape_SNR*shape_map,flat_cube_SNR)



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

    # Save a plot of the spectrum used
    spec_sampling = spec.get_gpi_wavelength_sampling(filter)
    plt.close(1)
    plt.figure(1)
    plt.plot(spec_sampling,spectrum_cpy,"rx-",markersize = 7, linewidth = 2)
    plt.title("Template spectrum use in this folder")
    plt.xlabel("Wavelength (mum)")
    plt.ylabel("Norm-2 Normalized")
    plt.savefig(outputDir+folderName+'template_spectrum.png', bbox_inches='tight')
    plt.close(1)

    hdulist2[1].data = [spec_sampling,spectrum_cpy]
    hdulist2.writeto(outputDir+folderName+'template_spectrum.fits', clobber=True)

    return 1
# END calculate_metrics() DEFINITION
