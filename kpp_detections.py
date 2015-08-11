__author__ = 'JB'

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


def candidate_detection(metrics_foldername,
                        mute = False,
                        metric = None):
    '''
    Run the candidate detection algorithm based on pre-calculated metric probability maps.

    What candidate_detection() needs is a folder in which the outputs of the function calculate_metrics() for a given
    data cube are saved. There should be only one *-shape_proba.fits file and one *-matchedFilter_proba.fits file.
    The convention is to have one folder for a given data cube with a given spectral template.

    Candidate_detection needs both files (*-shape_proba.fits and *-matchedFilter_proba.fits) in order to work even if
    only one of them is actually used. this could probably be corrected in the future.

    :param metrics_foldername: Name of the folder containing the *-shape_proba.fits file and the
                            *-matchedFilter_proba.fits.
    :param mute: If True prevent printed log outputs.
    :param metric: String matching either of the following: "shape", "matchedFilter", "maxShapeMF". It tells which
                metric should be used for the detection. The default value is "shape".
    :return: If successful the function returns 1 otherwise it returns None.
            If everything went well the function will have created the following files in the folder metrics_foldername:
            - <star_name>-detectionIm_all-<metric>.png:
                An image of the metric probablity map with an arrow on all the local maxima that have been tested.
                There is a limit of max_attempts = 60 (hard-coded in the function) local maxima. But there is also a
                lower limit on the acceptable probability value (2.5) so there might not be 60 iterations.
                You might think this output is useless and you were right. I just use it when I see that an object is
                not detected to check what probability it has (because it usually is still a local maximum but just
                fainter than the threshold).
            - <star_name>-detectionIm_candidates-<metric>.png
                Select only the points with a probability greater that 4. These are the candidates.
            - <star_name>-detections-<metric>.xml
                Store the position and probability for all local maxima and detections in a xml file.
                The tree is composed of root, then <star_name>, then candidates and all. The childs of either
                "candidates" or "all" are all "localMax" and their properties are id, max_val_criter, col_id, row_id,
                x_max_pos and y_max_pos.
    '''
    # Look for files in metrics_foldername that have the name matching *-shape_proba.fits and *-matchedFilter_proba.fits
    shape_filename_list = glob.glob(metrics_foldername+os.path.sep+"*-shape_proba.fits")
    matchedFilter_filename_list = glob.glob(metrics_foldername+os.path.sep+"*-matchedFilter_proba.fits")
    # Abort if the shape proba file couldn't be found.
    if len(shape_filename_list) == 0:
        if not mute:
            print("Couldn't find shape_proba map in "+metrics_foldername)
            return None
    else:
        # Read the shape proba file if it was found.
        hdulist_shape = pyfits.open(shape_filename_list[0])
    # Abort if the matched filter proba file couldn't be found.
    if len(matchedFilter_filename_list) == 0:
        if not mute:
            print("Couldn't find matchedFilter_SNR map in "+metrics_foldername)
            return None
    else:
        # Read the matched filter proba file if it was found.
        hdulist_matchedFilter = pyfits.open(matchedFilter_filename_list[0])

    # Try to grab the data and headers
    try:
        shape_map = hdulist_shape[1].data
        matchedFilter_map = hdulist_matchedFilter[1].data
        exthdr = hdulist_shape[1].header
        prihdr = hdulist_shape[0].header
    except:
        # This except was used for datacube not following GPI headers convention.
        # /!\ This is however not supported at the moment
        print("Return None. Couldn't read the fits file normally.")
        shape_map = hdulist_shape[0].data
        prihdr = hdulist_shape[0].header
        return None

    # Shape of the images
    if np.size(shape_map.shape) == 2:
        ny,nx = shape_map.shape

    # Get center of the image (star position)
    try:
        # Retrieve the center of the image from the fits headers.
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    except:
        # If the keywords could not be found the center is defined as the middle of the image
        if not mute:
            print("Couldn't find PSFCENTX and PSFCENTY keywords.")
        center = [(nx-1)/2,(ny-1)/2]


    try:
        # OBJECT: keyword in the primary header with the name of the star.
        star_name = prihdr['OBJECT'].strip().replace (" ", "_")
    except:
        # If the object name could nto be found cal lit unknown_object
        star_name = "UNKNOWN_OBJECT"

    # Default metric used is shape
    if metric is None:
        metric = "shape"

    # If the user defined a metric to use. Define the criterion_map accordingly as well as the threshold.
    if metric == "shape":
        criterion_map = shape_map
        threshold = 4
    elif metric == "matchedFilter":
        criterion_map = matchedFilter_map
        threshold = 4
    elif metric == "maxShapeMF":
        criterion_map = np.max([shape_map,matchedFilter_map],axis=0)
        threshold = 4
    else:
        return None

    # Make a copy of the criterion map because it will be modified in the following.
    # Local maxima are indeed masked out when checked
    criterion_map_cpy = copy(criterion_map)

    # Build as grids of x,y coordinates.
    # The center is in the middle of the array and the unit is the pixel.
    # If the size of the array is even 2n x 2n the center coordinate in the array is [n,n].
    x_grid, y_grid = np.meshgrid(np.arange(0,nx,1)-center[0],np.arange(0,ny,1)-center[1])


    # Definition of the different masks used in the following.
    stamp_nrow = 13
    stamp_ncol = 13
    # Mask to remove the spots already checked in criterion_map.
    stamp_x_grid, stamp_y_grid = np.meshgrid(np.arange(0,stamp_nrow,1)-6,np.arange(0,stamp_ncol,1)-6)
    stamp_mask = np.ones((stamp_nrow,stamp_ncol))
    r_stamp = abs((stamp_x_grid) +(stamp_y_grid)*1j)
    stamp_mask[np.where(r_stamp < 4.0)] = np.nan

    # Mask out a band of 10 pixels around the edges of the finite pixels of the image.
    # DISABLED: this is done in the metric calculation itself
    if 0:
        IWA,OWA,inner_mask,outer_mask = get_occ(criterion_map, centroid = center)
        conv_kernel = np.ones((10,10))
        flat_cube_wider_mask = convolve2d(outer_mask,conv_kernel,mode="same")
        criterion_map[np.where(np.isnan(flat_cube_wider_mask))] = np.nan


    # Number of rows and columns to add around a given pixel in order to extract a stamp.
    row_m = np.floor(stamp_nrow/2.0)    # row_minus
    row_p = np.ceil(stamp_nrow/2.0)     # row_plus
    col_m = np.floor(stamp_ncol/2.0)    # col_minus
    col_p = np.ceil(stamp_ncol/2.0)     # col_plus

    # Define the tree for the xml file
    root = ET.Element("root")
    star_elt = ET.SubElement(root, star_name)
    # Element where the candidates will be saved
    candidates_elt = ET.SubElement(star_elt, "candidates")
    # Element where all local maxima will be saved
    all_elt = ET.SubElement(star_elt, "all")

    # Count the number of valid detected candidates.
    N_candidates = 0.0

    # Maximum number of iterations on local maxima.
    max_attempts = 60
    ## START WHILE LOOP.
    # Each iteration looks at one local maximum in the criterion map.
    # Note: This loop is a bit dum. The same thing could be done more easily in a different way but it is an heritage
    # from previous test and I am lazy to change it.
    k = 0
    max_val_criter = np.nanmax(criterion_map)
    while max_val_criter >= 2.5 and k <= max_attempts:
        k += 1
        # Find the maximum value in the current criterion map. At each iteration the previous maximum is masked out.
        max_val_criter = np.nanmax(criterion_map)
        # Locate the maximum by retrieving its coordinates
        max_ind = np.where( criterion_map == max_val_criter )
        row_id,col_id = max_ind[0][0],max_ind[1][0]
        x_max_pos, y_max_pos = x_grid[row_id,col_id],y_grid[row_id,col_id]

        # Mask the spot around the maximum we just found.
        criterion_map[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] *= stamp_mask

        # If the criterion value is bigger than the threshold then it is considered as a candidate.
        potential_planet = max_val_criter > threshold

        # Store the current local maximum information in the xml
        ET.SubElement(all_elt,"localMax",
                      id = str(k),
                      max_val_criter = str(max_val_criter),
                      x_max_pos= str(x_max_pos),
                      y_max_pos= str(y_max_pos),
                      row_id= str(row_id),
                      col_id= str(col_id))

        if potential_planet:
            # If the current maximum has a value grater than the threshold its information is also stored in the xml as
            # a candidate.
            ET.SubElement(candidates_elt, "localMax",
                          id = str(k),
                          max_val_criter = str(max_val_criter),
                          x_max_pos= str(x_max_pos),
                          y_max_pos= str(y_max_pos),
                          row_id= str(row_id),
                          col_id= str(col_id))

            # Increment the number of detected candidates
            N_candidates += 1
    ## END WHILE LOOP.

    ## Save XML
    # Save the xml file from the tree
    tree = ET.ElementTree(root)
    tree.write(metrics_foldername+os.path.sep+star_name+'-detections-'+metric+'.xml')

    ## PLOTS to png
    # The following plots the criterion map and locate the candidates or local maxima before saving the result as a png.
    if not mute:
        print("Number of candidates = " + str(N_candidates))
    # Following paragraph plots the candidate image
    plt.close(3)
    plt.figure(3,figsize=(16,16))
    # plot the criterion map. One axis has to be mirrored to get North up.
    plt.imshow(criterion_map_cpy[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
    ax = plt.gca()
    # Loop over the candidates
    for candidate in candidates_elt:
        # get the position of the candidate and the criterion value from the candidate tree
        candidate_id = int(candidate.attrib["id"])
        max_val_criter = float(candidate.attrib["max_val_criter"])
        x_max_pos = float(candidate.attrib["x_max_pos"])
        y_max_pos = float(candidate.attrib["y_max_pos"])

        # Draw an arrow with some text to point at the candidate
        ax.annotate(str(candidate_id)+","+"{0:02.1f}".format(max_val_criter), fontsize=20, color = "black", xy=(x_max_pos+0.0, y_max_pos+0.0),
                xycoords='data', xytext=(x_max_pos+10, y_max_pos-10),
                textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                linewidth = 1.,
                                color = 'black')
                )
    plt.clim(0.,10.0)
    # Save the candidate plot as png in the same folder as the metric maps.
    # Filename is <star_name>-detectionIm_candidates-<metric>.png
    plt.savefig(metrics_foldername+os.path.sep+star_name+'-detectionIm_candidates-'+metric+'.png', bbox_inches='tight')
    plt.close(3)

    # Following paragraph plots the all local maxima image
    plt.close(3)
    # Show the local maxima in the criterion map
    plt.figure(3,figsize=(16,16))
    # plot the criterion map. One axis has to be mirrored to get North up.
    plt.imshow(criterion_map_cpy[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
    ax = plt.gca()
    # Loop over all local maxima
    for spot in all_elt:
        # get the position of the maximum and the criterion value from the all_elt tree
        candidate_id = int(spot.attrib["id"])
        max_val_criter = float(spot.attrib["max_val_criter"])
        x_max_pos = float(spot.attrib["x_max_pos"])
        y_max_pos = float(spot.attrib["y_max_pos"])
        ax.annotate(str(candidate_id)+","+"{0:02.1f}".format(max_val_criter), fontsize=20, color = 'black', xy=(x_max_pos+0.0, y_max_pos+0.0),
                xycoords='data', xytext=(x_max_pos+10, y_max_pos-10),
                textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                linewidth = 1.,
                                color = 'black')
                )
    plt.clim(0.,10.0)
    # Save the plot with all local maxima as png in the same folder as the metric maps.
    # Filename is <star_name>-detectionIm_all-<metric>.png
    plt.savefig(metrics_foldername+os.path.sep+star_name+'-detectionIm_all-'+metric+'.png', bbox_inches='tight')
    plt.close(3)

    return 1

def gather_detections(planet_detec_dir, PSF_cube_filename, mute = True,which_metric = None,GOI_list = None):
    '''
    Gather the candidates detected with different spectral templates for a given data cube into a single big image and
    a single xml file.

    Beside try to extract a very rough quicklook spectrum for these candidates in several png image (One per candidate).

    The algorithm should detect if the same object was detected by several templates and consider that as a single
    detection.

    Note: If the algorithm fails at merging neighboring detections (less than pixel apart) it means the centroid
    algorithm couldn't converge which means the detected object is not enough blob like in the flat cube.
    Bad sign for being a planet...



    :param planet_detec_dir: Directory of the detection folder. We recall that the convention is to create one folder
                            per data cube to store the outputs of the detection algorithm. This folder will contain
                            subfolder(s) for each spectral template.
    :param PSF_cube_filename: The name of the PSF cube fits file or directly a the numpy array of the PSF cube.
                    If PSF_cube_filename is an array then nl,ny_PSF,nx_PSF=PSF_cube_filename.shape, nl is the number of
                    wavelength samples and should be 37, ny_PSF and nx_PSF are the spatial dimensions of the PSF_cube.
    :param mute: If True prevent printed log outputs.
    :param which_metric: String matching either of the following: "shape", "matchedFilter", "maxShapeMF". It tells which
                metric should be used for the detection. The default value is "shape".
    :param GOI_list: XML file with the list of known object in the campaign. If GOI_list is not none and if there is
                    registered known GOI objects these objects will be marked with a circle in the png.
    :return: 1 if successful and None otherwise.
            Also creates some files in planet_detec_dir:
            - <star_name>-<filter>-<date>-candidates-<which_metric>.xml
                Non redundant detections in a xml file.
            - <star_name>-<filter>-<date>-candidates-<which_metric>.png
                All spectral templates candidates and non redundant ones as a png image.
            - One <star_name>-<filter>-<date>-candidates-<which_metric>_<ID>.png per candidate
                Rough spectrum of each candidates
    '''

    # Default value of the metric used for the detection is "shape".
    if which_metric is None:
        which_metric = "shape"


    # Check that the input directory is a valid path
    planet_detec_dir_glob = glob.glob(planet_detec_dir)
    if np.size(planet_detec_dir_glob) == 0:
        if not mute:
            print("/!\ Quitting! Couldn't find the following directory " + planet_detec_dir)
        return None
    else:
        planet_detec_dir = planet_detec_dir_glob[0]

    # First recover the name of the data cube file that was reduced for this folder.
    # The name of the file is in the folder name so we can split the input directory
    planet_detec_dir_splitted = planet_detec_dir.split(os.path.sep)
    # original_cube_filename is the filename of the original klipped cube
    original_cube_filename = planet_detec_dir_splitted[len(planet_detec_dir_splitted)-2].split("planet_detec_")[1]
    original_cube_filename += "-speccube.fits"

    # If PSF_cube_filename is a string read the corresponding fits file.
    if isinstance(PSF_cube_filename, basestring):
        hdulist = pyfits.open(PSF_cube_filename)
        PSF_cube = hdulist[1].data
        hdulist.close()
    else:
        # If PSF_cube_filename is an array just copy the reference into PSF_cube
        PSF_cube = PSF_cube_filename

    # Read the data cube fits with headers
    hdulist = pyfits.open(planet_detec_dir+os.path.sep+".."+os.path.sep+original_cube_filename)
    cube = hdulist[1].data
    exthdr = hdulist[1].header
    prihdr = hdulist[0].header
    hdulist.close()

    # Get shape of the data cube
    nl,ny,nx = np.shape(cube)

    # Get center of the image (star position)
    try:
        # Retrieve the center of the image from the fits headers.
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    except:
        # If the keywords could not be found the center is defined as the middle of the image
        if not mute:
            print("Couldn't find PSFCENTX and PSFCENTY keywords.")
        center = [(nx-1)/2,(ny-1)/2]


    try:
        prihdr = hdulist[0].header
        date = prihdr["DATE"]
        hdulist.close()
    except:
        date = "no_date"
        hdulist.close()

    # Get the filter of the image
    try:
        # Retrieve the filter used from the fits headers.
        filter = prihdr['IFSFILT'].split('_')[1]
    except:
        # If the keywords could not be found assume that the filter is H...
        if not mute:
            print("Couldn't find IFSFILT keyword.")
        filter = "no_filter"

    # Get current star name
    try:
        # OBJECT: keyword in the primary header with the name of the star.
        star_name = prihdr['OBJECT'].strip().replace (" ", "_")
    except:
        # If the object name could nto be found cal lit unknown_object
        star_name = "UNKNOWN_OBJECT"


    # Get the list of folders in planet_detec_dir
    # The list of folders should also be the list of spectral templates for which a detection has been ran.
    spectrum_folders_list = glob.glob(planet_detec_dir+os.path.sep+"*"+os.path.sep)
    N_spectra_folders = len(spectrum_folders_list)

    # Open the figure in which the detections for different spectra will be plotted
    plt.figure(1,figsize=(8*N_spectra_folders,16))

    if not mute:
        print("Looking for folders in " + planet_detec_dir + " ...")
        print("... They should contain spectral template based detection algorithm outputs.")
    all_templates_detections = []
    # Loop over the folders in spectrum_folders_list
    for spec_id,spectrum_folder in enumerate(spectrum_folders_list):
        # Get the spectrum name which should be the name the of the folder
        spectrum_folder_splitted = spectrum_folder.split(os.path.sep)
        spectrum_name = spectrum_folder_splitted[len(spectrum_folder_splitted)-2]
        if not mute:
            print("Found the folder " + spectrum_name)

        # Look for the metric, metric probability, detection xml file and spectral template in the folder
        candidates_log_file_list = glob.glob(spectrum_folder+os.path.sep+"*-detections-"+which_metric+".xml")
        #weightedFlatCube_file_list = glob.glob(spectrum_folder+os.path.sep+"*-weightedFlatCube_proba.fits")
        shape_proba_file_list = glob.glob(spectrum_folder+os.path.sep+"*-shape_proba.fits")
        matchedFilter_proba_file_list = glob.glob(spectrum_folder+os.path.sep+"*-matchedFilter_proba.fits")
        shape_file_list = glob.glob(spectrum_folder+os.path.sep+"*-shape.fits")
        matchedFilter_file_list = glob.glob(spectrum_folder+os.path.sep+"*-matchedFilter.fits")
        template_spectrum_file_list = glob.glob(spectrum_folder+os.path.sep+"template_spectrum.fits")

        # Only if they are all found we process with the following
        if len(candidates_log_file_list) == 1 and \
                        len(shape_proba_file_list) == 1 and \
                        len(shape_file_list) == 1 and \
                        len(template_spectrum_file_list) == 1 and \
                        len(matchedFilter_proba_file_list) == 1 and \
                        len(matchedFilter_file_list) == 1:

            # Get the path of the files from the results of the glob function
            candidates_log_file = candidates_log_file_list[0]
            shape_proba_file = shape_proba_file_list[0]
            shape_file = shape_file_list[0]
            matchedFilter_file = matchedFilter_file_list[0]
            matchedFilter_proba_file = matchedFilter_proba_file_list[0]
            template_spectrum_file = template_spectrum_file_list[0]

            # Read shape_file
            if which_metric == "shape":
                hdulist = pyfits.open(shape_proba_file)
                metric_proba = hdulist[1].data
            elif which_metric == "matchedFilter":
                hdulist = pyfits.open(matchedFilter_proba_file)
                metric_proba = hdulist[1].data
            elif which_metric == "maxShapeMF":
                hdulist = pyfits.open(shape_proba_file)
                shape_proba = hdulist[1].data
                hdulist = pyfits.open(matchedFilter_proba_file)
                matchedFilter_proba = hdulist[1].data
                metric_proba = np.max([shape_proba,matchedFilter_proba],axis=0)
            exthdr = hdulist[1].header
            prihdr = hdulist[0].header
            hdulist.close()
            x_grid, y_grid = np.meshgrid(np.arange(0,nx,1)-center[0],np.arange(0,ny,1)-center[1])

            # Plot on the top line of the figure the probability map. One axis has to be mirrored to get North up.
            plt.subplot(2,N_spectra_folders,spec_id+1)
            plt.imshow(metric_proba[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
            ax = plt.gca()

            # Get tree of the xml file with the candidates
            tree = ET.parse(candidates_log_file)
            root = tree.getroot()
            for candidate in root[0].find("candidates"):
                # Get the information of the candidate from the element attributes
                candidate_id = int(candidate.attrib["id"])
                max_val_criter = float(candidate.attrib["max_val_criter"])
                x_max_pos = float(candidate.attrib["x_max_pos"])
                y_max_pos = float(candidate.attrib["y_max_pos"])
                row_id = float(candidate.attrib["row_id"])
                col_id = float(candidate.attrib["col_id"])

                # Store the information in a simpler list (than the tree) for next time we will need it in this function
                all_templates_detections.append((spectrum_name,int(candidate_id),float(max_val_criter),float(x_max_pos),float(y_max_pos), int(row_id),int(col_id)))

                # Draw an arrow with some text to point at the candidate
                ax.annotate(str(int(candidate_id))+","+"{0:02.1f}".format(float(max_val_criter)), fontsize=30, color = "red", xy=(float(x_max_pos), float(y_max_pos)),
                        xycoords='data', xytext=(float(x_max_pos)+20, float(y_max_pos)-20),
                        textcoords='data',
                        arrowprops=dict(arrowstyle="->",
                                        linewidth = 2.,
                                        color = 'red')
                        )

            # Draw a circle around the known objects if GOI_list is not None.
            if GOI_list is not None:
                # Read the GOI_list xml file
                tree_GOI_list = ET.parse(GOI_list)
                root_GOI_list = tree_GOI_list.getroot()

                # Get the elements corresponding to the right star
                root_GOI_list_givenObject = root_GOI_list.findall(star_name)
                # Loop over these known elements and circle them
                for object_elt in root_GOI_list_givenObject:
                    #print(object_elt)
                    for candidate in object_elt.findall("candidate"):
                        col_centroid = float(candidate.attrib["col_centroid"])
                        row_centroid = float(candidate.attrib["row_centroid"])
                        circle=plt.Circle((float(col_centroid)-center[0], float(row_centroid)-center[1]),radius=7.,color='r',fill=False)
                        ax.add_artist(circle)

            plt.title(star_name +" "+ spectrum_name)
            plt.clim(0.,5.0)

            # Read metric fits file
            if which_metric == "shape":
                hdulist = pyfits.open(shape_file)
                metric = hdulist[1].data
                hdulist.close()
            elif which_metric == "matchedFilter":
                hdulist = pyfits.open(matchedFilter_file)
                metric = hdulist[1].data
                hdulist.close()
            elif which_metric == "maxShapeMF":
                metric = metric_proba

            # Plot the metric for the given spectral template in the lower row of the figure
            plt.subplot(2,N_spectra_folders,N_spectra_folders+spec_id+1)
            plt.imshow(metric[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
            ax = plt.gca()

            # Draw a circle around the known objects if GOI_list is not None.
            if GOI_list is not None:
                # Loop over the known elements and circle them
                for object_elt in root_GOI_list_givenObject:
                    #print(object_elt)
                    for candidate in object_elt.findall("candidate"):
                        col_centroid = float(candidate.attrib["col_centroid"])
                        row_centroid = float(candidate.attrib["row_centroid"])
                        circle=plt.Circle((float(col_centroid)-center[0], float(row_centroid)-center[1]),radius=7.,color='r',fill=False)
                        ax.add_artist(circle)

            plt.colorbar()


            if 0:
                # Read template_spectrum_file
                hdulist = pyfits.open(template_spectrum_file)
                spec_sampling = hdulist[1].data[0]
                spectrum = hdulist[1].data[1]
                hdulist.close()

                plt.subplot(3,N_spectra_folders,2*N_spectra_folders+spec_id+1)
                plt.plot(spec_sampling,spectrum,"rx-",markersize = 7, linewidth = 2)
                plt.title("Template spectrum use in this folder")
                plt.xlabel("Wavelength (mum)")
                plt.ylabel("Norm-2 Normalized")

    #plt.show()

    ##
    # Extract centroid for all detections. It will be used to determined if two detections are actually the same object.
    # It can be two detections for the same template if the candidate is really bright but it is usually to gather
    # detections from different templates.

    # Define a tree to store the detection where the duplicates have been removed
    root = ET.Element("root")
    star_elt = ET.SubElement(root, star_name)
    # candidate index that is incremented only when a real new detection is made (one with a different centroid)
    no_duplicates_id = 0
    for detection in all_templates_detections:
        # get the information on all the candidates detected (no matter what the spectrum was)
        spectrum_name,candidate_it,max_val_criter,x_max_pos,y_max_pos, row_id,col_id = detection

        # Try to extract the centroid for the current candidate
        row_cen,col_cen = spec.extract_planet_centroid(cube, (row_id,col_id), PSF_cube)

        # If the current has already been detected in another template candidate_already_seen will become True.
        candidate_already_seen = False
        # If the current has already been detected in within the same template candidate_seen_with_template will become
        # True.
        candidate_seen_with_template = False


        #Check that the candidate hasn't already been detected somewhere else
        for candidate in star_elt:
            if abs(col_cen-float(candidate.attrib['col_centroid'])) < 1. \
                and abs(row_cen-float(candidate.attrib['row_centroid'])) < 1.:
                candidate_already_seen = True
                candidate_elt = candidate

        # If the candidate is new then add it to the list
        if not candidate_already_seen:
            # Increment the index for dectections without duplicates
            no_duplicates_id+=1

            # Add this candidate to the tree
            candidate_elt = ET.SubElement(star_elt,"candidate",
                                          id = str(no_duplicates_id),
                                          x_max_pos= str(x_max_pos),
                                          y_max_pos= str(y_max_pos),
                                          col_centroid= str(col_cen),
                                          row_centroid= str(row_cen),
                                          row_id= str(row_id),
                                          col_id= str(col_id))
            # Add the information relative to the spectral template that could detect it
            ET.SubElement(candidate_elt,"spectrumTemplate",
                          candidate_it = str(candidate_it),
                          name = spectrum_name,
                          max_val_criter = str(max_val_criter))

            # Draw an arrow pointing to the candidate in all the lower plots of the figure. They are the plots with the
            # metric maps
            for spec_id in range(N_spectra_folders):
                plt.subplot(2,N_spectra_folders,N_spectra_folders+spec_id+1)
                ax = plt.gca()
                ax.annotate(str(int(no_duplicates_id)), fontsize=30, color = "black", xy=(float(x_max_pos), float(y_max_pos)),
                        xycoords='data', xytext=(float(x_max_pos)+20, float(y_max_pos)-20),
                        textcoords='data',
                        arrowprops=dict(arrowstyle="->",
                                        linewidth = 2.,
                                        color = 'black')
                        )
        else:
            # If the candidate has already been seen we still want to add the know with which templates it was indeed
            # detected.
            # Nex loop just check that the current candidate was not detected in the same spectrum.
            for template in candidate_elt:
                if template.attrib["name"] == spectrum_name:
                    candidate_seen_with_template = True

            # If the current spectrum correspond to a new spectrum then we add it to the tree
            if not candidate_seen_with_template:
                ET.SubElement(candidate_elt,"spectrumTemplate",
                              candidate_it = str(candidate_it),
                              name = spectrum_name,
                              max_val_criter = str(max_val_criter))

    # Try to extract spectrum for all the candidates after removing duplicates and create a png for all of them.
    for candidate in star_elt:
        # Extract rough spectrum with aperture photometry
        wave_samp,spectrum = spec.extract_planet_spectrum(planet_detec_dir+os.path.sep+".."+os.path.sep+original_cube_filename,
                                                          (float(candidate.attrib["row_centroid"]), float(candidate.attrib['col_centroid'])),
                                                          PSF_cube, method="aperture")

        # Create the png with the extracted spectrum as well as all the spectral templates used.
        plt.close(2)
        plt.figure(2)
        plt.plot(wave_samp,spectrum/np.nanmean(spectrum),"rx-",markersize = 7, linewidth = 2)
        # legend string list. Will be updated as things get plotted
        legend_str = ["candidate spectrum"]
        # Plot the spectral templates
        for spectrum_folder in spectrum_folders_list:
            spectrum_folder_splitted = spectrum_folder.split(os.path.sep)
            spectrum_name = spectrum_folder_splitted[len(spectrum_folder_splitted)-2]

            # Read the fits file containing the spectrum in the folders insides the planet detection input folder
            hdulist = pyfits.open(spectrum_folder+os.path.sep+'template_spectrum.fits')
            spec_data = hdulist[1].data
            hdulist.close()

            wave_samp = spec_data[0]
            spectrum_template = spec_data[1]

            plt.plot(wave_samp,spectrum_template/np.nanmean(spectrum_template),"--")
            legend_str.append(spectrum_name)
        plt.title("Candidate spectrum and templates used")
        plt.xlabel("Wavelength (mum)")
        plt.ylabel("Mean Normalized")
        ax = plt.gca()
        ax.legend(legend_str, loc = 'upper right', fontsize=12)
        plt.savefig(planet_detec_dir+os.path.sep+star_name+'-'+filter+'-'+date+'-candidate-'+which_metric+"_"+ \
                    str(candidate.attrib["id"]) +'.png', bbox_inches='tight')
        plt.close(2)


    # Save the xml file with the non redundant detection
    tree = ET.ElementTree(root)
    if not mute:
        print("Saving "+planet_detec_dir+os.path.sep+star_name+'-'+filter+'-'+date+'-candidates-'+which_metric+'.xml')
    tree.write(planet_detec_dir+os.path.sep+star_name+'-'+filter+'-'+date+'-candidates-'+which_metric+'.xml')

    # Save the summary png image with the detection of all the spectral templates.
    if not mute:
        print("Saving "+planet_detec_dir+os.path.sep+star_name+'-'+filter+'-'+date+'-candidates-'+which_metric+'.png')
    plt.savefig(planet_detec_dir+os.path.sep+star_name+'-'+filter+'-'+date+'-candidates-'+which_metric+'.png', bbox_inches='tight')
    plt.close(1)

    return 1