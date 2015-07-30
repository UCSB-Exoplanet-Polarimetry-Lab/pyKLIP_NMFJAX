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
                        confirm_candidates = None,
                        metric = None):
    '''

    Inputs:


    Outputs:

    '''
    shape_filename_list = glob.glob(metrics_foldername+os.path.sep+"*-shape_proba.fits")
    matchedFilter_filename_list = glob.glob(metrics_foldername+os.path.sep+"*-matchedFilter_proba.fits")
    if len(shape_filename_list) == 0:
        if not mute:
            print("Couldn't find shape_SNR map in "+metrics_foldername)
            return 0
    else:
        hdulist_shape = pyfits.open(shape_filename_list[0])
    if len(matchedFilter_filename_list) == 0:
        if not mute:
            print("Couldn't find matchedFilter_SNR map in "+metrics_foldername)
            return 0
    else:
        hdulist_matchedFilter = pyfits.open(matchedFilter_filename_list[0])

    #grab the data and headers
    try:
        shape_map = hdulist_shape[1].data
        matchedFilter_map = hdulist_matchedFilter[1].data
        exthdr = hdulist_shape[1].header
        prihdr = hdulist_shape[0].header
    except:
        print("Couldn't read the fits file normally. Try another way.")
        shape_map = hdulist_shape[0].data
        prihdr = hdulist_shape[0].header

    if np.size(shape_map.shape) == 2:
        ny,nx = shape_map.shape

    try:
        # Retrieve the center of the image from the fits keyword.
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    except:
        # If the keywords could not be found.
        if not mute:
            print("Couldn't find PSFCENTX and PSFCENTY keywords.")
        center = [(nx-1)/2,(ny-1)/2]


    try:
        # OBJECT: keyword in the primary header with the name of the star.
        star_name = prihdr['OBJECT'].strip().replace (" ", "_")
    except:
        star_name = "UNKNOWN_OBJECT"


    if metric is None:
        metric = "shape"

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

    criterion_map_cpy = copy(criterion_map)
    if 0:
        IWA,OWA,inner_mask,outer_mask = get_occ(criterion_map, centroid = center)

        # Ignore all the pixel too close from an edge with nans
        #flat_cube_nans = np.where(np.isnan(criterion_map))
        #flat_cube_mask = np.ones((ny,nx))
        #flat_cube_mask[flat_cube_nans] = np.nan
        #widen the nans region
        conv_kernel = np.ones((10,10))
        flat_cube_wider_mask = convolve2d(outer_mask,conv_kernel,mode="same")
        criterion_map[np.where(np.isnan(flat_cube_wider_mask))] = np.nan


    # Build as grids of x,y coordinates.
    # The center is in the middle of the array and the unit is the pixel.
    # If the size of the array is even 2n x 2n the center coordinates is [n,n].
    x_grid, y_grid = np.meshgrid(np.arange(0,nx,1)-center[0],np.arange(0,ny,1)-center[1])


    # Definition of the different masks used in the following.
    stamp_nrow = 13
    stamp_ncol = 13
    # Mask to remove the spots already checked in criterion_map.
    stamp_x_grid, stamp_y_grid = np.meshgrid(np.arange(0,stamp_nrow,1)-6,np.arange(0,stamp_ncol,1)-6)
    stamp_mask = np.ones((stamp_nrow,stamp_ncol))
    r_stamp = abs((stamp_x_grid) +(stamp_y_grid)*1j)
    stamp_mask[np.where(r_stamp < 4.0)] = np.nan

    row_m = np.floor(stamp_nrow/2.0)
    row_p = np.ceil(stamp_nrow/2.0)
    col_m = np.floor(stamp_ncol/2.0)
    col_p = np.ceil(stamp_ncol/2.0)

    root = ET.Element("root")
    star_elt = ET.SubElement(root, star_name)
    candidates_elt = ET.SubElement(star_elt, "candidates")
    all_elt = ET.SubElement(star_elt, "all")

    # Count the number of valid detected candidates.
    N_candidates = 0.0

    # Maximum number of iterations on local maxima.
    max_attempts = 60
                ## START FOR LOOP.
                ## Each iteration looks at one local maximum in the criterion map.
    k = 0
    max_val_criter = np.nanmax(criterion_map)
    while max_val_criter >= 2.5 and k <= max_attempts:
        k += 1
        # Find the maximum value in the current SNR map. At each iteration the previous maximum is masked out.
        max_val_criter = np.nanmax(criterion_map)
        # Locate the maximum by retrieving its coordinates
        max_ind = np.where( criterion_map == max_val_criter )
        row_id,col_id = max_ind[0][0],max_ind[1][0]
        x_max_pos, y_max_pos = x_grid[row_id,col_id],y_grid[row_id,col_id]

        # Mask the spot around the maximum we just found.
        criterion_map[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] *= stamp_mask

        potential_planet = max_val_criter > threshold


        ET.SubElement(all_elt,"localMax",
                      id = str(k),
                      max_val_criter = str(max_val_criter),
                      x_max_pos= str(x_max_pos),
                      y_max_pos= str(y_max_pos),
                      row_id= str(row_id),
                      col_id= str(col_id))

        if potential_planet:
            ET.SubElement(candidates_elt, "localMax",
                          id = str(k),
                          max_val_criter = str(max_val_criter),
                          x_max_pos= str(x_max_pos),
                          y_max_pos= str(y_max_pos),
                          row_id= str(row_id),
                          col_id= str(col_id))

            # Increment the number of detected candidates
            N_candidates += 1


    tree = ET.ElementTree(root)
    tree.write(metrics_foldername+os.path.sep+star_name+'-detections-'+metric+'.xml')

    # Highlight the detected candidates in the criterion map
    if not mute:
        print("Number of candidates = " + str(N_candidates))
    plt.close(3)
    plt.figure(3,figsize=(16,16))
    plt.imshow(criterion_map_cpy[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
    ax = plt.gca()
    for candidate in candidates_elt:
        candidate_id = int(candidate.attrib["id"])
        max_val_criter = float(candidate.attrib["max_val_criter"])
        x_max_pos = float(candidate.attrib["x_max_pos"])
        y_max_pos = float(candidate.attrib["y_max_pos"])

        ax.annotate(str(candidate_id)+","+"{0:02.1f}".format(max_val_criter), fontsize=20, color = "black", xy=(x_max_pos+0.0, y_max_pos+0.0),
                xycoords='data', xytext=(x_max_pos+10, y_max_pos-10),
                textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                linewidth = 1.,
                                color = 'black')
                )
    plt.clim(0.,10.0)
    plt.savefig(metrics_foldername+os.path.sep+star_name+'-detectionIm_candidates-'+metric+'.png', bbox_inches='tight')
    plt.close(3)

    plt.close(3)
    # Show the local maxima in the criterion map
    plt.figure(3,figsize=(16,16))
    plt.imshow(criterion_map_cpy[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
    ax = plt.gca()
    for spot in all_elt:
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
    plt.savefig(metrics_foldername+os.path.sep+star_name+'-detectionIm_all-'+metric+'.png', bbox_inches='tight')
    plt.close(3)

def gather_detections(planet_detec_dir, PSF_cube_filename, mute = True,which_metric = None):

    if which_metric is None:
        which_metric = "shape"

    spectrum_folders_list = glob.glob(planet_detec_dir+os.path.sep+"*"+os.path.sep)
    N_spectra_folders = len(spectrum_folders_list)
    plt.figure(1,figsize=(8*N_spectra_folders,16))


    # First let's recover the name of the file that was reduced for this folder.
    # The name of the file is in the folder name
    planet_detec_dir_glob = glob.glob(planet_detec_dir)
    if np.size(planet_detec_dir_glob) == 0:
        if not mute:
            print("/!\ Quitting! Couldn't find the following directory " + planet_detec_dir)
        return 0
    else:
        planet_detec_dir = planet_detec_dir_glob[0]

    planet_detec_dir_splitted = planet_detec_dir.split(os.path.sep)
    # Filename of the original klipped cube
    original_cube_filename = planet_detec_dir_splitted[len(planet_detec_dir_splitted)-2].split("planet_detec_")[1]
    original_cube_filename += "-speccube.fits"

    if isinstance(PSF_cube_filename, basestring):
        hdulist = pyfits.open(PSF_cube_filename)
        PSF_cube = hdulist[1].data
        hdulist.close()
    else:
        PSF_cube = PSF_cube_filename

    hdulist = pyfits.open(planet_detec_dir+os.path.sep+".."+os.path.sep+original_cube_filename)
    cube = hdulist[1].data
    exthdr = hdulist[1].header
    prihdr = hdulist[0].header
    hdulist.close()

    nl,ny,nx = np.shape(cube)

    try:
        # Retrieve the center of the image from the fits keyword.
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    except:
        # If the keywords could not be found.
        center = [(nx-1)/2,(ny-1)/2]


    try:
        prihdr = hdulist[0].header
        date = prihdr["DATE"]
        hdulist.close()
    except:
        date = "no_date"
        hdulist.close()

    try:
        filter = prihdr['IFSFILT'].split('_')[1]
    except:
        # If the keywords could not be found.
        filter = "no_filter"


    if not mute:
        print("Looking for folders in " + planet_detec_dir + " ...")
        print("... They should contain spectrum template based detection algorithm outputs.")
    all_templates_detections = []
    for spec_id,spectrum_folder in enumerate(spectrum_folders_list):
        spectrum_folder_splitted = spectrum_folder.split(os.path.sep)
        spectrum_name = spectrum_folder_splitted[len(spectrum_folder_splitted)-2]
        if not mute:
            print("Found the folder " + spectrum_name)


        # Gather the detection png in a single png
        candidates_log_file_list = glob.glob(spectrum_folder+os.path.sep+"*-detections-"+which_metric+".xml")
        #weightedFlatCube_file_list = glob.glob(spectrum_folder+os.path.sep+"*-weightedFlatCube_proba.fits")
        shape_proba_file_list = glob.glob(spectrum_folder+os.path.sep+"*-shape_proba.fits")
        matchedFilter_proba_file_list = glob.glob(spectrum_folder+os.path.sep+"*-matchedFilter_proba.fits")
        shape_file_list = glob.glob(spectrum_folder+os.path.sep+"*-shape.fits")
        matchedFilter_file_list = glob.glob(spectrum_folder+os.path.sep+"*-matchedFilter.fits")
        template_spectrum_file_list = glob.glob(spectrum_folder+os.path.sep+"template_spectrum.fits")
        if len(candidates_log_file_list) == 1 and \
                        len(shape_proba_file_list) == 1 and \
                        len(shape_file_list) == 1 and \
                        len(template_spectrum_file_list) == 1 and \
                        len(matchedFilter_proba_file_list) == 1 and \
                        len(matchedFilter_file_list) == 1:
            candidates_log_file = candidates_log_file_list[0]
            shape_proba_file = shape_proba_file_list[0]
            shape_file = shape_file_list[0]
            matchedFilter_file = matchedFilter_file_list[0]
            matchedFilter_proba_file = matchedFilter_proba_file_list[0]
            template_spectrum_file = template_spectrum_file_list[0]

            splitted_str =  candidates_log_file.split(os.path.sep)
            star_name = splitted_str[len(splitted_str)-1].split("-")[0]

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

            plt.subplot(2,N_spectra_folders,spec_id+1)
            plt.imshow(metric_proba[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
            ax = plt.gca()


            tree = ET.parse(candidates_log_file)
            root = tree.getroot()
            for candidate in root[0].find("candidates"):
                candidate_id = int(candidate.attrib["id"])
                max_val_criter = float(candidate.attrib["max_val_criter"])
                x_max_pos = float(candidate.attrib["x_max_pos"])
                y_max_pos = float(candidate.attrib["y_max_pos"])
                row_id = float(candidate.attrib["row_id"])
                col_id = float(candidate.attrib["col_id"])

                all_templates_detections.append((spectrum_name,int(candidate_id),float(max_val_criter),float(x_max_pos),float(y_max_pos), int(row_id),int(col_id)))

                ax.annotate(str(int(candidate_id))+","+"{0:02.1f}".format(float(max_val_criter)), fontsize=30, color = "red", xy=(float(x_max_pos), float(y_max_pos)),
                        xycoords='data', xytext=(float(x_max_pos)+20, float(y_max_pos)-20),
                        textcoords='data',
                        arrowprops=dict(arrowstyle="->",
                                        linewidth = 2.,
                                        color = 'red')
                        )
            plt.title(star_name +" "+ spectrum_name)
            plt.clim(0.,5.0)

            # Read flatCube_file
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

            plt.subplot(2,N_spectra_folders,N_spectra_folders+spec_id+1)
            plt.imshow(metric[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
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
    # Extract centroid for all detections
    root = ET.Element("root")
    star_elt = ET.SubElement(root, star_name)
    #position_no_duplicated_detec = []
    no_duplicates_id = 0
    for detection in all_templates_detections:
        spectrum_name,candidate_it,max_val_criter,x_max_pos,y_max_pos, row_id,col_id = detection

        row_cen,col_cen = spec.extract_planet_centroid(cube, (row_id,col_id), PSF_cube)

        candidate_already_seen = False
        candidate_seen_with_template = False


        #Check that the detection hasn't already been taken care of.
        for candidate in star_elt:
            if abs(col_cen-float(candidate.attrib['col_centroid'])) < 1. \
                and abs(row_cen-float(candidate.attrib['row_centroid'])) < 1.:
                candidate_already_seen = True
                candidate_elt = candidate

        if not candidate_already_seen:
            no_duplicates_id+=1
            candidate_elt = ET.SubElement(star_elt,"candidate",
                                          id = str(no_duplicates_id),
                                          x_max_pos= str(x_max_pos),
                                          y_max_pos= str(y_max_pos),
                                          col_centroid= str(col_cen),
                                          row_centroid= str(row_cen),
                                          row_id= str(row_id),
                                          col_id= str(col_id))
            ET.SubElement(candidate_elt,"spectrumTemplate",
                          candidate_it = str(candidate_it),
                          name = spectrum_name,
                          max_val_criter = str(max_val_criter))

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
            for template in candidate_elt:
                if template.attrib["name"] == spectrum_name:
                    candidate_seen_with_template = True

            if not candidate_seen_with_template:
                ET.SubElement(candidate_elt,"spectrumTemplate",
                              candidate_it = str(candidate_it),
                              name = spectrum_name,
                              max_val_criter = str(max_val_criter))

    for candidate in star_elt:
        wave_samp,spectrum = spec.extract_planet_spectrum(planet_detec_dir+os.path.sep+".."+os.path.sep+original_cube_filename,
                                                          (float(candidate.attrib["row_centroid"]), float(candidate.attrib['col_centroid'])),
                                                          PSF_cube, method="aperture")

        plt.close(2)
        plt.figure(2)
        plt.plot(wave_samp,spectrum/np.nanmean(spectrum),"rx-",markersize = 7, linewidth = 2)
        legend_str = ["candidate spectrum"]
        for spectrum_folder in spectrum_folders_list:
            spectrum_folder_splitted = spectrum_folder.split(os.path.sep)
            spectrum_name = spectrum_folder_splitted[len(spectrum_folder_splitted)-2]

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
        #print(planet_detec_dir)
        #plt.show()
        plt.savefig(planet_detec_dir+os.path.sep+star_name+'-'+filter+'-'+date+'-candidate-'+which_metric+"_"+ \
                    str(candidate.attrib["id"]) +'.png', bbox_inches='tight')
        plt.close(2)

        #myAttributes = {"proba":max_val_criter}
        #ET.SubElement(object_elt, "detec "+str(candidate_it), name="coucou").text = "some value1"


    tree = ET.ElementTree(root)
    print(planet_detec_dir+os.path.sep+star_name+'-'+filter+'-'+date+'-candidates-'+which_metric+'.xml')
    tree.write(planet_detec_dir+os.path.sep+star_name+'-'+filter+'-'+date+'-candidates-'+which_metric+'.xml')


    plt.savefig(planet_detec_dir+os.path.sep+star_name+'-'+filter+'-'+date+'-candidates-'+which_metric+'.png', bbox_inches='tight')
    plt.close(1)