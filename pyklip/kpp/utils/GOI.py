__author__ = 'JB'


import numpy as np
from copy import copy
from  glob import glob
import csv
import os

def mask_known_objects(cube,prihdr,exthdr,GOI_list_folder, mask_radius = 7):

    cube_cpy = copy(cube)

    if np.size(cube_cpy.shape) == 3:
        nl,ny,nx = cube_cpy.shape
    elif np.size(cube_cpy.shape) == 2:
        ny,nx = cube_cpy.shape
        cube_cpy = cube_cpy[None,:]
        nl = 1

    width = 2*mask_radius+1
    stamp_x_grid, stamp_y_grid = np.meshgrid(np.arange(0,width,1)-width/2,np.arange(0,width,1)-width/2)
    stamp_mask = np.ones((width,width))
    r_stamp = abs((stamp_x_grid) +(stamp_y_grid)*1j)
    stamp_mask[np.where(r_stamp < mask_radius)] = np.nan

    try:
        # OBJECT: keyword in the primary header with the name of the star.
        object_name = prihdr['OBJECT'].strip().replace (" ", "_")
    except:
        object_name = "UNKNOWN_OBJECT"

    # Get center of the image (star position)
    try:
        # Retrieve the center of the image from the fits headers.
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    except:
        # If the keywords could not be found the center is defined as the middle of the image
        print("Couldn't find PSFCENTX and PSFCENTY keywords.")
        center = [(nx-1)/2,(ny-1)/2]

    #Julian Day OBServation
    MJDOBS_fits = prihdr['MJD-OBS']

    row_m = np.floor(width/2.0)
    row_p = np.ceil(width/2.0)
    col_m = np.floor(width/2.0)
    col_p = np.ceil(width/2.0)

    object_GOI_filename = GOI_list_folder+os.path.sep+object_name+'_GOI.csv'

    if len(glob(object_GOI_filename)) != 0:
        with open(object_GOI_filename, 'rb') as csvfile_GOI_list:
            GOI_list_reader = csv.reader(csvfile_GOI_list, delimiter=';')
            GOI_csv_as_list = list(GOI_list_reader)
            N_objects = len(GOI_csv_as_list)-1
            attrib_name = GOI_csv_as_list[0]
            GOI_list = np.array(GOI_csv_as_list[1:len(GOI_csv_as_list)])

            pa_id = attrib_name.index("PA")
            sep_id = attrib_name.index("SEP")
            MJDOBS_id = attrib_name.index("MJDOBS")

            MJDOBS_arr = np.array([ float(it) for it in GOI_list[:,MJDOBS_id]])
            MJDOBS_unique = np.unique(MJDOBS_arr)
            MJDOBS_closest_id = np.argmin(np.abs(MJDOBS_unique-MJDOBS_fits))
            MJDOBS_closest = MJDOBS_unique[MJDOBS_closest_id]
            #Check that the closest MJDOBS is closer than 2 hours
            if abs(MJDOBS_closest-MJDOBS_fits) > 2./24.:
                # Skip if we couldn't find a matching date.
                return np.squeeze(cube_cpy)

            for obj_id in np.where(MJDOBS_arr == MJDOBS_closest)[0]:
                try:
                    pa = float(GOI_list[obj_id,pa_id])
                    radius = float(GOI_list[obj_id,sep_id])/0.01413
                    x_max_pos = float(radius)*np.cos(np.radians(90+pa))
                    y_max_pos = float(radius)*np.sin(np.radians(90+pa))
                    col_centroid = x_max_pos+center[0]
                    row_centroid = y_max_pos+center[1]
                    k = round(row_centroid)
                    l = round(col_centroid)

                    cube_cpy[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = np.tile(stamp_mask,(nl,1,1)) * cube_cpy[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)]

                except:
                    print("Missing data in GOI database for {0}".format(object_name))

    return np.squeeze(cube_cpy)


def get_pos_known_objects(prihdr,exthdr,GOI_list_folder,xy = False,pa_sep = False):


    try:
        # OBJECT: keyword in the primary header with the name of the star.
        object_name = prihdr['OBJECT'].strip().replace (" ", "_")
    except:
        object_name = "UNKNOWN_OBJECT"

    # Get center of the image (star position)
    # Retrieve the center of the image from the fits headers.
    center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]

    #Julian Day OBServation
    MJDOBS_fits = prihdr['MJD-OBS']


    object_GOI_filename = GOI_list_folder+os.path.sep+object_name+'_GOI.csv'

    x_vec = []
    y_vec = []
    col_vec = []
    row_vec = []
    pa_vec = []
    sep_vec = []
    if len(glob(object_GOI_filename)) != 0:
        with open(object_GOI_filename, 'rb') as csvfile_GOI_list:
            GOI_list_reader = csv.reader(csvfile_GOI_list, delimiter=';')
            GOI_csv_as_list = list(GOI_list_reader)
            attrib_name = GOI_csv_as_list[0]
            GOI_list = np.array(GOI_csv_as_list[1:len(GOI_csv_as_list)])

            pa_id = attrib_name.index("PA")
            sep_id = attrib_name.index("SEP")
            MJDOBS_id = attrib_name.index("MJDOBS")

            MJDOBS_arr = np.array([ float(it) for it in GOI_list[:,MJDOBS_id]])
            MJDOBS_unique = np.unique(MJDOBS_arr)
            MJDOBS_closest_id = np.argmin(np.abs(MJDOBS_unique-MJDOBS_fits))
            MJDOBS_closest = MJDOBS_unique[MJDOBS_closest_id]
            #Check that the closest MJDOBS is closer than 2 hours
            if abs(MJDOBS_closest-MJDOBS_fits) > 2./24.:
                # Skip if we couldn't find a matching date.
                return [],[]

            for obj_id in np.where(MJDOBS_arr == MJDOBS_closest)[0]:
                try:
                    pa = float(GOI_list[obj_id,pa_id])
                    radius = float(GOI_list[obj_id,sep_id])

                    pa_vec.append(pa)
                    sep_vec.append(radius)
                    x_max_pos = float(radius/0.01413)*np.cos(np.radians(90+pa))
                    y_max_pos = float(radius/0.01413)*np.sin(np.radians(90+pa))
                    x_vec.append(x_max_pos)
                    y_vec.append(y_max_pos)
                    row_vec.append(y_max_pos+center[1])
                    col_vec.append(x_max_pos+center[0])
                except:
                    print("Missing data in GOI database for {0}".format(object_name))

    print(sep_vec,pa_vec)
    if pa_sep:
        return sep_vec,pa_vec
    elif xy:
        return x_vec,y_vec
    else:
        return row_vec,col_vec