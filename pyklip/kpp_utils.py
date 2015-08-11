__author__ = 'JB'

import numpy as np
import astropy.io.fits as pyfits
from copy import copy
from scipy.stats import nanmedian
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import glob, os
import multiprocessing as mp
import multiprocessing.pool as mpPool
import xml.etree.cElementTree as ET
import shutil

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonPool(mpPool.Pool):
    Process = NoDaemonProcess

def get_campaign_candidates(campaign_dir = "."+os.path.sep,output_dir = "."+os.path.sep, metric = None, mute = True):
    #objectsDir_list = []
    for objectsDir in os.listdir(campaign_dir):
        if not objectsDir.startswith('.'):
            #objectsDir_list.append(campaign_dir+inputDir+os.path.sep+"autoreduced"+os.path.sep)

            objectsDir = campaign_dir+objectsDir+os.path.sep+"autoreduced"+os.path.sep
            #print(inputDir)

            planet_detec_dir_list = glob.glob(objectsDir+"planet_detec*k100a7s4m3-KL20")
            #print(planet_detec_dir_list)

            for planet_detec_dir in planet_detec_dir_list:
                spectrum_folders_list = glob.glob(planet_detec_dir+os.path.sep+"*"+os.path.sep)
                if metric is not None:
                    src_list = glob.glob(planet_detec_dir+os.path.sep+"*-candidates-"+metric+".png")
                else:
                    src_list = glob.glob(planet_detec_dir+os.path.sep+"*-candidates.png")

                for src in src_list:
                    src_splitted = src.split(os.path.sep)
                    dst = output_dir+src_splitted[len(src_splitted)-1]
                    if not mute:
                        print("Copying " + src + " to " + dst)
                    shutil.copyfile(src, dst)


    '''
    with open(GOI_list_filename, 'r') as GOI_list:
        for myline in GOI_list:
            #print([myline.rstrip()])
            if (not myline.startswith("#")) and myline.rstrip():
    '''


def clean_planet_detec_outputs(campaign_dir = "."+os.path.sep):
    print("/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\")
    print("VERY DANGEROUS FUNCTION. DO NOT USE IF NOT SURE.")
    print("Are you sure you want to delete files?")
    print("Please modify the function to enable files removal. Nothing will happen like that.")
    print("/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\")
    var = raw_input("Enter \"I want to\" to go on: ")
    print("You entered: "+str([var]))
    if 1 and var != "I want to":
        print("Won't delete files")
        return 0
    else:
        print("I warned you!!")


    for objectsDir in os.listdir(campaign_dir):
        if not objectsDir.startswith('.'):

            objectsDir = campaign_dir+objectsDir+os.path.sep+"autoreduced"+os.path.sep
            #print(inputDir)

            planet_detec_dir_list = glob.glob(objectsDir+"planet_detec*-KL*")
            #print(planet_detec_dir_list)

            for planet_detec_dir in planet_detec_dir_list:
                #print(planet_detec_dir)

                if 0:
                    spectrum_folder_list = glob.glob(planet_detec_dir+os.path.sep+"t800*")
                    for spectrum_folder in spectrum_folder_list:
                        print("Removing "+spectrum_folder)
                        files_list = glob.glob(spectrum_folder+os.path.sep+"*")
                        for file in files_list:
                            print("Removing "+file)
                            #os.remove(file)
                        print(spectrum_folder)
                        #os.rmdir(spectrum_folder)
                    #print(planet_detec_dir)
                    #os.rmdir(planet_detec_dir)



                        #os.remove(files_no_spectra)

                if 0:
                    files_no_spectra_list = glob.glob(planet_detec_dir+os.path.sep+"*-*-*-*-*-*_*_*.png")
                    for files_no_spectra in files_no_spectra_list:
                        print("Removing "+files_no_spectra)
                        os.remove(files_no_spectra)
                if 0:
                    spectrum_folders_list = glob.glob(planet_detec_dir+os.path.sep+"*"+os.path.sep)
                    for spectrum_folder in spectrum_folders_list:
                        #print(spectrum_folder)
                        proba_files_list = glob.glob(spectrum_folder+os.path.sep+"*.txt")
                        for proba_file in proba_files_list:
                            print("Removing "+proba_file)
                            os.remove(proba_file)
                if 0:
                    spectrum_folders_list = glob.glob(planet_detec_dir+os.path.sep+"*"+os.path.sep)
                    for spectrum_folder in spectrum_folders_list:
                        #print(spectrum_folder)
                        conflicted_files_list = glob.glob(spectrum_folder+os.path.sep+"*(Copie en conflit*")
                        for conflicted_file in conflicted_files_list:
                            print("Removing "+conflicted_file)
                            os.remove(conflicted_file)

def GOI_list_add_object_from_coord(GOI_list_filename, object_name, col_centroid,row_centroid):
    GOI_list_filename_list = glob.glob(GOI_list_filename)
    print(GOI_list_filename_list)
    if len(GOI_list_filename_list) == 0:
        print("Couldn't find GOI list xml file: "+GOI_list_filename)
        print("Try to create one")
        root_GOI_list = ET.Element("root")
        tree_GOI_list = ET.ElementTree(root_GOI_list)
        tree_GOI_list.write(GOI_list_filename)
    else:
        tree_GOI_list = ET.parse(GOI_list_filename)
        root_GOI_list = tree_GOI_list.getroot()

    object_elt_in_GOI_list = root_GOI_list.find(object_name)
    if object_elt_in_GOI_list is None:
        object_elt_in_GOI_list = ET.Element(object_name)
        root_GOI_list.append(object_elt_in_GOI_list)

    ET.SubElement(object_elt_in_GOI_list, "candidate",
                      col_centroid = str(col_centroid),
                      row_centroid = str(row_centroid))

    tree_GOI_list.write(GOI_list_filename)

def GOI_list_add_object_from_xml(GOI_list_filename, object_planet_detec_directory, detection_indices ):

    GOI_list_filename_list = glob.glob(GOI_list_filename)
    print(GOI_list_filename_list)
    if len(GOI_list_filename_list) == 0:
        print("Couldn't find GOI list xml file: "+GOI_list_filename)
        print("Try to create one")
        root_GOI_list = ET.Element("root")
        tree_GOI_list = ET.ElementTree(root_GOI_list)
        tree_GOI_list.write(GOI_list_filename)
    else:
        tree_GOI_list = ET.parse(GOI_list_filename)
        root_GOI_list = tree_GOI_list.getroot()

    object_detec_xml = glob.glob(object_planet_detec_directory+os.path.sep+os.path.sep+"*-candidates.xml")
    #print(object_detec_xml)
    if len(object_detec_xml) == 1:
        tree_object = ET.parse(object_detec_xml[0])
        root_object = tree_object.getroot()

        object_elt_in_GOI_list = root_GOI_list.find(root_object[0].tag)
        if object_elt_in_GOI_list is None:
            object_elt_in_GOI_list = ET.Element(root_object[0].tag)
            root_GOI_list.append(object_elt_in_GOI_list)


        for candidate in root_object[0]:
            if int(candidate.attrib["id"]) in detection_indices:
                object_elt_in_GOI_list.append(candidate)

        tree_GOI_list.write(GOI_list_filename)
    elif len(object_detec_xml) == 0:
        print("Couldn't find xml file: "+object_planet_detec_directory+os.path.sep+"*-candidates.xml")
        return None
    else:
        print("I found several files: "+object_planet_detec_directory+os.path.sep+"*-candidates.xml")
        print(object_detec_xml)
        print("Don't know what to do with them...")
        return None


def get_occ(image, centroid = None):
    '''
    Get the IWA (inner working angle) of the central disk of nans and return the mask corresponding to the inner disk.

    :param image: A GPI image with a disk full of nans at the center.
    :param centroid: center of the nan disk
    :return:
    '''
    ny,nx = image.shape

    if centroid is None :
        x_cen = np.ceil((nx-1)/2) ; y_cen = np.ceil((ny-1)/2)
    else:
        x_cen, y_cen = centroid

    IWA = 0
    while np.isnan(image[x_cen,y_cen+IWA]):
        IWA += 1

    # Build the x and y coordinates grids
    x, y = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
    # Calculate the radial distance of each pixel
    r = abs(x +y*1j)

    mask = np.ones((ny,nx))
    mask[np.where(np.isnan(image))] = np.nan

    inner_mask = copy(mask)
    inner_mask[np.where(r > IWA+2.)] = 1

    outer_mask = copy(mask)
    outer_mask[np.where(np.isnan(inner_mask))] = 1
    OWA = np.min(r[np.where(np.isnan(outer_mask))])

    return IWA,OWA,inner_mask,outer_mask

def mask_known_objects(cube,prihdr,GOI_list_filename, mask_radius = 7):

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

    candidates_list = []

    tree_GOI_list = ET.parse(GOI_list_filename)
    root_GOI_list = tree_GOI_list.getroot()

    row_m = np.floor(width/2.0)
    row_p = np.ceil(width/2.0)
    col_m = np.floor(width/2.0)
    col_p = np.ceil(width/2.0)

    for object_elt in root_GOI_list.findall(object_name):
        #print(object_elt)
        for candidate in object_elt.findall("candidate"):
            k = round(float(candidate.attrib["row_centroid"]))
            l = round(float(candidate.attrib["col_centroid"]))
            #print(k,l)
            cube_cpy[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)] = np.tile(stamp_mask,(nl,1,1)) * cube_cpy[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)]


    return np.squeeze(cube_cpy)


def extract_PSFs(filename, stamp_width = 10, mute = False):
    '''
    Extract the PSFs in a pyklip reduced cube in which fake planets have been injected.
    The position of the fake planets is stored in the headers when pyklip is used.
    A cube stamp is extracted for each radius and angle and they are all sorted in the out_stamp_PSFs array.

    /!\ The cube is normalized following cube /= np.nanstd(cube[10,:,:])
    This is because I don't want the very small numbers in Jason's output as he uses contrast units

    :param filename: Name of the file from which the fake planets cube stamps should be extracted.
    :param stamp_width: Spatial width of the stamps to be extracted around each fake planets.
    :param mute: If true print some text in the console.
    :return out_stamp_PSFs: A (nl,stamp_width,stamp_width,nth,nr) array with nl the number of wavelength of the cube,
            nth the number of section in the klip reduction and nr the number of annuli. Therefore the cube defined by
            out_stamp_PSFs[:,:,:,0,2] is a cube stamp of the planet in the first section of the third annulus. In order
            to know what position it exactly corresponds to please look at the FAKPLPAR keyword in the primary headers.
    '''
    hdulist = pyfits.open(filename)
    cube = hdulist[1].data
    cube /= np.nanstd(cube[10,:,:])
    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        cube = cube[None,:]
        nl = 1
    #slice = hdulist[1].data[2,:,:]
    #ny,nx = slice.shape
    prihdr = hdulist[0].header
    exthdr = hdulist[1].header

    try:
        # Retrieve the center of the image from the fits keyword.
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    except:
        # If the keywords could not be found.
        if not mute:
            print("Couldn't find PSFCENTX and PSFCENTY keywords.")
        center = [(nx-1)/2,(ny-1)/2]

    try:
        # Retrieve the position of the fake planets from the fits keyword.
        fakePlparams_str = prihdr['FAKPLPAR']
    except:
        # If the keywords could not be found.
        if not mute:
            print("ERROR. Couldn't find FAKPLPAR (Fake planets parameters) keyword. Has to quit extract_PSFs().")
        return 0

    fakePlparams_splitted_str = fakePlparams_str.split(";")
    planet_angles_str = fakePlparams_splitted_str[6]
    planet_radii_str = fakePlparams_splitted_str[7]
    planet_angles =eval(planet_angles_str.split("=")[1])
    planet_radii =eval(planet_radii_str.split("=")[1])

    nth = np.size(planet_angles)
    nr = np.size(planet_radii)

    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    #x.shape = (x.shape[0] * x.shape[1])
    #y.shape = (y.shape[0] * y.shape[1])
    x -= center[0]
    y -= center[1]
    x_planets = np.dot(np.array([planet_radii]).transpose(),np.array([np.cos(np.radians(planet_angles))]))
    y_planets = np.dot(np.array([planet_radii]).transpose(),np.array([np.sin(np.radians(planet_angles))]))
    #print(x_planets+center[0])
    #print(y_planets+center[1])

    out_stamp_PSFs = np.zeros((nl,stamp_width,stamp_width,nth,nr))

    for l_id in range(nl):
        for r_id,r_it in enumerate(planet_radii):
            for th_id, th_it in enumerate(planet_angles):
                x_plnt = np.round(x_planets[r_id,th_id]+center[0])
                y_plnt = np.round(y_planets[r_id,th_id]+center[1])

                out_stamp_PSFs[l_id,:,:,th_id,r_id] = cube[l_id,
                                                            (y_plnt-np.floor(stamp_width/2.)):(y_plnt+np.ceil(stamp_width/2.)),
                                                            (x_plnt-np.floor(stamp_width/2.)):(x_plnt+np.ceil(stamp_width/2.))]

        #print(l_id,r_id,r_it,th_id, th_it,x_plnt,y_plnt)
        #plt.imshow(out_stamp_PSFs[:,:,l_id,th_id,r_id],interpolation = 'nearest')
        #plt.show()

    return out_stamp_PSFs

def extract_merge_PSFs(filename, radii, thetas, stamp_width = 10):
    hdulist = pyfits.open(filename)
    cube = hdulist[1].data
    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        cube = cube[None,:]
        nl = 1
    nth = np.size(thetas)
    nr = np.size(radii)
    #slice = hdulist[1].data[2,:,:]
    #ny,nx = slice.shape
    prihdr = hdulist[0].header
    exthdr = hdulist[1].header
    try:
        # Retrieve the center of the image from the fits keyword.
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    except:
        # If the keywords could not be found.
        center = [(nx-1)/2,(ny-1)/2]

    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    #x.shape = (x.shape[0] * x.shape[1])
    #y.shape = (y.shape[0] * y.shape[1])
    x -= center[0]
    y -= center[1]
    x_planets = np.dot(np.array([radii]).transpose(),np.array([np.cos(np.radians(thetas))]))
    y_planets = np.dot(np.array([radii]).transpose(),np.array([np.sin(np.radians(thetas))]))

    out_stamp_PSFs = np.zeros((ny,nx,nth,nr,nl))

    dn = stamp_width
    for l_id in range(nl):
        for r_id,r_it in enumerate(radii):
            for th_id, th_it in enumerate(thetas):
                x_plnt = np.ceil(x_planets[r_id,th_id]+center[0])
                y_plnt = np.ceil(y_planets[r_id,th_id]+center[1])

                stamp = cube[l_id,(y_plnt-dn/2):(y_plnt+dn/2),(x_plnt-dn/2):(x_plnt+dn/2)]
                stamp_x, stamp_y = np.meshgrid(np.arange(dn, dtype=np.float32), np.arange(dn, dtype=np.float32))
                stamp_x += (x_planets[r_id,th_id]+center[0]-x_plnt)
                stamp_y += y_planets[r_id,th_id]+center[1]-y_plnt
                stamp = ndimage.map_coordinates(stamp, [stamp_y, stamp_x])

                #plt.imshow(stamp,interpolation = 'nearest')
                #plt.show()
                #return

                if k == 0 and l == 0:
                    PSFs_stamps = [stamp]
                else:
                    PSFs_stamps = np.concatenate((PSFs_stamps,[stamp]),axis=0)

    return np.mean(PSFs_stamps,axis=0)

def gauss2d(x, y, amplitude = 1.0, xo = 0.0, yo = 0.0, sigma_x = 1.0, sigma_y = 1.0, theta = 0, offset = 0):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g

def subtract_radialMed(image,w,l,center):
    ny,nx = image.shape

    n_area_x = np.floor(nx/w)
    n_area_y = np.floor(ny/w)

    x, y = np.meshgrid(np.arange(nx)-center[0], np.arange(ny)-center[1])
    r_grid = abs(x +y*1j)
    th_grid = np.arctan2(x,y)

    for p in np.arange(n_area_x):
        for q in np.arange(n_area_y):
            #stamp = image[(q*w):((q+1)*w),(p*w):((p+1)*w)]
            #image[(q*w):((q+1)*w),(p*w):((p+1)*w)] -= nanmedian(stamp)

            r = r_grid[((q+0.5)*w),((p+0.5)*w)]
            th = th_grid[((q+0.5)*w),((p+0.5)*w)]

            arc_id = np.where(((r-w/2.0) < r_grid) * (r_grid < (r+w/2.0)) * ((th-l/r) < th_grid) * (th_grid < (th+l/r)))
            image[(q*w):((q+1)*w),(p*w):((p+1)*w)] -= nanmedian(image[arc_id])

            if 0 and p == 50 and q == 50:
                image[arc_id] = 100
                print(image[arc_id].size)
                plt.figure(2)
                plt.imshow(image, interpolation="nearest")
                plt.show()


    return image