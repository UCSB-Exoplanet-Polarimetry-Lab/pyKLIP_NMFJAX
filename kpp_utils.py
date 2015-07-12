__author__ = 'JB'

import numpy as np
import astropy.io.fits as pyfits
from copy import copy
from scipy.stats import nanmedian
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import glob, os

def get_campaign_candidates(campaign_dir = "."+os.path.sep,output_dir = "."+os.path.sep):
    #objectsDir_list = []
    for objectsDir in os.listdir(campaign_dir):
        if not objectsDir.startswith('.'):
            #objectsDir_list.append(campaign_dir+inputDir+os.path.sep+"autoreduced"+os.path.sep)

            objectsDir = campaign_dir+objectsDir+os.path.sep+"autoreduced"+os.path.sep
            #print(inputDir)

            planet_detec_dir_list = glob.glob(objectsDir+"planet_detec*-KL20")
            #print(planet_detec_dir_list)

            for planet_detec_dir in planet_detec_dir_list:
                spectrum_folders_list = glob.glob(planet_detec_dir+os.path.sep+"*"+os.path.sep)
                N_spectra_folders = len(spectrum_folders_list)
                plt.figure(1,figsize=(4*N_spectra_folders,8))
                for spec_id,spectrum_folder in enumerate(spectrum_folders_list):
                    #print(spectrum_folder)
                    spectrum_folder_splitted = spectrum_folder.split(os.path.sep)
                    spectrum_name = spectrum_folder_splitted[len(spectrum_folder_splitted)-2]
                    #print(spectrum_name)
                    candidates_log_file_list = glob.glob(spectrum_folder+os.path.sep+"*-detections.xml")
                    #weightedFlatCube_file_list = glob.glob(spectrum_folder+os.path.sep+"*-weightedFlatCube_proba.fits")
                    shape_proba_file_list = glob.glob(spectrum_folder+os.path.sep+"*-shape_proba.fits")
                    shape_file_list = glob.glob(spectrum_folder+os.path.sep+"*-shape.fits")
                    if len(candidates_log_file_list) == 1 and len(shape_proba_file_list) == 1 and len(shape_file_list) == 1:
                        candidates_log_file = candidates_log_file_list[0]
                        shape_proba_file = shape_proba_file_list[0]
                        shape_file = shape_file_list[0]

                        splitted_str =  candidates_log_file.split(os.path.sep)
                        object_name = splitted_str[len(splitted_str)-1].split("-")[0]

                        # Read flatCube_file
                        hdulist = pyfits.open(shape_proba_file)
                        shape_proba = hdulist[1].data
                        exthdr = hdulist[1].header
                        prihdr = hdulist[0].header
                        hdulist.close()

                        ny,nx = np.shape(shape_proba)

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

                        x_grid, y_grid = np.meshgrid(np.arange(0,nx,1)-center[0],np.arange(0,ny,1)-center[1])

                        plt.subplot(2,N_spectra_folders,spec_id+1)
                        plt.imshow(shape_proba[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
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

                            ax.annotate(str(int(candidate_id))+","+"{0:02.1f}".format(float(max_val_criter)), fontsize=10, color = "red", xy=(float(x_max_pos), float(y_max_pos)),
                                    xycoords='data', xytext=(float(x_max_pos)+10, float(y_max_pos)-10),
                                    textcoords='data',
                                    arrowprops=dict(arrowstyle="->",
                                                    linewidth = 1.,
                                                    color = 'red')
                                    )
                        plt.title(object_name +" "+ spectrum_name)
                        plt.clim(0.,5.0)

                        # Read flatCube_file
                        hdulist = pyfits.open(shape_file)
                        shape = hdulist[1].data
                        hdulist.close()

                        plt.subplot(2,N_spectra_folders,N_spectra_folders+spec_id+1)
                        plt.imshow(shape[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
                        plt.colorbar()

                plt.savefig(output_dir+os.path.sep+object_name+'-'+filter+'-'+date+'-candidates.png', bbox_inches='tight')
                plt.close(1)
                #plt.show()


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
    if 0 and var != "I want to":
        print("Won't delete files")
        return 0
    else:
        print("I warned you!!")


    for objectsDir in os.listdir(campaign_dir):
        if not objectsDir.startswith('.'):

            objectsDir = campaign_dir+objectsDir+os.path.sep+"autoreduced"+os.path.sep
            #print(inputDir)

            planet_detec_dir_list = glob.glob(objectsDir+"planet_detec*")
            #print(planet_detec_dir_list)

            for planet_detec_dir in planet_detec_dir_list:
                #print(planet_detec_dir)

                if 0:
                    spectrum_folder_list = glob.glob(planet_detec_dir+os.path.sep+"t700*")
                    for spectrum_folder in spectrum_folder_list:
                        print("Removing "+spectrum_folder)
                        files_list = glob.glob(spectrum_folder+os.path.sep+"*")
                        for file in files_list:
                            print("Removing "+file)
                            os.remove(file)

                        os.rmdir(spectrum_folder)



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




def confirm_candidates(GOI_list_filename, logFilename_all, candidate_indices,candidate_status, object_name = None):
    #print(object_name)
    with open(GOI_list_filename, 'a') as GOI_list:
        #print("coucou")
        #print(candidate_indices)

        with open(logFilename_all, 'r') as logFile_all:
            #print("bonjour")
            for myline in logFile_all:
                if not myline.startswith("#"):
                    k,potential_planet,max_val_criter,x_max_pos,y_max_pos, row_id,col_id = myline.rstrip().split(",")
                    if int(k) in candidate_indices:
                        GOI_list.write(object_name+", "+candidate_status[np.where(np.array(candidate_indices)==int(k))[0]]+", "+myline)

    #"/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB/planet_detec_pyklip-S20141218-k100a7s4m3_KL20/t800g100nc/c_Eri-detectionLog_candidates.txt"

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

    with open(GOI_list_filename, 'r') as GOI_list:
        for myline in GOI_list:
            #print([myline.rstrip()])
            if (not myline.startswith("#")) and myline.rstrip():
                GOI_name, status, k,potential_planet,max_val_criter,x_max_pos,y_max_pos, row_id,col_id = myline.rstrip().split(",")
                if GOI_name.lower() == object_name.lower(): # case insensitive comparision
                    candidates_list.append((int(k),bool(potential_planet),float(max_val_criter),float(x_max_pos),float(y_max_pos), int(row_id),int(col_id)))


    row_m = np.floor(width/2.0)
    row_p = np.ceil(width/2.0)
    col_m = np.floor(width/2.0)
    col_p = np.ceil(width/2.0)

    for candidate in candidates_list:
        k,potential_planet,max_val_criter,x_max_pos,y_max_pos, k,l = candidate
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