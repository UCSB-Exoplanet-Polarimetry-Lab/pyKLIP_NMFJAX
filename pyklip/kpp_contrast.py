__author__ = 'JB'

import glob
import os
from copy import deepcopy

import pyklip.parallelized as parallelized
import pyklip.instruments.GPI as GPI
import pyklip.spectra_management as spec
import pyklip.fakes as fakes
from pyklip.klipPostProcessing import *

def generate_fakes(inputdir,
                   outputdir,
                   filename_filter = None,
                   N_repeat = 5,
                   init_sep_list = [0.0],
                   contrast_list = [1.0],
                   planet_spectrum_list = [None],
                   annuli_list = [7],
                   subsections_list = [4],
                   movement_list = [3],
                   klip_spectrum_list = [None],
                   numbasis = [1,10,20,100],
                   star_type = "G4",
                   compact_date = None,
                   filter_check = None,
                   satSpot_based_contrast = True,
                   GOI_list = None,
                   mute = False):

    if filename_filter is None:
        if compact_date is not None:
            filename_filter = "S"+compact_date+"S*_spdc_distorcorr.fits"
        else:
            filename_filter = "S*_spdc_distorcorr.fits"

    pyklip_dir = os.path.dirname(os.path.realpath(__file__))
    if planet_spectrum_list[0] is None:
        planet_spectrum_dir_list = [pyklip_dir+os.path.sep+"spectra"+os.path.sep+"g32ncflx"+os.path.sep+"t950g32nc.flx"]
    else:
        planet_spectrum_dir_list = []
        for plspec_it in planet_spectrum_list:
            planet_spectrum_dir_list.append(pyklip_dir+os.path.sep+"spectra"+os.path.sep+plspec_it)

    # generate database of klipped fakes
    filelist = glob.glob(inputdir+filename_filter)
    #print(inputdir+filename_filter)
    dataset = GPI.GPIData(filelist)

    prihdr = dataset.prihdrs[0]
    exthdr = dataset.exthdrs[0]

    if satSpot_based_contrast:
        spot_ratio = dataset.spot_ratio[dataset.prihdrs[0]['IFSFILT'].split('_')[1]]
        contrast_list = np.ndarray.tolist(spot_ratio*np.array(contrast_list))

    # Retrieve the filter used from the fits headers.
    filter = prihdr['IFSFILT'].split('_')[1]
    spot_ratio = dataset.spot_ratio[filter]

    if filter_check is not None:
        if filter_check != filter:
            raise ValueError("Filter in fits don't match expected filter")

    date = prihdr['DATE']
    compact_date = date.replace("-","")

    #Make sure it is the right PSF used
    PSF_cube_filename = glob.glob(inputdir+"*"+"S"+compact_date+"*-original_PSF_cube.fits")[0]
    hdulist = pyfits.open(PSF_cube_filename)
    dataset.psfs = hdulist[1].data
    hdulist.close()
    #radial_PSF_cube_filename = glob.glob(inputdir+"*"+"S"+compact_date+"*-original_radial_PSF_cube.fits")[0]
    #hdulist = pyfits.open(radial_PSF_cube_filename)
    #radial_psfs = hdulist[1].data /  (np.mean(dataset.spot_flux.reshape([dataset.spot_flux.shape[0]/37,37]), axis=0)[:, None, None])

    try:
        # OBJECT: keyword in the primary header with the name of the star.
        star_name = dataset.prihdrs[0]['OBJECT'].strip().replace (" ", "_")
    except:
        # If the object name could nto be found cal lit unknown_object
        star_name = "UNKNOWN_OBJECT"

    n_frames,ny,nx = dataset.input.shape
    try:
        # Retrieve the center of the image from the fits keyword.
        center = [dataset.exthdrs[0]['PSFCENTX'], dataset.exthdrs[0]['PSFCENTY']]
    except:
        # If the keywords could not be found.
        if not mute:
            print("Couldn't find PSFCENTX and PSFCENTY keywords.")
        center = [(nx-1)/2,(ny-1)/2]

    if star_name[0].isdigit():
        xml_star_name = "_"+star_name
    else:
        xml_star_name = star_name

    # Define the tree for the xml file
    xml_filename = outputdir+os.path.sep+star_name+"_"+compact_date+"_"+filter+"_"+'fakes.xml'
    xml_filename_list = glob.glob(xml_filename)
    if np.size(xml_filename_list) == 0:
        root = ET.Element("root")
        star_elt = ET.SubElement(root, xml_star_name)
    else:
        tree = ET.parse(xml_filename)
        root = tree.getroot()
        star_elt = root[0]

    cpy_dataset = deepcopy(dataset)
    for init_sep,fake_planet_contrast,planet_spectrum,annuli,subsections,movement,klip_spectrum in itertools.product(init_sep_list,
                                                                                          contrast_list,
                                                                                          planet_spectrum_dir_list,
                                                                                          annuli_list,
                                                                                          subsections_list,
                                                                                          movement_list,
                                                                                          klip_spectrum_list):

        # Calculate the radii of the annuli like in klip_adi_plus_sdi using the first image
        # We want to inject one planet per section where klip is independently applied.
        dims = dataset.input.shape
        x_grid, y_grid = np.meshgrid(np.arange(dims[2] * 1.0), np.arange(dims[1] * 1.0))
        nanpix = np.where(np.isnan(dataset.input[0]))
        OWA = np.sqrt(np.min((x_grid[nanpix] - dataset.centers[0][0]) ** 2 + (y_grid[nanpix] - dataset.centers[0][1]) ** 2))
        dr = float(OWA - dataset.IWA) / (annuli)
        delta_th = 360./subsections

        init_sep_list = [0.] # Should probably be random later when iterating
        fake_planet_contrast_list = [spot_ratio*0.3,spot_ratio*0.1,spot_ratio*0.03,spot_ratio*0.01]


        # Store the current local maximum information in the xml

        init_pa_list = np.arange(0,delta_th,delta_th/N_repeat) # Should probably be random later when iterating
        for init_pa in init_pa_list:

            # Extract the 37 points spectrum for the 37*N_cubes vector dataset.wvs.
            unique_wvs = np.unique(dataset.wvs)
            # Shoudl give numwaves=37
            numwaves = np.size(unique_wvs)
            # Number of cubes in dataset
            N_cubes = int(n_frames)/int(numwaves)
            # Peak value of the fake planet for each slice. To be define below.
            inputflux = np.zeros(n_frames)

            # Define the peak value of the fake planet for each slice depending if a star and a planet type is given.
            filter = dataset.prihdrs[0]['IFSFILT'].split('_')[1]
            # Interpolate a spectrum of the star based on its spectral type/temperature
            wv,star_sp = spec.get_star_spectrum(filter,star_type,None)
            # Interpolate the spectrum of the planet based on the given filename
            wv,planet_sp = spec.get_planet_spectrum(planet_spectrum,filter)
            ratio_spec_models = planet_sp/star_sp
            for k in range(N_cubes):
                inputflux[37*k:37*(k+1)] = dataset.spot_flux[37*k:37*(k+1)]*ratio_spec_models
                inputflux[37*k:37*(k+1)] *= fake_planet_contrast/dataset.spot_ratio[filter]*(np.mean(dataset.spot_flux[37*k:37*(k+1)])/np.mean(inputflux[37*k:37*(k+1)]))


            # Manage the shape of the PSF. the PSF is calculated from the sat spots.
            inputpsfs = np.tile(dataset.psfs,(N_cubes,1,1))
            for l in range(dataset.input.shape[0]):
                #peak value of the psfs is normalized to unity.
                inputpsfs[l,:,:] /= np.nanmax(inputpsfs[l,:,:])
                # then multiplied by the peak flux
                inputpsfs[l,:,:] *= inputflux[l]


            # Get parallactic angle of where to put fake planets
            # PSF_dist = 20 # Distance between PSFs. Actually length of an arc between 2 consecutive PSFs.
            # delta_pa = 180/np.pi*PSF_dist/radius
            #print(dataset.PAs)
            th_list = np.arange(-180.,180.1,delta_th) + init_pa - dataset.PAs[0]
            pa_list = fakes.covert_polar_to_image_pa(th_list, dataset.wcs[0])
            radii_list = np.array([dr * annuli_it + dataset.IWA + dr/2.for annuli_it in range(annuli-1)]) + init_sep
            pa_grid, radii_grid = np.meshgrid(pa_list,radii_list)
            th_grid, radii_grid = np.meshgrid(th_list,radii_list)
            pa_grid[range(1,annuli-1,2),:] += delta_th/2.
            th_grid[range(1,annuli-1,2),:] += delta_th/2.
            #print(th_list)
            #print(th_grid)
            #print(pa_grid)
            #print(radii_grid)

            planet_spectrum_str = planet_spectrum.split(os.path.sep)[-1].split(".")[0]
            if klip_spectrum is None:
                fileprefix = compact_date+"_"+filter+"_" + \
                         "_k{0}a{1}s{2}m{3}".format(np.max(numbasis),annuli,subsections,movement) + \
                         "_r{0:.2f}_a{1:.2f}_c{2:.2f}".format(init_sep,init_pa,-np.log10(fake_planet_contrast)) + \
                         planet_spectrum_str
            else:
                fileprefix = compact_date+"_"+filter + \
                         "_k{0}a{1}s{2}m{3}".format(np.max(numbasis),annuli,subsections,movement) + \
                         klip_spectrum + \
                         "_r{0:.2f}_a{1:.2f}_c{2:.2f}".format(init_sep,init_pa,-np.log10(fake_planet_contrast)) + \
                         planet_spectrum_str

            file_info = ET.SubElement(star_elt,"file",
                          fileprefix = fileprefix,
                          init_sep = str(init_sep),
                          fake_planet_contrast_log = str(-np.log10(fake_planet_contrast)),
                          planet_spectrum = planet_spectrum_str,
                          init_pa = str(init_pa),
                          outputdir = str(outputdir),
                          N_repeat = str(N_repeat),
                          star_type = star_type,
                          filter = filter,
                          compact_date = compact_date,
                          maxKL = str(np.max(numbasis)),
                          annuli=str(annuli),
                          subsections=str(subsections),
                          movement=str(movement),
                          klip_spectrum=str(klip_spectrum))

            # Loop for injecting fake planets. One planet per section of the image.
            # Too many hard-coded parameters because still work in progress.
            for pa, radius in itertools.izip(np.reshape(pa_grid,np.size(pa_grid)),np.reshape(radii_grid,np.size(radii_grid))):
                x_max_pos = radius*np.cos(np.radians(pa))
                y_max_pos = radius*np.sin(np.radians(pa))
                if not mute:
                    print("injecting planet for "+fileprefix+". Position ("+str(x_max_pos)+","+str(y_max_pos)+")")
                ET.SubElement(file_info,"fake",
                              pa = str(pa),
                              radius = str(radius),
                              col_centroid= str(x_max_pos+center[0]),
                              row_centroid= str(y_max_pos+center[1]),
                              x_max_pos=str(x_max_pos),
                              y_max_pos=str(y_max_pos))
                fakes.inject_planet(dataset.input, dataset.centers, inputpsfs, dataset.wcs, radius, pa)


            parallelized.klip_dataset(dataset,
                                      outputdir=outputdir,
                                      fileprefix=fileprefix,
                                      annuli=annuli, subsections=subsections, movement=movement,
                                      numbasis=numbasis, calibrate_flux=True,
                                      mode="ADI+SDI",lite=False,
                                      spectrum=klip_spectrum)
            dataset = deepcopy(cpy_dataset)

    # Save the xml file from the tree
    tree = ET.ElementTree(root)
    tree.write(xml_filename)