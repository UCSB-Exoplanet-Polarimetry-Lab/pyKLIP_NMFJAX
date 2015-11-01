__author__ = 'JB'

import glob
import os
from copy import deepcopy

from astropy.io import ascii

import pyklip.parallelized as parallelized
import pyklip.fakes as fakes
import pyklip.kpp_std as kpp_std
from pyklip.kpp.klipPostProcessing import *


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
                   use_stddev_as_first_guess = False,
                   fakes_for_rec_op_char = False,
                   GOI_list_folder = None,
                   mute = False,
                   N_cores = None,
                   start_and_stop = 'start_and_stop.txt'):

    if start_and_stop is not None:
        try:
            start_and_stop_txtfile = open(outputdir+os.path.sep+start_and_stop, 'r+')
            klipPara_loop_count_done = int(start_and_stop_txtfile.readline())
            contrast_point_loop_count_done = int(start_and_stop_txtfile.readline())
            per_file_loop_count_done = int(start_and_stop_txtfile.readline())
            print(klipPara_loop_count_done,contrast_point_loop_count_done,per_file_loop_count_done)
        except:
            start_and_stop_txtfile = open(outputdir+os.path.sep+start_and_stop, 'w+')
            klipPara_loop_count_done = 0
            contrast_point_loop_count_done = 0
            per_file_loop_count_done = 0
    else:
        klipPara_loop_count_done = 0
        contrast_point_loop_count_done = 0
        per_file_loop_count_done = 0



    if fakes_for_rec_op_char:
        N_repeat = 1
        init_sep_list = [0.0] #[0.0,5.0]
        contrast_list = 10**(np.linspace(-1.,1.,10))
        planet_spectrum_list=["g32ncflx"+os.path.sep+"t950g32nc.flx"]
        f_radial_cont = interp1d(np.array([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,2.0])/0.01414,
                                10**np.array([-4,-4.5,-5.5,-6.,-6.,-6.,-6.,-6.,-6.]),#/(dataset.spot_ratio[filter]/np.mean(dataset.spot_flux)),
                                kind='linear',bounds_error=False, fill_value=np.nan)


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


    if GOI_list_folder is not None:
        x_real_object_list,y_real_object_list = get_pos_known_objects(prihdr,exthdr,GOI_list_folder,xy = True)


    # Retrieve the filter used from the fits headers.
    filter = prihdr['IFSFILT'].split('_')[1]
    spot_ratio = dataset.spot_ratio[filter]

    if filter_check is not None:
        if filter_check != filter:
            raise ValueError("Filter in fits don't match expected filter")

    date = prihdr['DATE']
    compact_date = date.replace("-","")

    PSF_cube_filename = glob.glob(inputdir+"pyklip-S"+compact_date+"-*-original_radial_PSF_cube.fits")[0]
    dst = outputdir+os.path.sep+"S"+compact_date+"-original_radial_PSF_cube.fits"
    if not mute:
        print("Copying " + PSF_cube_filename + " to " + dst)
    shutil.copyfile(PSF_cube_filename, dst)

    #Make sure it is the right PSF used
    hdulist = pyfits.open(PSF_cube_filename)
    dataset.psfs = hdulist[1].data
    hdulist.close()
    nl,ny_PSF,nx_PSF = dataset.psfs.shape
    sat_spot_spec = np.nanmax(dataset.psfs,axis=(1,2))
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
    klipPara_loop_count = 0
    for annuli,subsections,movement,klip_spectrum in itertools.product(annuli_list,
                                                                      subsections_list,
                                                                      movement_list,
                                                                      klip_spectrum_list):
        klipPara_loop_count = klipPara_loop_count+1
        if klipPara_loop_count < klipPara_loop_count_done:#
            continue

        if klip_spectrum is None:
            fileprefix_ref = "noFakeRef_"+compact_date+"_"+filter+"_" + \
                     "k{0}a{1}s{2}m{3:1.2f}".format(np.max(numbasis),annuli,subsections,movement)
        else:
            fileprefix_ref = "noFakeRef_"+compact_date+"_"+filter+"_" + \
                     "k{0}a{1}s{2}m{3:1.2f}".format(np.max(numbasis),annuli,subsections,movement) + \
                     klip_spectrum

        ref_exist = True
        for KL_it in numbasis:
            check_filename = outputdir + fileprefix_ref+"-KL"+str(KL_it)+"-speccube.fits"
            check_filename_glob = glob.glob(check_filename)
            if len(check_filename_glob) == 0:
                ref_exist = False

        if not ref_exist:
            parallelized.klip_dataset(dataset,
                                      outputdir=outputdir,
                                      fileprefix=fileprefix_ref,
                                      annuli=annuli, subsections=subsections, movement=movement,
                                      numbasis=numbasis, calibrate_flux=False,
                                      mode="ADI+SDI",lite=False,
                                      spectrum=klip_spectrum,
                                      numthreads=N_cores)
            dataset = deepcopy(cpy_dataset)
        else:
            if not mute:
                print("Skipping ref computation. Already exist.")

        if use_stddev_as_first_guess:
            # Calculate standard deviation with respect to separation
            check_filename = outputdir + fileprefix_ref+"-KL"+str(20)+"-speccube.fits"
            check_filename_glob = glob.glob(check_filename)
            # Load klip files without fakes
            hdulist = pyfits.open(check_filename_glob[0])
            noFake_cube = hdulist[1].data
            exthdr = hdulist[1].header
            prihdr = hdulist[0].header
            center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
            hdulist.close()
            if GOI_list_folder is not None:
                noFake_cube = mask_known_objects(np.nanmean(noFake_cube,axis=0),prihdr,exthdr,GOI_list_folder, mask_radius = 7)
            radial_std,std_r_samp,tmp = kpp_std.radialStd(noFake_cube,2,5,centroid = center,rejection = False)
            #print(std_r_samp,radial_std)
            #plt.figure(1)
            #plt.plot(std_r_samp[np.where(np.isfinite(radial_std))],radial_std[np.where(np.isfinite(radial_std))])
            #ax = plt.gca()
            #ax.set_yscale('log')
            #plt.show()
            where_non_nans =np.where(np.isfinite(radial_std))
            f_radial_std = interp1d(std_r_samp[where_non_nans],
                                    radial_std[where_non_nans],#/(dataset.spot_ratio[filter]/np.mean(dataset.spot_flux)),
                                    kind='cubic',bounds_error=False, fill_value=np.nan)

        contrast_point_loop_count = 0
        for init_sep,fake_planet_contrast,planet_spectrum in itertools.product(init_sep_list,
                                                                              contrast_list,
                                                                              planet_spectrum_dir_list):
            contrast_point_loop_count = contrast_point_loop_count+1
            if contrast_point_loop_count < contrast_point_loop_count_done:#
                continue

            # Calculate the radii of the annuli like in klip_adi_plus_sdi using the first image
            # We want to inject one planet per section where klip is independently applied.
            dims = dataset.input.shape
            x_grid, y_grid = np.meshgrid(np.arange(dims[2] * 1.0), np.arange(dims[1] * 1.0))
            nanpix = np.where(np.isnan(dataset.input[0]))
            OWA = np.sqrt(np.min((x_grid[nanpix] - dataset.centers[0][0]) ** 2 + (y_grid[nanpix] - dataset.centers[0][1]) ** 2))
            dr = float(OWA - dataset.IWA) / (annuli)
            delta_th = 360./subsections

            init_sep_list = [0.] # Should probably be random later when iterating


            # Store the current local maximum information in the xml

            init_pa_list = np.arange(0,delta_th,delta_th/N_repeat) # Should probably be random later when iterating
            per_file_loop_count = 0
            for init_pa in init_pa_list:
                per_file_loop_count = per_file_loop_count+1
                if per_file_loop_count < per_file_loop_count_done:#
                    continue

                start_and_stop_txtfile.seek(0)
                start_and_stop_txtfile.write(str(klipPara_loop_count)+"\n")
                start_and_stop_txtfile.write(str(contrast_point_loop_count)+"\n")
                start_and_stop_txtfile.write(str(per_file_loop_count)+"\n")

                # Extract the 37 points spectrum for the 37*N_cubes vector dataset.wvs.
                unique_wvs = np.unique(dataset.wvs)
                # Shoudl give numwaves=37
                numwaves = np.size(unique_wvs)
                # Number of cubes in dataset
                N_cubes = int(n_frames)/int(numwaves)

                # Define the peak value of the fake planet for each slice depending if a star and a planet type is given.
                filter = dataset.prihdrs[0]['IFSFILT'].split('_')[1]
                # Interpolate a spectrum of the star based on its spectral type/temperature
                wv,star_sp = spec.get_star_spectrum(filter,star_type,None)
                # Interpolate the spectrum of the planet based on the given filename
                wv,planet_sp = spec.get_planet_spectrum(planet_spectrum,filter)
                ratio_spec_models = planet_sp/star_sp
                mean_satSpot = np.mean(sat_spot_spec)#np.reshape(dataset.spot_flux,(N_cubes,37))
                unit_mean_spec = (sat_spot_spec*ratio_spec_models)/np.mean((sat_spot_spec*ratio_spec_models))

                stamp_x_grid, stamp_y_grid = np.meshgrid(np.arange(0,nx_PSF,1)-nx_PSF/2,np.arange(0,ny_PSF,1)-ny_PSF/2)
                r_stamp = np.sqrt((stamp_x_grid)**2 +(stamp_y_grid)**2)
                stamp_mask = np.ones((ny_PSF,nx_PSF))
                stamp_mask[np.where(r_stamp > 3.)] = np.nan
                stamp_cube_mask = np.tile(stamp_mask[None,:,:],(nl,1,1))

                # Get parallactic angle of where to put fake planets
                # PSF_dist = 20 # Distance between PSFs. Actually length of an arc between 2 consecutive PSFs.
                # delta_pa = 180/np.pi*PSF_dist/radius
                th_list = np.arange(-180.,180.-0.01,delta_th) + init_pa - dataset.PAs[0]
                pa_list = fakes.covert_polar_to_image_pa(th_list, dataset.wcs[0])
                radii_list = np.array([dr * annuli_it + dataset.IWA + dr/2.for annuli_it in range(annuli-1)]) + init_sep
                pa_grid, radii_grid = np.meshgrid(pa_list,radii_list)
                th_grid, radii_grid = np.meshgrid(th_list,radii_list)
                pa_grid[range(1,annuli-1,2),:] += delta_th/2.
                th_grid[range(1,annuli-1,2),:] += delta_th/2.

                planet_spectrum_str = planet_spectrum.split(os.path.sep)[-1].split(".")[0]
                if klip_spectrum is None:
                    fileprefix = compact_date+"_"+filter+"_" + \
                             "k{0}a{1}s{2}m{3:1.2f}".format(np.max(numbasis),annuli,subsections,movement)+"_" + \
                             "r{0:.2f}_a{1:.2f}_c{2:.2f}".format(init_sep,init_pa,np.log10(fake_planet_contrast)) + \
                             "cm"+str(1*int(satSpot_based_contrast)+2*int(use_stddev_as_first_guess)+3*int(fakes_for_rec_op_char)) +\
                             planet_spectrum_str
                else:
                    fileprefix = compact_date+"_"+filter+"_" + \
                             "k{0}a{1}s{2}m{3:1.2f}".format(np.max(numbasis),annuli,subsections,movement) + \
                             klip_spectrum+"_" + \
                             "r{0:.2f}_a{1:.2f}_c{2:.2f}".format(init_sep,init_pa,np.log10(fake_planet_contrast)) + \
                             "cm"+str(1*int(satSpot_based_contrast)+2*int(use_stddev_as_first_guess)+3*int(fakes_for_rec_op_char)) +\
                             planet_spectrum_str

                
                print(fileprefix)
                file_info = ET.SubElement(star_elt,"file",
                              fileprefix = fileprefix,
                              init_sep = str(init_sep),
                              fake_planet_contrast_log = str(np.log10(fake_planet_contrast)),
                              satSpot_based_contrast = str(satSpot_based_contrast),
                              use_stddev_as_first_guess = str(use_stddev_as_first_guess),
                              fakes_for_rec_op_char = str(fakes_for_rec_op_char),
                              planet_spectrum = planet_spectrum_str,
                              init_pa = str(init_pa),
                              outputdir = str(outputdir),
                              N_repeat = str(N_repeat),
                              star_type = star_type,
                              filter = filter,
                              compact_date = compact_date,
                              numbasis = str(numbasis),
                              maxKL = str(np.max(numbasis)),
                              annuli=str(annuli),
                              subsections=str(subsections),
                              movement=str(movement),
                              klip_spectrum=str(klip_spectrum))

                # Loop for injecting fake planets. One planet per section of the image.
                # Too many hard-coded parameters because still work in progress.
                for pa, radius in itertools.izip(np.reshape(pa_grid,np.size(pa_grid)),np.reshape(radii_grid,np.size(radii_grid))):
                    x_max_pos = radius*np.cos(np.radians(90+pa))
                    y_max_pos = radius*np.sin(np.radians(90+pa))
                    if GOI_list_folder is not None:
                        too_close = False
                        for x_real_object,y_real_object  in zip(x_real_object_list,y_real_object_list):
                            if (x_max_pos-x_real_object)**2+(y_max_pos-y_real_object)**2 < 15.*15.:
                                too_close = True
                                if not mute:
                                    print("Skipping planet. Real object too close.")
                                break
                        if too_close:
                            continue
                    if not mute:
                        print("injecting planet for "+fileprefix+". Position ("+str(x_max_pos)+","+str(y_max_pos)+")")

                    # Peak value of the fake planet for each slice. To be define below.
                    #inputflux = np.zeros(n_frames)
                    #print(sat_spot_spec)
                    #print(np.reshape(dataset.spot_flux,(N_cubes,37)))
                    if use_stddev_as_first_guess:
                        inputflux = fake_planet_contrast*(f_radial_std(radius))/np.sqrt(N_cubes)*unit_mean_spec
                    elif satSpot_based_contrast:
                        inputflux = fake_planet_contrast*mean_satSpot*unit_mean_spec
                    elif fakes_for_rec_op_char:
                        inputflux = fake_planet_contrast*f_radial_cont(radius)*mean_satSpot/dataset.spot_ratio[filter]*unit_mean_spec
                    else:
                        inputflux = fake_planet_contrast*mean_satSpot/dataset.spot_ratio[filter]*unit_mean_spec
                    inputflux_tiled = np.tile(inputflux,(N_cubes))
                    inputpsfs = np.tile(dataset.psfs,(N_cubes,1,1))
                    inputpsfs_max_vec = np.nanmax(inputpsfs,axis=(1,2))
                    inputpsfs = inputpsfs*(inputflux_tiled/inputpsfs_max_vec)[:,None,None]
                    contrast_log = np.log10(np.mean(inputflux)/mean_satSpot*dataset.spot_ratio[filter])

                    aperture_flux = np.nansum(inputpsfs[0:nl,:,:]*stamp_cube_mask,axis=(0,1,2))


                    center = [140,140] # to clean
                    ET.SubElement(file_info,"fake",
                                  pa = str(pa),
                                  radius = str(radius),
                                  contrast_log = str(contrast_log),
                                  aperture_flux = str(aperture_flux),
                                  col_centroid= str(x_max_pos+center[0]),
                                  row_centroid= str(y_max_pos+center[1]),
                                  x_max_pos=str(x_max_pos),
                                  y_max_pos=str(y_max_pos))
                    print(pa,radius,x_max_pos,y_max_pos,x_max_pos+center[0],y_max_pos+center[1])
                    #print(inputpsfs.shape,np.sum(inputpsfs,axis=(1,2)).shape)
                    #print(np.sum(inputpsfs,axis=(1,2)))

                    fakes.inject_planet(dataset.input, dataset.centers, inputpsfs, dataset.wcs, radius, pa)

                parallelized.klip_dataset(dataset,
                                          outputdir=outputdir,
                                          fileprefix=fileprefix,
                                          annuli=annuli, subsections=subsections, movement=movement,
                                          numbasis=numbasis, calibrate_flux=False,
                                          mode="ADI+SDI",lite=False,
                                          spectrum=klip_spectrum,
                                          numthreads=N_cores)

                if start_and_stop_txtfile is not None:
                    # Save the xml file from the tree
                    tree = ET.ElementTree(root)
                    tree.write(xml_filename)
                    start_and_stop_txtfile.seek(0)
                    start_and_stop_txtfile.write(str(klipPara_loop_count)+"\n")
                    start_and_stop_txtfile.write(str(contrast_point_loop_count)+"\n")
                    start_and_stop_txtfile.write(str(per_file_loop_count+1)+"\n")

                dataset = deepcopy(cpy_dataset)

            if start_and_stop_txtfile is not None:
                start_and_stop_txtfile.seek(0)
                start_and_stop_txtfile.write(str(klipPara_loop_count)+"\n")
                start_and_stop_txtfile.write(str(contrast_point_loop_count+1)+"\n")
                start_and_stop_txtfile.write(str(0)+"\n")
        if start_and_stop_txtfile is not None:
            start_and_stop_txtfile.seek(0)
            start_and_stop_txtfile.write(str(klipPara_loop_count+1)+"\n")
            start_and_stop_txtfile.write(str(0)+"\n")
            start_and_stop_txtfile.write(str(0)+"\n")


    if start_and_stop_txtfile is None:
        # Save the xml file from the tree
        tree = ET.ElementTree(root)
        tree.write(xml_filename)



def analysis_contrast(inputdir,
                      annuli = 7,
                      subsections = 4,
                      movement = 3,
                      klip_spectrum = None,
                      N_KL = 10,
                      N_KL_max = 100,
                      contrast_metric = "matchedFilter",
                      spectrum_metric = None,
                      N = 3000,
                      stamp_width = 20,
                      mask_radius = 7,
                      dir_for_contrast = None,
                      GOI_list_folder = None,
                      mute = False,
                      N_cores = None):


    radial_PSF_cube_filename = glob.glob(inputdir+os.path.sep+"S*-original_radial_PSF_cube.fits")[0]
    # generate normalized PSF
    hdulist = pyfits.open(radial_PSF_cube_filename)
    radial_psfs = hdulist[1].data

    nl,ny_PSF,nx_PSF = radial_psfs.shape

    stamp_x_grid, stamp_y_grid = np.meshgrid(np.arange(0,nx_PSF,1)-nx_PSF/2,np.arange(0,ny_PSF,1)-ny_PSF/2)
    r_stamp = np.sqrt((stamp_x_grid)**2 +(stamp_y_grid)**2)
    stamp_mask = np.ones((ny_PSF,nx_PSF))
    stamp_mask[np.where(r_stamp > 3.)] = np.nan
    #stamp_cube_mask = np.tile(stamp_mask[None,:,:],(nl,1,1))

    if klip_spectrum is None:
        fileprefix_ref = "noFakeRef_"+"*"+"_"+"*"+"_" + \
                 "k{0}a{1}s{2}m{3:1.2f}".format(N_KL_max,annuli,subsections,movement)
    else:
        fileprefix_ref = "noFakeRef_"+"*"+"_"+"*"+"_" + \
                 "k{0}a{1}s{2}m{3:1.2f}".format(N_KL_max,annuli,subsections,movement) + \
                 klip_spectrum

    filename_ref= inputdir + fileprefix_ref+"-KL"+str(N_KL)+"-speccube.fits"
    print(filename_ref)
    filename_ref = glob.glob(filename_ref)[0]
    #noFakeRef_20141218_H__k50a7s4m3-KL10-speccube

    #hdulist = pyfits.open(ori_cube_filename)
    #ori_cube = hdulist[1].data
    #hdulist.close()

    pyklip_dir = os.path.dirname(os.path.realpath(__file__))
    if spectrum_metric is None:
        spectrum_metric = pyklip_dir+os.path.sep+"spectra"+os.path.sep+"g32ncflx"+os.path.sep+"t950g32nc.flx"
    else:
        spectrum_metric = pyklip_dir+os.path.sep+"spectra"+os.path.sep+spectrum_metric

    spectrum_metric_name = spectrum_metric.split(os.path.sep)[-1].split(".")[0]
    planet_detec_folder = inputdir+os.path.sep+"planet_detec_"+fileprefix_ref+"-KL"+str(N_KL)+os.path.sep+spectrum_metric_name+os.path.sep

    flatCube_glob = glob.glob(planet_detec_folder+"*"+"-flatCube.fits")
    matchedFilter_glob = glob.glob(planet_detec_folder+"*"+"-matchedFilter.fits")
    shape_glob = glob.glob(planet_detec_folder+"*"+"-shape.fits")
    if len(flatCube_glob) == 0 or \
       len(matchedFilter_glob) == 0 or \
       len(shape_glob) == 0:
        planet_detection_in_dir_per_file(filename_ref,
                                          metrics = ["shape", "matchedFilter","weightedFlatCube"],
                                          directory = inputdir,
                                          outputDir = inputdir,
                                          spectrum_model = [spectrum_metric],
                                          star_type = "G5",
                                          star_temperature = None,
                                          user_defined_PSF_cube = radial_PSF_cube_filename,
                                          metrics_only = True,
                                          planet_detection_only = False,
                                          mute = False,
                                          threads = True,
                                          GOI_list_folder = GOI_list_folder,
                                          overwrite_metric = False,
                                          overwrite_stat = False,
                                          proba_using_mask_per_pixel = False,
                                          SNR = True,
                                          probability = True,
                                          detection_metric = None,
                                          N_cores=N_cores)
    else:
        if not mute:
            print("found planet detec for ref klip")

    if "matchedFilter" in contrast_metric:
        metric_glob = glob.glob(planet_detec_folder+"*"+"-matchedFilter.fits")
        hdulist = pyfits.open(metric_glob[0])
    elif "shape" in contrast_metric:
        metric_glob = glob.glob(planet_detec_folder+"*"+"-shape.fits")
        hdulist = pyfits.open(metric_glob[0])
    elif "weightedFlatCube" in contrast_metric:
        metric_glob = glob.glob(planet_detec_folder+"*"+"-weightedFlatCube.fits")
        hdulist = pyfits.open(metric_glob[0])
    ori_metric_map = hdulist[1].data
    exthdr = hdulist[1].header
    prihdr = hdulist[0].header
    hdulist.close()

    #flatCube_glob = glob.glob(planet_detec_folder+"*"+"-flatCube.fits")
    #hdulist = pyfits.open(flatCube_glob[0])
    #ori_flatCube = hdulist[1].data
    #hdulist.close()

    try:
        # OBJECT: keyword in the primary header with the name of the star.
        star_name = prihdr['OBJECT'].strip().replace (" ", "_")
    except:
        # If the object name could nto be found cal lit unknown_object
        star_name = "UNKNOWN_OBJECT"

    filter = prihdr['IFSFILT'].split('_')[1]

    date = prihdr['DATE']
    compact_date = date.replace("-","")

    xml_filename = inputdir+os.path.sep+star_name+"_"+compact_date+"_"+filter+"_"+'fakes.xml'
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    star_elt = root[0]
    contrast_and_radius_list = []
    proba_table = []
    flux_ratio_table = []
    rec_op_char_proba_vector_shaped = []
    for file_elt in star_elt:
        # Get the information of the candidate from the element attributes
        curr_fileprefix = file_elt.attrib["fileprefix"]
        curr_fake_planet_contrast_log = float(file_elt.attrib["fake_planet_contrast_log"])
        curr_outputdir = file_elt.attrib["outputdir"]
        curr_maxKL = int(file_elt.attrib["maxKL"])
        curr_annuli = int(file_elt.attrib["annuli"])
        curr_subsections = int(file_elt.attrib["subsections"])
        curr_movement = float(file_elt.attrib["movement"])
        curr_klip_spectrum = file_elt.attrib["klip_spectrum"]
        fakes_for_rec_op_char = bool(file_elt.attrib["fakes_for_rec_op_char"])
        print(curr_fileprefix)

        #print(curr_maxKL,curr_N_KL,curr_annuli,curr_subsections,curr_movement,curr_klip_spectrum)
        #print(curr_maxKL,N_KL,annuli,subsections,movement,str(klip_spectrum))
        if N_KL_max != N_KL_max or \
            curr_annuli != annuli or \
            curr_subsections != subsections or \
            curr_movement != movement or \
            curr_klip_spectrum != str(klip_spectrum):
            continue

        fakePl_filelist = glob.glob(inputdir+curr_fileprefix+"*KL"+str(N_KL)+"-speccube.fits")
        #fakePl_filelist = glob.glob(inputdir+curr_fileprefix+"-speccube.fits")
        fakePl_filename = fakePl_filelist[0]

        #hdulist = pyfits.open(fakePl_filename)
        #cube = hdulist[1].data
        #hdulist.close()
        planet_detec_folder = inputdir+os.path.sep+"planet_detec_"+curr_fileprefix+"-KL"+str(N_KL)+os.path.sep+spectrum_metric_name+os.path.sep
        #planet_detec_folder = inputdir+os.path.sep+"planet_detec_"+curr_fileprefix+os.path.sep+spectrum_metric_name+os.path.sep

        print(planet_detec_folder+"*"+"-flatCube.fits")
        flatCube_glob = glob.glob(planet_detec_folder+"*"+"-flatCube.fits")
        matchedFilter_glob = glob.glob(planet_detec_folder+"*"+"-matchedFilter.fits")
        shape_glob = glob.glob(planet_detec_folder+"*"+"-shape.fits")
        if len(flatCube_glob) == 0 or \
           len(matchedFilter_glob) == 0 or \
           len(shape_glob) == 0:
            print("calculating")
            planet_detection_in_dir_per_file(fakePl_filename,
                                          metrics = ["shape", "matchedFilter","weightedFlatCube"],
                                          directory = inputdir,
                                          outputDir = inputdir,
                                          spectrum_model = [spectrum_metric],
                                          star_type = "G5",
                                          star_temperature = None,
                                          user_defined_PSF_cube = radial_PSF_cube_filename,
                                          metrics_only = True,
                                          planet_detection_only = False,
                                          mute = False,
                                          threads = True,
                                          GOI_list_folder = GOI_list_folder,
                                          overwrite_metric = False,
                                          overwrite_stat = False,
                                          proba_using_mask_per_pixel = False,
                                          SNR = False,
                                          probability = False,
                                          detection_metric = None,
                                             N_cores = N_cores)
            #sleep(0.1)
        else:
            print("skipping")

        if "matchedFilter" in contrast_metric:
            fakePl_metric_glob = glob.glob(planet_detec_folder+"*"+"-matchedFilter.fits")
            hdulist = pyfits.open(fakePl_metric_glob[0])
        elif "shape" in contrast_metric:
            fakePl_metric_glob = glob.glob(planet_detec_folder+"*"+"-shape.fits")
            hdulist = pyfits.open(fakePl_metric_glob[0])
        elif "weightedFlatCube" in contrast_metric:
            fakePl_metric_glob = glob.glob(planet_detec_folder+"*"+"-weightedFlatCube.fits")
            hdulist = pyfits.open(fakePl_metric_glob[0])
        fakePl_metric_map = hdulist[1].data
        hdulist.close()
        flatCube_glob = glob.glob(planet_detec_folder+"*"+"-flatCube.fits")
        hdulist = pyfits.open(flatCube_glob[0])
        fakePl_flatCube = hdulist[1].data
        hdulist.close()

        ## to remove
        #fakePl_shape_map = fakePl_flatCube

        if 0:
            plt.figure(1)
            plt.imshow(fakePl_metric_map,interpolation="nearest")
            plt.show()

        # If GOI_list is not None. Mask the known objects from the image that will be used for calculating the
        # PDF. This masked image is given separately to the probability calculation function.
        if GOI_list_folder is not None:
            ori_metric_map_without_planet = mask_known_objects(ori_metric_map,prihdr,exthdr,GOI_list_folder, mask_radius = 7)
        else:
            ori_metric_map_without_planet = ori_metric_map

        if 0:
            plt.figure(1)
            plt.imshow(ori_metric_map_without_planet)
            plt.show()

        ny,nx = ori_metric_map.shape
        try:
            # Retrieve the center of the image from the fits keyword.
            center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
        except:
            # If the keywords could not be found.
            if not mute:
                print("Couldn't find PSFCENTX and PSFCENTY keywords.")
            center = [(nx-1)/2,(ny-1)/2]
        x_cen, y_cen = center

        IWA,OWA,inner_mask,outer_mask = get_occ(ori_metric_map, centroid = center)

        image_without_planet_mask = np.ones((ny,nx))
        image_without_planet_mask[np.where(np.isnan(ori_metric_map_without_planet))] = 0

        # Build the x and y coordinates grids
        x_grid, y_grid = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
        # Calculate the radial distance of each pixel
        r_grid = abs(x_grid +y_grid*1j)
        th_grid = np.arctan2(x_grid,y_grid)

        r_min_firstZone,r_max_firstZone = (IWA,np.sqrt(N/np.pi+IWA**2))
        r_limit_firstZone = (r_min_firstZone + r_max_firstZone)/2.
        r_min_lastZone,r_max_lastZone = (OWA,np.max([ny,nx]))
        r_limit_lastZone = OWA - N/(4*np.pi*OWA)
        #(r_limit_firstZone,r_min_firstZone,r_max_firstZone)
        #(r_limit_lastZone,r_min_lastZone,r_max_lastZone


        for fake_elt in file_elt:
            # Get the information of the candidate from the element attributes
            x_max_pos = float(fake_elt.attrib["x_max_pos"])
            y_max_pos = float(fake_elt.attrib["y_max_pos"])
            col_centroid = float(fake_elt.attrib["col_centroid"])
            row_centroid = float(fake_elt.attrib["row_centroid"])
            pa = float(fake_elt.attrib["pa"])
            radius = float(fake_elt.attrib["radius"])
            contrast_log = float(fake_elt.attrib["contrast_log"])
            aperture_flux_ori = float(fake_elt.attrib["aperture_flux"])
            #print(pa,radius)


            if (contrast_log,radius) in contrast_and_radius_list:
                curr_proba_list = proba_table[contrast_and_radius_list.index((contrast_log,radius))]
                flux_ratio_list = flux_ratio_table[contrast_and_radius_list.index((contrast_log,radius))]

                #print(contrast_and_radius_list.index((fake_planet_contrast,radius)))
            else:
                contrast_and_radius_list.append((contrast_log,radius))
                curr_proba_list = []
                proba_table.append(curr_proba_list)
                flux_ratio_list = []
                flux_ratio_table.append(flux_ratio_list)

            #print(contrast_and_radius_list)
            #print(proba_table)
            #print(curr_proba_list)

            #print(radius,pa)
            #print(x_max_pos,y_max_pos,col_centroid,row_centroid)
            k = round(row_centroid)
            l = round(col_centroid)



            stamp = fakePl_flatCube[(k-np.floor(stamp_width/2.)):(k+np.ceil(stamp_width/2.)),
                                                        (l-np.floor(stamp_width/2.)):(l+np.ceil(stamp_width/2.))]
            aperture_flux_fake = np.nansum(stamp*stamp_mask,axis=(0,1))*nl
            aperture_flux_ratio = float(aperture_flux_fake)/float(aperture_flux_ori)
            #print(aperture_flux_fake,aperture_flux_ori,aperture_flux_ratio)
            flux_ratio_list.append(aperture_flux_ratio)

            x = x_grid[(k,l)]
            y = y_grid[(k,l)]
            #print(x,y)
            r = r_grid[(k,l)]

            if "SNR" in contrast_metric or "Raw" in contrast_metric:
                Dr= 2
                r_min,r_max = (r-Dr, r+Dr)
            else:
                if r < r_limit_firstZone:
                    #Calculate stat for pixels close to IWA
                    r_min,r_max = r_min_firstZone,r_max_firstZone
                elif r > r_limit_lastZone:
                    r_min,r_max = r_min_lastZone,r_max_lastZone
                else:
                    dr = N/(4*np.pi*r)
                    r_min,r_max = (r-dr, r+dr)

            where_ring = np.where((r_min< r_grid) * (r_grid < r_max) * image_without_planet_mask)
            where_ring_masked = np.where((((x_grid[where_ring]-x)**2 +(y_grid[where_ring]-y)**2) > mask_radius*mask_radius))
            #print(np.shape(where_ring_masked[0]))

            data = ori_metric_map_without_planet[(where_ring[0][where_ring_masked],where_ring[1][where_ring_masked])]
            print(r_min,r_max,data.size)

            #
            #print(radius, contrast_log,tmp_std,fakePl_shape_map[k,l] )
            if "SNR" in contrast_metric:
                tmp_std = np.nanstd(data)
                crit = np.nanmax(fakePl_metric_map[k-1:k+2,l-1:l+2])/tmp_std
            elif "Raw" in contrast_metric:
                crit = fakePl_metric_map[k-1:k+2,l-1:l+2]
            else:
                cdf_model, pdf_model, sampling, im_histo, center_bins  = get_cdf_model(data)
                cdf_fit = interp1d(sampling,cdf_model,kind = "linear",bounds_error = False, fill_value=1.0)
                #curr_proba_list.append(-np.log10(1-cdf_fit(fakePl_shape_map[k,l])))
                crit = -np.log10(1-cdf_fit(np.nanmax(fakePl_metric_map[k-1:k+2,l-1:l+2])))
                #print(radius, contrast_log,-np.log10(1-cdf_fit(fakePl_shape_map[k,l])))

            curr_proba_list.append(crit)
            if fakes_for_rec_op_char:
                rec_op_char_proba_vector_shaped.append(crit)

            if 0 and curr_movement == 3.0 and N_KL == 20 and radius == 36.4285084597 and contrast_log >= -5.5:
                print(shape_glob[0])
                print(contrast_log)
                #print(shape_glob_ref,shape_glob)
                #print(data)
                print(fakePl_metric_map[k,l])
                #print(-np.log10(1-cdf_fit(fakePl_metric_map[k,l])))
                stamp = fakePl_metric_map[(k-np.floor(stamp_width/2.)):(k+np.ceil(stamp_width/2.)),
                                        (l-np.floor(stamp_width/2.)):(l+np.ceil(stamp_width/2.))]
                #stamp = fakePl_flatCube[(k-np.floor(stamp_width/2.)):(k+np.ceil(stamp_width/2.)),
                #                        (l-np.floor(stamp_width/2.)):(l+np.ceil(stamp_width/2.))]
                plt.figure(1)
                plt.imshow(stamp,interpolation="nearest")
                plt.colorbar()
                plt.show()

    contrast_and_radius_list = np.array(contrast_and_radius_list)
    N_points = contrast_and_radius_list.shape[0]
    #proba_table = np.array(proba_table)
    ratio_detected_planets_list = np.zeros(N_points)
    N_samples_list = np.zeros(N_points)
    total_proba_list = np.zeros(N_points)
    mean_flux_ratio_list = np.zeros(N_points)
    stddev_flux_ratio_list = np.zeros(N_points)
    #all_proba_vector_shaped = []
    for id,prob_row in enumerate(proba_table):
        # prob_row is the list of proba for a given point with contrast and separation
        total_proba_list[id] = np.sum(prob_row)
        N_samples_list[id] = len(prob_row)
        #print(prob_row)
        detected_bool = np.array(prob_row) > 4.
        ratio_detected_planets_list[id] = float(np.sum(detected_bool))/float(N_samples_list[id])
        #all_proba_vector_shaped += prob_row
    for id,flux_ratio_row in enumerate(flux_ratio_table):
        # prob_row is the list of proba for a given point with contrast and separation
        mean_flux_ratio_list[id] = np.mean(flux_ratio_row)
        stddev_flux_ratio_list[id] = np.std(flux_ratio_row)
    #print(mean_flux_ratio_list)

    if klip_spectrum is None:
        rec_op_char_proba_vector_filename = "fakesProbaList_"+compact_date+"_"+filter+"_" + \
                 "k{0}a{1}s{2}m{3:1.2f}".format(N_KL_max,annuli,subsections,movement) +"_"+ contrast_metric+".csv"
        contrast_curve_filename = "contrastCurve_"+compact_date+"_"+filter+"_" + \
                 "k{0}a{1}s{2}m{3:1.2f}".format(N_KL_max,annuli,subsections,movement) +"_"+ contrast_metric+ ".csv"
    else:
        rec_op_char_proba_vector_filename = "fakesProbaList_"+compact_date+"_"+filter+"_" + \
                 "k{0}a{1}s{2}m{3:1.2f}".format(N_KL_max,annuli,subsections,movement) + \
                 klip_spectrum +"_"+ contrast_metric+ ".csv"
        contrast_curve_filename = "contrastCurve_"+compact_date+"_"+filter+"_" + \
                 "k{0}a{1}s{2}m{3:1.2f}".format(N_KL_max,annuli,subsections,movement) + \
                 klip_spectrum +"_"+ contrast_metric+ ".csv"


    with open(inputdir+os.path.sep+rec_op_char_proba_vector_filename, 'w+') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerow(rec_op_char_proba_vector_shaped)

    # Save contrast curve
    unique_sep = np.unique(contrast_and_radius_list[:,1])
    contrast_curve = np.zeros(unique_sep.shape)
    for id,sep_it in enumerate(unique_sep):
        indices_for_curr_sep = np.where(contrast_and_radius_list[:,1]==sep_it)[0]
        ratio_at_curr_sep = ratio_detected_planets_list[indices_for_curr_sep]
        contrasts_at_curr_sep = contrast_and_radius_list[indices_for_curr_sep,0]
        try:
            where_full_detec = np.where(ratio_at_curr_sep == 1.)[0]
            full_detec_limit_id = np.argmin(contrasts_at_curr_sep[where_full_detec])
            print(full_detec_limit_id,contrasts_at_curr_sep.size)
            if where_full_detec[full_detec_limit_id] != 0:
                cont1 = contrasts_at_curr_sep[where_full_detec[full_detec_limit_id]]
                ratio1 = ratio_at_curr_sep[where_full_detec[full_detec_limit_id]]
                cont2 = contrasts_at_curr_sep[where_full_detec[full_detec_limit_id]-1]
                ratio2 = ratio_at_curr_sep[where_full_detec[full_detec_limit_id]-1]
                print(ratio1,ratio2, cont1,cont2)
                contrast_curve[id] = (0.95-ratio1)*(cont2-cont1)/(ratio2-ratio1)+cont1
            else:
                contrast_curve[id] = contrasts_at_curr_sep[where_full_detec][full_detec_limit_id]

        except:
            contrast_curve[id]=np.nan


    with open(inputdir+os.path.sep+contrast_curve_filename, 'w+') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerow(unique_sep)
        csvwriter.writerow(contrast_curve)


    #print(contrast_and_radius_list)
    max_N_samples_list = np.max(N_samples_list)
    #marker_surface = 1000 - 100*(max_N_samples_list-N_samples_list)
    #print(ratio_detected_planets_list)
    #print(N_samples_list)
    #marker_surface[np.where(marker_surface == 0)] = 10
    #marker_color = ratio_detected_planets_list
    marker_surface = 1000*ratio_detected_planets_list
    marker_surface[np.where(marker_surface <= 10)] = 10
    marker_color = N_samples_list
    plt.figure(1)
    plt.scatter(contrast_and_radius_list[:,1]*0.01414,contrast_and_radius_list[:,0],c=marker_color,s=marker_surface,marker=".",)
    plt.colorbar()
    plt.clim(0,max_N_samples_list)
    plt.plot(unique_sep*0.01414,contrast_curve)
    if dir_for_contrast is not None:
        #data = ascii.read(inputdir_data+"contrast-S"+date+".txt")
        data = ascii.read(dir_for_contrast+os.path.sep+"contrast-S"+compact_date+".txt")
        sep_sampling = np.array(data['Seps'])
        contrast_flat = np.array(data['Flat Spectrum'])
        contrast_meth = np.array(data['Methane Spectrum'])
        plt.plot(sep_sampling,np.log10(contrast_flat))
    plt.xlabel('Separation (arcsec)', fontsize=20)
    plt.ylabel('Contrast', fontsize=20)
    plt.title(star_name+ " "+compact_date+" "+ contrast_metric)
    #plt.xlim((-30.* im_std,20.*im_std))
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    #ax.set_yscale('log')
    ax = plt.gca()
    #ax.invert_yaxis()
    ax.grid(True)
    #ax.set_yscale('log')
    #plt.ylim((10**-7,10**-3))
    #plt.show()

    if 0:
        plt.figure(2)
        plt.scatter(contrast_and_radius_list[:,1],mean_flux_ratio_list)
        plt.show()


    if 0:
        plt.figure(2)
        contrast_values = np.unique(contrast_and_radius_list[:,0])
        print(contrast_values)
        legend_str = []
        for id,contrast_it in enumerate(contrast_values):#enumerate([contrast_values[1],contrast_values[3],contrast_values[5]]):
            print(contrast_it)
            legend_str.append(str(contrast_it))
            indices = np.where(contrast_and_radius_list[:,0] == contrast_it)
            #print(contrast_and_radius_list[:,1].shape,mean_flux_ratio_list.shape)
            #print(contrast_and_radius_list[indices,1].shape,mean_flux_ratio_list[indices])
            x = np.squeeze(contrast_and_radius_list[indices,1])*0.01414
            y = mean_flux_ratio_list[indices]
            sorted_indices = sorted(range(len(x)),key=lambda k: x[k])
            plt.plot(x[sorted_indices],y[sorted_indices])#,s=300,marker=".")
            #plt.errorbar(np.squeeze(contrast_and_radius_list[indices,1])*0.01414,mean_flux_ratio_list[indices],yerr = stddev_flux_ratio_list[indices],linestyle="None")
        ax = plt.gca()
        ax.legend(legend_str, loc = 'upper right', fontsize=12)
        plt.ylim((0.,1.))
        plt.show()


def get_receiver_operating_characteristic_per_frame(inputdir,
                      threshold_sampling,
                      annuli = 7,
                      subsections = 4,
                      movement = 3,
                      klip_spectrum = None,
                      N_KL = 10,
                      N_KL_max = 100,
                      contrast_metric_list = ["matchedFilter"],
                      spectrum_metric = None,
                      GOI_list_folder = None,
                      mute = False):




    radial_PSF_cube_filename = glob.glob(inputdir+os.path.sep+"S*-original_radial_PSF_cube.fits")[0]
    # generate normalized PSF
    hdulist = pyfits.open(radial_PSF_cube_filename)
    radial_psfs = hdulist[1].data

    nl,ny_PSF,nx_PSF = radial_psfs.shape

    stamp_x_grid, stamp_y_grid = np.meshgrid(np.arange(0,nx_PSF,1)-nx_PSF/2,np.arange(0,ny_PSF,1)-ny_PSF/2)
    r_stamp = np.sqrt((stamp_x_grid)**2 +(stamp_y_grid)**2)
    stamp_mask = np.ones((ny_PSF,nx_PSF))
    stamp_mask[np.where(r_stamp > 3.)] = np.nan
    #stamp_cube_mask = np.tile(stamp_mask[None,:,:],(nl,1,1))

    if klip_spectrum is None:
        fileprefix_ref = "noFakeRef_"+"*"+"_"+"*"+"_" + \
                 "k{0}a{1}s{2}m{3:1.2f}".format(N_KL_max,annuli,subsections,movement)
    else:
        fileprefix_ref = "noFakeRef_"+"*"+"_"+"*"+"_" + \
                 "k{0}a{1}s{2}m{3:1.2f}".format(N_KL_max,annuli,subsections,movement) + \
                 klip_spectrum

    filename_ref= inputdir + fileprefix_ref+"-KL"+str(N_KL)+"-speccube.fits"
    filename_ref = glob.glob(filename_ref)[0]
    print(filename_ref)
    hdulist = pyfits.open(filename_ref)
    ori_klipped_cube = hdulist[1].data
    exthdr = hdulist[1].header
    prihdr = hdulist[0].header
    hdulist.close()



    try:
        # OBJECT: keyword in the primary header with the name of the star.
        star_name = prihdr['OBJECT'].strip().replace (" ", "_")
    except:
        # If the object name could nto be found cal lit unknown_object
        star_name = "UNKNOWN_OBJECT"

    filter = prihdr['IFSFILT'].split('_')[1]

    date = prihdr['DATE']
    compact_date = date.replace("-","")

    if klip_spectrum is None:
        fileprefix_ref = "noFakeRef_"+compact_date+"_"+filter+"_" + \
                 "k{0}a{1}s{2}m{3:1.2f}".format(N_KL_max,annuli,subsections,movement)
    else:
        fileprefix_ref = "noFakeRef_"+compact_date+"_"+filter+"_" + \
                 "k{0}a{1}s{2}m{3:1.2f}".format(N_KL_max,annuli,subsections,movement) + \
                 klip_spectrum

    if GOI_list_folder is not None:
        x_real_object_list,y_real_object_list = get_pos_known_objects(prihdr,exthdr,GOI_list_folder,xy = True)

    pyklip_dir = os.path.dirname(os.path.realpath(__file__))
    if spectrum_metric is None:
        spectrum_metric = pyklip_dir+os.path.sep+"spectra"+os.path.sep+"g32ncflx"+os.path.sep+"t950g32nc.flx"
    else:
        spectrum_metric = pyklip_dir+os.path.sep+"spectra"+os.path.sep+spectrum_metric

    spectrum_metric_name = spectrum_metric.split(os.path.sep)[-1].split(".")[0]
    planet_detec_folder = inputdir+os.path.sep+"planet_detec_"+fileprefix_ref+"-KL"+str(N_KL)+os.path.sep+spectrum_metric_name+os.path.sep

    legend_str_list = []
    plt.figure(1)
    for contrast_metric in contrast_metric_list:
        print(planet_detec_folder+star_name+"-detections-"+contrast_metric+".xml")
        candidates_log_file = planet_detec_folder+star_name+"-detections-"+contrast_metric+".xml"
        if len(glob.glob(candidates_log_file)) == 0 :
            candidate_detection(planet_detec_folder,
                                mute = False,
                                metric = contrast_metric,
                                noPlots = True)

        tree = ET.parse(candidates_log_file)
        root = tree.getroot()

        false_detec_proba_vec = []
        for candidate in root[0].find("all"):
            # Get the information of the candidate from the element attributes
            candidate_id = int(candidate.attrib["id"])
            max_val_criter = float(candidate.attrib["max_val_criter"])
            x_max_pos = float(candidate.attrib["x_max_pos"])
            y_max_pos = float(candidate.attrib["y_max_pos"])
            row_id = float(candidate.attrib["row_id"])
            col_id = float(candidate.attrib["col_id"])

            #remove the detection if it is a real object
            if GOI_list_folder is not None:
                too_close = False
                for x_real_object,y_real_object  in zip(x_real_object_list,y_real_object_list):
                    if (x_max_pos-x_real_object)**2+(y_max_pos-y_real_object)**2 < 15.*15.:
                        too_close = True
                        if not mute:
                            print("Skipping planet. Real object too close.")
                        break
                if too_close:
                    continue

            false_detec_proba_vec.append(max_val_criter)
            #print(false_detec_proba_vec)


        if klip_spectrum is None:
            fakes_proba_vector_filename = "fakesProbaList_"+compact_date+"_"+filter+"_" + \
                     "k{0}a{1}s{2}m{3:1.2f}".format(N_KL_max,annuli,subsections,movement) +"_"+ contrast_metric+  ".csv"
            rec_op_char_filename = "recOpChar_"+compact_date+"_"+filter+"_" + \
                     "k{0}a{1}s{2}m{3:1.2f}".format(N_KL_max,annuli,subsections,movement) +"_"+ contrast_metric+  ".csv"
        else:
            fakes_proba_vector_filename = "fakesProbaList_"+compact_date+"_"+filter+"_" + \
                     "k{0}a{1}s{2}m{3:1.2f}".format(N_KL_max,annuli,subsections,movement) + \
                     klip_spectrum +"_"+ contrast_metric+  ".csv"
            rec_op_char_filename = "recOpChar_"+compact_date+"_"+filter+"_" + \
                     "k{0}a{1}s{2}m{3:1.2f}".format(N_KL_max,annuli,subsections,movement) + \
                     klip_spectrum +"_"+ contrast_metric+  ".csv"

        with open(inputdir+os.path.sep+fakes_proba_vector_filename, 'rb') as csvfile_TID:
            TID_reader = csv.reader(csvfile_TID, delimiter=';')
            TID_csv_as_list = list(TID_reader)
            fakes_proba_vec = np.array(TID_csv_as_list[0], dtype='string').astype(np.float)
            print(fakes_proba_vec)

        #threshold_sampling = np.linspace(max(np.min(fakes_proba_vec),np.min(false_detec_proba_vec)),
        #                               min(np.max(fakes_proba_vec),np.max(false_detec_proba_vec)),100)

        N_false_pos = np.zeros(threshold_sampling.shape)
        N_true_detec = np.zeros(threshold_sampling.shape)
        for id,threshold_it in enumerate(threshold_sampling):
            N_false_pos[id] = np.sum(false_detec_proba_vec >= threshold_it)
            N_true_detec[id] = np.sum(fakes_proba_vec >= threshold_it)


        with open(inputdir+os.path.sep+rec_op_char_filename, 'w+') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerows([threshold_sampling]+[N_false_pos]+[N_true_detec])

        plt.plot(N_false_pos,N_true_detec)
        legend_str_list.append(contrast_metric)

    plt.legend(legend_str_list)
    plt.show()
