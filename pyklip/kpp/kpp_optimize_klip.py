__author__ = 'jruffio'

import glob
import os

from pyklip.kpp.klipPostProcessing import *


def optimize_klip(inputdir,
                    contrast_log,
                    separation,
                    contrast_metric = "matchedFilter",
                    spectrum_metric = None,
                    N = 3000,
                    stamp_width = 20,
                    mask_radius = 7,
                    dir_for_contrast = None,
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

    pyklip_dir = os.path.dirname(os.path.realpath(__file__))
    if spectrum_metric is None:
        spectrum_metric = pyklip_dir+os.path.sep+"spectra"+os.path.sep+"g32ncflx"+os.path.sep+"t950g32nc.flx"
    else:
        spectrum_metric = pyklip_dir+os.path.sep+"spectra"+os.path.sep+spectrum_metric

    spectrum_metric_name = spectrum_metric.split(os.path.sep)[-1].split(".")[0]

    xml_filename = glob.glob(inputdir+os.path.sep+"*"+"_"+'fakes.xml')[0]
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    star_elt = root[0]
    fakes_list = []
    for file_elt in star_elt:
        # Get the information of the candidate from the element attributes
        curr_fileprefix = file_elt.attrib["fileprefix"]
        curr_fake_planet_contrast_log = float(file_elt.attrib["fake_planet_contrast_log"])
        curr_outputdir = file_elt.attrib["outputdir"]
        curr_maxKL = int(file_elt.attrib["maxKL"])
        curr_annuli = int(file_elt.attrib["annuli"])
        numbasis = map(int,file_elt.attrib["numbasis"][1:-1].split(","))
        curr_subsections = int(file_elt.attrib["subsections"])
        curr_movement = float(file_elt.attrib["movement"])
        curr_klip_spectrum = file_elt.attrib["klip_spectrum"]
        #print([curr_klip_spectrum])
        print(curr_movement)
        print(curr_fileprefix)

        process_file = False
        for fake_elt in file_elt:
            # Get the information of the candidate from the element attributes
            fk_radius = float(fake_elt.attrib["radius"])
            fk_contrast_log = float(fake_elt.attrib["contrast_log"])

            if contrast_log == fk_contrast_log and separation == fk_radius:
                process_file = True
        #print(process_file)
        if process_file:
            if curr_klip_spectrum == str(None):
                fileprefix_ref = "noFakeRef_"+"*"+"_"+"*"+"_" + \
                         "k{0}a{1}s{2}m{3:1.2f}".format(curr_maxKL,curr_annuli,curr_subsections,curr_movement)
            else:
                fileprefix_ref = "noFakeRef_"+"*"+"_"+"*"+"_" + \
                         "k{0}a{1}s{2}m{3:1.2f}".format(curr_maxKL,curr_annuli,curr_subsections,curr_movement) + \
                         curr_klip_spectrum

            for N_KL in numbasis:
                filename_ref= inputdir + fileprefix_ref+"-KL"+str(N_KL)+"-speccube.fits"
                print(filename_ref) #noFakeRef_20141218_H_k50a7s4m3-KL10-speccube
                filename_ref = glob.glob(filename_ref)[0]

                planet_detec_folder_ref = inputdir+os.path.sep+"planet_detec_"+fileprefix_ref+"-KL"+str(N_KL)+os.path.sep+spectrum_metric_name+os.path.sep

                print(planet_detec_folder_ref+"*"+"-matchedFilter.fits")
                print(glob.glob(planet_detec_folder_ref+"*"+"-flatCube.fits"))
                flatCube_glob = glob.glob(planet_detec_folder_ref+"*"+"-flatCube.fits")
                matchedFilter_glob = glob.glob(planet_detec_folder_ref+"*"+"-matchedFilter.fits")
                shape_glob = glob.glob(planet_detec_folder_ref+"*"+"-shape.fits")
                if (len(flatCube_glob) == 0 or \
                   len(matchedFilter_glob) == 0 or \
                   len(shape_glob) == 0):
                    planet_detection_in_dir_per_file(filename_ref,
                                                      metrics = ["shape", "matchedFilter"],
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
                                                      detection_metric = None)
                else:
                    if not mute:
                        print("found planet detec for ref klip")

                flatCube_glob_ref = glob.glob(planet_detec_folder_ref+"*"+"-flatCube.fits")
                matchedFilter_glob_ref = glob.glob(planet_detec_folder_ref+"*"+"-matchedFilter.fits")
                shape_glob_ref = glob.glob(planet_detec_folder_ref+"*"+"-shape.fits")
                if contrast_metric == "matchedFilter":
                    hdulist = pyfits.open(matchedFilter_glob_ref[0])
                elif contrast_metric == "shape":
                    hdulist = pyfits.open(shape_glob_ref[0])
                ori_shape_map = hdulist[1].data
                exthdr = hdulist[1].header
                prihdr = hdulist[0].header
                hdulist.close()

                # If GOI_list is not None. Mask the known objects from the image that will be used for calculating the
                # PDF. This masked image is given separately to the probability calculation function.
                if GOI_list_folder is not None:
                    ori_shape_map_without_planet = mask_known_objects(ori_shape_map,prihdr,exthdr,GOI_list_folder, mask_radius = 7)
                else:
                    ori_shape_map_without_planet = ori_shape_map

                ny,nx = ori_shape_map.shape
                try:
                    # Retrieve the center of the image from the fits keyword.
                    center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
                except:
                    # If the keywords could not be found.
                    if not mute:
                        print("Couldn't find PSFCENTX and PSFCENTY keywords.")
                    center = [(nx-1)/2,(ny-1)/2]
                x_cen, y_cen = center

                IWA,OWA,inner_mask,outer_mask = get_occ(ori_shape_map, centroid = center)



                image_without_planet_mask = np.ones((ny,nx))
                image_without_planet_mask[np.where(np.isnan(ori_shape_map_without_planet))] = 0

                # Build the x and y coordinates grids
                x_grid, y_grid = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
                # Calculate the radial distance of each pixel
                r_grid = abs(x_grid +y_grid*1j)
                th_grid = np.arctan2(x_grid,y_grid)

                r_min_firstZone,r_max_firstZone = (IWA,np.sqrt(N/np.pi+IWA**2))
                r_limit_firstZone = (r_min_firstZone + r_max_firstZone)/2.
                r_min_lastZone,r_max_lastZone = (OWA,np.max([ny,nx]))
                r_limit_lastZone = OWA - N/(4*np.pi*OWA)

                fakePl_filelist = glob.glob(inputdir+curr_fileprefix+"-KL"+str(N_KL)+"-speccube.fits")
                fakePl_filename = fakePl_filelist[0]
                #N_KL = int(fakePl_filename.split("-")[-2][2::])

                planet_detec_folder = inputdir+os.path.sep+"planet_detec_"+curr_fileprefix+"-KL"+str(N_KL)+os.path.sep+spectrum_metric_name+os.path.sep

                flatCube_glob = glob.glob(planet_detec_folder+"*"+"-flatCube.fits")
                matchedFilter_glob = glob.glob(planet_detec_folder+"*"+"-matchedFilter.fits")
                shape_glob = glob.glob(planet_detec_folder+"*"+"-shape.fits")
                print(flatCube_glob)
                if (len(flatCube_glob) == 0 or \
                   len(matchedFilter_glob) == 0 or \
                   len(shape_glob) == 0):
                    print("calculating")
                    planet_detection_in_dir_per_file(fakePl_filename,
                                                  metrics = ["shape", "matchedFilter"],
                                                  directory = inputdir,
                                                  outputDir = inputdir,
                                                  spectrum_model = [spectrum_metric],
                                                  star_type = "G5",
                                                  star_temperature = None,
                                                  user_defined_PSF_cube = radial_PSF_cube_filename,
                                                  metrics_only = False,
                                                  planet_detection_only = False,
                                                  mute = False,
                                                  threads = True,
                                                  GOI_list_folder = GOI_list_folder,
                                                  overwrite_metric = True,
                                                  overwrite_stat = False,
                                                  proba_using_mask_per_pixel = False,
                                                  SNR = False,
                                                  probability = False,
                                                  detection_metric = None)
                    #sleep(0.1)
                else:
                    print("skipping")

                print(planet_detec_folder)
                flatCube_glob = glob.glob(planet_detec_folder+"*"+"-flatCube.fits")
                matchedFilter_glob = glob.glob(planet_detec_folder+"*"+"-matchedFilter.fits")
                shape_glob = glob.glob(planet_detec_folder+"*"+"-shape.fits")
                if contrast_metric == "shape":
                    hdulist = pyfits.open(shape_glob[0])
                elif contrast_metric == "matchedFilter":
                    hdulist = pyfits.open(matchedFilter_glob[0])
                fakePl_shape_map = hdulist[1].data
                hdulist.close()
                hdulist = pyfits.open(flatCube_glob[0])
                fakePl_flatCube = hdulist[1].data
                #print(center)
                #print([hdulist[1].header['PSFCENTX'], hdulist[1].header['PSFCENTY']])
                hdulist.close()

                for fake_elt in file_elt:
                    # Get the information of the candidate from the element attributes
                    fk_x_max_pos = float(fake_elt.attrib["x_max_pos"])
                    fk_y_max_pos = float(fake_elt.attrib["y_max_pos"])
                    fk_col_centroid = float(fake_elt.attrib["col_centroid"])
                    fk_row_centroid = float(fake_elt.attrib["row_centroid"])
                    fk_pa = float(fake_elt.attrib["pa"])
                    fk_radius = float(fake_elt.attrib["radius"])
                    fk_contrast_log = float(fake_elt.attrib["contrast_log"])
                    fk_aperture_flux_ori = float(fake_elt.attrib["aperture_flux"])
                    #print(fk_pa,fk_radius,fk_x_max_pos,fk_y_max_pos,fk_col_centroid,fk_row_centroid)


                    if 1 and contrast_log == fk_contrast_log and separation == fk_radius:
                        #print(fk_contrast_log,fk_radius,fk_pa,curr_fileprefix)

                        k = round(fk_col_centroid)
                        l = round(fk_row_centroid)


                        x = x_grid[(k,l)]
                        y = y_grid[(k,l)]
                        #print(x,y)
                        r = r_grid[(k,l)]


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

                        data = ori_shape_map_without_planet[(where_ring[0][where_ring_masked],where_ring[1][where_ring_masked])]

                        cdf_model, pdf_model, sampling, im_histo, center_bins  = get_cdf_model(data)

                        cdf_fit = interp1d(sampling,cdf_model,kind = "linear",bounds_error = False, fill_value=1.0)

                        proba = -np.log10(1-cdf_fit(fakePl_shape_map[k,l]))
                        fakes_list.append([proba,curr_annuli,curr_subsections,curr_movement,N_KL,curr_maxKL,curr_klip_spectrum])
                        #print(-np.log10(1-cdf_fit(fakePl_shape_map[k,l])))
                        if 0 and curr_movement == 2.0 and N_KL == 30:
                            #print(data)
                            print(fakePl_shape_map[k,l])
                            #print(shape_glob_ref[0],shape_glob[0])
                            print(-np.log10(1-cdf_fit(fakePl_shape_map[k,l])))
                            stamp = fakePl_shape_map[(k-np.floor(stamp_width/2.)):(k+np.ceil(stamp_width/2.)),
                                                    (l-np.floor(stamp_width/2.)):(l+np.ceil(stamp_width/2.))]
                            plt.figure(1)
                            plt.imshow(stamp,interpolation="nearest")
                            plt.colorbar()
                            plt.show()

    #print(np.array(fakes_list))
    fakes_list = np.array(fakes_list)
    #print(fakes_list[:,3],fakes_list[:,0])
    proba_list = np.array(map(float,fakes_list[:,0]))
    N_KL_list = np.array(map(int,fakes_list[:,4]))
    mvt_list = np.array(map(float,fakes_list[:,3]))
    mvt_list_unique = np.unique(mvt_list)
    N_KL_list_unique = np.unique(N_KL_list)
    mean_proba_mvt = np.zeros(mvt_list_unique.size)
    mean_proba_N_KL = np.zeros(N_KL_list_unique.size)
    for k in range(mvt_list_unique.size):
        mean_proba_mvt[k]=np.mean(proba_list[np.where(mvt_list == mvt_list_unique[k])])
    for k in range(N_KL_list_unique.size):
        mean_proba_N_KL[k]=np.mean(proba_list[np.where(N_KL_list == N_KL_list_unique[k])])
    print(proba_list)
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.scatter(N_KL_list[np.where(mvt_list == 2.5)],proba_list[np.where(mvt_list == 2.5)])
    #plt.scatter(N_KL_list,proba_list)
    #plt.plot(N_KL_list_unique,mean_proba_N_KL)
    ax = plt.gca()
    ax.grid(True)
    ax.set_xlabel('N_KL')
    ax.set_ylabel('proba')
    plt.xlim((0,110))
    plt.ylim((0,10))
    plt.subplot(2,1,2)
    #print(proba_list[np.where(N_KL_list == 20)])
    plt.scatter(mvt_list[np.where(N_KL_list == 40)],proba_list[np.where(N_KL_list == 40)])
    #plt.scatter(mvt_list,proba_list)
    #plt.plot(mvt_list_unique,mean_proba_mvt)
    ax = plt.gca()
    ax.grid(True)
    ax.set_xlabel('movement')
    ax.set_ylabel('proba')
    plt.xlim((0,6))
    plt.ylim((0,10))
    plt.show()