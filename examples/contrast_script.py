#!/home/anaconda/bin/python2.7
__author__ = 'jruffio'

import platform
import sys

import multiprocessing as mp
from pyklip.kpp.stat.statPerPix_utils import *
from pyklip.kpp.utils.oi import *
from pyklip.instruments import GPI
import pyklip.fakes as fakes
import astropy.io.fits as pyfits


def contrast_dataset(inputDir,outputDir,dir_fakes,mvt,reduc_spectrum,fakes_spectrum,approx_throughput,HPF=False):
    ###########################################################################################
    ## SECONDARY PARAMETERS
    ###########################################################################################
    numbasis=[5]
    maxnumbasis = 10
    # Less important parameters
    overwrite = True # Force rewriting the files even if they already exist
    label0 = "FMMF2018" # Name of the folder in which the outputs will be saved
    # Register all signal above "detec_threshold" in the final SNR map
    detec_threshold = 3.5
    # platescale
    pix2as = GPI.GPIData.lenslet_scale

    pa_shift_list = [0,45] # Position angle shift between the fakes in the different copies of the dataset

    IWA = 8.7
    OWA = 50
    N_pix_sector = 200 # Number of pixels in each inner sector
    # Definition of the annuli
    tmp = (np.arange(IWA,OWA,5)).tolist()
    annuli = [(rho1,rho2) for rho1,rho2 in zip(tmp[0:-1],tmp[1::])]
    print("N_annulus",len(annuli))

    if HPF:
        PSFname = "-hpf_PSF_cube.fits"
    else:
        PSFname = "-PSF_cube.fits"

    ###########################################################################################
    ## Generate PSF cube
    ###########################################################################################
    try:
        PSF_cube_filename = glob(os.path.join(inputDir,"*"+PSFname))[0]
        PSF_cube_obj = GPI.GPIData([PSF_cube_filename],highpass=False)
        PSF_cube_arr = PSF_cube_obj.input
        PSF_cube_wvs = PSF_cube_obj.wvs
    except:
        # Generate PSF cube for GPI from the satellite spots
        filenames = glob(os.path.join(inputDir,"S*distorcorr.fits"))
        # filenames = filelist_tormv
        dataset = GPI.GPIData(filenames,highpass=HPF)
        dataset.generate_psf_cube(20,same_wv_only=True)
        PSF_cube_filename = inputDir + os.path.sep + os.path.basename(filenames[0]).split(".fits")[0]+PSFname
        # Save the original PSF calculated from combining the sat spots
        dataset.output_wcs = np.array([w.deepcopy() if w is not None else None for w in dataset.wcs])
        dataset.output_centers = dataset.centers
        dataset.savedata(PSF_cube_filename, dataset.psfs, filetype="PSF Spec Cube")

        PSF_cube_arr = dataset.psfs
        PSF_cube_wvs = np.unique(dataset.wvs)

        if not os.path.exists(os.path.join(outputDir)):
            os.makedirs(os.path.join(outputDir))

        # Copy the PSF cube to the new folder
        import shutil
        src = glob(os.path.join(inputDir,"*"+PSFname))[0]
        shutil.copyfile(src,os.path.join(outputDir,os.path.basename(src)))

    ###########################################################################################
    ## Read the dataset
    ###########################################################################################
    print("~~ Reading data ~~")
    file_list = glob(os.path.join(inputDir,"S*distorcorr.fits"))
    dataset = GPI.GPIData(file_list,
                         meas_satspot_flux=True,
                         numthreads = None,
                         highpass=HPF,
                         PSF_cube="*"+PSFname)

    ###########################################################################################
    ## Define the spectrum template (transmission correction and flux calibration)
    ###########################################################################################
    nl,ny,nx = dataset.input.shape
    nl_PSF,ny_PSF,nx_PSF = PSF_cube_arr.shape
    dn_per_contrast = 1./dataset.calibrate_output(np.ones((nl,1,1)),spectral=True).squeeze()
    host_star_spec = dn_per_contrast/np.mean(dn_per_contrast)
    import pyklip.spectra_management as spec
    star_type = spec.get_specType(dataset.object_name)
    # Interpolate a spectrum of the star based on its spectral type/temperature
    wv,star_sp = spec.get_star_spectrum(dataset.wvs,star_type)
    pykliproot = os.path.dirname(os.path.realpath(spec.__file__))
    spectrum_filename = os.path.abspath(glob(os.path.join(pykliproot,"spectra","*",reduc_spectrum+".flx"))[0])
    # Interpolate the spectrum of the planet based on the given filename
    wv,planet_sp = spec.get_planet_spectrum(spectrum_filename,dataset.wvs)
    # Correct the ideal spectrum given in spectrum_filename for atmospheric and instrumental absorption.
    spectrum_vec = (host_star_spec/star_sp)*planet_sp
    # Make sure the total flux of each PSF is unity for all wavelengths
    # So the peak value won't be unity.
    PSF_cube_arr = PSF_cube_arr/np.nansum(PSF_cube_arr,axis=(1,2))[:,None,None]
    # Get the conversion factor from peak spectrum to aperture based spectrum
    aper_over_peak_ratio = 1/np.nanmax(PSF_cube_arr,axis=(1,2))
    aper_over_peak_ratio_tiled = np.zeros(nl)#wavelengths
    for k,wv in enumerate(dataset.wvs):
        aper_over_peak_ratio_tiled[k] = aper_over_peak_ratio[spec.find_nearest(PSF_cube_wvs,wv)[1]]
    # Summed DN flux of the star in the entire dataset calculated from dn_per_contrast
    star_flux = np.sum(aper_over_peak_ratio_tiled*dn_per_contrast)
    fake_contrast = 1. # ratio of flux of the planet/flux of the star (broad band flux)
    # normalize the spectra to unit contrast.
    spectrum_vec = spectrum_vec/np.sum(spectrum_vec)*star_flux*fake_contrast

    ###########################################################################################
    ## Reduce the dataset with FMMF
    ###########################################################################################
    print("~~ Reduce dataset ~~")
    # This section will take for ever due to the FMMF reduction
    import pyklip.fmlib.matchedFilter as mf
    fm_class = mf.MatchedFilter(dataset.input.shape,numbasis, PSF_cube_arr, PSF_cube_wvs,
                                     spectrallib = [spectrum_vec],
                                     ref_center=[np.mean(dataset.centers[:,0]), np.mean(dataset.centers[:,1])],
                                     flipx=dataset.flipx)

    # run KLIP-FM
    prefix = os.path.basename(file_list[0]).split(".fits")[0]+"_"+reduc_spectrum+"_mvt{0:0.2f}".format(mvt)
    curr_outputDir = os.path.join(outputDir,label0,reduc_spectrum)
    import pyklip.fm as fm
    if not os.path.exists(curr_outputDir):
        os.makedirs(curr_outputDir)

    fm.klip_dataset(dataset, fm_class, outputdir=curr_outputDir, fileprefix=prefix, numbasis=numbasis,
                    annuli=annuli,subsections=None,N_pix_sector=N_pix_sector, padding=nx_PSF//2, movement=mvt,OWA=dataset.OWA,maxnumbasis = maxnumbasis,
                    highpass = None,mute_progression=True)


    ###########################################################################################
    ## Calculate quick SNR map
    ###########################################################################################
    filename = glob(os.path.join(outputDir,label0,reduc_spectrum,"*_mvt{0:0.2f}-FMMF-KL{1}.fits".format(mvt,numbasis[0])))[0]
    hdulist = pyfits.open(filename)
    FMMF_map = hdulist[1].data
    prihdr = hdulist[0].header
    exthdr = hdulist[1].header
    hdulist.close()
    center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    SNR_map = get_image_stat_map_perPixMasking(FMMF_map,
                                               centroid = center,
                                               mask_radius=5,
                                               Dr = 2,
                                               type = "SNR")
    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(header=prihdr))
    hdulist.append(pyfits.ImageHDU(header=exthdr, data=SNR_map, name="Sci"))
    prefix = os.path.basename(filename).split(".fits")[0]
    try:
        hdulist.writeto(os.path.join(outputDir,label0,reduc_spectrum,prefix+"_SNR.fits"), overwrite=True)
    except TypeError:
        hdulist.writeto(os.path.join(outputDir,label0,reduc_spectrum,prefix+"_SNR.fits"), clobber=True)
    hdulist.close()

    ###########################################################################################
    ## Initial guess for the contrast curve (to determine the contrast of the fakes)
    ###########################################################################################
    # This section can be user defined as long as sep_bins_center and cont_stddev are set.
    filename = glob(os.path.join(outputDir,label0,reduc_spectrum,"*_mvt{0:0.2f}-FMCont-KL{1}.fits".format(mvt,numbasis[0])))[0]
    hdulist = pyfits.open(filename)
    FMCont_map = hdulist[1].data
    prihdr = hdulist[0].header
    exthdr = hdulist[1].header
    hdulist.close()
    center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    hdulist.close()

    cont_stddev,sep_bins = get_image_stddev(FMCont_map, centroid = center)
    # Separation samples in pixels
    sep_bins_center =  (np.array([r_tuple[0] for r_tuple in sep_bins]))
    # Approximative contrast curve at these separations
    approx_cont_curve = 5*np.array(cont_stddev)/approx_throughput

    ############################################################################################
    ## Build fake datasets to be used to calibrate the conversion factor
    ###########################################################################################
    print("~~ Injecting fakes ~~")

    # Load the PSF cube that has been calculated from the sat spots
    PSF_cube_filename = glob(os.path.join(inputDir,"*"+PSFname))[0]
    PSF_cube_obj = GPI.GPIData([PSF_cube_filename],highpass=False)
    PSF_cube_arr = PSF_cube_obj.input
    PSF_cube_wvs = PSF_cube_obj.wvs

    if not os.path.exists(dir_fakes):
        os.makedirs(dir_fakes)
    import shutil
    shutil.copyfile(PSF_cube_filename,os.path.join(dir_fakes,os.path.basename(PSF_cube_filename)))


    sep_vec,pa_vec = get_pos_known_objects(prihdr,prihdr["OBJECT"],GPI.GPIData.lenslet_scale,
                                   center,MJDOBS=prihdr["MJD-OBS"],pa_sep=True)
    real_planets_pos = [(sep/GPI.GPIData.lenslet_scale,pa) for sep,pa in zip(sep_vec,pa_vec)]

    for pa_shift in pa_shift_list:
        #Define the fakes position and contrast
        fake_flux_dict = dict(mode = "SNR",SNR=5,sep_arr = sep_bins_center, contrast_arr=approx_cont_curve)
        # fake_flux_dict = dict(mode = "contrast",contrast=GPI.GPIData.spot_ratio["J"]/20)#0.00017975
        fake_position_dict = dict(mode = "spirals",pa_shift=pa_shift,annuli=10)

        # Inject the fakes
        spdc_glob = glob(inputDir+os.path.sep+"S*_spdc_distorcorr.fits")
        if overwrite or len(glob(os.path.join(dir_fakes,"S*_spdc_distorcorr_{0}_PA*.fits").format(fakes_spectrum))) != len(pa_shift_list)*len(spdc_glob):
            dataset = GPI.GPIData(spdc_glob,
                                  meas_satspot_flux=True,
                                  numthreads=mp.cpu_count(),
                                  highpass=HPF,
                                  PSF_cube = PSF_cube_arr)

            dataset,extra_keywords = fakes.generate_dataset_with_fakes(dataset,
                                                                       fake_position_dict,
                                                                       fake_flux_dict,
                                                                       spectrum = fakes_spectrum,
                                                                       PSF_cube = PSF_cube_arr,
                                                                       PSF_cube_wvs=PSF_cube_wvs,
                                                                       mute = False,
                                                                       real_planets_pos = real_planets_pos,
                                                                       pa_skip_real_pl= 45,
                                                                       sep_skip_real_pl= 10)

            numwaves = np.size(np.unique(dataset.wvs))
            N_cubes = np.size(dataset.wvs)//numwaves
            suffix = fakes_spectrum+"_PA{0:02d}".format(pa_shift)
            #Save each cube with the fakes
            for cube_id in range(N_cubes):
                spdc_filename = dataset.filenames[(cube_id*numwaves)].split(os.path.sep)[-1].split(".")[0]
                print("Saving file: "+dir_fakes + os.path.sep + spdc_filename+"_"+suffix+".fits")
                dataset.output_wcs = np.array([w.deepcopy() if w is not None else None for w in dataset.wcs])
                dataset.output_centers = dataset.centers
                dataset.savedata(dir_fakes + os.path.sep + spdc_filename+"_"+suffix+".fits",
                                 dataset.input[(cube_id*numwaves):((cube_id+1)*numwaves),:,:],
                                 filetype="raw spectral cube with fakes", more_keywords =extra_keywords,
                                 user_prihdr=dataset.prihdrs[cube_id], user_exthdr=dataset.exthdrs[cube_id])

    ###########################################################################################
    ## Reduce the fake dataset
    ###########################################################################################
    print("~~ Reduce SIMULATED dataset ~~")

    for pa_shift in pa_shift_list:
        # Object to reduce the fake dataset with FMMF
        file_list = glob(os.path.join(dir_fakes,"S*_spdc_distorcorr_{0}_PA{1:02d}.fits".format(fakes_spectrum,pa_shift)))
        dataset = GPI.GPIData(file_list,
                             meas_satspot_flux=True,
                             numthreads = None,
                             highpass=False,
                             butterfly_rdi = False,
                             PSF_cube="*"+PSFname)

        #####
        # Define the spectrum
        nl,ny,nx = dataset.input.shape
        nl_PSF,ny_PSF,nx_PSF = PSF_cube_arr.shape
        dn_per_contrast = 1./dataset.calibrate_output(np.ones((nl,1,1)),spectral=True).squeeze()
        host_star_spec = dn_per_contrast/np.mean(dn_per_contrast)
        import pyklip.spectra_management as spec
        star_type = spec.get_specType(dataset.object_name)
        # Interpolate a spectrum of the star based on its spectral type/temperature
        wv,star_sp = spec.get_star_spectrum(dataset.wvs,star_type)
        pykliproot = os.path.dirname(os.path.realpath(spec.__file__))
        spectrum_filename = os.path.abspath(glob(os.path.join(pykliproot,"spectra","*",reduc_spectrum+".flx"))[0])
        # Interpolate the spectrum of the planet based on the given filename
        wv,planet_sp = spec.get_planet_spectrum(spectrum_filename,dataset.wvs)
        # Correct the ideal spectrum given in spectrum_filename for atmospheric and instrumental absorption.
        spectrum_vec = (host_star_spec/star_sp)*planet_sp
        # Make sure the total flux of each PSF is unity for all wavelengths
        # So the peak value won't be unity.
        PSF_cube_arr = PSF_cube_arr/np.nansum(PSF_cube_arr,axis=(1,2))[:,None,None]
        # Get the conversion factor from peak spectrum to aperture based spectrum
        aper_over_peak_ratio = 1/np.nanmax(PSF_cube_arr,axis=(1,2))
        aper_over_peak_ratio_tiled = np.zeros(nl)#wavelengths
        # PSF_cube_4norma = np.zeros(nl)
        for k,wv in enumerate(dataset.wvs):
            wv_id = spec.find_nearest(PSF_cube_wvs,wv)[1]
            aper_over_peak_ratio_tiled[k] = aper_over_peak_ratio[wv_id]
            # PSF_cube_4norma[k] = np.nansum(PSF_cube_arr[wv_id,:,:])
        # Summed DN flux of the star in the entire dataset calculated from dn_per_contrast
        star_flux = np.sum(aper_over_peak_ratio_tiled*dn_per_contrast)
        fake_contrast = 1. # ratio of flux of the planet/flux of the star (broad band flux)
        # normalize the spectra to unit contrast.
        spectrum_vec = spectrum_vec/np.sum(spectrum_vec)*star_flux*fake_contrast

        sep_list, pa_list = get_pos_known_objects(dataset.prihdrs[0],dataset.object_name,GPI.GPIData.lenslet_scale,
                                                  OI_list_folder=None,
                                                  xy = False,pa_sep = True,fakes_only=True)
        fakes_sepPa_list = [(1./GPI.GPIData.lenslet_scale*sep,pa) for sep,pa in zip(sep_list, pa_list)]
        # fakes_sepPa_list=None

        import pyklip.fmlib.matchedFilter as mf
        fm_class = mf.MatchedFilter(dataset.input.shape,numbasis, PSF_cube_arr, PSF_cube_wvs,
                                     spectrallib = [spectrum_vec],
                                     ref_center=[np.mean(dataset.centers[:,0]), np.mean(dataset.centers[:,1])],
                                     flipx=dataset.flipx,
                                     fakes_sepPa_list = fakes_sepPa_list)

        # run KLIP-FM
        prefix = os.path.basename(file_list[0]).split(".fits")[0]+"_"+reduc_spectrum+"_mvt{0:0.2f}".format(mvt)
        curr_outputDir = os.path.join(dir_fakes,label0+"_cont_PA{0:02d}".format(pa_shift),reduc_spectrum)
        import pyklip.fm as fm
        if not os.path.exists(curr_outputDir):
            os.makedirs(curr_outputDir)
        fm.klip_dataset(dataset, fm_class, outputdir=curr_outputDir, fileprefix=prefix, numbasis=numbasis,
                        annuli=annuli,subsections=None,N_pix_sector=N_pix_sector, padding=nx_PSF//2, movement=mvt,OWA=dataset.OWA,maxnumbasis = maxnumbasis,
                        highpass = None,mute_progression=True)

        # Calculate SNR for fake reductions
        filename = glob(os.path.join(outputDir,label0,reduc_spectrum,"*_mvt{0:0.2f}-FMMF-KL{1}.fits".format(mvt,numbasis[0])))[0]
        hdulist = pyfits.open(filename)
        FMMF_map_nofakes = hdulist[1].data
        hdulist.close()
        filename = glob(os.path.join(dir_fakes,label0+"_cont_PA{0:02d}".format(pa_shift),reduc_spectrum,
                                     "*_mvt{0:0.2f}-FMMF-KL{1}.fits".format(mvt,numbasis[0])))[0]
        hdulist = pyfits.open(filename)
        FMMF_map_fakes = hdulist[1].data
        prihdr = hdulist[0].header
        exthdr = hdulist[1].header
        hdulist.close()
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
        SNR_map = get_image_stat_map_perPixMasking(FMMF_map_fakes,FMMF_map_nofakes,
                                                   centroid = center,
                                                   mask_radius=5,
                                                   Dr = 2,
                                                   type = "SNR")
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(header=prihdr))
        hdulist.append(pyfits.ImageHDU(header=exthdr, data=SNR_map, name="Sci"))
        prefix = os.path.basename(filename).split(".fits")[0]
        try:
            hdulist.writeto(os.path.join(dir_fakes,label0+"_cont_PA{0:02d}".format(pa_shift),reduc_spectrum,prefix+"_SNR.fits"), overwrite=True)
        except TypeError:
            hdulist.writeto(os.path.join(dir_fakes,label0+"_cont_PA{0:02d}".format(pa_shift),reduc_spectrum,prefix+"_SNR.fits"), clobber=True)
        hdulist.close()

    spdc_glob = glob(os.path.join(dir_fakes,"S*_spdc_distorcorr_{0}_*.fits".format(fakes_spectrum)))
    for filename in spdc_glob:
        print("Removing {0}".format(filename))
        os.remove(filename)


    ###########################################################################################
    ## Combine all the data to build the contrast curve
    ###########################################################################################
    FMMF_map_filename = os.path.join(outputDir,label0,reduc_spectrum,
                                       "*_mvt{0:0.2f}-FMMF-KL{1}.fits".format(mvt,numbasis[0]))
    hdulist = pyfits.open(glob(FMMF_map_filename)[0])
    FMMF_map = hdulist[1].data
    prihdr = hdulist[0].header
    exthdr = hdulist[1].header
    target_name = hdulist[0].header["OBJECT"].replace(" ","_")
    center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    hdulist.close()
    FMCont_filename = os.path.join(outputDir,label0,reduc_spectrum,
                                       "*_mvt{0:0.2f}-FMCont-KL{1}.fits".format(mvt,numbasis[0]))
    hdulist = pyfits.open(glob(FMCont_filename)[0])
    FMCont = hdulist[1].data
    hdulist.close()
    fakes_FMMF_list = []
    fakes_FMCont_list = []
    fakeinfohdr_list = []
    for pa_shift in pa_shift_list:
        fakes_filename = os.path.join(dir_fakes,label0+"_cont_PA{0:02d}".format(pa_shift),reduc_spectrum,
                                               "*_mvt{0:0.2f}-FMMF-KL{1}.fits".format(mvt,numbasis[0]))
        hdulist = pyfits.open(glob(fakes_filename)[0])
        fakes_FMMF_list.append(hdulist[1].data)
        fakeinfohdr_list.append(hdulist[0].header)
        hdulist.close()

        fakes_filename = os.path.join(dir_fakes,label0+"_cont_PA{0:02d}".format(pa_shift),reduc_spectrum,
                                               "*_mvt{0:0.2f}-FMCont-KL{1}.fits".format(mvt,numbasis[0]))
        hdulist = pyfits.open(glob(fakes_filename)[0])
        fakes_FMCont_list.append(hdulist[1].data)
        hdulist.close()

    # Mask any real astrophysical signal here!
    FMMF_map_masked = FMMF_map
    FMCont_masked = FMCont

    FMCont_calib,throughput_map,fake_info = calibrate_contrast_map_1D(FMCont,FMCont_masked,fakes_FMCont_list,fakeinfohdr_list,center,GPI.GPIData.lenslet_scale,target_name)
    FMMF_calib,calib_sig_map = calibrate_SNR_map_1D(FMMF_map,FMMF_map_masked,center=center)

    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(header=prihdr))
    hdulist.append(pyfits.ImageHDU(header=exthdr, data=FMCont_calib, name="Sci"))
    prefix = os.path.basename(glob(FMCont_filename)[0]).split(".fits")[0]
    try:
        hdulist.writeto(os.path.join(outputDir,label0,reduc_spectrum,prefix+"_calib.fits"), overwrite=True)
    except TypeError:
        hdulist.writeto(os.path.join(outputDir,label0,reduc_spectrum,prefix+"_calib.fits"), clobber=True)
    hdulist.close()
    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(header=prihdr))
    hdulist.append(pyfits.ImageHDU(header=exthdr, data=FMMF_calib, name="Sci"))
    prefix = os.path.basename(glob(FMMF_map_filename)[0]).split(".fits")[0]
    try:
        hdulist.writeto(os.path.join(outputDir,label0,reduc_spectrum,prefix+"_calib.fits"), overwrite=True)
    except TypeError:
        hdulist.writeto(os.path.join(outputDir,label0,reduc_spectrum,prefix+"_calib.fits"), clobber=True)
    hdulist.close()


    ###########################################################################################
    ## Save contrast curve as csv
    ###########################################################################################
    cont_stddev,sep_bins = get_image_stddev(FMCont_calib, centroid = center)
    # Separation samples in pixels
    sep_bins_center =  (np.array([r_tuple[0] for r_tuple in sep_bins]))
    # Approximative contrast curve at these separations
    contrast_curve = 5*np.array(cont_stddev)/approx_throughput
    savecontrast = os.path.join(outputDir,label0,reduc_spectrum,prefix+"_FMMF_contrast.csv")
    with open(savecontrast, 'w+') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerows([["Seps",fakes_spectrum]])
        contrast_curve[np.where(np.isnan(contrast_curve))] = -1.
        not_neg = np.where(contrast_curve>0)
        csvwriter.writerows([[a,b] for a,b in zip(sep_bins_center*pix2as,
                                contrast_curve[not_neg])])



    ###########################################################################################
    ## Save detection table as csv
    ###########################################################################################
    from pyklip.kpp.detection.detection import point_source_detection
    # list of the local maxima with their info
    #         Description by column: ["index","value","PA","Sep (pix)","Sep (as)","x","y","row","col"]
    #         1/ index of the candidate
    #         2/ Value of the maximum
    #         3/ Position angle in degree from North in [0,360]
    #         4/ Separation in pixel
    #         5/ Separation in arcsec
    #         6/ x position in pixel
    #         7/ y position in pixel
    #         8/ row index
    #         9/ column index
    candidates_table = point_source_detection(FMMF_calib, center,detec_threshold,pix2as=pix2as,
                                             mask_radius = 15,maskout_edge=10,IWA=None, OWA=None)
    savedetections = os.path.join(outputDir,label0,reduc_spectrum,prefix+"_FMMF_detections.csv")
    with open(savedetections, 'w+') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerows([["index","value","PA","Sep (pix)","Sep (as)","x","y","row","col"]])
        csvwriter.writerows(candidates_table)

    return 1

def calibrate_SNR_map_1D(FMMF_map,FMMF_map_noOI=None,center=None):

    stddev_map = get_image_stat_map_perPixMasking(FMMF_map,FMMF_map_noOI,
                                               mask_radius=15,centroid=center,
                                               Dr = 2,
                                               type = "stddev")

    return FMMF_map/stddev_map,stddev_map


def calibrate_contrast_map_1D(FMCont_map,FMCont_map_masked,fakes_FMCont_list,fakeinfohdr_list,center,pix2as,target_name,MJDOBS=None,IOWA=None):

    if IOWA is None:
        IWA,OWA,inner_mask,outer_mask = get_occ(FMCont_map, centroid = center)
        IOWA = (IWA,OWA)
        IOWA_as = (pix2as*IWA,pix2as*OWA)

    real_contrast_list = []
    sep_list = []
    pa_list = []
    metric_fakes_val = []
    metric_nofakes_val = []

    for fakes_FMCont,fakeinfohdr in zip(fakes_FMCont_list,fakeinfohdr_list):
        row_list,col_list = get_pos_known_objects(fakeinfohdr,target_name,pix2as,MJDOBS=MJDOBS,center=center,fakes_only=True)
        sep,pa = get_pos_known_objects(fakeinfohdr,target_name,pix2as,MJDOBS=MJDOBS,center=center,pa_sep=True,fakes_only=True)
        sep_list.extend(sep)
        pa_list.extend(pa)
        for fake_id in range(100):
            try:
                real_contrast_list.append(fakeinfohdr["FKCONT{0:02d}".format(fake_id)])
            except:
                continue
        for (row,col) in zip(row_list,col_list):
            try:
                metric_fakes_val.append(fakes_FMCont[int(np.round(row)),int(np.round(col))])
                metric_nofakes_val.append(FMCont_map_masked[int(np.round(row)),int(np.round(col))])
            except:
                metric_fakes_val.append(np.nan)
                metric_nofakes_val.append(np.nan)
    metric_fakes_val =  np.array(metric_fakes_val) - np.array(metric_nofakes_val)

    whereNoNans = np.where(np.isfinite(metric_fakes_val))
    metric_fakes_val = metric_fakes_val[whereNoNans]
    sep_list = np.array(sep_list)[whereNoNans]
    pa_list = np.array(pa_list)[whereNoNans]
    real_contrast_list =  np.array(real_contrast_list)[whereNoNans]

    sep_list,pa_list,metric_fakes_val,real_contrast_list = zip(*sorted(zip(sep_list,pa_list,metric_fakes_val,real_contrast_list)))
    metric_fakes_val = np.array(metric_fakes_val)
    sep_list =  np.array(sep_list)
    pa_list =  np.array(pa_list)
    real_contrast_list =  np.array(real_contrast_list)

    fake_info = (sep_list,pa_list,metric_fakes_val,real_contrast_list)

    whereInRange = np.where((sep_list>IOWA_as[0])*(sep_list<IOWA_as[1]))
    metric_fakes_in_range = metric_fakes_val[whereInRange]
    cont_in_range = real_contrast_list[whereInRange]
    unique_sep = np.unique(sep_list[whereInRange])
    mean_throughput = np.zeros(len(unique_sep))
    std_throughput = np.zeros(len(unique_sep))
    for k,sep_it in enumerate(unique_sep):
        where_sep = np.where(sep_list==sep_it)
        mean_throughput[k] = np.nanmean(metric_fakes_in_range[where_sep]/cont_in_range[where_sep])
        var_throughput = np.nanvar(metric_fakes_in_range[where_sep]/cont_in_range[where_sep])
        std_throughput[k] = np.sqrt(var_throughput)
    from  scipy.interpolate import interp1d
    throughput_func = interp1d(unique_sep,mean_throughput,bounds_error=False, fill_value=np.nan)
    std_throughput_func = interp1d(unique_sep,std_throughput,bounds_error=False, fill_value=np.nan)

    ny,nx = FMCont_map.shape
    x_grid, y_grid = np.meshgrid(np.arange(nx * 1.)-center[0], np.arange(ny * 1.)-center[1])
    r_grid = abs(x_grid +y_grid*1j)
    pa_grid = np.arctan2( -x_grid,y_grid) % (2.0 * np.pi)

    throughput_map = throughput_func(pix2as*r_grid)

    if 0:
        import matplotlib.pyplot as plt
        conversion = metric_fakes_val/real_contrast_list
        ###########################################################################################
        ## PLOT Conversion Factor
        ###########################################################################################
        sep_cont_samples = np.linspace(IOWA_as[0],IOWA_as[1],100)
        contrast_conv = throughput_func(sep_cont_samples)

        plt.figure(1)
        plt.subplot(1,3,1)
        plt.title("Throughput", fontsize=10,color="black") #y=1.08
        # [i.set_color("white") for i in ax.spines.itervalues()]
        plt.scatter(sep_list,conversion,c="black")
        plt.plot(sep_cont_samples,contrast_conv,linestyle="-",linewidth=1,color="black")
        plt.fill_between(sep_cont_samples,contrast_conv-std_throughput_func(sep_cont_samples),contrast_conv+std_throughput_func(sep_cont_samples),color="grey",alpha = 0.5)
        plt.xlabel("Separation (Arcsec)", fontsize=10,color="black")
        plt.ylabel("Throughput", fontsize=10,color="black")
        # plt.xlim(IOWA_as)
        plt.ylim([np.min(conversion),np.max(conversion)])
        ax = plt.gca()
        ax.grid(which='major',axis="both",linestyle="-",color="grey")
        ax.grid(which='minor',axis="both",linestyle="--",color="grey")

        ###########################################################################################
        ## PLOT Conversion Factor map
        ###########################################################################################
        plt.subplot(1,3,2,projection="polar")
        plt.title("Throughput Map", fontsize=10,color="black",y=-0.75)
        sc = plt.scatter(np.radians(pa_list),sep_list,
                         c=100*(conversion),
                         s=50*(conversion),cmap="viridis")
        ax = plt.gca()
        ax.set_rmax(1.)
        ax.set_theta_zero_location("N")
        cbar = plt.colorbar(sc,orientation="horizontal")
        cbar.set_label("Relative Deviation (%)", fontsize=10,color="black")
        cbar.ax.tick_params(axis='x', labelsize=10,colors="black")
        plt.clim([95,105])
        plt.grid(which='major',axis="both",linestyle="-",color="black")
        ax.set_thetagrids(np.arange(0,360,45),frac=1.3)

        ###########################################################################################
        plt.subplot(1,3,3)
        plt.imshow(throughput_map,interpolation="nearest")
        plt.show()


    return FMCont_map/throughput_map,throughput_map,fake_info


if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    print("CPU COUNT: {0}".format(mp.cpu_count()))

    ###########################################################################################
    ## Contrast curve parameters
    ###########################################################################################

    mypath = os.path.dirname(os.path.realpath(__file__))
    inputDir = os.path.join(mypath,"..","tests","data")
    outputDir =  os.path.join(mypath,"..","tests","data_kpop")
    dir_fakes =  os.path.join(mypath,"..","tests","data-Fakes")

    mvt = 0.7
    ## T-type
    reduc_spectrum = "t600g100nc"
    fakes_spectrum = "t1000g100nc"
    # The flux of the fake is based on the flux map of the first reduction but we don't know the throughput yet. So this value is used instead.
    approx_throughput = 0.5
    ## L-type
    # reduc_spectrum = "t1300g100f2"
    # fakes_spectrum = "t1300g100f2"
    # approx_throughput = 1.0

    HPF = False

    contrast_dataset(inputDir,outputDir,dir_fakes,mvt,reduc_spectrum,fakes_spectrum,approx_throughput,HPF=HPF)
