__author__ = 'jruffio'

from klipPostProcessing import *

from scipy.special import erf
import platform
import os

if __name__ == "__main__":
    print(platform.system())

    OS = platform.system()
    if OS == "Windows":
        print("Using WINDOWS!!")
    else:
        print("I hope you are using a UNIX OS")

    print(os.environ['PYTHONPATH'].split(os.pathsep))
    print("path sep: " + os.path.sep)

    if 1:
        if OS == "Windows":
            campaign_dir = "C:\\Users\\JB\\Dropbox (GPI)\\GPIDATA\\"
        else:
            campaign_dir = "/Users/jruffio/Dropbox (GPI)/GPIDATA/"
        planet_detection_campaign(campaign_dir)


    if 0:
        outputDir = ''
        star_type = ''
        metrics = None


        if OS == "Windows":
            #inputDir = "C:\\Users\\JB\\Dropbox (GPI)\\GPIDATA\\HD\\autoreduced\\"
            #inputDir = "C:\\Users\\JB\\Dropbox (GPI)\\GPIDATA\\HD_100491\\autoreduced\\"
            inputDir = "C:\\Users\\JB\\Dropbox (GPI)\\GPIDATA\\bet_cir\\autoreduced\\"
            #inputDir = "C:\\Users\\JB\\Dropbox (GPI)\\GPIDATA\\c_Eri\\autoreduced\\"
            #inputDir = "/Users/jruffio/Dropbox (GPI)/GPIDATA/HD19467/autoreduced/"
            #inputDir = "/Users/jruffio/gpi/pyklip/outputs/baade"
            #inputDir = "/Users/jruffio/gpi/pyklip/outputs/test_detec_folder"
            #inputDir = "/Users/jruffio/gpi/pyklip/outputs/dropbox_prior_test/"
            #inputDir = "/Users/jruffio/gpi/pyklip/outputs/Baade_5im"
            #outputDir = "C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\"
            spectrum_model = ["C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\code\\spectra\\t800g100nc.flx",""] #"C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\code\\spectra\\g100ncflx\\t2400g100nc.flx",
            user_defined_PSF_cube = "C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\code\\pyklipH-S20141218-k100a7s4m3-original_radial_PSF_cube.fits"
        else:
            inputDir = "/Users/jruffio/Dropbox (GPI)/GPIDATA/beta_pictoris/autoreduced/"
            #inputDir = "/Users/jruffio/Dropbox (GPI)/GPIDATA/bet_cir/autoreduced/"
            #inputDir = "/Users/jruffio/Dropbox (GPI)/GPIDATA/c_Eri/autoreduced/"
            #inputDir = "/Users/jruffio/Dropbox (GPI)/GPIDATA/HD19467/autoreduced/"
            #inputDir = "/Users/jruffio/gpi/pyklip/outputs/baade"
            #inputDir = "/Users/jruffio/gpi/pyklip/outputs/test_detec_folder"
            #inputDir = "/Users/jruffio/gpi/pyklip/outputs/dropbox_prior_test/"
            #inputDir = "/Users/jruffio/gpi/pyklip/outputs/Baade_5im"
            #outputDir = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB"
            #spectrum_model = ""
            spectrum_model = ["/Users/jruffio/gpi/pyklip/spectra/t800g100nc.flx",""]
            user_defined_PSF_cube = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB/code/pyklipH-S20141218-k100a7s4m3-original_radial_PSF_cube.fits"

        numbasis = 20
        star_type = "G4"
        #filename_filter = "pyklip-S20141218-k100a7s4m3"
        filename_filter = "pyklip-*-k100a7s4m3"
        #filename_filter = "*baade*"
        #spectrum_model = ""
        metrics = []#"weightedFlatCube","matchedFilter","shape"
        planet_detection_in_dir(inputDir,
                                outputDir=outputDir,
                                filename_prefix_is=filename_filter,
                                spectrum_model=spectrum_model,
                                star_type=star_type,
                                metrics = metrics,
                                numbasis=numbasis,
                                user_defined_PSF_cube=user_defined_PSF_cube,
                                metrics_only = False,
                                planet_detection_only = True,
                                threads = True,
                                mute = False,
                                SNR_only = False,
                                probability_only = True)
        #'C:\\Users\\JB\\Dropbox (GPI)\\GPIDATA\\c_Eri\\autoreduced\\\\\\planet_detec_pyklip-S20141218-k100a7s4m3_KL20\\t800g100nc\\\\c_Eri-shape.fits'

    if 0:
        GOI_list = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB/GOI_list.txt"

        logfilename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/c_Eri/autoreduced/planet_detec_pyklip-S20141218-k100a7s4m3_KL20/t800g100nc/c_Eri-detectionLog_all.txt"
        candidate_indices = [1]
        candidate_status = ["pl"]
        object_name = "c_eri"
        confirm_candidates(GOI_list, logfilename, candidate_indices,candidate_status,object_name)

        logfilename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/bet_Cir/autoreduced/planet_detec_pyklip-S20150408-k100a7s4m3_KL20/t800g100nc/bet_Cir-detectionLog_all.txt"
        candidate_indices = [1,2]
        candidate_status = ["bg","bg"]
        object_name = "bet_Cir"
        confirm_candidates(GOI_list, logfilename, candidate_indices,candidate_status,object_name)

        logfilename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/beta_pictoris/autoreduced/planet_detec_pyklip-S20150402-k100a7s4m3_KL20/t800g100nc/beta_pictoris-detectionLog_all.txt"
        candidate_indices = [1]
        candidate_status = ["pl"]
        object_name = "beta_pictoris"
        confirm_candidates(GOI_list, logfilename, candidate_indices,candidate_status,object_name)

        logfilename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/HD_28287/autoreduced/planet_detec_pyklip-S20141112-k100a7s4m3_KL20/t800g100nc/HD_28287-detectionLog_all.txt"
        candidate_indices = [1]
        candidate_status = ["bg"]
        object_name = "HD_28287"
        confirm_candidates(GOI_list, logfilename, candidate_indices,candidate_status,object_name)

        logfilename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/HD_88117/autoreduced/planet_detec_pyklip-S20150129-k100a7s4m3_KL20/t800g100nc/HD_88117-detectionLog_all.txt"
        candidate_indices = [1]
        candidate_status = ["tbd"]
        object_name = "HD_88117"
        confirm_candidates(GOI_list, logfilename, candidate_indices,candidate_status,object_name)

        logfilename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/HD_100491/autoreduced/planet_detec_pyklip-S20150129-k100a7s4m3_KL20/t800g100nc/HD_100491-detectionLog_all.txt"
        candidate_indices = [1]
        candidate_status = ["tbd"]
        object_name = "HD_100491"
        confirm_candidates(GOI_list, logfilename, candidate_indices,candidate_status,object_name)

        logfilename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/HD_118991_A/autoreduced/planet_detec_pyklip-S20150404-k100a7s4m3_KL20/t800g100nc/HD_118991_A-detectionLog_all.txt"
        candidate_indices = [1]
        candidate_status = ["tbd"]
        object_name = "HD_118991_A"
        confirm_candidates(GOI_list, logfilename, candidate_indices,candidate_status,object_name)

        if 0:
            logfilename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/HD_155114/autoreduced/planet_detec_pyklip-S20150408-k100a7s4m3_KL20/t800g100nc/HD_155114-detectionLog_all.txt"
            candidate_indices = [1]
            candidate_status = ["tbd"]
            object_name = "HD_155114"
            confirm_candidates(GOI_list, logfilename, candidate_indices,candidate_status,object_name)
        # Copy paste next line in GOI_list
        #HD_155114, tbd edge, 1, True, 5.87464777187, -76.0, 90.0, 230, 64

        logfilename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/HD_164249_A/autoreduced/planet_detec_pyklip-S20150501-k100a7s4m3_KL20/t800g100nc/HD_164249_A-detectionLog_all.txt"
        candidate_indices = [1]
        candidate_status = ["tbd"]
        object_name = "HD_164249_A"
        confirm_candidates(GOI_list, logfilename, candidate_indices,candidate_status,object_name)

        logfilename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/HR_4669/autoreduced/planet_detec_pyklip-S20150504-k100a7s4m3_KL20/t800g100nc/HR_4669-detectionLog_all.txt"
        candidate_indices = [1,3]
        candidate_status = ["tbd","tbd"]
        object_name = "HR_4669"
        confirm_candidates(GOI_list, logfilename, candidate_indices,candidate_status,object_name)
        # Add this object too
        #HR_4669, tbd edge, 1, True, 8.10705215173, -118.0, -11.0, 129, 22

        logfilename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/V371_Nor/autoreduced/planet_detec_pyklip-S20150504-k100a7s4m3_KL20/t800g100nc/V371_Nor-detectionLog_all.txt"
        candidate_indices = [1,2]
        candidate_status = ["tbd","tbd"]
        object_name = "V371_Nor"
        confirm_candidates(GOI_list, logfilename, candidate_indices,candidate_status,object_name)

        logfilename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/HD_161719/autoreduced/planet_detec_pyklip-S20150404-k100a7s4m3_KL20/t800g100nc/HD_161719-detectionLog_all.txt"
        candidate_indices = [1,2,3,4,5] # There shoudl be 3 more candidates but they are very faint...
        candidate_status = ["tbd","tbd","tbd","tbd","tbd"]
        object_name = "HD_161719"
        confirm_candidates(GOI_list, logfilename, candidate_indices,candidate_status,object_name)

        logfilename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/HD19467/autoreduced/planet_detec_pyklip-S20150201-k100a7s4m3_KL20/t800g100nc/HD19467-detectionLog_all.txt"
        candidate_indices = [1]
        candidate_status = ["tbd"]
        object_name = "HD19467"
        confirm_candidates(GOI_list, logfilename, candidate_indices,candidate_status,object_name)


    if 0:
        if OS == "Windows":
            campaign_dir = "C:\\Users\\JB\\Dropbox (GPI)\\GPIDATA\\"
        else:
            campaign_dir = "/Users/jruffio/Dropbox (GPI)/GPIDATA/"
        clean_planet_detec_outputs(campaign_dir)

####################################################################################
####################### analyze single no Jason's file ###############################################
####################################################################################

    if 0:
        filename = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/rameauj/GOI-2/goi2_HDec_cadi_speccube.fits"
            #filename = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/rameauj/GOI-2/goi2_JJan_loci_speccube.fits"
            #filename = "/Users/jruffio/gpi/pyklip/histo_inout/pyklip-S20141218-k100a7s4m3-KL1-speccube_H.fits"
        outputDir = "/Users/jruffio/gpi/pyklip/outputs/dropbox_prior_test/"
        folderName = "/planet_detec_"+"goi2_HDec_cadi_speccube"+"/"
        tofits = "goi2_HDec_cadi_speccube"
            #folderName = "/planet_detec_"+"goi2_JJan_loci_speccube"+"/"
            #tofits = "goi2_JJan_loci_speccube"
            #folderName = "/planet_detec_"+"goi2_H_KL1"+"/"
            #tofits = "goi2_H_KL1"
        PSF_filename = "/Users/jruffio/gpi/pyklip/histo_inout/pyklip-baade-original_radial_PSF_cube.fits"
        spectrum_model = "/Users/jruffio/gpi/pyklip/t800g100nc.flx"
        star_type = "G4"
        filter = "H"
        #filter = "J"
        pipeline_dir = "/Users/jruffio/gpi/pipeline/"

        hdulist = pyfits.open(PSF_filename)
        PSF_cube = hdulist[1].data[:,::-1,:]
        sat_spot_spec = np.nanmax(PSF_cube,axis=(1,2))
        nl_PSF, ny_PSF, nx_PSF = PSF_cube.shape
        sat_spot_spec /= np.sqrt(np.nansum(sat_spot_spec**2))
        wv,planet_sp = spec.get_planet_spectrum(spectrum_model,filter)
        planet_sp /= np.sqrt(np.nansum(planet_sp**2))
        # Interpolate a spectrum of the star based on its spectral type/temperature
        wv,star_sp = spec.get_star_spectrum(pipeline_dir,filter,star_type)
        star_sp /= np.sqrt(np.nansum(star_sp**2))
        spectrum = (sat_spot_spec/star_sp)*planet_sp
        spectrum /= np.sqrt(np.nansum(spectrum**2))


        PSF_planet = copy(PSF_cube)
        for l in range(nl_PSF):
            PSF_planet[l,:,:] /= sat_spot_spec[l]
        PSF_planet /= np.sqrt(np.sum(PSF_planet**2))

        candidate_detection(filename,
                            PSF = PSF_planet,
                            outputDir = outputDir,
                            folderName = folderName,
                            toDraw=False,
                            toFits=tofits,
                            toPNG=tofits,
                            logFile=tofits,
                            spectrum=spectrum,
                            mute = False)




####################################################################################
####################### CODE GARBAGE ###############################################
####################################################################################

#[16.60295306150426, 32.41140515339537, 48.219857245286484, 64.02830933717762, 79.83676142906873]
#[ 362.14315877,  272.14315877,  182.14315877,   92.14315877]

'''
    thetas = 90+np.array([54.7152978, -35.2847022, -125.2847022, -215.2847022])
    radii = np.array([16.57733255996081, 32.33454364876502, 48.09175473756924, 63.84896582637346, 79.60617691517767])
    filename_PSFs = "/Users/jruffio/gpi/pyklip/outputs/baade-PSFs-KLmodes-all-PSFs.fits"
    #filename_PSFs = "/Users/jruffio/gpi/pyklip/outputs/baade-PSFs2-KLmodes-all-PSFs.fits"
    #filename_PSFs = "/Users/jruffio/gpi/pyklip/outputs/baade-PSFs15-KLmodes-all-PSFs.fits"
    #filename_PSFs = "/Users/jruffio/gpi/pyklip/outputs/baade-PSFs20-KLmodes-all-PSFs.fits"
    if 0:
        PSF = extract_merge_PSFs(filename_PSFs, radii, thetas)
    #print(PSF.shape)
    #plt.figure(20)
    #plt.imshow(PSF,interpolation = 'nearest')
    #plt.show()


    #filelist = glob.glob("/Users/jruffio/gpi/pyklip/outputs/HD114174-KL20-speccube.fits")

    #dataset = GPI.GPIData(filelist)


    filename = "/Users/jruffio/gpi/pyklip/outputs/baade/baade-h-k300a7s4m3-KL20-speccube.fits"
    filename = "/Users/jruffio/Dropbox/GPIDATA/c_Eri/autoreduced/pyklip-S20141218-k100a5s4m3-KL20-speccube.fits"

    tofits = ''
    outputDir = "/Users/jruffio/gpi/pyklip/outputs/"
    folderName = "/baade_out"
    folderName = "/c_Eri"
    tofits = folderName
    spectrumHEri = np.array([141.9296769,   198.73423838 , 242.22243818,  299.85427039,  335.37585698, # spec c_eri H
  370.88441648 , 398.06479267 , 411.32024875 , 424.47542857,  429.61704089,
  421.74684239 , 377.85252056 , 305.35767397 , 247.08267071, 196.72520657,
  138.06660696 , 104.14917438 , 104.38252497 ,  95.81786937 ,  73.66560944,
   37.55078903 ,  37.32665492  , 46.10064065  , 35.90442611  , 18.09452635,
   25.52058438  , 34.30831346  , 13.11016915  ,  0.95286082  ,  7.54171253,
   10.2595994 ,   10.93531723  , 15.62122062 ,   7.88143819   , 4.68372895])
    spectrumH = np.array([0.0,141.9296769,   198.73423838 , 242.22243818,  299.85427039,  335.37585698, # spec c_eri H
  370.88441648 , 398.06479267 , 411.32024875 , 424.47542857,  429.61704089,
  421.74684239 , 377.85252056 , 305.35767397 , 247.08267071, 196.72520657,
  138.06660696 , 104.14917438 , 104.38252497 ,  95.81786937 ,  73.66560944,
   37.55078903 ,  37.32665492  , 46.10064065  , 35.90442611  , 18.09452635,
   25.52058438  , 34.30831346  , 13.11016915  ,  0.95286082  ,  7.54171253,
   10.2595994 ,   10.93531723  , 15.62122062 ,   7.88143819   , 4.68372895,0.0])
    spectrumJ = np.array([ -0.11320329 , -0.07596916 , -2.7990458 ,  -3.79993844 , -4.38923454,  # spec c_eri J
  -5.78503418 , -6.53705502 , -4.93471336 , -4.2245512 ,  -2.4959271,
   0.46197677 ,  0.99514222 ,  4.06949472 , 10.77979755 , 11.6944828,
  20.47476196 , 20.85378075 , 19.65199089 , 24.29648018 , 27.36954117,
  34.65512085 , 40.13297653 , 35.40809631 , 31.18817902 , 32.92845154,
  36.96004486 , 39.82872009 , 41.06404877 , 35.63868332 , 30.73090744,
  27.26691818 , 23.48490906 , 21.04096222 , 14.74596024 ,  8.45730782,
   3.58565426 ,  0.78945696])
    #spectrum_flat = np.ones(37)
    #spectrum = spectrum[1:36]
    #spectrum = np.ones(5)
    #spectrum[0] = 0.0

    #candidate_detection(filename, outputDir = outputDir, folderName = folderName, PSF = None, toDraw=True, toFits=tofits,toPNG=tofits,logFile=tofits, spectrum=spectrumHEri)#toPNG="Baade", logFile='Baade')
        #candidate_detection(filename,PSF = None, toDraw=True, toFits="HD114174")#toPNG="HD114174", logFile='HD114174')
'''


def flatten_annulus(im,rad_bounds,center):
    ny,nx = im.shape

    x, y = np.meshgrid(np.arange(ny), np.arange(nx))
    #x.shape = (x.shape[0] * x.shape[1])
    #y.shape = (y.shape[0] * y.shape[1])
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    phi = np.arctan2(y - center[1], x - center[0])

    r_min, r_max = rad_bounds
    annulus_indices = np.where((r >= r_min) & (r < r_max))
    ann_y_id = annulus_indices[0]
    ann_x_id = annulus_indices[1]

    annulus_pix = slice_PSFs[ann_y_id,ann_x_id]
    annulus_x = x[ann_y_id,ann_x_id] - center[0]
    annulus_y = y[ann_y_id,ann_x_id]- center[1]
    annulus_r = r[ann_y_id,ann_x_id]
    annulus_phi = phi[ann_y_id,ann_x_id]

    annulus_pix = annulus_pix.reshape(np.size(annulus_pix))
    annulus_x = annulus_x.reshape(np.size(annulus_pix))
    annulus_y = annulus_y.reshape(np.size(annulus_pix))
    annulus_r = annulus_r.reshape(np.size(annulus_pix))
    annulus_phi = annulus_phi.reshape(np.size(annulus_pix))

    #slice_PSFs[ann_y_id,ann_x_id] = 100
    #plt.imshow(slice_PSFs,interpolation = 'nearest')
    #plt.show()

    n_phi = np.floor(2*np.pi*r_max)
    dphi = 2*np.pi/n_phi
    #r_arr, phi_arr = np.meshgrid(np.arange(np.ceil(r_min),np.ceil(r_min)+n_r,0.01),np.arange(-np.pi,np.pi,dphi))
    r_arr, phi_arr = np.meshgrid(np.arange(np.min(annulus_r),np.max(annulus_r),1),np.arange(-np.pi,np.pi,dphi))

    points = np.array([annulus_r,(r_max+r_min)/2.*annulus_phi])
    points = points.transpose()
    grid_z2 = griddata(points, annulus_pix, (r_arr, (r_max+r_min)/2.*phi_arr), method='cubic')

    '''
    tck = bisplrep(annulus_x, annulus_y, annulus_pix)
    nrow, ncol = r_arr.shape
    #znew = bisplev(35, 35, tck)
    #print(znew)
    #print(np.min(annulus_x))
    #print(np.max(annulus_x))
    znew = np.zeros([nrow, ncol])
    for i_it in range(nrow):
        for j_it in range(ncol):
            xp = r_arr[i_it,j_it]*np.cos(phi_arr[i_it,j_it])
            yp = r_arr[i_it,j_it]*np.sin(phi_arr[i_it,j_it])
            znew[i_it,j_it] = bisplev(xp, yp, tck)
    #xnew, ynew = np.meshgrid(np.arange(np.min(annulus_x),np.max(annulus_x),1), np.arange(np.min(annulus_x),np.max(annulus_x),1))
    #znew = bisplev(xnew[:,0], ynew[0,:], tck)
    '''

    plt.figure(2)
    plt.imshow(grid_z2.transpose(),interpolation = 'nearest',extent=[-np.pi*((r_max+r_min)/2.),np.pi*((r_max+r_min)/2.),np.min(annulus_r),np.max(annulus_r)], origin='lower')
    plt.plot(points[:,1], points[:,0],'b.')
    #plt.imshow(znew,interpolation = 'nearest')
    plt.show()

    return grid_z2

def badPixelFilter(cube,scale = 2,maxDeviation = 10):

    cube_cpy = cube[:]

    if np.size(cube_cpy.shape) == 3:
        nl,ny,nx = cube_cpy.shape
    elif np.size(cube_cpy.shape) == 2:
        ny,nx = cube_cpy.shape
        cube_cpy = cube_cpy[None,:]
        nl = 1

    smooth_cube = np.zeros((nl,ny,nx))

    ker = np.ones((3,3))/8.0
    n_bad = np.zeros((nl,))
    ker[1,1] = 0.0
    for l_id in np.arange(nl):
        smooth_cube[l_id,:,:] = convolve2d(cube_cpy[l_id,:,:], ker, mode='same')
        slice_diff = cube_cpy[l_id,:,:] - smooth_cube[l_id,:,:]
        stdmap = radialStdMap(slice_diff,2,10)
        bad_pixs = np.where(abs(slice_diff) > maxDeviation*stdmap)
        n_bad[l_id] = np.size(bad_pixs)
        bad_pixs_x, bad_pixs_y = bad_pixs
        cube_cpy[l_id,bad_pixs_x, bad_pixs_y] = np.nan ;

    return cube_cpy,bad_pixs

def radialMed(cube,dr,Dr,centroid = None, r = None, r_samp = None):
    '''
    Return the mean with respect to the radius computed in annuli of radial width Dr separated by dr.
    :return:
    '''
    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        cube = cube[None,:]
        nl = 1

    if centroid is None :
        x_cen = np.ceil((nx-1)/2) ; y_cen = np.ceil((ny-1)/2)
    else:
        x_cen, y_cen = centroid

    if r is None:
        r = [np.nan]
    if r_samp is None:
        r_samp = [np.nan]

    x, y = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
    r[0] = abs(x +y*1j)
    r_samp[0] = np.arange(dr,max(r[0].reshape(np.size(r[0]))),dr)

    radial_std = np.zeros((nl,np.size(r_samp[0])))

    for r_id, r_it in enumerate(r_samp[0]):
        selec_pix = np.where( ((r_it-Dr/2.0) < r[0]) * (r[0] < (r_it+Dr/2.0)) )
        selec_y, selec_x = selec_pix
        radial_std[:,r_id] = nanmedian(cube[:,selec_y, selec_x],1)

    radial_std = np.squeeze(radial_std)

    return radial_std


def radialMean(cube,dr,Dr,centroid = None, r = None, r_samp = None):
    '''
    Return the mean with respect to the radius computed in annuli of radial width Dr separated by dr.
    :return:
    '''
    if np.size(cube.shape) == 3:
        nl,ny,nx = cube.shape
    elif np.size(cube.shape) == 2:
        ny,nx = cube.shape
        cube = cube[None,:]
        nl = 1

    #TODO centroid should be different for each slice?
    if centroid is None :
        x_cen = np.ceil((nx-1)/2) ; y_cen = np.ceil((ny-1)/2)
    else:
        x_cen, y_cen = centroid

    if r is None:
        r = [np.nan]
    if r_samp is None:
        r_samp = [np.nan]

    x, y = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
    r[0] = abs(x +y*1j)
    r_samp[0] = np.arange(dr,max(r[0].reshape(np.size(r[0]))),dr)

    radial_std = np.zeros((nl,np.size(r_samp[0])))

    for r_id, r_it in enumerate(r_samp[0]):
        selec_pix = np.where( ((r_it-Dr/2.0) < r[0]) * (r[0] < (r_it+Dr/2.0)) )
        selec_y, selec_x = selec_pix
        radial_std[:,r_id] = np.nanmean(cube[:,selec_y, selec_x],1)

    radial_std = np.squeeze(radial_std)

    return radial_std



def candidate_detection(metrics_foldername,
                        PSF = None,
                        outputDir = None,
                        folderName = None,
                        toPNG='',
                        toFits='',
                        toDraw=False,
                        logFile='',
                        spectrum = None,
                        mute = False ):
    '''
    Should take into account PSF wavelength dependence.
    3d convolution to take into account spectral shift if useful
    but 3d conv takes too long

    Inputs:
        filename: Path and name of the fits file to be analyzed.
        PSF: User-defined 2D PSF. If None, gaussian PSF is assumed.
        allmodes:

        outputDir: Directory where to save the outputs
        toPNG: Save some plots as PNGs. toPNG must be a string being a prefix of the filename of the images.
        toFits: Save some fits files in memory. toFits must be a string being a prefix of the filename of the images.
        toDraw: Plot some figures using matplotlib.pyplot. First a SNR map with the candidate list and a criterion map
                with all the checked spots. toDraw is a string ie the prefix of the filename.
        logFile: Log the result of the detection in text files. logFile is a string ie the prefix of the filename.

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
        if PSF is not None:
            PSF_cube = PSF
            if np.size(np.shape(PSF)) != 3:
                if not mute:
                    print("Wrong PSF dimensions. Image is 3D.")
                return 0
            # The PSF is user-defined.
            nl, ny_PSF, nx_PSF = PSF_cube.shape
            tmp_spectrum = np.nanmax(PSF_cube,axis=(1,2))
        else:
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
            PSF_cube = np.tile(PSF,(nl,1,1))

        if spectrum is not None:
            for k in range(nl):
                PSF_cube[k,:,:] *= spectrum[k]
        else:
            if PSF is not None:
                spectrum = tmp_spectrum
            else:
                spectrum = np.ones(nl)

        spectrum /= np.sqrt(np.nansum(spectrum**2))
        # normalize PSF with norm 2.
        PSF_cube /= np.sqrt(np.sum(PSF_cube**2))

    else: # Assuming 2D image
        flat_cube = cube

        # Build the PSF.
        if PSF is not None:
            if np.size(np.shape(PSF)) != 2:
                if not mute:
                    print("Wrong PSF dimensions. Image is 2D.")
                return 0
            # The PSF is user-defined.
            # normalize PSF with norm 2.
            PSF /= np.sqrt(np.sum(PSF**2))
            ny_PSF, nx_PSF = PSF.shape
        else:
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


    try:
        # Retrieve the center of the image from the fits keyword.
        center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    except:
        # If the keywords could not be found.
        if not mute:
            print("Couldn't find PSFCENTX and PSFCENTY keywords.")
        center = [(nx-1)/2,(ny-1)/2]

    candidates_KLs_list = []


    # Smoothing of the image. Remove the median of an arc centered on each pixel.
    # Actually don't do pixel per pixel but consider small boxes.
    # This function has to be cleaned.
    #flat_cube = subtract_radialMed(flat_cube,2,20,center)
    flat_cube_cpy = copy(flat_cube)
    flat_cube_nans = np.where(np.isnan(flat_cube))


    # Build as grids of x,y coordinates.
    # The center is in the middle of the array and the unit is the pixel.
    # If the size of the array is even 2n x 2n the center coordinates is [n,n].
    x_grid, y_grid = np.meshgrid(np.arange(0,nx,1)-center[0],np.arange(0,ny,1)-center[1])

            # Replace nans by zeros.
            # Otherwise we loose the border of the image because of the convolution which will give NaN if there is any NaNs in
            # the area.
            # /!\ Desactivated because there is no hope in real life to get anything there anyway. Only for Baade's window...
            #flat_cube[np.where(np.isnan(flat_cube))] = 0.0
            #flat_cube = copy(flat_cube_cpy)

            # Perform a "match-filtering". Simply the convolution of the transposed PSF with the image.
            # It should still be zero if there is no signal. Assuming a zero mean noise after KLIP.
            # The value at a point centered on a planet should be the L2 norm of the planet.
            # /!\ Desactivated because matched filtering doesn't work on correlated images.
            #flat_cube_convo = convolve2d(flat_cube,PSF,mode="same")
            # The 3d convolution takes a while so the idea is to detect the interesting spot in the 2d flat cube and then
            # perform the 3d convolution on the cube stamp around it.

    # Calculate the standard deviation map.
    # the standard deviation is calculated on annuli of width Dr. There is an annulus centered every dr.
    dr = 2 ; Dr = 5 ;
    flat_cube_std = radialStdMap(flat_cube,dr,Dr, centroid=center)

    # Divide the convolved flat cube by the standard deviation map to get the SNR.
    flat_cube_SNR = flat_cube/flat_cube_std


    # Definition of the different masks used in the following.
    stamp_nrow = 13
    stamp_ncol = 13
    # Mask to remove the spots already checked in criterion_map.
    stamp_x_grid, stamp_y_grid = np.meshgrid(np.arange(0,stamp_nrow,1)-6,np.arange(0,stamp_ncol,1)-6)
    stamp_mask = np.ones((stamp_nrow,stamp_ncol))
    r_stamp = abs((stamp_x_grid) +(stamp_y_grid)*1j)
    stamp_mask[np.where(r_stamp < 4.0)] = np.nan
    stamp_mask_small = np.ones((stamp_nrow,stamp_ncol))
    stamp_mask_small[np.where(r_stamp < 2.0)] = 0.0
    stamp_cube_small_mask = np.tile(stamp_mask_small[None,:,:],(nl,1,1))




    shape_crit_map = -np.ones((ny,nx))
        #ortho_criterion_map = np.zeros((ny,nx))
    row_m = np.floor(ny_PSF/2.0)
    row_p = np.ceil(ny_PSF/2.0)
    col_m = np.floor(nx_PSF/2.0)
    col_p = np.ceil(nx_PSF/2.0)

    # Calculate the criterion map.
    # For each pixel calculate the dot product of a stamp around it with the PSF.
    # We use the PSF cube to consider also the spectrum of the planet we are looking for.
    if not mute:
        print("Calculate the criterion map. It is done pixel per pixel so it might take a while...")
    if nl !=1:
        stamp_PSF_x_grid, stamp_PSF_y_grid = np.meshgrid(np.arange(0,nx_PSF,1)-nx_PSF/2,np.arange(0,ny_PSF,1)-ny_PSF/2)
        stamp_PSF_mask = np.ones((nl,ny_PSF,nx_PSF))
        r_PSF_stamp = abs((stamp_PSF_x_grid) +(stamp_PSF_y_grid)*1j)
        #r_PSF_stamp = np.tile(r_PSF_stamp,(nl,1,1))
        stamp_PSF_mask[np.where(r_PSF_stamp < 2.5)] = np.nan

        #plt.figure(1)
        #plt.imshow(stamp_PSF_mask[5,:,:], interpolation="nearest")
        #plt.show()

        stdout.write("\r%d" % 0)
        for k in np.arange(10,ny-10):
            stdout.flush()
            stdout.write("\r%d" % k)

            for l in np.arange(10,nx-10):
                stamp_cube = copy(cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)])
                for slice_id in range(nl):
                    stamp_cube[slice_id,:,:] -= np.nanmean(stamp_cube[slice_id,:,:]*stamp_PSF_mask)
                ampl = np.nansum(PSF_cube*stamp_cube)
                if ampl != 0.0:
                    square_norm_stamp = np.nansum(stamp_cube**2)
                    shape_crit_map[k,l] = np.sign(ampl)*ampl**2/square_norm_stamp
                    #shape_crit_map[k,l] = ampl
                else:
                    shape_crit_map[k,l] = -1.0
                #ortho_criterion_map[k,l] = np.nansum((stamp_cube-ampl*PSF_cube)**2)/square_norm_stamp
    else:
        print("planet detection Not ready for 2D images to be corrected first")
        return
        for k in np.arange(10,ny-10):
            for l in np.arange(10,nx-10):
                stamp = cube[(k-row_m):(k+row_p), (l-col_m):(l+col_p)]
                ampl = np.nansum(PSF*stamp)
                if ampl != 0.0:
                    square_norm_stamp = np.nansum(stamp**2)
                    shape_crit_map[k,l] = np.sign(ampl)*ampl**2/square_norm_stamp
                else:
                    shape_crit_map[k,l] = -1.0
                #ortho_criterion_map[k,l] = np.nansum((stamp_cube-ampl*PSF_cube)**2)/square_norm_stamp


        ## ortho_criterion is actually the sine between the two vectors
        # ortho_criterion_map = 1 - criterion_map
    # criterion is here a cosine squared so we take the square root to get something similar to a cosine.
    shape_crit_map = np.sign(shape_crit_map)*np.sqrt(abs(shape_crit_map))
    #Save a copy of the flat cube because we will mask the detected spots as the algorithm goes.

    ratio_shape_SNR = 10

    #criterion_map = np.minimum(ratio_shape_SNR*shape_crit_map,flat_cube_SNR)
    criterion_map = shape_crit_map/radialStdMap(shape_crit_map,dr,Dr, centroid=center)
    #criterion_map = flat_cube_SNR
    criterion_map[flat_cube_nans] = np.nan
    #criterion_map /= radialStdMap(flat_cube,dr,Dr, centroid=center)
    criterion_map_cpy = copy(criterion_map)

    # List of local maxima
    checked_spots_list = []
    # List of local maxima that are valid candidates
    candidates_list = []

    if outputDir is None:
        outputDir = "./"
    else:
        outputDir = outputDir+"/"


    if folderName is None:
        folderName = "/default_out/"
    else:
        folderName = folderName+"/"

    if not os.path.exists(outputDir+folderName): os.makedirs(outputDir+folderName)

    if logFile:
        logFile_all = open(outputDir+folderName+logFile+'-log_allLocalMax.txt', 'w')
        logFile_candidates = open(outputDir+folderName+logFile+'-log_candidates.txt', 'w')

        myStr = "# Log some values for each local maxima \n" +\
                "# Meaning of the columns from left to right. \n" +\
                "# 1/ Index \n" +\
                "# 2/ Boolean. True if the local maximum is a valid candidate. \n" +\
                "# 3/ Value of the criterion at this local maximum. \n"+\
                "# 3/ Value of the shape criterion at this local maximum. \n"+\
                "# 3/ Value of the SNR at this local maximum. \n"+\
                "# 4/ Error check of the gaussian fit. \n"+\
                "# 5/ Distance between the fitted centroid and the original max position. \n"+\
                "# 6/ x-axis width of the gaussian. \n"+\
                "# 7/ y-axis width of the gaussian. \n"+\
                "# 8/ Amplitude of the gaussian. \n"+\
                "# 9/ Row index of the maximum. (y-coord in DS9) \n"+\
                "# 10/ Column index of the maximum. (x-coord in DS9) \n"
        logFile_all.write(myStr)

        myStr2 = "# Log some values for each valid candidates. \n" +\
                "# Meaning of the columns from left to right. \n" +\
                "# 1/ Index \n" +\
                "# 2/ Boolean. True if the centroid of the maximum is stable-ish. \n" +\
                "# 3/ Value of the criterion at this local maximum. \n"+\
                "# 3/ Value of the shape criterion at this local maximum. \n"+\
                "# 3/ Value of the SNR at this local maximum. \n"+\
                "# 4/ Error check of the gaussian fit. \n"+\
                "# 5/ Distance between the fitted centroid and the original max position. \n"+\
                "# 6/ x-axis width of the gaussian. \n"+\
                "# 7/ y-axis width of the gaussian. \n"+\
                "# 8/ Amplitude of the gaussian. \n"+\
                "# 9/ Row index of the maximum. (y-coord in DS9) \n"+\
                "# 10/ Column index of the maximum. (x-coord in DS9) \n"
        logFile_candidates.write(myStr2)


    flat_cube[flat_cube_nans] = 0.0

    # Count the number of valid detected candidates.
    N_candidates = 0.0

    # Maximum number of iterations on local maxima.
    max_attempts = 60
                ## START FOR LOOP.
                ## Each iteration looks at one local maximum in the criterion map.
                ## Then it verifies some other criteria to check if it is worse looking at.
                #for k in np.arange(max_attempts):
    k = 0
    max_val_criter = np.nanmax(criterion_map)
    while max_val_criter >= 2.0 and k <= max_attempts:
        k += 1
        # Find the maximum value in the current SNR map. At each iteration the previous maximum is masked out.
        max_val_criter = np.nanmax(criterion_map)
        # Locate the maximum by retrieving its coordinates
        max_ind = np.where( criterion_map == max_val_criter )
        row_id,col_id = max_ind[0][0],max_ind[1][0]
        x_max_pos, y_max_pos = x_grid[row_id,col_id],y_grid[row_id,col_id]

        # Check if the maximum is next to a nan. If so it should not be considered.
        if not np.isnan(np.sum(criterion_map[(row_id-1):(row_id+2), (col_id-1):(col_id+2)])):
            valid_potential_planet = max_val_criter > 3.0
        else:
            valid_potential_planet = False

        #Extract a stamp around the maximum in the flat cube (without the convolution)
        row_m = np.floor(stamp_nrow/2.0)
        row_p = np.ceil(stamp_nrow/2.0)
        col_m = np.floor(stamp_ncol/2.0)
        col_p = np.ceil(stamp_ncol/2.0)
        stamp = copy(flat_cube[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)])
        stamp_SNR = copy(flat_cube_SNR[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)])
        stamp_x_grid = x_grid[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
        stamp_y_grid = y_grid[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
        max_val_SNR = np.nanmax(flat_cube_SNR[(row_id-2):(row_id+3), (col_id-2):(col_id+3)])

        stamp[np.where(np.isnan(stamp))] = 0.0

        # Definition of a 2D gaussian fitting to be used on the stamp.
        g_init = models.Gaussian2D(max_val_SNR,x_max_pos,y_max_pos,1.5,1.5)
        fit_g = fitting.LevMarLSQFitter()
        # Fit the 2d Gaussian to the stamp
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        g = fit_g(g_init, stamp_x_grid, stamp_y_grid, stamp,np.abs(stamp)**0.5)

            # Calculate the fitting residual using the square root of the summed squared error.
            #ampl = np.nansum(stamp*g(stamp_x_grid, stamp_y_grid))
            #khi = np.sign(ampl)*ampl**2/(np.nansum(stamp**2)*np.nansum(g(stamp_x_grid, stamp_y_grid)**2))

        # JB Todo: Should explore the meaning of 'ierr' but I can't find a good clear documentation of astropy.fitting
        sig_min = 1.0 ; sig_max = 3.0 ;
        # The condition for a local maximum to be considered as a candidate are:
        #       - Positive criterion. Always verified because we are looking at maxima...
        #       - Reasonable SNR. ie greater than one.
        #         I prefer to be conservative on the SNR because we never know and I think it isn't the best criterion.
        #       - The gaussian fit had to succeed.
        #       - Reasonable width of the gaussian. Not wider than 3.5pix and not smaller than 0.5pix in both axes.
        #       - Centroid of the Gaussian fit not too far from the center of the stamp.
        #       - Amplitude of the Gaussian fit should be positive.
        valid_gaussian = (flat_cube_SNR[row_id,col_id] > 1.0 and
                                 #fit_g.fit_info['ierr'] <= 3 and
                                 sig_min < g.x_stddev < sig_max and
                                 sig_min < g.y_stddev < sig_max and
                                 np.sqrt((g.x_mean-x_max_pos)**2+(g.y_mean-y_max_pos)**2)<1.5 and
                                 g.amplitude > 0.0)
                                        #khi/khi0 < 1.0 and # Check that the fit was good enough. Not a weird looking speckle.

        #fit_g.fit_info['ierr'] == 1 and # Check that the fitting actually occured. Actually I have no idea what the number mean but it looks like when it succeeds it is 1.

        # If the spot verifies the conditions above it is considered as a valid candidate.
        checked_spots_list.append((k,row_id,col_id,max_val_criter,max_val_SNR,x_max_pos,y_max_pos,g))

        # Mask the spot around the maximum we just found.
        criterion_map[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] *= stamp_mask
        flat_cube[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)] *= stamp_mask

        # Todo: Remove prints, for debugging purpose only
        if not mute:
            print(k,row_id,col_id,max_val_criter,ratio_shape_SNR*shape_crit_map[row_id,col_id],max_val_SNR, g.x_stddev+0.0, g.y_stddev+0.0,g.x_mean-x_max_pos,g.y_mean-y_max_pos,flat_cube_SNR[row_id,col_id])
        if k == 79 or 0:
            plt.figure(1)
            plt.imshow(stamp, interpolation="nearest")
            plt.figure(2)
            plt.imshow(g(stamp_x_grid, stamp_y_grid), interpolation="nearest")
            plt.figure(3)
            plt.imshow(stamp_SNR, interpolation="nearest")
            plt.show()

        if logFile:
            myStr = str(k)+', '+\
                    str(valid_potential_planet)+', '+\
                    str(max_val_criter)+', '+\
                    str(ratio_shape_SNR*shape_crit_map[row_id,col_id])+', '+\
                    str(max_val_SNR)+', '+\
                    str(fit_g.fit_info['ierr'])+', '+\
                    str(np.sqrt((g.x_mean-x_max_pos)**2+(g.y_mean-y_max_pos)**2))+', '+\
                    str(g.x_stddev+0.0)+', '+\
                    str(g.y_stddev+0.0)+', '+\
                    str(g.amplitude+0.0)+', '+\
                    str(row_id)+', '+\
                    str(col_id)+'\n'
            logFile_all.write(myStr)

        # If the spot is a valid candidate we add it to the candidates list
        stable_cent = True
        if valid_potential_planet:
            # Check that the centroid is stable over all the slices of the cube.
            # It basically fit a gaussian on each slice and verifies that the resulting centroid is not chaotic
            # and that it doesn't move with wavelength.
            if nl != 1:
                stamp_cube = cube[:,(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
                stamp_x_grid = x_grid[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
                stamp_y_grid = y_grid[(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]

                h_init = models.Gaussian2D(g.amplitude+0.0,g.x_mean+0.0,g.y_mean+0.0,g.x_stddev+0.0,g.y_stddev+0.0)
                fit_h = fitting.LevMarLSQFitter()
                warnings.simplefilter('ignore')

                stamp_cube_cent_x = np.zeros(nl)
                stamp_cube_cent_y = np.zeros(nl)
                wave_step = 0.00841081142426 # in mum
                lambdaH0 = 1.49460536242  # in mum
                spec_sampling = np.arange(nl)*wave_step + lambdaH0

                fit_weights = np.sqrt(np.abs(spectrum))

                for k_slice in np.arange(nl):
                    h = fit_h(h_init, stamp_x_grid, stamp_y_grid, stamp_cube[k_slice,:,:],np.abs(stamp)**0.5)
                    if fit_h.fit_info['ierr'] <= 3:
                        stamp_cube_cent_x[k_slice] = h.x_mean+0.0
                        stamp_cube_cent_y[k_slice] = h.y_mean+0.0
                    else:
                        fit_weights[k_slice] = 0.0

                    if 0 and k == 7:
                        print(fit_h.fit_info['ierr'])
                    if 0 and k == 7:
                        plt.imshow(stamp_cube[k_slice,:,:], interpolation="nearest")
                        plt.show()
                stamp_cube_cent_r = np.sqrt(stamp_cube_cent_x**2 + stamp_cube_cent_y**2)
                fit_weights[abs(stamp_cube_cent_r - np.nanmean(stamp_cube_cent_r)) > 3 * np.nanstd(stamp_cube_cent_r)] = 0.0
                #print(stamp_cube_cent_r - np.nanmean(stamp_cube_cent_r),3 * np.nanstd(stamp_cube_cent_r))
                #print(stamp_cube_cent_r,fit_weights)
                fit_coefs_r = np.polynomial.polynomial.polyfit(spec_sampling, stamp_cube_cent_r, 1, w=fit_weights)
                #print(fit_coefs_r)
                r_poly_fit = np.poly1d(fit_coefs_r[::-1])
                r_fit = r_poly_fit(spec_sampling)

                candidate_radius = np.sqrt(x_grid[row_id,col_id]**2 + y_grid[row_id,col_id]**2)
                fit_sig = np.sqrt(np.nansum((r_fit-stamp_cube_cent_r)**2*fit_weights**2)/r_fit.size)
                # Todo: Remove prints, for debugging purpose only
                if not mute:
                    print(fit_coefs_r[1]*37*wave_step,0.5*candidate_radius/lambdaH0*37*wave_step,0.5*30/lambdaH0*37*wave_step,candidate_radius, fit_sig)
                #if fit_coefs_r[1] > 0.5*min(30/lambdaH0,candidate_radius/lambdaH0):
                if abs(fit_coefs_r[1])*37*wave_step > 1.5 or fit_sig>0.4:
                    stable_cent = False

                if 0 and k ==13:
                    stamp_cube_cent_r[np.where(abs(fit_weights) == 0.0)] = np.nan
                    plt.figure(1)
                    plt.plot(spec_sampling,r_fit, 'g-',spec_sampling,stamp_cube_cent_r, 'go')
                    plt.show()

            if logFile:
                myStr = str(k)+', '+\
                        str(stable_cent)+', '+\
                        str(max_val_criter)+', '+\
                        str(ratio_shape_SNR*shape_crit_map[row_id,col_id])+', '+\
                        str(max_val_SNR)+', '+\
                        str(fit_g.fit_info['ierr'])+', '+\
                        str(np.sqrt((g.x_mean-x_max_pos)**2+(g.y_mean-y_max_pos)**2))+', '+\
                        str(g.x_stddev+0.0)+', '+\
                        str(g.y_stddev+0.0)+', '+\
                        str(g.amplitude+0.0)+', '+\
                        str(row_id)+', '+\
                        str(col_id)+'\n'
                logFile_candidates.write(myStr)


        if 0 and k==12:
            stamp_cube = cube[:,(row_id-row_m):(row_id+row_p), (col_id-col_m):(col_id+col_p)]
            stamp_cube[np.where(stamp_cube_small_mask != 0)] = 0.0
            spectrum = np.nansum(np.nansum(stamp_cube,axis=1),axis=1)
            print(spectrum)
            #plt.figure(2)
            #plt.imshow(stamp_cube[5,:,:], interpolation="nearest")
            plt.figure(3)
            plt.plot(spectrum)
            plt.show()


        # Todo: Remove prints, for debugging purpose only
        if not mute:
            print(valid_potential_planet,stable_cent,fit_g.fit_info['ierr'])


        if valid_potential_planet:
            # Increment the number of detected candidates
            N_candidates += 1
            # Append the useful things about the candidate in the list.
            candidates_list.append((k,valid_gaussian,stable_cent,x_max_pos, y_max_pos,max_val_criter,max_val_SNR,g))

    # END FOR LOOP

    candidates_KLs_list.append(candidates_list)

    if logFile:
        logFile_all.close()
        logFile_candidates.close()

    # START IF STATEMENT
    if toDraw or toPNG:
        # Highlight the detected candidates in the criterion map
        if not mute:
            print(N_candidates)
        criterion_map_checkedArea = criterion_map_cpy
        plt.figure(3,figsize=(16,16))
        #*flat_cube_mask[::-1,:]
        plt.imshow(criterion_map_checkedArea[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
        ax = plt.gca()
        for candidate in candidates_list:
            candidate_it,valid_gaussian,stable_cent,x_max_pos, y_max_pos,max_val_criter,max_val_SNR,g = candidate
            if valid_gaussian:
                color = 'green'
                if not stable_cent:
                    color = 'red'
            else:
                color = 'black'

            ax.annotate(str(candidate_it)+","+"{0:02.1f}".format(max_val_criter)+","+"{0:02.1f}".format(max_val_SNR), fontsize=20, color = color, xy=(x_max_pos+0.0, y_max_pos+0.0),
                    xycoords='data', xytext=(x_max_pos+10, y_max_pos-10),
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    linewidth = 1.,
                                    color = 'black')
                    )

        # Show the local maxima in the criterion map
        plt.figure(4,figsize=(16,16))
        plt.imshow(criterion_map_checkedArea[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
        ax = plt.gca()
        for spot in checked_spots_list:
            spot_it,row_id,col_id,max_val_criter,max_val_SNR,x_max_pos,y_max_pos,g = spot
            ax.annotate(str(spot_it)+","+"{0:02.1f}".format(max_val_criter), fontsize=20, color = 'black', xy=(x_max_pos+0.0, y_max_pos+0.0),
                    xycoords='data', xytext=(x_max_pos+10, y_max_pos-10),
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    linewidth = 1.,
                                    color = 'black')
                    )

    # END IF STATEMENT

    if toPNG:
        plt.figure(3,figsize=(16,16))
        plt.clim(-5.,10.0)
        plt.savefig(outputDir+folderName+toPNG+'_candidates_SNR.png', bbox_inches='tight')
        plt.clf()
        plt.close(3)
        plt.figure(4,figsize=(16,16))
        plt.clim(-5.,10.0)
        plt.savefig(outputDir+folderName+toPNG+'_allSpots_criterion.png', bbox_inches='tight')
        plt.clf()
        plt.close(4)
        plt.figure(5,figsize=(16,16))
        where_shape_bigger_than_SNR = np.zeros((ny,nx))
        where_shape_bigger_than_SNR[np.where(ratio_shape_SNR*shape_crit_map > flat_cube_SNR)] = 1
        plt.imshow(where_shape_bigger_than_SNR[::-1,:], interpolation="nearest",extent=[x_grid[0,0],x_grid[0,nx-1],y_grid[0,0],y_grid[ny-1,0]])
        plt.savefig(outputDir+folderName+toPNG+'_shape_biggerThan_SNR.png', bbox_inches='tight')
        plt.clf()
        plt.close(5)


    if toFits:
        hdulist2 = pyfits.HDUList()
        try:
            hdulist2.append(pyfits.PrimaryHDU(header=prihdr))
            hdulist2.append(pyfits.ImageHDU(header=exthdr, data=criterion_map_cpy, name="Sci"))
            hdulist2.writeto(outputDir+folderName+toFits+'-criterion.fits', clobber=True)
            hdulist2[1].data = flat_cube_cpy
            hdulist2.writeto(outputDir+folderName+toFits+'-flatCube.fits', clobber=True)
            hdulist2[1].data = shape_crit_map
            hdulist2.writeto(outputDir+folderName+toFits+'-shape.fits', clobber=True)
            hdulist2[1].data = flat_cube_SNR
            hdulist2.writeto(outputDir+folderName+toFits+'-SNR.fits', clobber=True)
        except:
            print("Couldn't save using the normal way so trying something else.")
            hdulist2.append(pyfits.PrimaryHDU(header=prihdr))
            hdulist2[0].data = criterion_map_cpy
            hdulist2.writeto(outputDir+folderName+toFits+'-criterion.fits', clobber=True)
            hdulist2[0].data = flat_cube_cpy
            hdulist2.writeto(outputDir+folderName+toFits+'-flatCube.fits', clobber=True)
            hdulist2[0].data = shape_crit_map
            hdulist2.writeto(outputDir+folderName+toFits+'-shape.fits', clobber=True)
            hdulist2[0].data = flat_cube_SNR
            hdulist2.writeto(outputDir+folderName+toFits+'-SNR.fits', clobber=True)
        hdulist2.close()


    if toDraw:
        plt.figure(3,figsize=(16,16))
        plt.clim(-5.,10.0)
        plt.figure(4,figsize=(16,16))
        plt.clim(-5.,10.0)
        plt.show()

    return 1
# END candidate_detection() DEFINITION