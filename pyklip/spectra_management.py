__author__ = 'jruffio'

import numpy as np
from scipy.interpolate import interp1d
import astropy.io.fits as pyfits
import platform
#import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.optimize import leastsq
from scipy.optimize import minimize
import glob
import os


# First and last wavelength of each band
band_sampling = {'Z' : (0.9444, 1.1448, 37),
                'Y' : (0.9444, 1.1448, 37),
                'J' : (1.1108, 1.353, 37),
                'H' : (1.4904, 1.8016, 37),
                'K1' : (1.8818, 2.1994, 37),
                'K2' : (2.1034, 2.4004, 37)}

def get_gpi_filter(filter_name):
    """
    Extract the spectrum of a given gpi filter with the sampling of pipeline reduced cubes.

    Inputs:
        filter_name: 'H', 'J', 'K1', 'K2', 'Y'

    Output:
        (wavelengths, spectrum) where
            wavelengths: is the gpi sampling of the considered band in mum.
            spectrum: is the transmission spectrum of the filter for the given band.
    """

    # get the path to the file containing the spectrum in the pipeline directory
    if platform.system() == "Windows":
        filename = ".\\filters\\GPI-filter-"+filter_name+".fits"
    else:
        filename = "./filters/GPI-filter-"+filter_name+".fits"


    # load the fits array
    hdulist = pyfits.open(filename)
    cube = hdulist[1].data
    wavelengths = cube[0][0]
    spectrum = cube[0][1]


    w_start, w_end, N_sample = band_sampling[filter_name]
    dw = (w_end-w_start)/N_sample
    sampling_pip = np.arange(w_start,w_end,dw)

    counts_per_bin, bin_edges = np.histogram(wavelengths, bins=N_sample, range=(w_start-dw/2.,w_end+dw/2.), weights=spectrum)
    N_samples_per_bin, bin_edges = np.histogram(wavelengths, bins=N_sample, range=(w_start-dw/2.,w_end+dw/2.), weights=None)

    # if 0:
    #     plt.figure(1)
    #     plt.plot(wavelengths,spectrum,'b')
    #     plt.plot(sampling_pip,counts_per_bin/N_samples_per_bin,'r')
    #     plt.show()

    return (sampling_pip,counts_per_bin/N_samples_per_bin)


def find_upper_nearest(array,value):
    """
    Find the upper nearest element to value in array.

    :param array: Array of value
    :param value: Value for which one wants the upper value.
    :return: (up_value, id) with up_value the closest upper value and id its index.
    """
    diff = array-value
    diff[np.where(diff<0.0)] = np.nan
    idx = np.nanargmin(diff)
    return array[idx], idx

def find_lower_nearest(array,value):
    """
    Find the lower nearest element to value in array.

    :param array: Array of value
    :param value: Value for which one wants the lower value.
    :return: (low_value, id) with low_value the closest lower value and id its index.
    """
    diff = array-value
    diff[np.where(diff>0.0)] = np.nan
    idx = np.nanargmax(diff)
    return array[idx], idx

def get_star_spectrum(filter_name,star_type = None, temperature = None):
    """
    Get the spectrum of a star with given spectral type interpolating in the pickles database.
    The sampling is the one of pipeline reduced cubes.
    The spectrum is normalized to unit mean.
    Work only for V (ie brown dwarf) star

    Inputs:
        filter_name: 'H', 'J', 'K1', 'K2', 'Y'
        star_type: 'A5','F4',... Assume type V star. Is ignored of temperature is defined.
        temperature: temperature of the star. Overwrite star_type if defined.

    Output:
        (wavelengths, spectrum) where
            wavelengths: is the gpi sampling of the considered band in mum.
            spectrum: is the spectrum of the star for the given band.
    """

    #filename_emamajek_lookup = "emamajek_star_type_lookup.txt" #/Users/jruffio/gpi/pyklip/emamajek_star_type_lookup.rtf

    pykliproot = os.path.dirname(os.path.realpath(__file__))
    filename_temp_lookup = pykliproot+os.path.sep+"pickles"+os.path.sep+"mainseq_colors.txt"
    filename_pickles_lookup = pykliproot+os.path.sep+"pickles"+os.path.sep+"AA_README"

    #a = np.genfromtxt(filename_temp_lookup, names=True, delimiter=' ', dtype=None)

    if temperature is None:
        #Read pickles list
        dict_temp = dict()
        with open(filename_temp_lookup, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    pass
                else:
                    splitted_line = line.split()
                    # splitted_line[0]: spectral type F5 G0...
                    # splitted_line[2]: Temperature in K
                    dict_temp[splitted_line[0]] = splitted_line[2]

        target_temp = float(dict_temp[star_type])
    else:
        target_temp = temperature

    dict_filename = dict()
    with open(filename_pickles_lookup, 'r') as f:
        for line in f:
            if line.startswith('pickles_uk_'):
                splitted_line = line.split()
                # splitted_line[0]: Filename
                # splitted_line[1]: spectral type F5V G0III...
                # splitted_line[2]: Temperature in K

                #Check that the last character is numeric
                spec_type = splitted_line[1]
                if splitted_line[0][len(splitted_line[0])-1].isdigit() and not (spec_type.endswith('IV') or spec_type.endswith('I')):
                    dict_filename[float(splitted_line[2])] = splitted_line[0]

    temp_list = np.array(dict_filename.keys())
    #print(temp_list)
    # won't work for the hottest and coldest spectra.
    upper_temp, upper_temp_id = find_upper_nearest(temp_list,target_temp)
    lower_temp, lower_temp_id = find_lower_nearest(temp_list,target_temp)
    #print( upper_temp, upper_temp_id,lower_temp, lower_temp_id)

    upper_filename = dict_filename[upper_temp]
    lower_filename = dict_filename[lower_temp]

    upper_filename = pykliproot+os.path.sep+"pickles"+os.path.sep+upper_filename+".fits"
    lower_filename = pykliproot+os.path.sep+"pickles"+os.path.sep+lower_filename+".fits"


    hdulist = pyfits.open(upper_filename)
    cube = hdulist[1].data
    upper_wave = []
    upper_spec = []
    for wave_value,spec_value in cube:
        upper_wave.append(wave_value) # in angstrom
        upper_spec.append(spec_value)
    delta_wave = upper_wave[1]-upper_wave[0]
    upper_wave = np.array(upper_wave)/10**4 # in mum
    # upper_spec is a density spectrum in flux.A-1 so we need to multiply by delta_wave to integrate and get a flux.
    upper_spec = np.array(upper_spec)*delta_wave

    hdulist = pyfits.open(lower_filename)
    cube = hdulist[1].data
    lower_wave = []
    lower_spec = []
    for wave_value,spec_value in cube:
        lower_wave.append(wave_value) # in angstrom
        lower_spec.append(spec_value)
    lower_wave = np.array(lower_wave)/10**4 # in mum
    # lower_spec is a density spectrum in flux.A-1 so we need to multiply by delta_wave to integrate and get a flux.
    lower_spec = np.array(lower_spec)*delta_wave

    w_start, w_end, N_sample = band_sampling[filter_name]
    dw = (w_end-w_start)/N_sample
    sampling_pip = np.arange(w_start,w_end,dw)

    upper_counts_per_bin, bin_edges = np.histogram(upper_wave, bins=N_sample, range=(w_start-dw/2.,w_end+dw/2.), weights=upper_spec)
    lower_counts_per_bin, bin_edges = np.histogram(lower_wave, bins=N_sample, range=(w_start-dw/2.,w_end+dw/2.), weights=lower_spec)
    N_samples_per_bin, bin_edges = np.histogram(upper_wave, bins=N_sample, range=(w_start-dw/2.,w_end+dw/2.), weights=None)

    upper_spec_pip = upper_counts_per_bin/N_samples_per_bin
    lower_spec_pip = lower_counts_per_bin/N_samples_per_bin

    spec_pip = ((target_temp-lower_temp)*upper_spec_pip+(upper_temp-target_temp)*lower_spec_pip)/(upper_temp-lower_temp)

    # if 0:
    #     plt.figure(1)
    #     plt.plot(upper_wave,upper_spec,'r')
    #     plt.plot(lower_wave,lower_spec,'g')
    #     plt.plot(spec_pip,spec_pip,'b')
    #
    #     plt.figure(2)
    #     print(lower_temp,target_temp,upper_temp)
    #     where_gpi_band = np.where((upper_wave < w_end) * (upper_wave > w_start) )
    #     plt.plot(upper_wave[where_gpi_band],upper_spec[where_gpi_band],'r')
    #     plt.plot(lower_wave[where_gpi_band],lower_spec[where_gpi_band],'g')
    #     plt.plot(sampling_pip,upper_spec_pip,'r--')
    #     plt.plot(sampling_pip,lower_spec_pip,'g--')
    #     plt.plot(sampling_pip,spec_pip,'b--')
    #     plt.show()

    return (sampling_pip,spec_pip/np.nanmean(spec_pip))

def get_planet_spectrum(filename,filter_name):
    """
    Get the spectrum of a planet from a given file. Files are Mark Marleys'.
    The sampling is the one of pipeline reduced cubes.
    The spectrum is normalized to unit mean.
    I should check that I actually do the right operation on the spectra.

    Inputs:
        filename: Directory of the gpi pipeline.
        filter_name: 'H', 'J', 'K1', 'K2', 'Y'

    Output:
        (wavelengths, spectrum) where
            wavelengths: is the gpi sampling of the considered band in mum.
            spectrum: is the spectrum of the planet for the given band.
    """


    spec_data = []
    with open(filename, 'r') as f:
        for line in f:
            splitted_line = line.split()
            # splitted_line[0]: index
            # splitted_line[1]: wavelength (mum)
            # splitted_line[2]: T_brt
            # splitted_line[2]: flux in units of erg cm-2 sec-1 Hz-1 at the top of the planet's atmosphere

            spec_data.append([float(splitted_line[0]),float(splitted_line[1]),float(splitted_line[2]),float(splitted_line[3])])

    spec_data = np.array(spec_data)
    N_samp = spec_data.shape[0]
    wave = spec_data[:,1]
    #wave_intervals = np.zeros(N_samp)
    #wave_intervals[1:N_samp-1] = wave[2::] - wave[0:(N_samp-2)]
    #wave_intervals[0] = wave[1]-wave[0]
    #wave_intervals[N_samp-1] = wave[N_samp-1]-wave[N_samp-2]
    spec = spec_data[:,3]

    # if 0:
    #     plt.figure(1)
    #     plt.plot(spec_data[0:100,1],spec_data[0:100,3],'r')
    #     plt.show()

    w_start, w_end, N_sample = band_sampling[filter_name]
    dw = (w_end-w_start)/N_sample
    sampling_pip = np.arange(w_start,w_end,dw)

    # I think this isn't rigorous. The spectrum is not well binned maybe
    #counts_per_bin, bin_edges = np.histogram(wave, bins=N_sample, range=(w_start-dw/2.,w_end+dw/2.), weights=spec)
    #weights_per_bin, bin_edges = np.histogram(wave, bins=N_sample, range=(w_start-dw/2.,w_end+dw/2.), weights=wave_intervals)
    #spec_pip = dw * counts_per_bin/weights_per_bin

    f = interp1d(wave, spec)
    spec_pip = f(sampling_pip)

    # if 0:
    #     plt.figure(2)
    #     plt.plot(wave[50:100],spec[50:100],'r')
    #     plt.plot(sampling_pip,spec_pip,'b.')
    #     plt.show()

    return (sampling_pip,spec_pip/np.nanmean(spec_pip))


def get_gpi_wavelength_sampling(filter_name):

    w_start, w_end, N_sample = band_sampling[filter_name]
    dw = (w_end-w_start)/N_sample
    sampling_pip = np.arange(w_start,w_end,dw)

    return sampling_pip



def place_model_PSF(PSF_template,x_cen,y_cen,output_shape, x_grid = None, y_grid = None):

    ny_template, nx_template = PSF_template.shape
    if x_grid is None and y_grid is None:
        x_grid, y_grid = np.meshgrid(np.arange(0,output_shape[1],1),np.arange(0,output_shape[0],1))

    x_grid = x_grid.astype(np.float)
    y_grid = y_grid.astype(np.float)

    x_grid -= x_cen - nx_template/2
    y_grid -= y_cen - ny_template/2

    return ndimage.map_coordinates(PSF_template, [y_grid,x_grid], mode='constant', cval=0.0)

def LSQ_place_model_PSF(PSF_template,x_cen,y_cen,planet_image, x_grid = None, y_grid = None):
    model = place_model_PSF(PSF_template,x_cen,y_cen,planet_image.shape, x_grid = x_grid, y_grid = y_grid)
    return np.nansum((planet_image-model)**2,axis = (0,1))#/y_model


def extract_planet_centroid(cube, position, PSF_cube):


    nl,ny,nx = cube.shape
    row_id,col_id = position
    nl_PSF,ny_PSF,nx_PSF = PSF_cube.shape

    row_m = np.floor(ny_PSF/2.0)
    row_p = np.ceil(ny_PSF/2.0)
    col_m = np.floor(nx_PSF/2.0)
    col_p = np.ceil(nx_PSF/2.0)
    #print(np.max([0,(row_id-row_m)]),np.min([ny_PSF-1,(row_id+row_p)]),np.max([0,(col_id-col_m)]),np.min([nx_PSF-1,(col_id+col_p)]))
    cube_stamp = cube[:,np.max([0,(row_id-row_m)]):np.min([ny-1,(row_id+row_p)]), np.max([0,(col_id-col_m)]):np.min([nx-1,(col_id+col_p)])]
    flatCube_stamp = np.nansum(cube_stamp,axis=0)
    #plt.figure(5)
    #print(flatCube_stamp.shape,flatCube_stamp)
    #plt.imshow(flatCube_stamp,interpolation="nearest")
    #plt.show()
    flatCube_stamp /= np.nanmax(flatCube_stamp)
    flatPSF = np.nansum(PSF_cube,axis=0)
    flatPSF /= np.nanmax(flatPSF)

    nanargmax_flat_stamp= np.nanargmax(flatCube_stamp)
    max_row_id = np.floor(nanargmax_flat_stamp/nx_PSF)
    max_col_id = nanargmax_flat_stamp-nx_PSF*max_row_id

    param0 = (float(max_col_id),float(max_row_id)+1)

    LSQ_func = lambda para: LSQ_place_model_PSF(flatPSF,para[0],para[1],flatCube_stamp)
    param_fit = minimize(LSQ_func,param0, method="Nelder-Mead").x

    # if 0:
        # plt.figure(1)
        # plt.subplot(2,2,1)
        # plt.imshow(flatCube_stamp,interpolation="nearest")
        # plt.subplot(2,2,2)
        # plt.imshow(flatPSF,interpolation="nearest")
        # plt.subplot(2,2,3)
        # plt.imshow(flatCube_stamp-place_model_PSF(flatPSF,param0[0],param0[1],(ny_PSF,nx_PSF)),interpolation="nearest")
        # plt.subplot(2,2,4)
        # plt.imshow(flatCube_stamp-place_model_PSF(flatPSF,param_fit[0],param_fit[1],(ny_PSF,nx_PSF)),interpolation="nearest")
        # plt.show()

    return (param_fit[1]+row_id-row_m),(param_fit[0]+col_id-col_m)


def LSQ_scale_model_PSF(PSF_template,planet_image,a):
    return np.nansum((planet_image-a*PSF_template)**2,axis = (0,1))#/y_model

def extract_planet_spectrum(cube_para, position, PSF_cube_para, method = None,filter = None, mute = True):

    if isinstance(cube_para, basestring):
        hdulist = pyfits.open(cube_para)
        cube = hdulist[1].data
        exthdr = hdulist[1].header
        prihdr = hdulist[0].header
        hdulist.close()

        try:
            filter = prihdr['IFSFILT'].split('_')[1]
        except:
            if not mute:
                print("Couldn't find IFSFILT keyword in headers.")

    else:
        cube = cube_para

    if isinstance(PSF_cube_para, basestring):
        hdulist = pyfits.open(PSF_cube_para)
        PSF_cube = hdulist[1].data
        exthdr = hdulist[1].header
        prihdr = hdulist[0].header
        hdulist.close()

        sat_spot_spec = np.nanmax(PSF_cube,axis=(1,2))
        for l_id in range(PSF_cube.shape[0]):
            PSF_cube[l_id,:,:] /= sat_spot_spec[l_id]
    else:
        PSF_cube = PSF_cube_para

    row_cen,col_cen = extract_planet_centroid(cube, position, PSF_cube)

    nl,ny,nx = cube.shape
    row_id = np.round(row_cen)
    col_id = np.round(col_cen)
    #row_id,col_id = position
    nl_PSF,ny_PSF,nx_PSF = PSF_cube.shape

    row_m = np.floor(ny_PSF/2.0)
    row_p = np.ceil(ny_PSF/2.0)
    col_m = np.floor(nx_PSF/2.0)
    col_p = np.ceil(nx_PSF/2.0)
    cube_stamp = cube[:,np.max([0,(row_id-row_m)]):np.min([ny-1,(row_id+row_p)]), np.max([0,(col_id-col_m)]):np.min([nx-1,(col_id+col_p)])]
    nl_stamp, ny_stamp,nx_stamp = cube_stamp.shape


    # Mask to remove the spots already checked in criterion_map.
    stamp_x_grid, stamp_y_grid = np.meshgrid(np.arange(0,nx_stamp,1),np.arange(0,ny_stamp,1))
    r_stamp = np.sqrt((stamp_x_grid-(col_cen-(col_id-col_m)))**2 +(stamp_y_grid-(row_cen-(row_id-row_m)))**2)
    #stamp_mask = np.ones((stamp_nrow,stamp_ncol))
    #stamp_mask[np.where(r_stamp < 4.0)] = np.nan
    stamp_mask_small = np.ones((ny_stamp,nx_stamp))
    stamp_mask_small[np.where(r_stamp > 2.)] = np.nan
    stamp_cube_small_mask = np.tile(stamp_mask_small[None,:,:],(nl,1,1))


    # if 0:
        # plt.figure(6)
        # plt.subplot(2,2,1)
        # plt.imshow(stamp_mask_small,interpolation="nearest")
        # plt.subplot(2,2,2)
        # plt.imshow(cube_stamp[10,:,:]*stamp_mask_small,interpolation="nearest")
        # plt.show()

    if method is None or method == "max":
        spectrum = np.nanmax(cube_stamp*stamp_cube_small_mask,axis=(1,2))
    elif method == "aperture":
        spectrum = np.nansum(cube_stamp*stamp_cube_small_mask,axis=(1,2))
    elif method == "fit":
        spectrum = np.zeros((nl,))
        for k in range(nl):
            PSF_cube_slice = PSF_cube[k,:,:]/np.nanmax(PSF_cube[k,:,:])
            cube_stamp_slice = cube_stamp[k,:,:]
            param0 = np.nanmax(cube_stamp_slice)
            LSQ_func = lambda para: LSQ_scale_model_PSF(PSF_cube_slice,cube_stamp_slice,para)
            spectrum_fit[k] = minimize(LSQ_func,param0).x #, method="Nelder-Mead"
            # if 0:
                # plt.figure(1)
                # plt.subplot(2,2,1)
                # plt.imshow(cube_stamp_slice,interpolation="nearest")
                # plt.subplot(2,2,2)
                # plt.imshow(cube_stamp_slice*stamp_mask_small,interpolation="nearest")
                # plt.show()

    if 0:
        print(spectrum)
        # plt.figure(3)
        # plt.plot(get_gpi_wavelength_sampling(filter), spectrum)
        # plt.show()

    return get_gpi_wavelength_sampling(filter), spectrum





if __name__ == "__main__":

    OS = platform.system()
    if OS == "Windows":
        print("Using WINDOWS!!")
    else:
        print("I hope you are using a UNIX OS")


    if 0:
        if OS == "Windows":
            #cube_filename = "C:\\Users\\JB\\Dropbox (GPI)\\GPIDATA\\HD_100491\\autoreduced\\"
            #cube_filename = "C:\\Users\\JB\\Dropbox (GPI)\\GPIDATA\\bet_cir\\autoreduced\\"
            cube_filename = "C:\\Users\\JB\\Dropbox (GPI)\\GPIDATA\\c_Eri\\autoreduced\\pyklip-S20141218-k100a7s4m3-KL20-speccube.fits"
            #cube_filename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/HD19467/autoreduced/"
            #outputDir = "C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\"
            spectrum_model = ["","C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\code\\spectra\\t800g100nc.flx",""] #"C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\code\\spectra\\g100ncflx\\t2400g100nc.flx",
            user_defined_PSF_cube = "C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\code\\pyklipH-S20141218-k100a7s4m3-original_radial_PSF_cube.fits"
        else:
            cube_filename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/beta_pictoris/autoreduced/"
            #cube_filename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/bet_cir/autoreduced/"
            #cube_filename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/c_Eri/autoreduced/"
            #cube_filename = "/Users/jruffio/Dropbox (GPI)/GPIDATA/HD19467/autoreduced/"
            #outputDir = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB"
            #spectrum_model = ""
            spectrum_model = ["/Users/jruffio/gpi/pyklip/spectra/t800g100nc.flx",""]
            user_defined_PSF_cube = "/Users/jruffio/Dropbox (GPI)/SCRATCH/Scratch/JB/code/pyklipH-S20141218-k100a7s4m3-original_radial_PSF_cube.fits"

        wave_samp,spectrum = extract_planet_spectrum(cube_filename, (110, 135), user_defined_PSF_cube, method="aperture")
        # plt.plot(wave_samp,spectrum)
        # plt.show()

    if 1:
        if OS == "Windows":
            #spectrum_model = "C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\code\\spectra\\t800g100nc.flx"
            filelist = glob.glob("C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\code\\spectra\\g100ncflx\\*.flx")
            filelist = glob.glob("C:\\Users\\JB\\Dropbox (GPI)\\SCRATCH\\Scratch\\JB\\code\\spectra\\g18ncflx\\*.flx")
        else:
            spectrum_model = "/Users/jruffio/gpi/pyklip/spectra/t800g100nc.flx"

        for spectrum_model in filelist:
            print(spectrum_model)
            #wv,sp = get_star_spectrum(pipeline_dir,'H','F3')
            wv,spectrum  = get_planet_spectrum(spectrum_model,'H')

            # plt.plot(wv,spectrum/np.nanmean(spectrum))
            # plt.show()

        # if 0:
        #     filename = "/Users/jruffio/gpi/pipeline/config/pickles/pickles_uk_23.fits"
        #     hdulist = pyfits.open(filename)
        #     cube = hdulist[1].data
        #     N_samples = np.size(cube)
        #     c = []
        #     d = []
        #     for a,b in cube:
        #         c.append(a)
        #         d.append(b)
        #     print(c,d)
        #     wavelengths = cube[0][0]
        #     filter_spec = cube[0][1]
        #     plt.figure(1)
        #     plt.plot(wavelengths,filter_spec)
        #     plt.show()