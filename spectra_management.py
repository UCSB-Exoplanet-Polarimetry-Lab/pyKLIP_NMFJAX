__author__ = 'jruffio'

import numpy as np
from scipy.interpolate import interp1d
import astropy.io.fits as pyfits
import platform


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

    if platform.system() == "Windows":
        filename_temp_lookup = '.\\pickles\\mainseq_colors.txt'
        filename_pickles_lookup = '.\\pickles\\AA_README'
    else:
        filename_temp_lookup = './pickles/mainseq_colors.txt'
        filename_pickles_lookup = './pickles/AA_README'

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

    if platform.system() == "Windows":
        upper_filename = ".\\pickles\\"+upper_filename+".fits"
        lower_filename = ".\\pickles\\"+lower_filename+".fits"
    else:
        upper_filename = "./pickles/"+upper_filename+".fits"
        lower_filename = "./pickles/"+lower_filename+".fits"

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



if __name__ == "__main__":

    #wv,sp = get_star_spectrum(pipeline_dir,'H','F3')
    filename = "/Users/jruffio/gpi/pyklip/t800g100nc.flx"
    get_planet_spectrum(filename,'H')

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