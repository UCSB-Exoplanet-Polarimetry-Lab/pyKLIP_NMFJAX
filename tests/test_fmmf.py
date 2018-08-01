import os
import glob
import numpy as np
import scipy.interpolate as sinterp
import astropy.io.fits as fits

import pyklip.spectra_management as specmanage
import pyklip.fm as fm
import pyklip.instruments.GPI as GPI
import pyklip.fmlib.matchedFilter as mf
import pyklip.kpp.metrics.matchedfilter as kppmf

testdir = os.path.dirname(os.path.abspath(__file__)) + os.path.sep

def test_fmmf():
    """
    Test the FMMF on the beta Pic dataset. Make sure we can see beta Pic :P
    """

    ####### Read the files in #######
    # grab the files
    filelist = glob.glob(testdir + os.path.join("data", "S20131210*distorcorr.fits"))
    filelist.sort()

    # hopefully there is still 3 filelists
    assert(len(filelist) == 3)

    # read in data
    # only read in every 3rd channel
    skipslices = [i for i in range(0,37) if (i+1) % 3 > 0]
    dataset = GPI.GPIData(filelist, highpass=True, skipslices=skipslices)

    # generate PSF from satellite spots
    dataset.generate_psfs(boxrad=25//2)
    dataset.psfs /= (np.mean(dataset.spot_flux.reshape([dataset.spot_flux.shape[0]//dataset.numwvs, dataset.numwvs]), axis=0)[:, None, None])

    # read in model spectrum
    model_file = os.path.join(testdir, "..", "pyklip", "spectra", "cloudy", "t1600g100f2.flx")
    spec_dat = np.loadtxt(model_file)
    spec_wvs = spec_dat[1]
    spec_f = spec_dat[3]
    spec_interp = sinterp.interp1d(spec_wvs, spec_f, kind='nearest')
    inputspec = spec_interp(dataset.wvs)

    # we are only going to process a thin annulus around the planet
    guesssep = 0.4267 / GPI.GPIData.lenslet_scale
    guesspa = 212.15
    annuli_bounds = [[guesssep-1, guesssep+1]]

    # set PCA basis
    numbasis = np.array([10])

    # create forward model matched filter object
    fm_class = mf.MatchedFilter(dataset.input.shape, numbasis, dataset.psfs, np.unique(dataset.wvs), [inputspec])

    # run KLIP-FMMF
    outputdir = testdir
    fileprefix = "betpic-131210-j-fmmf"
    fm.klip_dataset(dataset, fm_class, outputdir=outputdir, fileprefix=fileprefix, numbasis=numbasis, annuli=annuli_bounds, subsections=1, 
                    padding=dataset.psfs.shape[-1]//2, movement=1.5, mode="ADI+SDI", maxnumbasis=150)

    outputfile = os.path.join(outputdir, fileprefix + "-FMMF-KL{0}.fits".format(numbasis[0]))
    assert os.path.exists(outputfile)

    with fits.open(outputfile) as hdulist:
        dat = hdulist[1].data
        assert len(dat.shape) == 2

        planet_x = -guesssep * np.sin(np.radians(guesspa)) + hdulist[1].header['PSFCENTX']
        planet_y = guesssep * np.cos(np.radians(guesspa)) + hdulist[1].header['PSFCENTY']

        planet_x_min = int(np.floor(planet_x))
        planet_x_max = int(np.ceil(planet_x))
        planet_y_min = int(np.floor(planet_y))
        planet_y_max = int(np.ceil(planet_y))

        planet_signal = np.max(dat[planet_y_min:planet_y_max+1, planet_x_min:planet_x_max+1])

        y,x = np.indices(dat.shape)
        distance_from_planet = np.sqrt((x - planet_x)**2 + (y - planet_y)**2)
        away_from_planet = np.where(distance_from_planet > 15)

        noise = np.nanstd(dat[away_from_planet])

        snr_planet = planet_signal/noise
        print("FMMF SNR", snr_planet)
        assert snr_planet > 25

    # compare against a regular matched filter
    klip_outputfile = os.path.join(outputdir, fileprefix + "-klipped-KL{0}-speccube.fits".format(numbasis[0]))
    assert os.path.exists(klip_outputfile)
    with fits.open(klip_outputfile) as hdulist:
        klipcube = hdulist[1].data

        # convert the template spectrum into DN
        star_spectrum = specmanage.get_star_spectrum(np.unique(dataset.wvs), "A6V")[1]
        inputspec_contrast = np.unique(inputspec)/star_spectrum
        inputspec_dn = inputspec_contrast * (np.mean(dataset.spot_flux.reshape([dataset.spot_flux.shape[0]//dataset.numwvs, dataset.numwvs]), axis=0))

        # create psf template
        psfmf = dataset.psfs * inputspec_dn[:, None, None]
        mf_map, cc_map, flux_map = kppmf.run_matchedfilter(klipcube, psfmf)

        # use same method to comptue SNR
        mf_planet_signal = np.max(mf_map[planet_y_min:planet_y_max+1, planet_x_min:planet_x_max+1])
        mf_noise = np.nanstd(mf_map[away_from_planet])
        mf_snr_planet = mf_planet_signal/mf_noise
        print("noFM MF SNR", mf_snr_planet)

        # make sure it's better than snr 5
        assert mf_snr_planet > 5

        # but it should be worse than FMMF
        assert mf_snr_planet < snr_planet


if __name__ == "__main__":
    test_fmmf()


