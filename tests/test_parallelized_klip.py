__author__ = 'jwang'

import os
import glob
from time import time
import numpy as np
import astropy.io.fits as fits

import pyklip
import pyklip.instruments
import pyklip.parallelized as parallelized
import pyklip.instruments.GPI as GPI
import pyklip.fakes as fakes

testdir = os.path.dirname(os.path.abspath(__file__)) + os.path.sep

def test_exmaple_gpi_klip_dataset():
    """
    Tests standard pykip.parallelized.klip_dataset() with GPI data from the tutorial. Uses no spectral template

    """
    # time it
    t1 = time()

    # grab the files
    filelist = glob.glob(testdir + os.path.join("data", "S20131210*distorcorr.fits"))
    # hopefully there is still 3 filelists
    assert(len(filelist) == 3)

    # create the dataset object
    dataset = GPI.GPIData(filelist)

    # run klip parallelized
    outputdir = testdir
    prefix = "example-betapic-j-k100a9s4m1"
    parallelized.klip_dataset(dataset, outputdir=outputdir, fileprefix=prefix,
                          annuli=9, subsections=4, movement=1, numbasis=[1,20,50,100],
                          calibrate_flux=True, mode="ADI+SDI")

    # look at the output data. Validate the spectral cube
    spec_hdulist = fits.open("{out}/{pre}-KL20-speccube.fits".format(out=outputdir, pre=prefix))
    speccube_kl20 = spec_hdulist[1].data

    # check to make sure it's the right shape
    assert(speccube_kl20.shape == (37, 281, 281))


    # look at the output data. Validate the KL mode cube
    kl_hdulist = fits.open("{out}/{pre}-KLmodes-all.fits".format(out=outputdir, pre=prefix))
    klcube = kl_hdulist[1].data

    # check to make sure it's the right shape
    assert(klcube.shape == (4, 281, 281))

    # try to retrieve beta pic b.
    # True astrometry taken from Wang et al.(2016)
    true_sep = 426.6 / 1e3 / GPI.GPIData.lenslet_scale # in pixels
    true_pa = 212.2 # degrees
    # guessing flux and FWHM
    true_flux = 1.7e-5
    true_fwhm = 2.3 # ~lambda/D for lambda=1.25 microns, D=8 m

    # find planet in collapsed cube
    collapsed_kl20 = klcube[1]
    flux_meas, x_meas, y_meas, fwhm_meas = fakes.retrieve_planet(collapsed_kl20, dataset.centers[0], dataset.wcs[0],
                                                                 true_sep, true_pa, searchrad=4, guesspeak=2.e-5,
                                                                 guessfwhm=2)
    print(flux_meas, x_meas, y_meas, fwhm_meas)

    # error thresholds
    # flux error
    assert np.abs((flux_meas - true_flux)/true_flux) < 0.4
    # positonal error
    theta = fakes.covert_pa_to_image_polar(true_pa, dataset.wcs[0])
    true_x = true_sep * np.cos(np.radians(theta)) + dataset.centers[0,0]
    true_y = true_sep * np.sin(np.radians(theta)) + dataset.centers[0,1]
    assert np.abs(true_x - x_meas) < 0.4
    assert np.abs(true_y - y_meas) < 0.4
    # fwhm error
    assert np.abs(true_fwhm - fwhm_meas) < 0.4

    # measure SNR of planet


    print("{0} seconds to run".format(time()-t1))


def test_adi_gpi_klip_dataset_with_fakes(filelist=None):
    """
    Tests ADI reduction with fakes injected at certain position angles

    Args:
        filelist: if not None, supply files to test on. Otherwise use standard beta pic data
    """
    # time it
    t1 = time()

    # grab the files
    if filelist is None:
        filelist = glob.glob(testdir + os.path.join("data", "S20131210*distorcorr.fits"))

        # hopefully there is still 3 filelists
        assert(len(filelist) == 3)

    # create the dataset object
    dataset = GPI.GPIData(filelist)

    # inject fake planet
    fake_seps = [20, 50, 40, 30] # pixels
    fake_pas = [-50, -165, 130, 10] # degrees
    fake_contrasts = np.array([1.e-4, 3.e-5, 5.e-5, 1.e-4]) # bright planet
    fake_injected_fluxes = fake_contrasts/np.mean(dataset.contrast_scaling)
    for fake_sep, fake_pa, fake_flux in zip(fake_seps, fake_pas, fake_injected_fluxes):
        fakes.inject_planet(dataset.input, dataset.centers, fake_flux, dataset.wcs, fake_sep, fake_pa)


    # run klip parallelized
    outputdir = testdir
    prefix = "adionly-betapic-j-k100a9s4m1-fakes50pa50"
    parallelized.klip_dataset(dataset, outputdir=outputdir, fileprefix=prefix,
                          annuli=9, subsections=4, movement=1, numbasis=[1,20,50,100],
                          calibrate_flux=True, mode="ADI")

    # look at the output data. Validate the spectral cube
    spec_hdulist = fits.open("{out}/{pre}-KL20-speccube.fits".format(out=outputdir, pre=prefix))
    speccube_kl20 = spec_hdulist[1].data

    # check to make sure it's the right shape
    assert(speccube_kl20.shape == (37, 281, 281))

    # collapse data
    collapsed_kl20 = np.nanmean(speccube_kl20, axis=0)

    # try to retrieve fake planet
    for fake_sep, fake_pa, fake_contrast in zip(fake_seps, fake_pas, fake_contrasts):
        peakflux = fakes.retrieve_planet_flux(collapsed_kl20, dataset.centers[0], dataset.wcs[0], fake_sep,
                                              fake_pa, refinefit=True)

        assert (np.abs((peakflux/0.7-fake_contrast)/fake_contrast) < 0.5)

    print("{0} seconds to run".format(time()-t1))


if __name__ == "__main__":
    test_exmaple_gpi_klip_dataset()