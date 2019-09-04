"""testDiskFM.py
author: johan mazoyer
"""
# pylint: disable=C0103
import os
import glob

import numpy as np
import astropy.io.fits as fits
from astropy.convolution import convolve

import pyklip.instruments.GPI as GPI
from pyklip.fmlib.diskfm import DiskFM
import pyklip.fm as fm

os.environ["OMP_NUM_THREADS"] = "1"

########################################################
########################################################

TESTDIR = os.path.dirname(os.path.abspath(__file__)) + os.path.sep


def make_phony_disk(dim):
    """
    Create a very simple disk model

    Args:
        dim: Dimension of the array

    Returns:
        centered ellisp disk

    """

    phony_disk = np.zeros((dim, dim))
    PA_rad = 0.4712388980  # 27 deg

    x = np.arange(dim, dtype=np.float)[None, :] - dim // 2
    y = np.arange(dim, dtype=np.float)[:, None] - dim // 2

    x1 = x * np.cos(PA_rad) + y * np.sin(PA_rad)
    y1 = -x * np.sin(PA_rad) + y * np.cos(PA_rad)

    x2 = x1
    y2 = y1 / np.cos(np.radians(76))
    rho2dellip = np.sqrt(x2**2 + y2**2)

    phony_disk[np.where((rho2dellip > 80) & (rho2dellip < 85))] = 1

    return phony_disk

def run_test_diskFM(just_loading=False, ext=".h5", nwls=1, annulitest=1):
    """
    Test DiskFM package. Creata Model disk. Create a disk model class.
    Measure and save the KL basis + measure a FM. Load the KL basis.
    Re-measuure a FM from loaded KL basis. Make sure the 2 FMs are not
    zero and that they are identical

    Args:
        just_loading: if True we are not measuring the KL basis, just
                        loading it from file
        ext: type of saving (h5 or pickle)
        nwls: number of wavelength when we collaps the data

    Returns:
        None

    """

    # grab the files
    filelist = sorted(
        glob.glob(TESTDIR + os.path.join("data", "S20131210*distorcorr.fits")))
    dataset = GPI.GPIData(filelist, quiet=True)

    # set a few parameters
    mov_here = 8
    numbasis = [3]
    [xcen, ycen] = [140, 140]
    fileprefix = "DiskFM_test_nwls{0}_ann{1}".format(nwls, annulitest)

    dataset.spectral_collapse(collapse_channels=nwls, align_frames=True)

    # create a phony disk model and convovle it by the instrument psf
    phony_disk_model = make_phony_disk(281)
    dataset.generate_psfs(boxrad=12)
    instrument_psf = dataset.psfs[0]
    instrument_psf[np.where(instrument_psf < 0)] = 0

    model_convolved = convolve(phony_disk_model,
                               instrument_psf,
                               boundary="wrap")

    if not just_loading:
        diskobj = DiskFM(
            dataset.input.shape,
            numbasis,
            dataset,
            model_convolved,
            basis_filename=TESTDIR + fileprefix + "_KLbasis" + ext,
            save_basis=True,
            aligned_center=[xcen, ycen],
        )

        fm.klip_dataset(
            dataset,
            diskobj,
            numbasis=numbasis,
            maxnumbasis=100,
            annuli=annulitest,
            subsections=1,
            mode="ADI",
            outputdir=TESTDIR,
            fileprefix=fileprefix,
            aligned_center=[xcen, ycen],
            mute_progression=True,
            highpass=False,
            minrot=mov_here,
            calibrate_flux=False,
        )

    if nwls == 1:
        fmout_klip_dataset = fits.getdata(TESTDIR + fileprefix +
                                          "-fmpsf-KLmodes-all.fits")
    else:
        fmout_klip_dataset = fits.getdata(
            TESTDIR + fileprefix +
            "-fmpsf-KL{0}-speccube.fits".format(numbasis[0]))

    diskobj = DiskFM(
        dataset.input.shape,
        numbasis,
        dataset,
        model_convolved,
        basis_filename=TESTDIR + fileprefix + "_KLbasis" + ext,
        load_from_basis=True,
    )

    diskobj.update_disk(model_convolved)
    modelfm_here = diskobj.fm_parallelized()

    if nwls == 1:
        return_klip_dataset = fmout_klip_dataset[0]  # first KL
        return_by_fm_parallelized = modelfm_here[0]  # first KL
    else:
        return_klip_dataset = fmout_klip_dataset[0]  # first KL
        return_by_fm_parallelized = modelfm_here[0][0]  # first KL, first WL

    # fits.writeto(
    #     TESTDIR + fileprefix + "_fm_parallelized-fmpsf.fits",
    #     return_by_fm_parallelized,
    #     overwrite=True,
    # )

    # # print(fmout_klip_dataset[0].shape)
    # # print(modelfm_here[0][0].shape)
    # fits.writeto(
    #     TESTDIR + fileprefix + "_res.fits",
    #     return_klip_dataset - return_by_fm_parallelized,
    #     overwrite=True
    # )

    # test that the FM models are not zero everywhere
    assert np.nanmax(np.abs(return_klip_dataset)) > 0.0
    assert np.nanmax(np.abs(return_by_fm_parallelized)) > 0.0

    # test that fm.klip_dataset and diskobj.fm_parallelized
    # give very similar result
    assert (np.nanmax(
        np.abs((return_klip_dataset - return_by_fm_parallelized) /
               return_klip_dataset)) < 1)



def test_disk_helper():
    run_test_diskFM(just_loading=False, ext=".h5", nwls=1, annulitest=1)
    run_test_diskFM(just_loading=False, ext=".h5", nwls=2, annulitest=1)
    run_test_diskFM(just_loading=False, ext=".h5", nwls=2, annulitest=2)

    run_test_diskFM(just_loading=True, ext=".h5", nwls=1, annulitest=1)
    run_test_diskFM(just_loading=True, ext=".h5", nwls=2, annulitest=1)
    run_test_diskFM(just_loading=True, ext=".h5", nwls=2, annulitest=2)

    run_test_diskFM(just_loading=False, ext=".pkl", nwls=1)
    run_test_diskFM(just_loading=False, ext=".pkl", nwls=2)

    run_test_diskFM(just_loading=True, ext=".pkl", nwls=1)
    run_test_diskFM(just_loading=True, ext=".pkl", nwls=2)


if __name__ == "__main__":
    test_disk_helper()


