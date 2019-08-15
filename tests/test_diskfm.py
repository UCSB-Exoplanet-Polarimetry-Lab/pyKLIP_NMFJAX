"""test DiskFM
author: johan mazoyer
"""

import os
import glob
import warnings

import numpy as np
import astropy.io.fits as fits
from astropy.convolution import convolve

import pyklip.instruments.GPI as GPI
from pyklip.fmlib.diskfm import DiskFM
import pyklip.fm as fm


warnings.filterwarnings("ignore", category=RuntimeWarning)

########################################################
########################################################

TESTDIR = os.path.dirname(os.path.abspath(__file__)) + os.path.sep


def test_diskfm():

    """
    Test DiskFM package. Creata Model disk. Create a disk model class. Measure and save
    the KL basis + measure a FM. Load the KL basis. Re-measuure a FM from loaded KL
    basis. Make sure the 2 FMs are not zero and that they are identical

    Args:
        None

    Returns:
        None

    """

    ext = ".pkl"
    # ext = ".h5"

    # grab the files
    filelist = glob.glob(TESTDIR + os.path.join("data", "S20131210*distorcorr.fits"))
    dataset = GPI.GPIData(filelist, quiet=True)

    # set a few parameters
    mov_here = 8
    numbasis = [3]
    [xcen, ycen] = [140, 140]
    fileprefix = "DiskFM_test"

    # run it for 2 WL to make it harder
    dataset.spectral_collapse(collapse_channels=2, align_frames=True)

    # create a phomny disk model and convovle it by the instrument psf
    phony_disk_model = make_phony_disk(281)
    dataset.generate_psfs(boxrad=12)
    instrument_psf = dataset.psfs[0]
    instrument_psf[np.where(instrument_psf < 0)] = 0

    model_convolved = convolve(phony_disk_model, instrument_psf, boundary="wrap")

    diskobj = DiskFM(
        dataset.input.shape,
        numbasis,
        dataset,
        model_convolved,
        annuli=1,
        subsections=1,
        basis_filename=TESTDIR + fileprefix + "_KLbasis" + ext,
        save_basis=True,
        aligned_center=[xcen, ycen],
    )

    fm.klip_dataset(
        dataset,
        diskobj,
        numbasis=numbasis,
        maxnumbasis=100,
        annuli=2,
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

    fmout_klip_dataset = fits.getdata(
        TESTDIR
        + fileprefix
        + "-fmpsf-KL{0}-speccube.fits".format(numbasis[0])
    )

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
    # fits.writeto(
    #     TESTDIR + fileprefix + "_fm_parallelized-fmpsf.fits",
    #     modelfm_here[0][0],
    #     overwrite=True,
    # )

    # print(fmout_klip_dataset[0].shape)
    # print(modelfm_here[0][0].shape)
    # fits.writeto(
    #     TESTDIR + fileprefix + "_res.fits",
    #     fmout_klip_dataset[0] - modelfm_here[0][0],
    #     overwrite=True,
    # )

    # test that the FM models are not zero everywhere

    return_klip_dataset = fmout_klip_dataset[0]
    return_by_fm_parallelized = modelfm_here[0][0]
    assert np.nanmax(np.abs(return_klip_dataset)) > 0.0
    assert np.nanmax(np.abs(return_by_fm_parallelized)) > 0.0

    # test that fm.klip_dataset and diskobj.fm_parallelized give very similar result
    assert (
        np.nanmax(
            np.abs(
                (return_klip_dataset - return_by_fm_parallelized) / return_klip_dataset
            )
        )
        < 1
    )


def make_phony_disk(dim):

    """
    Create a very simple disk model

    Args:
        dim: Dimension of the array

    Returns:
        centered ellisp disk

    """

    phony_disk = np.zeros((dim, dim))
    PA_rad = np.radians(27)

    x = np.arange(dim, dtype=np.float)[None, :] - dim // 2
    y = np.arange(dim, dtype=np.float)[:, None] - dim // 2

    x1 = x * np.cos(PA_rad) + y * np.sin(PA_rad)
    y1 = -x * np.sin(PA_rad) + y * np.cos(PA_rad)

    x2 = x1
    y2 = y1 / np.cos(np.radians(76))
    rho2dellip = np.sqrt(x2 ** 2 + y2 ** 2)

    phony_disk[np.where((rho2dellip > 80) & (rho2dellip < 85))] = 1

    return phony_disk


if __name__ == "__main__":
    test_diskfm()
