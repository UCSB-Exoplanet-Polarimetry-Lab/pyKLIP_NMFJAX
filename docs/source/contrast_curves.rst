.. _contrast-label:

Calibrating Algoirthm Throughput & Generating Contrast Curves
==================

Due to oversubtraction and selfsubtraction (see `Pueyo (2016) <http://arxiv.org/abs/1604.06097>`_ for a good
explaination), the shape, flux, and spectrum of the signal of a planet or disk is distoed by PSF subtraction.
To calibrate algorithm throughput after KLIP in this tutorial, we will use the standard fake injection technique,
which basically is injecting fake planets/disks in the data of known shape, flux, and spectrum to measure the
algorithm throughput.

In this tutorial, we will calibrate the throughput of the previous exmaple (:ref:`basic-tutorial-label`) for the
purpose of generating a contrast curve. Note that this same general process can be used to character a planet or disk
(e.g. measure astrometry and spectrum of an imaged exoplanet).


Contrast Curves
---------------

To measure the contrast (ignoring algorithm throughput), the :meth:`pyklip.klip.meas_contrast` function can do this.
This will give us a sense of the contrast to inject fake planets into the data (algorithm throughput is ~50%).
We are calculating broadband contrast so we want to spectrally-collapsed data (if applicable). You can do this
by reading back in the KL mode cube and picking a KL mode cutoff. Here is how to do that for pyKLIP output of
GPI data::

    import astropy.io.fits as fits
    hdulist = fits.open("myobject-KLmodes-all.fits")
    # pick the 50 KL mode cutoff frame out of [1,20,50,100]
    avgframe = hdulist[1].data[2]


Then, a convenient pyKLIP function will calculate the contrast, accounting for small
sample statistics. We are picking 1.1 arcseconds as the outer boundary of our contrast curve. We also need
to specify the FWHM of the PSF in order to account for small sample statistics::

    import numpy as np
    import pyklip.klip as klip

    avgframe = np.nanmean(dataset.output[1], axis=(0,1))
    seps, contrast = klip.meas_contrast(avgframe, dataset.IWA, 1.4/GPI.GPIData.lenslet_scale, 3.5)



TODO: picture here of contrast curve?

Injecting Fake Planets
----------------------
To calibrate our sensitivity to planets, we need to inject some fake planets at known brightness into our data.
