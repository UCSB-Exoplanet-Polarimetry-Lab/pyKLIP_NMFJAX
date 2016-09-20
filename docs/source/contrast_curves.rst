.. _contrast-label:

Calibrating Algoirthm Throughput & Generating Contrast Curves
=============================================================

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

To measure the contrast (ignoring algorithm throughput), we use :meth:`pyklip.klip.meas_contrast`, which assumes
azimuthally symmetric noise and computes the 5σ noise level at each separation. It uses a Gaussian matched filter to
compute the noise as a small optimization to smooth out high frequency noise (since we know our planet is not going to
be smaller than on λ/D scales). It also corrects for small number statistics (i.e. by assuming a Student's
t-distribution rather than a Gaussian).
This will give us a sense of the contrast to inject fake planets into the data (algorithm throughput is ~50%).
We are calculating broadband contrast so we want to spectrally-collapsed data (if applicable). You can do this
by reading back in the KL mode cube and picking a KL mode cutoff. Here is how to do that for pyKLIP output of
GPI data::

    import astropy.io.fits as fits
    hdulist = fits.open("myobject-KLmodes-all.fits")
    # pick the 50 KL mode cutoff frame out of [1,20,50,100]
    kl50frame = hdulist[1].data[2]


Then, a convenient pyKLIP function will calculate the contrast, accounting for small
sample statistics. We are picking 1.1 arcseconds as the outer boundary of our contrast curve. We also need
to specify the FWHM of the PSF in order to account for small sample statistics::

    import numpy as np
    import pyklip.klip as klip

    dataset_iwa = GPI.GPIData.fpm_diam['J']/2 # radius of occulter in GPI J band
    dataset_owa = 1.5/GPI.GPIData.lenslet_scale # 1.5" is the furtherest out we will go
    dataset_fwhm = 3.5 # fwhm of PSF roughly
    seps, contrast = klip.meas_contrast(kl50frame, dataset_iwa, dataset_owa, dataset_fwhm)

TODO: picture here of contrast curve?

Injecting Fake Planets
----------------------
To calibrate our sensitivity to planets, we need to inject some fake planets at known brightness into our data. In this
tutorial, we only only inject a few fakes once into the data just to demonstrate the technique with pyKLIP. For your
data, it is suggested you inject many planets to explore the attenuation factor as a function of brightness,
separation, and KLIP parameters (more aggressive reductions increase attenuation of flux due to KLIP).


