# pyKLIP README #

A python library for PSF subtraction for both exoplanet and disk imaging. Development led by Jason Wang. Contributions made by Jonathan Aguilar, JB Ruffio, Rob de Rosa, Schuyler Wolff, Abhijith Rajan, Zack Briesemeister, and Laurent Pueyo (see contributors.txt for a detailed list).
If you use pyKLIP in your research, please cite the Astrophysical Source Code Library record of it: [ASCL](http://ascl.net/1506.001) or [ADS](http://adsabs.harvard.edu/abs/2015ascl.soft06001W).

> Wang, J. J., Ruffio, J.-B., De Rosa, R. J., et al. 2015, Astrophysics Source Code Library, ascl:1506.001

### Overview ###

* Implementation of [KLIP](http://arxiv.org/abs/1207.4197) and [KLIP-FM](http://arxiv.org/abs/1604.06097) in Python
* Capable of running ADI, SDI, ADI+SDI with spectral templates to optimize the PSF subtraction
* Library of KLIP-FM capabilties including forward-modelling a PSF, detection algorithms, and spectral extraction.
* Post-processing planet detection algorithms included
* Parallelized with both a quick memory-intensive mode and a slower memory-lite mode
* Initially built for [P1640](http://www.amnh.org/our-research/physical-sciences/astrophysics/research/project-1640) and 
[GPI](http://planetimager.org/) data reduction, but modularized so that interfaces can be written for other instruments too (e.g. a NIRC2 interface has been added)
* If confused about what a function is doing, read the docstring for it. Full documentation site coming!
* Version 1.1 - see ``release_notes.txt`` for update notes

### Dependencies ###

* numpy
* scipy
* astropy
* python2.7 or python3.4
* Recommended: a computer with lots of cores (16+) and lots of memory (20+ GB for a standard GPI 1hr sequence without using lite mode)

### Installation ###

To install the most up to date developer version of *pyklip*, clone this repository if you haven't already. 

    :::bash
        $ git clone git@bitbucket.org:pyKLIP/pyklip.git

Then ``cd`` into it and run ``setup.py`` with the ``develop`` option

    :::bash
        $ python setup.py develop

If you use multiple versions of python, you will need to run ``setup.py`` with each version of python (this should not apply to most people).

### Note on parallelized performance ###

Due to the fact that numpy compiled with BLAS and MKL also parallelizes linear algebra routines across multiple cores, performance can actually sharply decrease when multiprocessing and BLAS/MKL both try to parallelize the KLIP math. If you are noticing your load averages greatly exceeding the number of threads/CPUs, try disabling the BLAS/MKL optimization when running pyKLIP.

To disable OpenBLAS, just set the following environment variable before running pyKLIP:

    :::bash
       $ export OPENBLAS_NUM_THREADS=1

[A recent update to anaconda](https://www.continuum.io/blog/developer-blog/anaconda-25-release-now-mkl-optimizations) included some MKL optimizations which may cause load averages to greatly exceed the number of threads specified in pyKLIP. As with the OpenBLAS optimizations, this can be avoided by setting the maximum number of threads the MKL-enabled processes can use. 

    :::bash
       $ export MKL_NUM_THREADS=1

As these optimizations may be useful for other python tasks, you may only want MKL_NUM_THREADS=1 only when pyKLIP is called, rather than on a system-wide level. By defaulf in ``parallelized.py``, if ``mkl-service`` is installed, the original maximum number of threads for MKL is saved, and restored to its original value after pyKLIP has finished. You can also modify the number of threads MKL uses on a per-code basis by running the following piece of code (assuming ``mkl-service`` is installed).

    :::python
      import mkl
      mkl.set_num_threads(1)


### Bugs/Feature Requests ###

Please use the Bitbucket Issue Tracker to submit bugs and new feature requests

### Quick GPI Tutorial ###

You'll need some GPI reduced spectral datacubes to start (with satellite spots located). First we need to parse through the data. This is done with the ``instruments.GPI`` module.

    :::python
        import glob
        import pyklip.instruments.GPI as GPI

        filelist = glob.glob("path/to/dataset/*.fits")
        dataset = GPI.GPIData(filelist)

This returns ``dataset``, an implementation of the abstract class ``Instrument.Data`` with fields such as ``data``,
``wvs``, ``PAs`` that are needed to perform the KLIP subtraction, none of which are instrument specific.
 Please read the docstring for ``GPI`` or ``Instrument.Data``for a full description of what is in ``dataset``.

Next, we will perform the actual KLIP ADI+SDI subtraction. To take advantage of the easily parallelizable computation, we will use the
``parallelized`` module to perform the KLIP subtraction, which uses the python ``multiprocessing`` library to parallelize the code.

    :::python
        import pyklip.parallelized as parallelized

        parallelized.klip_dataset(dataset, outputdir="path/to/save/dir/", fileprefix="myobject", annuli=9, subsections=4, movement=3, numbasis=[1,20,100], calibrate_flux=True, mode="ADI+SDI")

This will save the processed KLIP images in the field ``dataset.output`` and as FITS files saved using the directory and fileprefix
 specified. The FITS files contain two different kinds of outputs. The first is a single 3D datacube where the z-axis is all the
 different KL mode cutoffs used for subtraction. The second is a series of 3D datacubes with the z-axis is wavelength and each datacube
  uses a different KL mode cutoff as specified by its filename.

To measure the contrast (ignoring algorithm throughput), the ``klip.meas_contrast`` function can do this. First we have to take collapse the output, ``dataset.output``, in both the file and wavelength dimensions. We also are going to pick the reduction using 20 KL modes, ``dataset.output[1]``, to calculate the contrast.

    :::python
        import numpy as np
        import pyklip.klip as klip

        avgframe = np.nanmean(dataset.output[1], axis=(0,1))
        calib_frame = dataset.calibrate_output(avgframe)
        seps, contrast = klip.meas_contrast(calib_frame, dataset.IWA, 1.1/GPI.GPIData.lenslet_scale, 3.5)