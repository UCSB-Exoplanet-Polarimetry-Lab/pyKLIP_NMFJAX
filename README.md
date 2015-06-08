# pyKLIP README #

A python library for PSF subtraction for both exoplanet and disk imaging. Development led by Jason Wang. Contributions made by Jonathan Aguilar, JB Ruffio, Rob de Rosa, Schuyler Wolff, and Laurent Pueyo.
If you use pyKLIP in your research, please cite the Astrophysical Source Code Library record of it: [http://ascl.net/1506.001](http://ascl.net/1506.001).

### Overview ###

* Implementation of [KLIP](http://arxiv.org/abs/1207.4197) in Python
* Capable of running ADI, SDI, ADI+SDI with spectral templates to optimize the PSF subtraction
* Parallelized with both a quick memory-intensive mdoe and a slower memory-lite mode
* Initially built for [P1640](http://www.amnh.org/our-research/physical-sciences/astrophysics/research/project-1640) and 
[GPI](http://planetimager.org/) data reduction, but modularized so that interfaces can be written for other instruments too
* If confused about what a function is doing, read the docstring for it.
* Version 1.0

### Dependencies ###

* numpy
* scipy
* astropy
* python2.7 or python3
* Recommended: a computer with lots of cores (16+) and lots of memory (40 GB for a standard GPI 1hr sequence)

### Installation ###

Put all of the files into a folder called *pyklip* and put the *pyklip* folder into either the directory in which your python script that will call ``pyklip`` resies or into a directory that is in your ``PYTHONPATH``.

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
        seps, contrast = klip.meas_contrast(avgframe, dataset.IWA, 1.1/GPI.GPIData.lenslet_scale, 3.5)