# README #

### Overview ###

* Implementation of [KLIP](http://arxiv.org/abs/1207.4197) in Python
* Initially built for [P1640](http://www.amnh.org/our-research/physical-sciences/astrophysics/research/project-1640) and [GPI](http://planetimager.org/) data reduction
* Version 0.1

### Dependencies ###

* numpy
* scipy
* astropy
* python2.7 or python3
* Recommended: a computer with lots of cores (16+) and lots of memory (~100 GB?)

### Installation ###

Put all of the files into a folder called *pyklip* and put the *pyklip* folder into either the directory in which your python script that will call ``pyklip`` resies or into a directory that is in your ``PYTHONPATH``.

### TODO ###

* Remove GPI specific code
* Read in instrument parameters for an .ini file
* Smarter way of breaking image into annuli (related to above)
* Post processing analysis functions
* Spectral Recovery

### GPI Example ###

You'll need some GPI reduced spectral datacubes to start (with satellite spots located). First we need to parse through the data. This is done with the ``readdata`` module.

    :::python
        import glob
        import pyklip.instruments.GPI as GPI

        filelist = glob.glob("path/to/dataset/*.fits")
        dataset = GPI.GPIData(filelist)

This returns ``dataset``, an implementation of the abstract class ``Instrument.Data`` with fields such as ``data``,
``wvs``, ``PAs`` that are needed to perform the KLIP subtraction. Please read the docstring for ``GPI`` or
``Instrument.Data``for a full description of what is in ``dataset``.

Next, we will perform the actual KLIP ADI+SDI subtraction. To take advantage of the easily parallelizable computation, we will use the
``parallelized`` module to perform the KLIP subtraction, which uses the python ``multiprocessing`` library to parallelize the code.

    :::python
        import pyklip.parallelized as parallelized

        parallelized.klip_dataset(dataset, outputdir="path/to/save/dir/", fileprefix="myobject")

This will save the processed KLIP images in the field ``dataset.output`` and as FITS files saved using the directory and fileprefix
 specified. The FITS files contain two different kinds of outputs. The first is a single 3D datacube where the z-axis is all the
 different KL mode cutoffs used for subtraction. The second is a series of 3D datacubes with the z-axis is wavelength and each datacube
  uses a different KL mode cutoff as specified by its filename.