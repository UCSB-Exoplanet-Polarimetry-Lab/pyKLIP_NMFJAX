# README #

### Overview ###

* Implementation of [KLIP](http://arxiv.org/abs/1207.4197) in Python
* Initially built for [P1640](http://www.amnh.org/our-research/physical-sciences/astrophysics/research/project-1640) and [GPI](http://planetimager.org/) data reduction
* Version 0.0

### Dependencies ###

* numpy
* scipy
* pyfits
* python2.7 or python3

### TODO ###

* Remove GPI specific code
* Read in instrument parameters for an .ini file
* Smarter way of breaking image into annuli (related to above)
* Post processing analysis functions
* Spectral Recovery

### GPI Example ###

You'll need some GPI reduced spectral datacubes to start. First we need to parse through the data. This is done with the ``readdata`` module.

    :::python
        import glob
        import readdata

        filelist = glob.glob("path/to/dataset/*.fits")
        dataset = readdata.gpi_readdata(filelist)

This returns ``dataset``, a dictionary with arrays such as ``data``, ``wvs``, ``PAs`` that are needed to perform the KLIP subtraction.
Please read the docstring for ``dataset`` for a full description of what is in ``dataset``.

Next, we will perform the actual KLIP ADI+SDI subtraction. To take advantage of the easily parallelizable computation, we will use the
``parallelized`` module to perform the KLIP subtraction, which uses the python ``multiprocessing`` library to parallelize the code.

    :::python
        import parallelized

        subtracted_imgs = parallelized.klip_dataset(dataset, outputdir="path/to/save/dir/", fileprefix="myobject")

This will both give us the KLIP processed images as ``subtracted_imgs`` and as FITS files saved using the directory and fileprefix
 specified. The FITS files contain a 3D datacube where the z-axis is all the different KL mode cutoffs used for subtraction and a series
 of 3D datacubes with the z-axis is wavelength and each datacube uses a different KL mode cutoff as specified by its filename.
