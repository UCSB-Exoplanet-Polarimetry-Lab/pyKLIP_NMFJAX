KLIP Tutorial
==============
Here, we will explain how to run PSF subtraciton using the KLIP algorithm in pyKLIP. We will use GPI
data to explain the process, but other than reading in the data, all the PSF subtraction steps are the same for
any other dataset.

GPI Tutorial
------------

You'll need some GPI reduced spectral datacubes to start (with satellite spots located).
If you don't have any, you can use the beta Pic datacubes from the
`GPI Public Data Release <https://www.gemini.edu/sciops/instruments/gpi/public-data>`_.

Once you have reduced some data, we need to parse through the GPI data::

    import glob
    import pyklip.instruments.GPI as GPI

    filelist = glob.glob("path/to/dataset/*.fits")
    dataset = GPI.GPIData(filelist)

This returns ``dataset``, an implementation of the abstract class :class:`pyklip.Instrument.Data` with standardized fields
that are needed to perform the KLIP subtraction, none of which are instrument specific.
Please read the docstring for :py:class:`pyklip.instruments.GPI.GPIData` to more information on the the fields for GPI data.

Next, we will perform the actual KLIP ADI+SDI subtraction. To take advantage of the easily parallelizable computation, we will use the
:mod:`pyklip.parallelized` module to perform the KLIP subtraction, which uses the python ``multiprocessing`` library to parallelize the code::

    import pyklip.parallelized as parallelized

    parallelized.klip_dataset(dataset, outputdir="path/to/save/dir/", fileprefix="myobject",
                              annuli=9, subsections=4, movement=1, numbasis=[1,20,100],
                              calibrate_flux=True, mode="ADI+SDI")

This will save the processed KLIP images in the field ``dataset.output`` and as FITS files saved using the directory and fileprefix
specified. The FITS files contain two different kinds of outputs. The first is a single 3D datacube where the z-axis is all the
different KL mode cutoffs used for subtraction. The second is a series of 3D datacubes with the z-axis is wavelength and each datacube
uses a different KL mode cutoff as specified by its filename.

To measure the contrast (ignoring algorithm throughput), the :meth:`pyklip.klip.meas_contrast` function can do this.
First we have to take collapse the output, ``dataset.output``, in both the file and wavelength
dimensions. We also are going to pick the reduction using 20 KL modes, ``dataset.output[1]``,
to calculate the contrast::

    import numpy as np
    import pyklip.klip as klip

    avgframe = np.nanmean(dataset.output[1], axis=(0,1))
    calib_frame = dataset.calibrate_output(avgframe)
    seps, contrast = klip.meas_contrast(calib_frame, dataset.IWA, 1.1/GPI.GPIData.lenslet_scale, 3.5)

