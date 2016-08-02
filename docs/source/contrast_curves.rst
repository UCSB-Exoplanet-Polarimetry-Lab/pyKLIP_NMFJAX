.. _contrast-label:

Generating Contrast Curves
==================

To measure the contrast (ignoring algorithm throughput), the :meth:`pyklip.klip.meas_contrast` function can do this.
First we have to take collapse the output, ``dataset.output``, in both the file and wavelength
dimensions. We also are going to pick the reduction using 20 KL modes, ``dataset.output[1]``,
to calculate the contrast::

    import numpy as np
    import pyklip.klip as klip

    avgframe = np.nanmean(dataset.output[1], axis=(0,1))
    calib_frame = dataset.calibrate_output(avgframe)
    seps, contrast = klip.meas_contrast(calib_frame, dataset.IWA, 1.1/GPI.GPIData.lenslet_scale, 3.5)