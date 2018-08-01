.. _kpop-label:


Klip POst Processing (KPOP)
=====================================================


.. note::
    The object architecture of KPOP has been removed. It won't be available in future commits.
    The original idea was to abstract as many things as possible to make the user's life easier but it became counter
    productive when we started reducing data from different instruments as the formats and headers vary too much.
    However most features are still available as stand alone functions but may require more setup.


Klip POst Processing (KPOP) is a module with tools to calculate:

    * matched filter maps,
    * SNR maps,
    * detections

.. note::
    KPOP is the framework developped in the context of `Ruffio et al. (2016) <https://arxiv.org/pdf/1705.05477.pdf>`_.

PyKLIP can be installed following :ref:`install-label`.
KPOP functions
-----------------

Please find an example ipython notebook (pyklip/examples/kpop_tutorial.ipynb) using beta Pictoris test data available in the test directory of pyklip.

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you want a quick example to test the following pieces of code, run the following.
It will reduce the Beta Pictoris pyklip test data using KLIP.

.. code-block:: python

    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        print("Your code might run slowly.")
        print("/!\ Please Read http://pyklip.readthedocs.io/en/latest/install.html#note-on-parallelized-performance")

    import os
    import glob
    import numpy as np
    import pyklip.instruments.GPI as GPI
    import pyklip.parallelized as parallelized

    pykliproot = os.path.dirname(os.path.realpath(parallelized.__file__))
    inputDir = os.path.join(pykliproot,"..","tests","data")
    outputDir = inputDir

    # Read the datacubes using the GPIData object
    filelist = glob.glob(os.path.join(inputDir,"*spdc_distorcorr.fits"))
    dataset = GPI.GPIData(filelist,highpass=True,meas_satspot_flux=False,numthreads=None)

    parallelized.klip_dataset(dataset, outputdir=outputDir, fileprefix="bet_Pic_test",
                              annuli=9, subsections=4, movement=1, numbasis=[1,20,50,100],
                              calibrate_flux=True, mode="ADI+SDI")

Cross-correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    ########################
    ## cross correlation of speccube

    from pyklip.kpp.metrics.crossCorr import calculate_cc
    import astropy.io.fits as pyfits

    # Definition of the cross correlation object
    filename = os.path.join(outputDir,"bet_Pic_test-KLmodes-all.fits")
    hdulist = pyfits.open(filename)
    cube = hdulist[1].data
    prihdr = hdulist[0].header
    exthdr = hdulist[1].header
    hdulist.close()

    # Definition of the planet spectrum (not corrected )
    import pyklip.instruments.GPI as GPI
    import pyklip.spectra_management as spec
    from glob import glob
    pykliproot = os.path.dirname(os.path.realpath(spec.__file__))
    reduc_spectrum = "t600g100nc" # sharp methane feature
    spectrum_filename = os.path.abspath(glob(os.path.join(pykliproot,"spectra","*",reduc_spectrum+".flx"))[0])
    # Interpolate the spectrum of the planet based on the given filename
    wv,planet_sp = spec.get_planet_spectrum(spectrum_filename,GPI.get_gpi_wavelength_sampling("J"))

    %matplotlib inline
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(wv,planet_sp)

    # Definition of the PSF
    from pyklip.kpp.utils.mathfunc import *
    x_grid,y_grid= np.meshgrid(np.arange(-10,10),np.arange(-10,10))
    PSF = gauss2d(x_grid,y_grid, amplitude = 1.0, xo = 0.0, yo = 0.0, sigma_x = 1.0, sigma_y = 1.0)


    image_cc = calculate_cc(cube, PSF,spectrum = planet_sp, nans2zero=True)

Matched filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    ########################
    # matched filter of speccube

    from pyklip.kpp.metrics.matchedfilter import run_matchedfilter
    from pyklip.kpp.metrics.crossCorr import calculate_cc
    from pyklip.kpp.stat.statPerPix_utils import get_image_stat_map_perPixMasking
    import astropy.io.fits as pyfits

    # Definition of the cross correlation object
    filename = os.path.join(outputDir,"bet_Pic_test-KL20-speccube.fits")
    hdulist = pyfits.open(filename)
    cube = hdulist[1].data
    prihdr = hdulist[0].header
    exthdr = hdulist[1].header
    center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    nl,ny,nx = cube.shape
    hdulist.close()

    # Definition of the planet spectrum (not corrected )
    import pyklip.instruments.GPI as GPI
    import pyklip.spectra_management as spec
    from glob import glob
    pykliproot = os.path.dirname(os.path.realpath(spec.__file__))
    reduc_spectrum = "t1300g100f2" # L-type
    spectrum_filename = os.path.abspath(glob(os.path.join(pykliproot,"spectra","*",reduc_spectrum+".flx"))[0])
    # Interpolate the spectrum of the planet based on the given filename
    wv,planet_sp = spec.get_planet_spectrum(spectrum_filename,GPI.get_gpi_wavelength_sampling("J"))

    %matplotlib inline
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(wv,planet_sp)

    # Definition of the PSF
    from pyklip.kpp.utils.mathfunc import *
    x_grid,y_grid= np.meshgrid(np.arange(-10,10),np.arange(-10,10))
    PSF = gauss2d(x_grid,y_grid, amplitude = 1.0, xo = 0.0, yo = 0.0, sigma_x = 1.0, sigma_y = 1.0)
    PSF = np.tile(PSF,(nl,1,1))*planet_sp[:,None,None]

    mf_map,cc_map,flux_map = run_matchedfilter(cube, PSF,N_threads=None,maskedge=True)

SNR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    ########################
    # SNRs

    from pyklip.kpp.stat.statPerPix_utils import get_image_stat_map_perPixMasking
    from pyklip.kpp.stat.stat_utils import get_image_stat_map
    from pyklip.kpp.metrics.crossCorr import calculate_cc
    import astropy.io.fits as pyfits

    # Definition of the cross correlation object
    filename = os.path.join(outputDir,"bet_Pic_test-KLmodes-all.fits")
    hdulist = pyfits.open(filename)
    cube = hdulist[1].data
    prihdr = hdulist[0].header
    exthdr = hdulist[1].header
    center = [exthdr['PSFCENTX'], exthdr['PSFCENTY']]
    hdulist.close()

    # Definition of the PSF
    from pyklip.kpp.utils.mathfunc import *
    x_grid,y_grid= np.meshgrid(np.arange(-10,10),np.arange(-10,10))
    PSF = gauss2d(x_grid,y_grid, amplitude = 1.0, xo = 0.0, yo = 0.0, sigma_x = 1.0, sigma_y = 1.0)
    # Run cross correlation first
    image_cc = calculate_cc(cube[2,:,:], PSF,spectrum = None, nans2zero=True)

    SNR_map1 = get_image_stat_map(image_cc,
                               centroid = center,
                               r_step=2,
                               Dr = 2,
                               type = "SNR")

    SNR_map2 = get_image_stat_map_perPixMasking(image_cc,
                                               centroid = center,
                                               mask_radius=5,
                                               Dr = 2,
                                               type = "SNR")

Point-source detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    ########################
    # Detection
    import csv
    from pyklip.kpp.detection.detection import point_source_detection
    # list of the local maxima with their info
    #         Description by column: ["index","value","PA","Sep (pix)","Sep (as)","x","y","row","col"]
    #         1/ index of the candidate
    #         2/ Value of the maximum
    #         3/ Position angle in degree from North in [0,360]
    #         4/ Separation in pixel
    #         5/ Separation in arcsec
    #         6/ x position in pixel
    #         7/ y position in pixel
    #         8/ row index
    #         9/ column index
    detec_threshold = 3
    pix2as = 0.014166
    candidates_table = point_source_detection(SNR_map, center,detec_threshold,pix2as=pix2as,
                                             mask_radius = 15,maskout_edge=10,IWA=None, OWA=None)
    savedetections = os.path.join(outputDir,"detections.csv")
    with open(savedetections, 'w+') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerows([["index","value","PA","Sep (pix)","Sep (as)","x","y","row","col"]])
        csvwriter.writerows(candidates_table)

