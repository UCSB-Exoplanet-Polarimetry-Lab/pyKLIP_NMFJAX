.. _CHARIS-label:

CHARIS Tutorials
================
This tutorial will walk you through the steps of a standard post processing reduction on CHARIS IFS data, including
PSF subtraction, forward modeling, and spectral extraction. This tutorial will also provide basic reduction steps and
explanations for the reduction parameters to allow you to tweak the reduction for your specific data set for the
optimal outcome.

The pyKLIP CHARIS interface supports IFS data produced from the
`CHARIS Data Reduction Pipeline <https://github.com/PrincetonUniversity/charis-dep>`_. You can use the
`CHARIS Data Reduction Pipeline documentation <http://princetonuniversity.github.io/charis-dep/>`_ to learn how to use
it to extract CHARIS data cubes.
To obtain data cubes for pyKLIP reductions, please download the aforementioned pipeline and refer to its tutorials to
extract data cubes from raw CHARIS data. The extracted data are 3-D data cubes where the third dimension is wavelength.

Package Requirements
--------------------
Refer to :ref:`install-label` for package dependencies.

Reading CHARIS Data and Centroiding
-----------------------------------
Once you have the extracted data cubes using the
`CHARIS Data Reduction Pipeline <https://github.com/PrincetonUniversity/charis-dep>`_, we first need to initialize the
data set and measure the centroid of the images. This will allow us to re-register and align the images for PSF modeling
and subtraction.

This tutorial is aimed at processing CHARIS data taken with a coronagraph and astrogrid, which block out the central
star and creates fainter copies of the central star psf at fixed off-center locations. These copies are called satallite
spots, and we measure the positions of these satellite spots to triangulate the image centroid. This centroiding step
is done automaticallly when initializing data with default arguments, the measured satellite spot positions will be
added to the headers of the original data cubes. pyKLIP will also automatically detect these header keywords and will
not carry out the centroiding measurement again if the measurement already exists.

You can initialize and measure the centroids of a CHARIS data set using the following code:

.. code-block:: python

    import glob
    import numpy as np
    import pyklip.instruments.CHARIS as CHARIS

    filepath = '/home/minghan/CHARIS/charis-work/reduced_data/HD33632/CRSA*_cube.fits'
    filelist = sorted(glob.glob(filepath))

    dataset = CHARIS.CHARISData(filelist, IWA=None, OWA=None)

However, by default, the centroid measurement is not done on a frame by frame basis, but rather takes into account the
relative offsets across all data cubes and fits for all image centroids simultaneously. As a consequence, if there are
bad data cubes (bad AO, intrument failure, different astrogrid settings etc.) in the sequence of data cubes you are
using, they will affect the centroid measurements for all cubes to varying extent. So, if you need to re-run the
centroid measurement due to an updated data cubes selection, you will need to specify it manually as it will
by default detect that the keywords already exist and skip the centroid step. You can do so using the "update_hdrs"
keyword:

.. code-block:: python

    dataset = CHARIS.CHARISData(filelist, IWA=None, OWA=None, update_hdrs=True)

There is also a simpler method that measures the satellite spot positions one by one, if the previously mentioned
global fitting fails for some reason. You can also specify to use this local fitting method by setting the keyword
"sat_fit_method" to "local", which is by default "global":

.. code-block:: python

    dataset = CHARIS.CHARISData(filelist, IWA=None, OWA=None, update_hdrs=True, sat_fit_method='local')

Now we are ready to perform KLIP subtraction using Angular Differential Imaging (ADI) and/or Spectral Differential
Imaging (SDI). First, we'll briefly go through some most common parameters used in the reduction, and then we'll show
examples for running the reduction using these parameters.

Reduction Parameters
--------------------
Please refer to :ref:`basic-tutorial-label` for detailed explanation on some of the parameters and how
you should pick them.

annuli: the number of annulus to divide the image into

subsection: the number of azimuthal section to divide the image intno

movement: the number of minimum pixel movements of a potential source for an image to be selected as template.

numbasis: the number(s) of KL basis cutoffs to use for PSF subtraction, this can be an array so you can experiment with
multiple KL basis numbers in a single reduction.

maxnumbasis: the maximum number of KL modes used for the realization of the speckle noise.

mode: for CHARIS, use either 'ADI', 'SDI' or 'ADI+SDI'

guessspec:

Running KLIP
------------
Now we are ready to perform the KLIP algorithm with the following code and recommended default parameters:

.. code-block:: python

    import pyklip.parallelized as parallelized

    outpath = '/path/to/output/directory'
    prefix = 'object name'
    numbasis = [1, 20 , 50]
    annuli = 9
    subsec = 4
    movement = 1
    maxnumbasis = 150
    mode = 'ADI+SDI'
    parallelized.klip_dataset(dataset, outputdir=outpath, fileprefix=prefix, annuli=annuli, subsections=subsec,
                              movement=movement, numbasis=numbasis, maxnumbasis=maxnumbasis, mode=mode,
                              time_collapse='weighted-mean', wv_collapse='trimmed-mean')

`pyklip.parallelized.klip_dataset` will save the processed KLIP images in the field ``dataset.output`` and as FITS files
in the specified directory. To learn about the two types of outputs, please refer to :ref:`basic-tutorial-label`

Forward-Model Astrometry and Photometry
---------------------------------------
Once we have a detection and a known approximate location for the companion of interest, we can use forward modeling
to model the companion PSF and fit for the astrometry and photometry.
For a more detailed description of forward modeling and fitting for astrometry and photometry, please refer to
:ref:`bka-label`.

You can run the forward modeled reduction with the following code. Note that for this section and the next section
(spectral extraction), we redefined all the reduction parameters, because as long as the dataset object is initialized,
these sections are independent from each other and can be run individually.

.. code-block:: python

    import pyklip.fm as fm
    import pyklip.fmlib.fmpsf as fmpsf

    fm_outpath = '/path/to/forward/model/output'
    prefix = "object_name-fmpsf" # fileprefix for the output files

    # setup FM guesses, change these to the numbers suited for your data.
    # radius from primary star centroid in pixels
    guesssep = 45.5
    # position angle in degrees (fairly certain it's degrees)
    guesspa = 261.25
    guessflux = 0.0249 # in units of contrast to dataset.psfs
    star_type = 'F8V'
    guessspec = your_spectrum # should be 1-D array with number of elements = np.size(np.unique(dataset.wvs))

    # reduction parameters
    numbasis = np.array([5, 20]) # KL basis cutoffs you want to try
    maxnumbasis = 150
    mode = 'ADI+SDI'
    # since we now already know where the companion is roughly, we only have to reduce the region around the companion
    # annuli and subsections can be specified as pixel and radian boundaries, respectively, instead of as integers.
    annuli = [[guesssep-15, guesssep+15]] # one annulus centered on the planet
    phi_section_size = 30 / guesssep # radians
    subsections = [[np.radians(guesspa) - phi_section_size / 2.,
                    np.radians(guesspa) + phi_section_size / 2.]] # one section centered on the planet
    padding = 0 # we are not padding our zones
    movement = 2

    # generate a background-subtracted satellite spot PSFs with shape (nwv, boxsize, boxsize), averaged over exposures
    boxsize = 15
    dataset.generate_psfs(boxrad=boxsize // 2)

    # sets up the contrast to data number conversion, further explained after this code block
    wvs = np.unique(dataset.wvs)
    dataset_gridamp = float(dataset.prihdrs[0]['X_GRDAMP'])
    star_to_spot_ratio = 1. / ((dataset_gridamp / dataset.ref_spot_contrast_gridamp) ** 2
                               * dataset.ref_spot_contrast * dataset.ref_spot_contrast_wv ** 2 / (wvs ** 2))
    dn_per_contrast = np.tile(star_to_spot_ratio, (dataset.input.shape[0]//22))

    fm_class = fmpsf.FMPlanetPSF(dataset.input.shape, numbasis, guesssep, guesspa, guessflux, dataset.psfs,
                                 np.unique(dataset.wvs), dn_per_contrast, star_spt=star_type,
                                 spectrallib=guessspec, spectrallib_units='contrast')

    fm.klip_dataset(dataset, fm_class, mode=mode, outputdir=fm_outpath, fileprefix=prefix, numbasis=numbasis,
                    maxnumbasis=maxnumbasis, annuli=annuli, subsections=subsections, padding=padding,
                    movement=movement, time_collapse='weighted-mean')

Some further explanation of some parameters:

dn_per_contrast: scaling factor(s) that convert contrasts (such as the parameter `guessflux`) to data units. The
convention that we adopted in this tutorial is to set dn_per_contrast to the contrast of the host star over the
satellite spots, as measured by Thayne and reported in `this paper <http://dx.doi.org/10.1117/12.2576349>`_,
and the `guessflux` parameter be specified as the contrast between the companion and the satellite spot.

For fitting astrometry and photometry: please refer to :ref:`bka-label`

Spectral Extraction
-------------------
Once you have detected and fitted for the astrometry of the companion in previous sections, you now have the information
required for spectral extraction, which is done using the extractSpec module in the forward modeling lilbrary.

The example below is an extraction for the example data set, which will return the extracted spectrum with shape
(len(numbasis), number of wavelength channels) in units of contrast to the satellite spot psfs ``dataset.psfs``.

.. code-block:: python

    import pyklip.fmlib.extractSpec as es

    exspec_outpath = '/path/to/extracted/spectrum/output'
    prefix = "object_name-fmspect" # fileprefix for the output files

    # use the known planet separation and PA, for example, from the previous forward-model fitted astrometry
    planet_sep = 45.94 # pixels
    planet_pa = 261.12 # degrees
    planet_stamp_size = 10 # how big of a stamp around the companion in pixels, stamp will be stamp_size**2 pixels
    stellar_template = None # a stellar template spectrum, if you want

    # reduction parameters
    numbasis = np.array([5, 20])
    maxnumbasis =150
    mode = 'ADI+SDI'
    annuli=[[planet_sep-planet_stamp_size, planet_sep+planet_stamp_size]]
    phi_section_size = 2 * planet_stamp_size / planet_sep # radians
    subsections=[[np.radians(planet_pa) - phi_section_size / 2.,
                  np.radians(planet_pa) + phi_section_size / 2.]],
    movement = 2

    # generate a background-subtracted satellite spot PSFs with shape (nwv, boxsize, boxsize), averaged over exposures
    boxsize = 15
    dataset.generate_psfs(boxrad=boxsize//2)

    fm_class = es.ExtractSpec(dataset.input.shape,
                              numbasis,
                              planet_sep,
                              planet_pa,
                              dataset.psfs,
                              np.unique(dataset.wvs),
                              stamp_size = planet_stamp_size)


    fm.klip_dataset(dataset, fm_class,
                    mode=mode,
                    fileprefix=prefix,
                    annuli=annuli,
                    subsections=subsections,
                    movement=movement,
                    numbasis=numbasis,
                    maxnumbasis=maxnumbasis,
                    spectrum=stellar_template,
                    outputdir=exspec_outpath, time_collapse='weighted-mean')

    # If you want to scale your spectrum by a calibration factor:
    units = "scaled"
    scaling_factor = your_calibration_factor
    #e.g. you could set scaling_factor to the star_to_spot_ratio variable in the previous section, which will convert
    # the extracted spectrum from units of contrast to the satellite spot PSF to units of data number
    # otherwise, the defaults are:
    units = "natural" # (default) returned relative to input PSF model
    scale_factor=1.0 # (default) not used if units not set to "scaled"

    fmout_nanzero = np.copy(dataset.fmout)
    fmout_nanzero[np.isnan(fmout_nanzero)] = 0.
    exspect, fm_matrix = es.invert_spect_fmodel(fmout_nanzero, dataset, units=units, scaling_factor=scaling_factor,
                                                method='leastsq')