.. _fmspect-label:

Spectrum Extraction using extractSpec FM
========================================

This document describes how to use KLIP-FM to extract a spectrum,
described in 
`Pueyo et al. (2016) <http://adsabs.harvard.edu/abs/2016ApJ...824..117P>`_ 
to account the effect of the companion signal in the reference library
when measuring its spectrum.

Running gen_fm and invert_spect_fmodel
--------------------------------------
gen_fm and invert_spect_fm are modules in pyklip/fmlib/extractSpec

gen_fm generates the forward model array given a pyklip instrument 
dataset and invert_spect_fm returns a spectrum in contrast units 
given the forward model array.

gen_fm usage::
 
    import glob
    import pyklip.instruments.GPI as GPI
    import pyklip.fmlib.extractSpec as es

    files = glob.glob("path/to/dataset/*.fits")
    dataset = GPI.GPIData(files, highpass=True)
    dataset.generate_psf_cube(20)

    pars = (45, 222) # separation and pa
    # other optional parameters shown below w/ default values
    fmarr = es.gen_fm(dataset, pars, numbasis=20, mv=2.0, stamp=10, numthreads=4,
                      spectra_template=None)
    # numbasis is K_klip
    # mv is movement parameter for reference library selection
    # stamp is postage stamp size
    # numthreads is specific to your machine
    # spectra_template is default None

    spectrum, fm_matrix = es.invert_spect_fmodel(fmarr, dataset, method="JB")
    # method indicates which matrix inversion method to use,
    # "JB" matrix inversion adds up over all exposures, then inverts
    # "LP" inversion adds over frames and one wavelength axis, then inverts

One way to calculate a spectrum with errorbars (we are getting rid of this though?)::

    import glob
    import pyklip.instruments.GPI as GPI
    import pyklip.fmlib.extractSpec as es

    files = glob.glob("path/to/dataset/*.fits")
    dataset = GPI.GPIData(files, highpass=True)
    dataset.generate_psf_cube(20)

    pars = (45, 222) # separation and pa
    # optional parameters shown w/ default values
    # This will take a long time - it is running the fm for 11 fake injections
    # and inverting the matrix by two different methods. It returns a dictionary
    # containing the spectrum by each method, and the measured error.
    spectrum_dict = es.get_spectrum_with_errorbars(dataset, pars, movement=1.0,
                                                   stamp=10, numbasis=3)
    
    
Some diagnostics you can run to check the FM
--------------------------------------------
1) First step is to look at the postage stamp of the klipped data and make sure
the companion signal is in there.::

    
    # useful values
    N_frames = len(dataset.input)
    N_cubes = len(dataset.exthdrs)
    nl = int(N_frames / N_cubes)
    num_k_klip = len(numbasis)

    fmarr = es.gen_fm(dataset, pars, numbasis=20, mv=2.0, stamp=10, numthreads=4,
                      spectra_template=None)
    klipped_data = fmarr[:,:,-1, :]
    klipped_coadd = np.zeros((num_k_klip, nl, stamp*stamp))
    for k in range(N_cubes):
        klipped_coadd = klipped_coadd + klipped[:, k*nl:(k+1)*nl, :]
    klipped_coadd.shape = [num_k_klip, nl, int(stamp), int(stamp)]
    # you can save this as an attribute of dataset...
    dataset.klipped = klipped_coadd

    import matplotlib.pyplot as plt
    plt.figure()
    # pick a wavelength slice slc
    plt.imshow(dataset.klipped[slc], interpolation="nearest")
    plt.show()

2) You can compare the klipped PSF to the forward model::

    spectrum, fm_matrix = es.invert_spect_fmodel(fmarr, dataset, method="JB")
    # fm_matrix has shape (n_k_klip, npix, nwav)
    # spectrum has shape (n_k_klip, nwav)
    # To get the FM for kth element of numbasis:
    fm_image_k = np.dot(fm_matrix[k,:,:], spectrum[k].transpose()).reshape(nl, stamp, stamp)
    fm_image_combined = np.zeros((stamp, stamp))

    plt.figure()
    # compared the same wavelength slice slc
    plt.imshow(fm_image_combined[slc], interpolation="nearest")
    plt.show()

Do the two look the same? If yes -- this is a good sign. If not, something went wrong.



