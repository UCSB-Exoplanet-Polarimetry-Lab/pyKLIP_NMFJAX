.. _diskfm_gpi-label:

Disk Foward Modelling Tutorial with GPI (DiskFM)
=====================================================
This tutorial presents how to use forward modelling routines specific to disk modelling
and disk parameter retrieval.

Why DiskFM?
--------------------------

As noted in `Pueyo (2016) <http://arxiv.org/abs/1604.06097>`_, "in practice Forward
Modeling with disks is complicated by the
fact that [it] cannot be simplified using a simple PSF as the astrophysical model:
every hypothetical disk morphology must be explored." Indeed, because of their complex
geometries, the forward modelling of disks have to be repeated a lot of time on disks
with slightly different parameters. All these geometries are compared
to the klipped reduced image of disk, within an MCMC or a Chi-square wrapper.

However, once measure for a set of reduction parameters the klip basis do not change.
one can save the KLIP forward model basis vectors in a file once so they do not have to be
recomputed every time. For a new disk model, the forward modelling is therfore only a
array reformating and a matrix multiplication, which can be optimized to be only a few
seconds. This code provide you with these routines.

This code currently only support KLIP ADI and KLIP SDI (or ADI + SDI) reduction (but not RDI).

DiskFM Requirements
--------------------------

diskFM on a single model can be done on a personnal computer. The full parameter space
exploration with the Chi-square or MCMC wrapper (not described in this tutorial) can be
computationally very intensive, taking easily a few days, even parallized on a large
server.

You also need the following pieces of data to forward model the data.

- A set of to run PSF subtraction on
- A model of disk, which we will assume is already convolved by the PSF of your instrument


Set up
--------------------------
First import an instrument data set:
.. code-block:: python
    import glob
    import numpy as np
    import pyklip.instruments.GPI as GPI

    # read in the data into a dataset
    filelist = glob.glob("path/to/dataset/*.fits")
    dataset = GPI.GPIData(filelist)

We also need a 2D model of disk convolved by the PSF of the instrument :
.. code-block:: python
    from astropy.convolution import convolve

    disk_model_convolved = convolve(disk_model,instrument_psf, boundary = 'wrap')

Simple disk Forward Modelling
--------------------------
This code shows how to initialize the DiskFM object and to do a forward modelling:

.. code-block:: python
    import pyklip.fm as fm
    numbasis = [1, 2, 3] # different KL number we applied to the disk.
    aligned_center=[140, 140] # indicate the position of the star
    diskobj = DiskFM(dataset.input.shape, numbasis, dataset, disk_model_convolved,
                    aligned_center=aligned_center)


In practice the "best KLIP parameters" have already be determined in advanced and we
recommand for speed to use a single KL number, e.g.:
.. code-block:: python
    numbasis = [3]


To run the forward modelling, just run:
.. code-block:: python
    fmout = FM.klip_dataset(dataset, diskobj,
                            outputdir="path/", fileprefix="my_favorite_disk"
                            numbasis=numbasis, maxnumbasis=100,
                            mode='ADI', annuli=2, subsections=1, minrot=3
                            aligned_center=aligned_center
                            )
fmout will contain the forward model, and the code will save two fits files in outputdir
containing the klipped data and the associated forward model of your disk.

Most of the parameters implemented for psf forward model KLIP correction with
pyklip can be set with the following exceptions:
- no spectrum specific keyword (spectrum, flux_overlap, calibrate_flux)
- specific correction mode filtering the data (corr_smooth, highpass)
- other specific correction parameters (N_pix_sector, padding, annuli_spacing)

Mode parameter can be set only to 'ADI', 'SDI' and 'ADI+SDI'.

aligned_center is the position were the klip reduction will center the reduced image.
The code will raise an error if it is not set to the position to which you set the star
of your model (see previous section).


DIskFM for MCMC or Chi-Square
--------------------------

For an MCMC or Chi-Square you can create the basis vectors and then save them so that
they do not need to be recomputed every time. If you would like to forward model
multiple models on a dataset, then you will need to signal it during the initialization
of the DiskFM object, then apply FM.klip_dataset to measure and save the forward model
basis and parameters:
.. code-block:: python
    diskobj = DiskFM(dataset.input.shape, numbasis, dataset,
                    disk_model_convolved, aligned_center=aligned_center,
                    basis_filename = 'klip-basis.h5', save_basis = True)

    fmout = FM.klip_dataset(dataset, diskobj,
                            outputdir="path/", fileprefix="my_favorite_disk"
                            numbasis=numbasis, maxnumbasis=100,
                            mode='ADI', annuli=2, subsections=1, minrot=3
                            aligned_center=aligned_center
                            )


Then, in any python session you can create a disk object with the loaded basis vectors,
and you can forward model disks without needing to create a new DiskFM object.
The forward modelled disk will be output to fmout:
.. code-block:: python
    diskobj = DiskFM(dataset.input.shape, numbasis, dataset,
                    disk_model_convolved, aligned_center=aligned_center,
                    basis_filename='klip-basis.h5', load_from_basis=True)

    # do the forward modelling on a new model
    new_disk_model_convolved=convolve(new_disk_model,instrument_psf, boundary='wrap')
    diskobj.update_disk(new_disk_model_convolved)
    fmout=diskobj.fm_parallelized()

    # do the forward modelling on a third model
    third_disk_model_convolved=convolve(third_disk_model,instrument_psf, boundary='wrap')
    diskobj.update_disk(third_disk_model_convolved)
    fmout=diskobj.fm_parallelized()

These last 3 lines are specifically the thing that should be be repeated withinin the MCMC
or Chi-Square wapper.


Note that even if you have already created a DiskFM object to save the FM
(runned diskFM with save_basis = True) in this python session, you still need to re-create
the DiskFM object and load it (runned diskFM with load_from_basis = True).

The time is a key element here if you want to produce hundreds of thousands of forward
modelling models. A smart choice of pyklip parameters can reduce the time for a single
model to forward model:
- use OWA to limit only in the zone where the disk is.
- limit the number of sections (small annuli and subsections number).
- reduce the number of wavelenghts. We recall this very usefull pyklip function to rebin
quickly the number of wavelength:
.. code-block:: python
    dataset.spectral_collapse(collapse_channels=1, align_frames=True)
which should be applied immediatly after loading the dataset.

Specific issues for multiwavelength disk forward modelling
--------------------------

If you put a multi-wavelenght dataset (e.g. IFS), the code will produce a multi-wavelenght forward
modelling

In that case, you can use a simple 2D model for the disk and the code will duplicate this model
and apply the forward modelling separately on each of those at every wavelengths. Or you can use a 3D model
(n_wl, x, y) and the code will apply the forward modelling separately on each of those at every wavelengths.

Alhtough everything we said earlier on basis saving still apply multiwavelength disk
forward modelling is long (it can take up to a few minutes or hours
for a single forward modelling depending on the number of wavelenghts) and we do not
recommand to use this in an MCMC wrapper.






