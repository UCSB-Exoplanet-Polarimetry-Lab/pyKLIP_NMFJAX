.. _diskfm_gpi-label:

Disk Foward Modelling Tutorial with GPI
=====================================================
Disk forward modelling is intended for use in cases where you would
like to model a variety of different model disks on the same dataset. This
can be used with an MCMC that is fitting for model parameters. It
works by saving the KLIP basis vectors in a file so they do not have
to be recomputed every time. 

Running
--------------------------
How to use::

    import glob
    import pyklip.parallelized.GPI as GPI
    from pyklip.fmlib.diskfm import DiskFM
    import pyklip.fm as FM
    
    filelist = glob.glob("path/to/dataset/*.fits")
    dataset = GPI.GPIData(filelist)
    model = [some 2D image array]


You will then need to create a disk object::

    numbasis = n.array([1, 2, ... ])
    diskobj = DiskFM(n.array([n_files, data_xshape, data_yshape]), numbasis, dataset, model_disk, annuli = 2, subsections = 1)

To run the forward modelling, run::

    fmout = fm.klip_dataset(dataset, diskobj, numbasis = numbasis, annuli = 2, subsections = 1, mode = 'ADI')

Note that in the case that annuli = 1, you will need to set padding = 0.001 in klip_dataset



DIskFM for MCMC
--------------------------

For an MCMC you can create the basis vectors and then save them so that they do not need to be recomputed every time. If you would like to forward model multiple models on a dataset, then you will need to save the eigenvalues and eigenvectors using the last 4 keywords::

    diskobj = DiskFM([n_files, data_xshape, data_yshape], numbasis, dataset, model_disk, annuli = 2, subsections = 1, basis_filename = 'klip-basis.p', save_basis = True, load_from_basis = False)


Then, in any session you can create a disk object with the loaded basis vectors. Note that even if you have created a disk object with the above command in the same session that you will need to create a new disk object.::
  
    diskobj = DiskFM([n_files, data_xshape, data_yshape], numbasis, dataset, model_disk, annuli = 2, subsections = 1, basis_filename = 'klip-basis.p', load_from_basis = True, save_basis = False)

Then, you can forward model disks without needing to create a new diskobj. The forward modelled disk will be output to fmout::

    diskobj.update_disk(newmodel)
    fmout = diskobj.fm_parallelized()
    diskobj.update_disk(othermodel)
    otherfmout = diskobj.fm_parallelized()



Current Works in Progress
------------------------------------
* Does not support SDI mode
* Is not parallized 
