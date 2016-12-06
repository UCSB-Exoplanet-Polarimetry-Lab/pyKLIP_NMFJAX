.. _diskfm_gpi-label:

Disk Foward Modelling Tutorial with GPI
=====================================================


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
For a single run::
    diskobj = DiskFM([n_files, data_xshape, data_yshape], numbasis, dataset, annuli = 2, subsections = 1)
If you would like to forward model multiple models on a dataset::
    diskobj = DiskFM([n_files, data_xshape, data_yshape], numbasis, dataset, annuli = 2, subsections = 1, save_basis = True, load_from_basis = False)
In both cases you then run::
    fmout = fm.klip_dataset(dataset, diskobj, numbasis = nummbasis, annuli = 2, subsections = 1)
In order to forward model another disk::
    diskobj.update_disk(newmodel)
    fmout = diskobj.fm_parallelized()
