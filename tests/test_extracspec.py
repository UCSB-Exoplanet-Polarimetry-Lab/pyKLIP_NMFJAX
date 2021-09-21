import glob
import os
import time
import pytest
import numpy as np

import pyklip.instruments.GPI as GPI
import pyklip.fmlib.extractSpec as es
import pyklip.fm as fm

testdir = os.path.dirname(os.path.abspath(__file__)) + os.path.sep

@pytest.mark.skip(reason="No assertion tests and takes a while to run")
def test_spectral_extract():
    """
    Tests FM astrometry using MCMC + GP Regression

    """
    # time it
    t1 = time.time()

    # # open up already generated FM and data_stamp
    # fm_hdu = fits.open("/home/jwang/GPI/betapic/fm_models/final_altpsf/pyklip-131118-h-k100m4-dIWA8-nohp-klipfm-KL7cube.fits")
    # data_hdu = fits.open("/home/jwang/GPI/betapic/klipped/final_altpsf/pyklip-131118-h-k100m4-dIWA8-nohp-onezone-KL7cube.fits")

    ########### generate FM ############
    # grab the files
    filelist = glob.glob(testdir + os.path.join("data", "S20131210*distorcorr.fits"))
    filelist.sort()

    # hopefully there is still 3 filelists
    assert(len(filelist) == 3)

    # only read in one spectral channel
    # skipslices = [i for i in range(37) if i != 7 and i != 33]\
    skipslices = [0,1,2,34,35,36]
    # read in data
    dataset = GPI.GPIData(filelist, highpass=9, skipslices=skipslices)

    unique_wvs = np.unique(dataset.wvs)
    unique_pas = dataset.PAs.reshape(dataset.PAs.shape[0]//dataset.numwvs, dataset.numwvs)[:,0]

    # generate PSF
    dataset.generate_psfs(boxrad=25//2, time_collapse=False)



    # setup FM guesses
    numbasis = np.array([1, 7, 100])
    guesssep = 0.4267 / GPI.GPIData.lenslet_scale
    guesspa = 212.15
    stamp_size=10
    print(guesssep, guesspa)
    ###### The forward model class ######
    fm_class = es.ExtractSpec(dataset.input.shape,
                       numbasis,
                       guesssep,
                       guesspa,
                       dataset.psfs,
                       unique_wvs,
                       stamp_size = stamp_size,
                       input_psfs_pas=unique_pas)
    # run KLIP-FM
    prefix = "betpic-131210-j-fmspec"
    fm.klip_dataset(dataset, fm_class, outputdir=testdir, fileprefix=prefix, numbasis=numbasis,
                    annuli=[[guesssep-stamp_size,guesssep+stamp_size]], 
                    subsections=[[(guesspa-45)/180.*np.pi,(guesspa+45)/180.*np.pi]], 
                    padding=0, movement=1, time_collapse="mean")


    klipped = dataset.fmout[:,:,-1,:]

    #e.g., for GPI this could be the star-to-spot ratio
    # otherwise, the defaults are:
    units = "natural" # (default) returned relative to input PSF model
    scaling_factor=1.0 # (default) not used if units not set to "scaled"


    exspect, fm_matrix = es.invert_spect_fmodel(dataset.fmout, dataset, units=units,
                                                scaling_factor=scaling_factor,
                                                method="leastsq")

    import matplotlib.pylab as plt
    plt.plot(np.unique(dataset.wvs), exspect[1])
    plt.show()

if __name__ == "__main__":
    test_spectral_extract()
                                                
