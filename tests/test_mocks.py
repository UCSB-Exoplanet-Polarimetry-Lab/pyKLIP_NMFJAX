import os
import glob
from time import time
import numpy as np
import astropy.io.fits as fits

import pyklip
import pyklip.instruments
import pyklip.parallelized as parallelized
import pyklip.instruments.GPI as GPI
import pyklip.fakes as fakes

import unittest.mock as mock
from unittest.mock import patch
import numpy as np

testdir = os.path.dirname(os.path.abspath(__file__)) + os.path.sep

@patch('pyklip.parallelized.klip_parallelized')
def test_mock_SDI(mock_klip_parallelized):
    #create a mocked return value for klip_parallelized that returns a4d array of size (b,N,y,x)
    mock_klip_parallelized.return_value = np.zeros((4,111,281,281))

    # time it
    t1 = time()

    # grab the files
    filelist = glob.glob(testdir + os.path.join("data", "S20131210*distorcorr.fits"))
    assert(len(filelist) == 3)

    # create the dataset object
    dataset = GPI.GPIData(filelist)

    # run klip parallelized in SDI mode
    outputdir = testdir
    prefix = "mock"
    parallelized.klip_dataset(dataset, outputdir=outputdir, fileprefix=prefix,
                          annuli=9, subsections=4, movement=1, numbasis=[1,20,50,100],
                          calibrate_flux=True, mode="SDI")
    
    mocked_glob = glob.glob(testdir + 'mock*')
    assert(len(mocked_glob)==5)

    print("{0} seconds to run".format(time()-t1))



if __name__ == "__main__":
    test_mock_SDI()