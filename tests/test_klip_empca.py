#!/usr/bin/env python

import os
import glob
import warnings
import numpy as np
import astropy.io.fits as fits
import pyklip
import pyklip.instruments.CHARIS as CHARIS
import pyklip.klip as klip
import pyklip.empca.empca as empca
import pytest
import sys
if sys.version_info < (3,3):
    import mock
    import unittest
else:
    import unittest
    import unittest.mock as mock

# this script contains tests for some functions in pyklip.klip and pyklip.empca.empca

class klip_functions_TestCase(unittest.TestCase):

    '''
    tests for functions added to klip.py since the introduction of empca
    '''

    def test_make_polar_coordinates(self):

        x, y = np.meshgrid(np.arange(10), np.arange(10))
        x.shape = (x.shape[0]*x.shape[1])
        y.shape = (y.shape[0]*y.shape[1])

        # test for center at [0,0]
        center = [0, 0]
        r, phi = klip.make_polar_coordinates(x, y, center)
        ind = np.where((x==0) & (y==0))
        assert r[ind] == 0
        ind = np.where(y==0)
        testarray = np.zeros(10)
        testarray.fill(-np.pi)
        assert np.array_equal(phi[ind], testarray)

        # test for center at [5,5]
        center = [5, 5]
        r, phi = klip.make_polar_coordinates(x, y, center)
        ind = np.where((x==0) & (y==0))
        assert r[ind] == np.sqrt(50)
        ind = np.where((y>5) & (x==5))
        testarray = np.zeros(4)
        testarray.fill(-np.pi/2)
        assert np.array_equal(phi[ind], testarray)

class empca_functions_TestCase(unittest.TestCase):

    '''
    tests for functions added to the original empca (mostly helper functions)
    '''

    def test_set_pixel_weights(self):
        pass