#!/usr/bin/env python

import os
import glob
import warnings
import numpy as np
import astropy.io.fits as fits
import pyklip
import pyklip.instruments
import pyklip.instruments.CHARIS as CHARIS
import pyklip.parallelized as parallelized
import pyklip.klip as klip
import pytest
import sys
if sys.version_info < (3,3):
    import mock
    import unittest
else:
    import unittest
    import unittest.mock as mock

# this script contains additional tests for parallelized
# mostly for functions and features added since weighted empca is introduced

def test_select_algo():

    assert parallelized._select_algo('empca') == parallelized._weighted_empca_section
    assert parallelized._select_algo('EMpca') == parallelized._weighted_empca_section
    assert parallelized._select_algo('klip') == parallelized._klip_section_multifile

def test_median_collapse():

    test_cube = np.reshape(np.arange(9), (3,3))
    weights = np.reshape(np.zeros(9), (3,3))
    weights[0] = 1.

    ans = parallelized._median_collapse(test_cube)
    assert np.array_equal(ans, np.array([1., 4., 7.]))
    ans = parallelized._median_collapse(test_cube, weights)
    assert np.array_equal(ans, np.array([1., 0., 0.]))
    ans = parallelized._median_collapse(test_cube, weights, axis=0)
    assert np.array_equal(ans, np.array([0., 0., 0.]))
    ans = parallelized._median_collapse(test_cube, weights, axis=1)
    assert np.array_equal(ans, np.array([1., 0., 0.]))

def test_mean_collapse():

    test_cube = np.reshape(np.arange(9), (3, 3))
    weights = np.reshape(np.ones(9), (3, 3))

    ans = parallelized._mean_collapse(test_cube)
    assert np.array_equal(ans, np.array([1., 4., 7.]))

    ans = parallelized._mean_collapse(test_cube, weights, axis=0)
    assert np.array_equal(ans, np.array([3., 4., 5.]))

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        ans = parallelized._mean_collapse(test_cube, weights)
        # Verify no warning's been triggered
        assert len(w) == 0
        assert np.array_equal(ans, np.array([1., 4., 7.]))

    weights[0, 1] = 2.
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        ans = parallelized._mean_collapse(test_cube, weights, axis=0)
        # Verify no warning's been triggered
        assert len(w) == 1
        assert "normalized" in str(w[-1].message)

def test_weighted_mean_collapse():

    test_cube = np.reshape(np.arange(9), (3, 3))
    weights = np.reshape(np.ones(9), (3, 3))

    ans = parallelized._weighted_mean_collapse(test_cube)
    assert np.array_equal(ans, np.array([1., 4., 7.]))

    ans = parallelized._weighted_mean_collapse(test_cube, weights, axis=0)
    assert np.array_equal(ans, np.array([3., 4., 5.]))

    weights = test_cube

    ans = parallelized._weighted_mean_collapse(test_cube, weights, axis=1)
    assert np.array_equal(ans, (np.nanmean(test_cube*weights, axis=1) / np.nanmean(weights, axis=1)))

def test_trimmed_mean_collapse():

    test_cube = np.array([[2., 1., 4., 0., 3.],
                          [5., 9., 7., 6., 8.]])

    ans = parallelized._trimmed_mean_collapse(test_cube, axis = 1)
    assert np.array_equal(ans, np.array([2, 7]))

    test_cube = np.array([[2., 1., 4., 0., 3., 5.],
                          [10., 9., 7., 6., 8., 11.]])
    ans = parallelized._trimmed_mean_collapse(test_cube, axis = 1)
    assert np.array_equal(ans, np.array([2.5, 8.5]))

def test_select_collapse():

    assert parallelized._select_collapse('mediAN') == parallelized._median_collapse
    assert parallelized._select_collapse('meaN') == parallelized._mean_collapse
    assert parallelized._select_collapse('weighted mean') == parallelized._weighted_mean_collapse
    assert parallelized._select_collapse('WeIGhTed_MeAN') == parallelized._weighted_mean_collapse
    assert parallelized._select_collapse('WeIGhTed_MeDIan') == parallelized._mean_collapse
    assert parallelized._select_collapse('trIMmEd_MeAN') == parallelized._trimmed_mean_collapse

@mock.patch.object(CHARIS.CHARISData, 'savedata')
@mock.patch('pyklip.parallelized._mean_collapse')
@mock.patch('pyklip.parallelized.os')
def test_save_spectral_cubes(mock_os, mock_mean_collapse, mock_savedata):

    filelist = glob.glob('./tests/data/CHARISData_test_cube*.fits')
    assert filelist
    nimg = len(filelist)
    numbasis = [1]
    dataset = CHARIS.CHARISData(filelist, None, None, update_hdrs=False)
    inputshape = (nimg, 22, 201, 201)
    dataset.input = dataset.input.reshape(inputshape)
    outputshape = (len(numbasis),) + inputshape
    pixel_weights = np.ones(outputshape)

    # output shape expected error test
    dataset.output = np.array(dataset.input) # shape (N, wv, y, x)
    with pytest.raises(ValueError):
        parallelized._save_spectral_cubes(dataset, pixel_weights, 'mean', numbasis, False,
                                          'anydir', 'anyprefix')

    # savedata test
    dataset.output = np.array([dataset.input]) # shape (1, N, wv, y, x)
    mock_spectral_cubes = ['speccube1']
    mock_mean_collapse.return_value = mock_spectral_cubes
    mock_os.path.join.return_value = 'anyfilepath'
    dataset.klipparams = 'numbasis={numbasis}'
    parallelized._save_spectral_cubes(dataset, pixel_weights, 'mean', numbasis, False,
                                      'anydir', 'anyprefix')
    mock_savedata.assert_called_with('anyfilepath', mock_spectral_cubes[0], klipparams='numbasis=1',
                                     filetype='PSF Subtracted Spectral Cube')

    # flux calibration test
    pass

def test_save_wv_collapsed_images():
    pass

class klip_functions_TestCase(unittest.TestCase):

    '''
    test functions added to klip.py since the introduction of empca
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

class parallelized_empca_TestCase(unittest.TestCase):

    '''
    test empca related features in parallelized
    '''

class weighted_empca_section_TestCase(unittest.TestCase):

    '''
    test _weighted_empca_section
    '''

    @mock.patch('pyklip.parallelized._arraytonumpy')
    def test_section_too_small(self, mock_arraytonumpy):

        numbasis = [1,2]
        b = len(numbasis)
        mock_arraytonumpy.side_effect = [np.ones((22, 189, 201*201)), np.ones((189, 201*201, b))]
        parallelized.original_shape = (189, 201, 201)
        parallelized.aligned = np.zeros((22, 189, 201, 201))
        parallelized.aligned_shape = (22, 189, 201, 201)
        parallelized.output = np.zeros((189, 201, 201, b))
        parallelized.output_shape = (189, 201, 201, b)

        assert not parallelized._weighted_empca_section(scidata_indices_void=None, wv_value_void=None, wv_index=0,
                                                        numbasis=numbasis, maxnumbasis_void=None, radstart=10, radend=10,
                                                        phistart=0, phiend=0.1, movement=None, ref_center=[100, 100],
                                                        minrot_void=None, maxrot_void=None, spectrum_void=None,
                                                        mode_void=None, corr_smooth_void=None, psf_library_good_void=None,
                                                        psf_library_corr_void=None, lite_void=None, dtype=None,
                                                        algo_void='empca', niter=15)

        assert parallelized._weighted_empca_section(scidata_indices_void=None, wv_value_void=None, wv_index=0,
                                                    numbasis=numbasis, maxnumbasis_void=None, radstart=10, radend=12,
                                                    phistart=0, phiend=1, movement=None, ref_center=[100, 100],
                                                    minrot_void=None, maxrot_void=None, spectrum_void=None,
                                                    mode_void=None, corr_smooth_void=None, psf_library_good_void=None,
                                                    psf_library_corr_void=None, lite_void=None, dtype=None,
                                                    algo_void='empca', niter=15)

class empca_TestCase(unittest.TestCase):
    pass