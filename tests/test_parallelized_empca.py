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

class test_collapse_data(unittest.TestCase):

    def test_median_collapse(self):

        test_cube = np.reshape(np.arange(9), (3,3))
        weights = 2.
        ans = parallelized._collapse_data(test_cube, collapse_method='median')
        assert np.array_equal(ans, np.array([1., 4., 7.]))
        ans = parallelized._collapse_data(test_cube, weights, axis=0, collapse_method='median')
        assert np.array_equal(ans, np.array([3., 4., 5.]))

    def test_mean_collapse(self):

        test_cube = np.reshape(np.arange(9), (3, 3))
        weights = 2.
        ans = parallelized._collapse_data(test_cube, collapse_method='mean')
        assert np.array_equal(ans, np.array([1., 4., 7.]))
        ans = parallelized._collapse_data(test_cube, weights, axis=0, collapse_method='mean')
        assert np.array_equal(ans, np.array([3., 4., 5.]))

    def test_weighted_mean_collapse(self):

        test_cube = np.reshape(np.arange(9), (3, 3))
        weights = np.reshape(np.ones(9), (3, 3))
        ans = parallelized._collapse_data(test_cube, collapse_method='weighted_mean')
        assert np.array_equal(ans, np.array([1., 4., 7.]))
        ans = parallelized._collapse_data(test_cube, weights, axis=0, collapse_method='weighted-mean')
        assert np.array_equal(ans, np.array([3., 4., 5.]))
        weights = test_cube
        ans = parallelized._collapse_data(test_cube, weights, axis=1, collapse_method='weighted mean')
        assert np.array_equal(ans, (np.nanmean(test_cube*weights, axis=1) / np.nanmean(weights, axis=1)))

    def test_trimmed_mean_collapse(self):

        test_cube = np.array([[2., 1., 4., 0., 3.],
                              [5., 9., 7., 6., 8.]])
        ans = parallelized._collapse_data(test_cube, axis=1, collapse_method='Trimmed-mean')
        assert np.array_equal(ans, np.array([2, 7]))
        test_cube = np.array([[2., 1., 4., 0., 3., 5.],
                              [10., 9., 7., 6., 8., 11.]])
        ans = parallelized._collapse_data(test_cube, axis=1, collapse_method='trimmed_mean')
        assert np.array_equal(ans, np.array([2.5, 8.5]))

# @mock.patch.object(CHARIS.CHARISData, 'savedata')
# @mock.patch('pyklip.parallelized.os')
# def test_save_spectral_cubes(mock_os, mock_dataset_init, mock_savedata):
#
#     nimg = 10
#     numbasis = [1, 2]
#     mock_dataset_init.return_value =
#     dataset = CHARIS.CHARISData()
#     inputshape = (nimg, 22, 201, 201)
#     dataset.input = dataset.input.reshape(inputshape)
#     outputshape = (len(numbasis),) + inputshape
#     pixel_weights = np.ones(outputshape)
#
#     # output shape expected error test
#     dataset.output = np.array(dataset.input) # shape (N, wv, y, x)
#     with pytest.raises(ValueError):
#         parallelized._save_spectral_cubes(dataset, pixel_weights, 'mean', numbasis, False,
#                                           'anydir', 'anyprefix')
#
#     # savedata test
#     dataset.output = np.array([dataset.input]) # shape (1, N, wv, y, x)
#     mock_spectral_cubes = ['speccube1']
#     mock_mean_collapse.return_value = mock_spectral_cubes
#     mock_os.path.join.return_value = 'anyfilepath'
#     dataset.klipparams = 'numbasis={numbasis}'
#     parallelized._save_spectral_cubes(dataset, pixel_weights, 'mean', numbasis, False,
#                                       'anydir', 'anyprefix')
#     mock_savedata.assert_called_with('anyfilepath', mock_spectral_cubes[0], klipparams='numbasis=1',
#                                      filetype='PSF Subtracted Spectral Cube')
#
#     # flux calibration test
#     pass
#
# @mock.patch('pyklip.parallelized._mean_collapse')
# @mock.patch('pyklip.parallelized._collapse_method')
# @mock.patch.object(CHARIS.CHARISData, 'savedata')
# def test_save_wv_collapsed_images(mock_savedata, mock_collapse_method, mock_mean_collapse):
#
#     filelist = glob.glob('./tests/data/CHARISData_test_cube*.fits')
#     assert filelist
#     nimg = len(filelist)
#     numbasis = [1]
#     dataset = CHARIS.CHARISData(filelist, None, None, update_hdrs=False)
#     inputshape = (nimg, 22, 201, 201)
#     dataset.input = dataset.input.reshape(inputshape)
#     outputshape = (len(numbasis),) + inputshape
#     pixel_weights = np.ones(outputshape)
#
#     # output shape expected error test
#     dataset.output = np.array(dataset.input)  # shape (N, wv, y, x)
#     with pytest.raises(ValueError):
#         parallelized._save_wv_collapsed_images(dataset, pixel_weights, 'median', numbasis,
#                                                'mean', None, None, False,
#                                                'anydir', 'anyprefix')
#
#     # test spectrum is None case
#     dataset.output = np.array([dataset.input])  # shape (1, N, wv, y, x)
#     mock_collapse_method.return_value = 'any spectral cubes'
#     mock_mean_collapse.return_value = ['KLmode_cube']
#     dataset.klipparams = 'numbasis={numbasis}'
#     parallelized._save_wv_collapsed_images(dataset, pixel_weights, 'median', numbasis, 'mean',
#                               None, None, False, 'anydir', 'anyprefix')
#     mock_mean_collapse.assert_called_with('any spectral cubes', axis=1)
#     mock_savedata.assert_called_with('anydir/anyprefix-KL1modes-all.fits', ['KLmode_cube'], klipparams='numbasis=[1]',
#                                      filetype='KL Mode Cube', zaxis=numbasis)
#
#     # test spectrum is not None case
#
#     # test flux calibration

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
                                                        psf_library_corr_void=None, lite=False, dtype=None,
                                                        algo_void='empca', niter=15)

        assert parallelized._weighted_empca_section(scidata_indices_void=None, wv_value_void=None, wv_index=0,
                                                    numbasis=numbasis, maxnumbasis_void=None, radstart=10, radend=12,
                                                    phistart=0, phiend=1, movement=None, ref_center=[100, 100],
                                                    minrot_void=None, maxrot_void=None, spectrum_void=None,
                                                    mode_void=None, corr_smooth_void=None, psf_library_good_void=None,
                                                    psf_library_corr_void=None, lite=False, dtype=None,
                                                    algo_void='empca', niter=15)

    @mock.patch('pyklip.parallelized._arraytonumpy')
    @mock.patch('pyklip.empca.empca.set_pixel_weights')
    @mock.patch('pyklip.empca.empca.weighted_empca')
    def test_functions_called(self, mock_weighted_empca, mock_set_pixel_weights, mock_arraytonumpy):

        numbasis = [1, 2]
        b = len(numbasis)
        wv_index = 0
        ref_center = [100, 100]
        mock_aligned = np.ones((22, 189, 201 * 201))
        mock_arraytonumpy.side_effect = [mock_aligned, np.ones((189, 201 * 201, b))]
        mock_set_pixel_weights.return_value = 'some weights'
        mock_weighted_empca.return_value = mock_aligned[wv_index]
        parallelized.original_shape = (189, 201, 201)
        parallelized.aligned = np.zeros((22, 189, 201, 201))
        parallelized.aligned_shape = (22, 189, 201, 201)
        parallelized.output = np.zeros((189, 201, 201, b))
        parallelized.output_shape = (189, 201, 201, b)
        x, y = np.meshgrid(np.arange(parallelized.original_shape[2]), np.arange(parallelized.original_shape[1]))
        x.shape = (x.shape[0] * x.shape[1])
        y.shape = (y.shape[0] * y.shape[1])
        r, phi = klip.make_polar_coordinates(x, y, ref_center)
        rflat = np.reshape(r[:], -1)

        parallelized._weighted_empca_section(scidata_indices_void=None, wv_value_void=None, wv_index=wv_index,
                                             numbasis=numbasis, maxnumbasis_void=None, radstart=0, radend=1000,
                                             phistart=-2*np.pi, phiend=2*np.pi, movement=None, ref_center=ref_center,
                                             minrot_void=None, maxrot_void=None, spectrum_void=None,
                                             mode_void=None, corr_smooth_void=None, psf_library_good_void=None,
                                             psf_library_corr_void=None, lite=False, dtype=None,
                                             algo_void='empca', niter=15)

        np.testing.assert_array_equal(mock_aligned[wv_index], mock_set_pixel_weights.call_args[0][0])
        np.testing.assert_array_equal(rflat, mock_set_pixel_weights.call_args[0][1])
        np.testing.assert_array_equal(mock_weighted_empca.call_args[0][0], mock_aligned[wv_index])
        assert mock_weighted_empca.call_args[1]['weights'] == 'some weights'
        assert mock_weighted_empca.call_args[1]['niter'] == 15
        assert mock_weighted_empca.call_args[1]['nvec'] == 2