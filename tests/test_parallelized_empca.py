#!/usr/bin/env python

import os
import glob
import warnings
import copy
import numpy as np
import astropy.io.fits as fits
import pyklip
import pyklip.instruments
import pyklip.instruments.CHARIS as CHARIS
import pyklip.parallelized
import pyklip.klip as klip
import pytest
import sys
if sys.version_info < (3,3):
    import mock
    import unittest
else:
    import unittest
    import unittest.mock as mock

# this script contains tests for the empca feature in parallelized, and for some refactored helper functions

class weighted_empca_section_TestCase(unittest.TestCase):

    '''
    test _weighted_empca_section
    '''

    @mock.patch('pyklip.empca.weighted_empca')
    @mock.patch('pyklip.parallelized._arraytonumpy')
    def test_weighted_empca_called_with_expected_ref_psfs(self, mock_arraytonumpy, mock_weighted_empca):
        '''
        Test empca.weighted_empca is called with ref_psfs that is truncated to have
        only pixel indices that have enough non-nan/nonzero values across cubes
        '''

        numbasis = [2]
        maxnumbasis = 2
        wv_index = 0
        ref_center = [1, 1]
        aligned_slice = np.arange(9)*1. # flattened 3*3 image slices
        aligned_cubes = np.tile(aligned_slice, (2, 5, 1)) # shape (nwv, N, y*x)
        aligned_cubes[0, 0:3, 1] = 0. # index 1 in slice should fail good_ind selection
        aligned_cubes[0, 0, 2] = 0. # index 2 in slice should pass good_ind selection
        # for wavelengh index 0, slice indices 0 and 1 should fail good_ind selection
        expected_ref_psfs = copy.copy(aligned_cubes[0, :, 2:])
        aligned_cubes[aligned_cubes == 0.] = np.nan
        mock_arraytonumpy.side_effect = [aligned_cubes, np.ones((5, 3*3, 1))]
        mock_weighted_empca.return_value = expected_ref_psfs
        pyklip.parallelized.original_shape = (5, 3, 3)
        pyklip.parallelized.aligned_shape = (2, 5, 3, 3)
        pyklip.parallelized.aligned = 'aligned data'
        pyklip.parallelized.output_shape = (5, 3, 3, len(numbasis))
        pyklip.parallelized.output = 'output data'

        return_value = pyklip.parallelized._klip_section_multifile('scidata_indices', 'wavelength', wv_index, numbasis, maxnumbasis,
                                                    radstart=0., radend=100., phistart=-np.pi*2., phiend=np.pi*2, minmove=0,
                                                    ref_center=ref_center, minrot=0, maxrot=None, spectrum=None,
                                                    mode='ADI', corr_smooth=1, psflib_good=None, psflib_corr=None,
                                                    lite=False, dtype=None, algo='empca')

        np.testing.assert_array_equal(mock_weighted_empca.call_args[0][0], expected_ref_psfs)
        assert mock_weighted_empca.call_args[1]['niter'] == 15
        assert mock_weighted_empca.call_args[1]['nvec'] == 2
        assert return_value

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