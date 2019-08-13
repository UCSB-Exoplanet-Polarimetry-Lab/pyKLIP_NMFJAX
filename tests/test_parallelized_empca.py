#!/usr/bin/env python

import os
import sys
import glob
import warnings
import numpy as np
import astropy.io.fits as fits
import pyklip
import pyklip.instruments
import pyklip.instruments.CHARIS as CHARIS
import pyklip.parallelized as parallelized

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


def test_save_spectral_cubes():

    pass


def test_save_wv_collapsed_images():

    pass


def test_weighted_empca_section():

    pass


def test_klip_dataset_empca_arguments():

    pass


def test_klip_parallelized_emcpa_arguments():

    pass


#test that **kwargs works as expected for klip_dataset, paralleilized, klip_multifile, and weighted_empca_section (e.g. niter passed through **kwargs to weighted_empca_section is effective)

#test the non void args of _weighted_empca_section are correct
