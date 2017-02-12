.. _fmmf-label:


Klip POst Processing (KPOP)
=====================================================
TODO
The kpop objects. The architecture. How does it work.


Matched Filter and Point Source Detection
=====================================================



This tutorial will provide the necessary steps to run the matched filters as described in Ruffio et al. (2017, in prep.) and detect point sources for direct imaging.

The code discussed in the page part of pyKLIP in a subfolder called 'kpp', for KLIP Post Processing, and can be installed following :ref:`install-label`.
It has only been tested with python2.7 but if you need python 3.5, simply ask and we will make it happen quickly.

Why using a matched filter?
--------------------------

In signal processing, a matched filter is the linear filter maximizing the Signal to Noise Ratio (SNR) of a known signal in the presence of additive noise.

Matched filters are used in Direct imaging to detect point sources using the expected shape of the planet Point Spread Function (PSF) as a template.

The distortion makes the definition of the template somewhat challenging,the planet PSF doesn't look like the instrumental PSF, but reasonable results can be obtained by using approximations.

Forward Model `Pueyo (2016) <http://arxiv.org/abs/1604.06097>`_

Requirements
~~~~~~~~~~~~~~~~~~~~~~~~
Same requirement as pyKLIP.

Input Data for Classical Cross Correlation and Classical Matched Filter:
* Speckle subtracted image or cube
Input Data for Forward Model Matched Filter:
* Data to run PSF subtraction on

Classical Cross Correlation
--------------------------

Classical Matched Filter
--------------------------



a
~~~~~~~~~~~~~~~~~~~~~~~~

b
^^^^^^^^^^^^^^^^^^^^^^^^

c
--------------------------


Forward Model Matched Filter (FMMF) Tutorial with GPI
--------------------------
The Forward Model Matched Filter (FMMF) is an algorithm aimed at improved exo-planet detection for direct imaging.
The current implementation only works for GPI and this tutorial will explain how to use it.
A reference paper J.-B. Ruffio et al. 2016/2017 with the description of the method is currently in preparation.

.. note::
    If you ask me enough, I will try to make the code more general to work for different instruments.


Input Data
~~~~~~~~~~~~~~~~~~~~~~~~
What are the input data needed to run the FMMF pyklip implementation.

Running FMMF
~~~~~~~~~~~~~~~~~~~~~~~~
How to run the code.

Contrast Curves Tutorial with GPI
--------------------------


Utilities
=====================================================

Spectra
--------------------------

Simulated Planet injection
--------------------------