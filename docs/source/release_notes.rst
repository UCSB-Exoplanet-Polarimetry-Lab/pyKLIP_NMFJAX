.. _release-notes:

Release Notes
==============

Version 2.1
 * RDI support in forward modeling framework (currently works for DiskFM, support for other FM modules coming)
 * GenericData is more feature rich (better saving, automatic wcs generation)
 * Minor bug fixes and documentation updates

Version 2.0.1
 * Update Python 3 version to Python 3.6

Version 2.0
 * Forgot to update for a long while. Lots of new changes. A few key summaries below.
 * Forward modeling for planet detection, astrometry, photometry, spectral extraction, and disk forward modeling
 * Support for Keck/NIRC2, Keck/OSIRIS, Subaru/CHARIS, VLT/SPHERE, MagAO/VisAO, and a generic instrument interface for all else
 * Alternative algorithms to KLIP: emperically weighted PCA, non-negative matrix factorization
 * RDI library support
 * Automated tests to ensure correctness of main features
 * Now released on PyPI/pip

Version 1.1
 * Updated installation to be much easier
 * Reorganized repo structure to match standard python repos
 * Improvements to automatic planet detection code

Version 1.0
 * Initial Release
 * Fully-functional KLIP implementation for ADI and SDI
 * Interface for GPI data in both spectral and polarimetry mode
 * Utility functions like fake injection and contrast calculation
