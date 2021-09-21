.. _release-notes:

Release Notes
==============

Version 2.4.1
 * Use pyKLIP version number rather than git commit to track versioning in headers (Jason Wang)

Version 2.4
 * Forward modeling can handle time dependent PSFs now (Jason Wang)
 * Added STIS.py interface and demo notebook (Robert Thompson)
 * Removed an extra 2x scaling in `klip.nan_gaussian_filter()` (Jason Wang)
 * Fixed RDI bug where the reference library only has 1 image (Aarynn Carter)
 * Fixed bug in background subtraction in `GPIData.generate_psf_cube()` (JB Ruffio)

Version 2.3
 * GPI interface improvements: coronagrpahic throughput, updated astrometric calibration, edge cases (Jason Wang, Rob De Rosa)
 * GPI interface: Removed wind butterfly PCA subtraction has it was not effective (JB Ruffio)
 * For PFS library, fixed diagonal elements of correlation matrix (JB Ruffio)
 * Improvements to DiskFM implementation and python > 3.7 compatability (Johan Mazoyer)
 * Fixed bug where pyKLIP crashes if you only have one science frame (Aarynn Carter)
 * Added warning for debug mode, and supressing print statements if not in verbose mode (Jea Adams)
 * Reorganized navigation bar for docs (Jason Wang)

Version 2.2
 * Field dependent throughput to account for changes in the off-axis PSF due to e.g., coronagraphic throughput (Jea Adams)
 * Added `verbose` flag that can be used to turn off print statements within pyklip (Jea Adams)
 * Various bug fixes (Jason Wang, Johan Mazoyer)
 * Added for explanatory material to docs so that they are more accessible (Jea Adams)

Version 2.1
 * RDI support in forward modeling framework (currently works for DiskFM, support for other FM modules coming) (Johan Mazoyer)
 * GenericData is more feature rich (better saving, automatic wcs generation) (Jason Wang)
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
