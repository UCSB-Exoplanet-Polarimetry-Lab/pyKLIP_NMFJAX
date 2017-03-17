.. _kpop-label:


Klip POst Processing (KPOP)
=====================================================
Klip POst Processing (KPOP) is a module with tools to calculate:
* matched filter maps,
* SNR maps,
* detection,
* ROC curves,
* contrast curves.
We will go over these application in this tutorial.
KPOP has been implemented with surveys application in mind and therefore make the reduction of a dataset with a structured file organization easier.

.. note::
    The ipython notebook ``pyklip.examples.kpop_tutorial.ipynb`` go through most of the applications with a GPI example based on the beta Pictoris test files in tests/data.

.. note::
    KPOP is the framework developped for Ruffio et al. (2017, in prep.).

PyKLIP can be installed following :ref:`install-label`.
It has only been tested with python 2.7 even though it should work for python 3.

Architecture
--------------------------
Each task (i.e. calculate matched filter, calculate SNR, ...) is represented with an object.
All KPOP inherit from the same object :py:class:`pyklip.kpp.utils.kppSuperClass`, which normalizes the function calls.
The parameter of the task are defined when instantiating the object.
The :meth:`pyklip.kpp.utils.kppSuperClass.initialize` method will then read the files and update the object's attributes.
Then, :meth:`pyklip.kpp.utils.kppSuperClass.calculate()` will process the file(s) and return the final product.
To finish, :meth:`pyklip.kpp.utils.kppSuperClass.save()` will save the final product following the class convention.
After initialize has been ran, it possible to check if the file has already been reduced by calling :meth:`pyklip.kpp.utils.kppSuperClass.check_existence()`.
The method :meth:`pyklip.kpp.utils.kppSuperClass.init_new_spectrum()` allows to change the reduction spectrum if needed.

In order to simplify the reduction of survey data, the filenames are defined with wild characters.
During the initialization, the object will read the file matching the filename pattern.
When several files match the filename pattern, it is possible to simply call initialize() in sequence and the object will automatically read the matchign files one by one.

The function :meth:`pyklip.kpp.kpop_wrapper.kpop_wrapper()` will take a list of objects (ie tasks) and a list of spectra as an input and run all the
task as many time as necessary to reduce all the matching files with all the spectra.

Matched Filter and Planet Detection
--------------------------
In signal processing, a matched filter is the linear filter maximizing the Signal to Noise Ratio (SNR) of a known signal in the presence of additive noise.

Matched filters are used in Direct imaging to detect point sources using the expected shape of the planet Point Spread Function (PSF) as a template.

The distortion makes the definition of the template somewhat challenging,the planet PSF doesn't look like the instrumental PSF, but reasonable results can be obtained by using approximations.

Forward Model `Pueyo (2016) <http://arxiv.org/abs/1604.06097>`_

ROC Curves
--------------------------
ROC curves can be built following the GPI script ``pyklip.examples.roc_script.py`` and adapting to it any instrument or data reduction.
This might include changing the PSF cube calculation, or the platescale and other details.
This script calculate the ROC curve for a single dataset but using different matched filters.
By running this script on several datasets and by combining the final product one can build a ROC curve for an entire survey.
One should consider modify the script for a different instrument.


Contrast Curves and Completeness
--------------------------
Contrast curves can be built following the GPI script ``pyklip.examples.contrast_script.py`` and adapting to it any instrument or data reduction.


