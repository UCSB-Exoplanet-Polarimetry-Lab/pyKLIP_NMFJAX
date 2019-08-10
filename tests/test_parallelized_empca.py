#!/usr/bin/env python

import os
import sys
import glob
import numpy as np
import astropy.io.fits as fits
import pyklip
import pyklip.instruments
import pyklip.instruments.CHARIS as CHARIS
import pyklip.parallelized as parallelized


#test that **kwargs works as expected for klip_dataset, paralleilized, klip_multifile, and weighted_empca_section (e.g. niter passed through **kwargs to weighted_empca_section is effective)

#test the non void args of _weighted_empca_section are correct

#test _select_algo()
#test _trimmed_mean()
