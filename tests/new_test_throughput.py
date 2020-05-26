import os
import astropy.io.fits as fits
import numpy as np
import scipy
import glob
import scipy.ndimage as ndi
import matplotlib.pylab as plt
import pytest
import pyklip.klip
import pyklip.fakes as fakes
import pyklip.fm as fm
import pyklip.instruments.GPI as GPI
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fitpsf as fitpsf
import scipy.interpolate as sinterp

testdir = os.path.dirname(os.path.abspath(__file__)) + os.path.sep

filelist = glob.glob(testdir + os.path.join("data", "S20131210*distorcorr.fits"))
filelist.sort()
skipslices = [i for i in range(37) if i != 7 and i != 33]
dataset = GPI.GPIData(filelist, highpass=9, skipslices=skipslices)

numwvs = np.size(np.unique(dataset.wvs))
print(numwvs)
# generate PSF
dataset.generate_psfs(boxrad=25//2)
dataset.psfs /= (np.mean(dataset.spot_flux.reshape([dataset.spot_flux.shape[0] // numwvs, numwvs]), axis=0)[:, None, None])

# read in model spectrum
model_file = os.path.join(testdir, "..", "pyklip", "spectra", "cloudy", "t1600g100f2.flx")
spec_dat = np.loadtxt(model_file)
spec_wvs = spec_dat[1]
spec_f = spec_dat[3]
spec_interp = sinterp.interp1d(spec_wvs, spec_f, kind='nearest')
inputspec = spec_interp(np.unique(dataset.wvs))

# setup FM guesses
numbasis = np.array([1, 7, 100])
guesssep = 0.4267 / GPI.GPIData.lenslet_scale
guesspa = 212.15
guessflux = 5e-5

#Specify transmission correction parameters
trans = np.ones(100)
trans[0:30]=10000
rad = np.arange(start = 0, stop =100, step = 1)

def transmission_corrected(input_stamp, input_dx, input_dy):
    """
    input_dx: should be 2d 
    input_dy: should be 2d
    """
    distance_from_center = np.sqrt((input_dx)**2+(input_dy)**2)
    trans_at_dist = np.interp(distance_from_center, rad, trans)
    transmission_stamp = trans_at_dist.reshape(input_stamp.shape)
    output_stamp = transmission_stamp*input_stamp
    return output_stamp

fm_class = fmpsf.FMPlanetPSF(dataset.input.shape, numbasis, guesssep, guesspa, guessflux, dataset.psfs,
                                np.unique(dataset.wvs), dataset.dn_per_contrast, star_spt='A6',
                                spectrallib=[inputspec], field_dependent_correction=transmission_corrected)

    # run KLIP-FM
prefix = "betpic-131210-j-fmpsf"
fm.klip_dataset(dataset, fm_class, outputdir=testdir, fileprefix=prefix, numbasis=numbasis,
                annuli=[[guesssep-15, guesssep+15]], subsections=1, padding=0, movement=2, 
                time_collapse="weighted-mean")

# read in outputs
output_prefix = os.path.join(testdir, prefix)
with fits.open(output_prefix + "-fmpsf-KLmodes-all.fits") as fm_hdu:
# get FM frame, use KL=7
    fm_frame = fm_hdu[1].data[1]
    fm_centx = fm_hdu[1].header['PSFCENTX']
    fm_centy = fm_hdu[1].header['PSFCENTY']

planet_x_pos = guesssep*np.cos((guesspa+90))
planet_y_pos = guesssep*np.sin((guesspa+90))
planet_flux_inner = fm_frame[int(planet_y_pos-4)][int(planet_x_pos-4)]
planet_flux_outer = fm_frame[int(planet_y_pos+4)][int(planet_x_pos+4)]


# Check that flux is 0 at center
#assert(pytest.approx(fm_frame[int(fm_centx)][int(fm_centx)] == 0, rel=1e-2))