import os
import astropy.io.fits as fits
import numpy as np
import scipy
import time
import glob
import scipy.ndimage as ndi
import matplotlib.pylab as plt
import scipy.interpolate as sinterp
import pytest
import pyklip.klip
import pyklip.fakes as fakes
import pyklip.fm as fm
import pyklip.instruments.GPI as GPI
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fitpsf as fitpsf

testdir = os.path.dirname(os.path.abspath(__file__)) + os.path.sep

def test_throughput():
    """
    Tests FM coronagraphic throughput correction

    This function assigns a throughput of 10,000 for pixel positions within the planet radius,
    and a throughput of 0 outside the planet radius. It then tests that this field dependent
    correction was made by checking that the median flux within the planet radius is greater 
    than the flux outside the radius post-KLIPfm.  
    """
    t1 = time.time()

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
         Args:
            input_stamp (array): 2D array of the region surrounding the fake planet injection site
            input_dx (array): 2D array specifying the x distance of each stamp pixel from the center
            input_dy (array): 2D array specifying the y distance of each stamp pixel from the center
        
        Returns:
            output_stamp (array): 2D array of the throughput corrected planet injection site.
        """
         # Calculate the distance of each pixel in the input stamp from the center
        distance_from_center = np.sqrt((input_dx)**2+(input_dy)**2)

         # Interpolate to find the transmission value for each pixel in the input stamp
        trans_at_dist = np.interp(distance_from_center, rad, trans)

        # Reshape the interpolated array to have the same dimensions as the input stamp
        transmission_stamp = trans_at_dist.reshape(input_stamp.shape)

         # Make the throughput correction
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

    print("{0} seconds to run".format(time.time()-t1))

    # Find the distance from the center of the frame to the planet psf (notice that the axes are flipped by 90 degrees)
    planet_dy = guesssep*np.cos((guesspa+90))
    planet_dx = guesssep*np.sin((guesspa+90))

    # Calculate planet psf coordinates wrt image (subtract from y b/c planet is at the bottom of the image)
    planet_x_pos = int(fm_centx + planet_dx)
    planet_y_pos = int(fm_centy - planet_dy)

    # Find the flux within 5 pixels inside and outside of transmission boundary
    inner_range = fm_frame[(planet_y_pos):(planet_y_pos+5), (planet_x_pos-5):planet_x_pos]
    outer_range = fm_frame[(planet_y_pos-5):(planet_y_pos), planet_x_pos:(planet_x_pos+5)]

    # Check that the flux within the planet distance is less than the flux outside
    assert(abs(np.median(inner_range))/abs(np.median(outer_range))>10)

if __name__ == "__main__":
    test_throughput()


                            
