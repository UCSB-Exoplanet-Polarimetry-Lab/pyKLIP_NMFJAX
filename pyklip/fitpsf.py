import warnings
import pickle
import math

import numpy as np
import scipy.linalg as linalg
import scipy.ndimage as ndi
import scipy.ndimage.interpolation as sinterp
import scipy.optimize as optimize

import pyklip.covars as covars
import astropy.stats.circstats as circstats

# emcee more MCMC sampling
import emcee



class FitPSF(object):
    """
    Base class to perform astrometry on direct imaging data_stamp using GP regression. Can utilize a Bayesian framework with MCMC or a frequentist framework with least squares. 

    Args:
        guess_x1 (float): guess location, first coordiante. Either separation or x (pixels). Default is separation
        guess_x2 (float): guess location, second coordinate. Either position angle (degrees) or y (pixels). Default is PA
        fitboxsize (int): fitting box side length (pixels)
        method (str): either 'mcmc' or 'maxl' depending on framework you want. Defaults to 'mcmc'. 
        fmt (str): either 'seppa' or 'xy' depending on how you want to input the guess coordiantes

    Attributes:
        guess_sep (float): (initialization) guess separation for planet [pixels]
        guess_pa (float): (initialization) guess PA for planet [degrees]
        guess_RA_offset (float): (initialization) guess RA offset [pixels]
        guess_Dec_offset (float): (initialization) guess Dec offset [pixels]
        raw_RA_offset (:py:class:`pyklip.fitpsf.ParamRange`): (result) the raw result from the MCMC fit for the planet's location [pixels]
        raw_Dec_offset (:py:class:`pyklip.fitpsf.ParamRange`): (result) the raw result from the MCMC fit for the planet's location [pixels]
        raw_flux (:py:class:`pyklip.fitpsf.ParamRange`): (result) factor to scale the FM to match the flux of the data
        covar_params (list of :py:class:`pyklip.fitpsf.ParamRange`): (result) hyperparameters for the Gaussian process
        raw_sep(:py:class:`pyklip.fitpsf.ParamRange`): (result) the inferred raw result from the MCMC fit for the planet's location [pixels]
        raw_PA(:py:class:`pyklip.fitpsf.ParamRange`): (result) the inferred raw result from the MCMC fit for the planet's location [degrees]
        RA_offset(:py:class:`pyklip.fitpsf.ParamRange`): (result) the RA offset of the planet that includes all astrometric errors [pixels or mas]
        Dec_offset(:py:class:`pyklip.fitpsf.ParamRange`): (result) the Dec offset of the planet that includes all astrometric errors [pixels or mas]
        sep(:py:class:`pyklip.fitpsf.ParamRange`): (result) the separation of the planet that includes all astrometric errors [pixels or mas]
        PA(:py:class:`pyklip.fitpsf.ParamRange`): (result) the PA of the planet that includes all astrometric errors [degrees]
        fm_stamp (np.array): (fitting) The 2-D stamp of the forward model (centered at the nearest pixel to the guessed location)
        data_stamp (np.array): (fitting) The 2-D stamp of the data (centered at the nearest pixel to the guessed location)
        noise_map (np.array): (fitting) The 2-D stamp of the noise for each pixel the data computed assuming azimuthally similar noise
        padding (int): amount of pixels on one side to pad the data/forward model stamp
        sampler (emcee.EnsembleSampler): an instance of the emcee EnsambleSampler. Only for Bayesian fit. See emcee docs for more details. 
    

    """
    def __init__(self, guess_x1, guess_x2, fitboxsize, method='mcmc', fmt="seppa"):
        """
        Initilaizes the FitPSF class
        """
        # check method
        if method.lower() == "maxl":
            self.isbayesian = False
        elif method.lower() == "mcmc":
            self.isbayesian = True
        else:
            raise ValueError("method needs to be either 'maxl' or 'mcmc'. Received {0}.".format(method))

        # store initailization
        if fmt.lower() == "seppa":  
            self.guess_sep = guess_x1
            self.guess_pa = guess_x2
            self.fitboxsize = fitboxsize

            # derive delta RA and delta Dec
            # in pixels
            self.guess_RA_offset = self.guess_sep * np.sin(np.radians(self.guess_pa))
            self.guess_Dec_offset = self.guess_sep * np.cos(np.radians(self.guess_pa))
        elif fmt.lower() == "xy":
            # assume image is North up East left. 
            self.guess_RA_offset = -guess_x1
            self.guess_Dec_offset = guess_x2

        # stuff that isn't generated yet
        # stamps of the data_stamp and the forward model
        self.fm_stamp = None # Forward Model
        self.padding = 0 # padding for FM. You kinda need this to shift the FM around
        self.data_stamp = None # Data
        self.noise_map = None # same shape as self.data_stamp
        self.data_stamp_RA_offset = None # RA offset of data_stamp (in pixels)
        self.data_stamp_Dec_offset = None # Dec offset (in pixels)
        self.data_stamp_RA_offset_center = None # RA offset of center pixel (stampsize // 2)
        self.data_stamp_Dec_offset_center = None # Dec offset of center pixel (stampsize // 2)

        # guess flux (a hyperparameter)
        self._guess_flux = None

        # covariance paramters. Use the covariance initilizer function to initilize them
        self.covar = None
        self.covar_param_guesses = None
        self.covar_param_labels = None
        self.include_readnoise = False

        # MCMC fit params
        self.bounds = None
        self.sampler = None
        # Max-Likelihood fit
        self.hess_inv = None

        # best fit
        self.raw_RA_offset = None
        self.raw_Dec_offset = None
        self.raw_flux = None
        self.covar_params = None
        # best fit infered parameters
        self.raw_sep = None
        self.raw_PA = None
        self.RA_offset = None
        self.Dec_offset = None
        self.sep = None
        self.PA = None

    # automatically guess flux if it hasn't been defined
    @property
    def guess_flux(self):
        # if it was already set return it
        if self._guess_flux is not None:
            return self._guess_flux
        # if the data and fm stamps haven't been created yet, we can't do much 
        elif self.data_stamp is None or self.fm_stamp is None:
            return None
        # guess it based on the max value in both stamps
        self._guess_flux = np.max(self.data_stamp) / np.max(self.fm_stamp)
        return self._guess_flux
    @guess_flux.setter
    def guess_flux(self, newval):
        self._guess_flux = newval

    def generate_fm_stamp(self, fm_image, fm_center=None, fm_wcs=None, extract=True, padding=5):
        """
        Generates a stamp of the forward model and stores it in self.fm_stamp
        Args:
            fm_image: full imgae containing the fm_stamp
            fm_center: [x,y] center of image (assuing fm_stamp is located at sep/PA) corresponding to guess_sep and guess_pa
            fm_wcs: if not None, specifies the sky angles in the image. If None, assume image is North up East left
            extract: if True, need to extract the forward model from the image. Otherwise, assume the fm_stamp is already
                    centered in the frame (fm_image.shape // 2)
            padding: number of pixels on each side in addition to the fitboxsize to extract to pad the fm_stamp
                        (should be >= 1)

        Returns:

        """
        # cheeck the padding to make sure it's valid
        if not isinstance(padding, int):
            raise TypeError("padding must be an integer")
        if padding < 1:
            warnings.warn("Padding really should be >= 1 pixel so we can shift the FM around", RuntimeWarning)
        self.padding = padding


        if extract:
            if fm_wcs is not None:
                raise NotImplementedError("Have not implemented rotation using WCS")

            # image is now rotated North up east left
            # find the location of the FM
            thistheta = np.radians(self.guess_pa + 90)
            psf_xpos = self.guess_sep * np.cos(thistheta) + fm_center[0]
            psf_ypos = self.guess_sep * np.sin(thistheta) + fm_center[1]

        else:
            # PSf is already cenetered
            psf_xpos = fm_image.shape[1]//2
            psf_ypos = fm_image.shape[0]//2

        # now we found the FM in the image, extract out a centered stamp of it
        # grab the coordinates of the image
        stampsize = 2 * self.padding + self.fitboxsize # full stamp needs padding around all sides
        x_stamp, y_stamp = np.meshgrid(np.arange(stampsize * 1.) - stampsize //2,
                                       np.arange(stampsize * 1.) - stampsize// 2)

        x_stamp += psf_xpos
        y_stamp += psf_ypos

        # zero nans because it messes with interpolation
        fm_image[np.where(np.isnan(fm_image))] = 0

        fm_stamp = ndi.map_coordinates(fm_image, [y_stamp, x_stamp])
        self.fm_stamp = fm_stamp



    def generate_data_stamp(self, data, data_center, data_wcs=None, noise_map=None, dr=4, exclusion_radius=10):
        """
        Generate a stamp of the data_stamp ~centered on planet and also corresponding noise map
        Args:
            data: the final collapsed data_stamp (2-D)
            data_center: location of star in the data_stamp.  
            data_wcs: sky angles WCS object. To rotate the image properly [NOT YET IMPLMETNED]
                      if None, data_stamp is already rotated North up East left
            noise_map: if not None, noise map for each pixel in the data_stamp (2-D).
                        if None, one will be generated assuming azimuthal noise using an annulus widthh of dr
            dr: width of annulus in pixels from which the noise map will be generated
            exclusion_radius: radius around the guess planet location which doens't get factored into noise estimate

        Returns:

        """
        # rotate image North up east left if necessary
        if data_wcs is not None:
            # rotate
            raise NotImplementedError("Rotating based on WCS is not currently implemented yet")

        xguess = -self.guess_RA_offset + data_center[0]
        yguess = self.guess_Dec_offset + data_center[1]

        # round to nearest pixel
        xguess_round = int(np.round(xguess))
        yguess_round = int(np.round(yguess))

        # get index bounds for grabbing pixels from data_stamp
        ymin = yguess_round - self.fitboxsize//2
        xmin = xguess_round - self.fitboxsize//2
        ymax = yguess_round + self.fitboxsize//2 + 1
        xmax = xguess_round + self.fitboxsize//2 + 1
        if self.fitboxsize % 2 == 0:
            # for even fitbox sizes, need to truncate ymax/xmax by 1
            ymax -= 1
            xmax -= 1

        data_stamp = data[ymin:ymax, xmin:xmax]
        self.data_stamp = data_stamp

        # store coordinates of stamp also
        dy_img, dx_img = np.indices(data.shape, dtype=float)
        dy_img -= data_center[1]
        dx_img -= data_center[0]

        dx_data_stamp = dx_img[ymin:ymax, xmin:xmax]
        dy_data_stamp = dy_img[ymin:ymax, xmin:xmax]
        self.data_stamp_RA_offset = -dx_data_stamp
        self.data_stamp_Dec_offset = dy_data_stamp
        self.data_stamp_RA_offset_center = self.data_stamp_RA_offset[0, self.fitboxsize // 2]
        self.data_stamp_Dec_offset_center = self.data_stamp_Dec_offset[self.fitboxsize // 2, 0]

        # generate noise map if necessary
        if noise_map is None:
            # blank map
            noise_stamp = np.zeros(data_stamp.shape)

            # define exclusion around planet.
            distance_from_planet = np.sqrt((dx_img - (xguess - data_center[0]))**2 +
                                           (dy_img - (yguess - data_center[1]))**2)
            # define radial coordinate
            rimg = np.sqrt(dx_img**2 + dy_img**2)

            # calculate noise for each pixel in the data_stamp stamp
            for y_index, x_index in np.ndindex(data_stamp.shape):
                r_pix = np.sqrt(dy_data_stamp[y_index, x_index]**2 + dx_data_stamp[y_index, x_index]**2)
                pixels_for_noise = np.where((np.abs(rimg - r_pix) <= dr/2.) & (distance_from_planet > exclusion_radius))
                noise_stamp[y_index, x_index] = np.nanstd(data[pixels_for_noise])

        else:
            noise_stamp = noise_map[ymin:ymax, xmin:xmax]

        self.noise_map = noise_stamp


    def set_kernel(self, covar, covar_param_guesses, covar_param_labels, include_readnoise=False,
                   read_noise_fraction=0.01):
        """
        Set the Gaussian process kernel used in our fit

        Args:
            covar: Covariance kernel for GP regression. If string, can be "matern32" or "sqexp"
                    Can also be a function: cov = cov_function(x_indices, y_indices, sigmas, cov_params)
            covar_param_guesses: a list of guesses on the hyperparmeteres (size of N_hyperparams)
            covar_param_labels: a list of strings labelling each covariance parameter
            include_readnoise: if True, part of the noise is a purely diagonal term (i.e. read/photon noise)
            read_noise_fraction: fraction of the total measured noise is read noise (between 0 and 1)

        Returns:

        """
        if isinstance(covar, str):
            if covar.lower() == "matern32":
                self.covar = covars.matern32
            elif covar.lower() == "sqexp":
                self.covar = covars.sq_exp
            else:
                raise ValueError("Covariance matricies currently supported are 'matern32' and 'sqexp'")
        else:
            # this better be a covariance function. We're trusting you
            self.covar = covar

        self.covar_param_guesses = covar_param_guesses
        self.covar_param_labels = covar_param_labels

        if include_readnoise:
            self.include_readnoise = True
            self.covar_param_guesses.append(read_noise_fraction)
            self.covar_param_labels.append(r"K_{\delta}")


    def set_bounds(self, dRA, dDec, df, covar_param_bounds, read_noise_bounds=None):
        """
        Set bounds on Bayesian priors. All paramters can be a 2 element tuple/list/array that specifies
        the lower and upper bounds x_min < x < x_max. Or a single value whose interpretation is specified below
        If you are passing in both lower and upper bounds, both should be in linear scale!
        Args:
            dRA: Distance from initial guess position in pixels. For a single value, this specifies the largest distance
                form the initial guess (i.e. RA_guess - dRA < x < RA_guess + dRA)
            dDec: Same as dRA except with Dec
            df: Flux range. If single value, specifies how many orders of 10 the flux factor can span in one direction
                (i.e. log_10(guess_flux) - df < log_10(guess_flux) < log_10(guess_flux) + df
            covar_param_bounds: Params for covariance matrix. Like df, single value specifies how many orders of
                                magnitude parameter can span. Otherwise, should be a list of 2-elem touples
            read_noise_bounds: Param for read noise term. If single value, specifies how close to 0 it can go
                                based on powers of 10 (i.e. log_10(-read_noise_bound) < read_noise < 1 )

        Returns:

        """
        self.bounds = []

        # x/RA bounds
        if np.size(dRA) == 2:
            self.bounds.append(dRA)
        else:
            self.bounds.append([self.guess_RA_offset - dRA, self.guess_RA_offset + dRA])

        # y/Dec bounds
        if np.size(dDec) == 2:
            self.bounds.append(dDec)
        else:
            self.bounds.append([self.guess_Dec_offset - dDec, self.guess_Dec_offset + dDec])


        if np.size(df) == 2:
            self.bounds.append(df)
        else:
            self.bounds.append([self.guess_flux / (10.**df), self.guess_flux * (10**df)])

        # hyperparam bounds
        if np.ndim(covar_param_bounds) == 2:
            for covar_param_bound in covar_param_bounds:
                self.bounds.append(covar_param_bound)
        else:
            # this is a 1-D list, with each param specified by one paramter
            for covar_param_bound, covar_param_guess in zip(covar_param_bounds, self.covar_param_guesses):
                self.bounds.append([covar_param_guess / (10.**covar_param_bound),
                                    covar_param_guess * (10**covar_param_bound)])

        if read_noise_bounds is not None:
        # read noise
            if np.size(read_noise_bounds) == 2:
                self.bounds.append(read_noise_bounds)
            else:
                self.bounds.append([self.covar_param_guesses[-1]/10**read_noise_bounds, 1])


    def fit_psf(self, nwalkers=100, nburn=200, nsteps=800, save_chain=True, chain_output="bka-chain.pkl",
                       numthreads=None):
        """
        Fits the PSF of the planet in either a frequentist or Bayesian way depending on initialization. 

        Args:
            nwalkers: number of walkers (mcmc-only)
            nburn: numbe of samples of burn-in for each walker (mcmc-only)
            nsteps: number of samples each walker takes (mcmc-only)
            save_chain: if True, save the output in a pickled file (mcmc-only)
            chain_output: filename to output the chain to (mcmc-only)
            numthreads: number of threads to use (mcmc-only)

        Returns:

        """
        if self.isbayesian:
            return self._mcmc_fit_psf(nwalkers=nwalkers, nburn=nburn, nsteps=nsteps, save_chain=save_chain, 
                                        chain_output=chain_output, numthreads=numthreads)
        else:
            return self._lsqr_fit_psf()   

    def _mcmc_fit_psf(self, nwalkers=100, nburn=200, nsteps=800, save_chain=True, chain_output="bka-chain.pkl",
                       numthreads=None):
        """
        Run a Bayesian fit of the astrometry using MCMC
        Saves to self.chian

        Args:
            nwalkers: number of walkers
            nburn: numbe of samples of burn-in for each walker
            nsteps: number of samples each walker takes
            save_chain: if True, save the output in a pickled file
            chain_output: filename to output the chain to
            numthreads: number of threads to use

        Returns:

        """
        # create array of initial guesses
        # array of guess RA, Dec, and flux
        # for everything that's not RA/Dec offset, should be converted to log space for MCMC sampling
        init_guess = np.array([self.guess_RA_offset, self.guess_Dec_offset, math.log(self.guess_flux)])
        # append hyperparams for covariance matrix, which also need to be converted to log space
        init_guess = np.append(init_guess, np.log(self.covar_param_guesses))
        # number of dimensions of MCMC fit
        ndim = np.size(init_guess)

        # initialize walkers in a ball around the best fit value
        pos = [init_guess + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

        # prior bounds also need to be put in log space
        sampler_bounds = np.copy(self.bounds)
        sampler_bounds[2:] = np.log(sampler_bounds[2:])

        global lnprob
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(self, sampler_bounds, self.covar),
                                        kwargs={'readnoise' : self.include_readnoise}, threads=numthreads)

        # burn in
        print("Running burn in")
        pos, _, _ = sampler.run_mcmc(pos, nburn)
        # reset sampler
        sampler.reset()
         
        # chains should hopefulyl have converged. Now run MCMC
        print("Burn in finished. Now sampling posterior")
        sampler.run_mcmc(pos, nsteps)
        print("MCMC sampler has finished")

        # convert chains in log space back in linear space
        sampler.chain[:,:,2:] = np.exp(sampler.chain[:,:,2:])

        # save state
        self.sampler = sampler

        # save best fit values
        # percentiles has shape [ndims, 3]
        percentiles = np.swapaxes(np.percentile(sampler.flatchain, [16, 50, 84], axis=0), 0, 1)
        self.raw_RA_offset = ParamRange(percentiles[0][1], np.array([percentiles[0][2], percentiles[0][0]]) - percentiles[0][1])
        self.raw_Dec_offset = ParamRange(percentiles[1][1], np.array([percentiles[1][2], percentiles[1][0]]) - percentiles[1][1])
        self.raw_flux =  ParamRange(percentiles[2][1], np.array([percentiles[2][2], percentiles[2][0]]) -  percentiles[2][1])
        self.covar_params = [ParamRange(thispercentile[1], np.array([thispercentile[2], thispercentile[0]]) - thispercentile[1] ) for thispercentile in percentiles[3:]]

        if save_chain:
            pickle_file = open(chain_output, 'wb')
            pickle.dump(sampler.chain, pickle_file)
            pickle.dump(sampler.lnprobability, pickle_file)
            pickle.dump(sampler.acceptance_fraction, pickle_file)
            #pickle.dump(sampler.acor, pickle_file)
            pickle_file.close()

    def _lsqr_fit_psf(self):
        """
        Do a frequentist maximum likelihood fit to the data. Approximate errors using the Hessian of the likelihood function.
        """
        # create array of initial guesses
        # array of guess RA, Dec, and flux
        # for everything that's not RA/Dec offset, should be converted to log space for MCMC sampling
        init_guess = np.array([self.guess_RA_offset, self.guess_Dec_offset, math.log(self.guess_flux)])
        # append hyperparams for covariance matrix, which also need to be converted to log space
        init_guess = np.append(init_guess, np.log(self.covar_param_guesses))

        init_guess = np.array([self.guess_RA_offset, self.guess_Dec_offset, (self.guess_flux)])
        # append hyperparams for covariance matrix, which also need to be converted to log space
        init_guess = np.append(init_guess, (self.covar_param_guesses))

        if self.bounds is None:
            cost_function = lnprob
            cost_function_args = (self, None, self.covar)
        else:
            # prior bounds also need to be put in log space
            sampler_bounds = np.copy(self.bounds)
            #sampler_bounds[2:] = np.log(sampler_bounds[2:])

            cost_function = lnprob
            cost_function_args = (self, sampler_bounds, self.covar)

        cost_function_args += (self.include_readnoise, True)


        #global cost_function
        nm_result = optimize.minimize(cost_function, init_guess, args=cost_function_args, method="Nelder-Mead")

        # BFGS will only fit for position and flux, and their uncertainties.
        new_init_guess = nm_result.x[:3]
        cost_function_args += tuple(nm_result.x[3:])
        #if cost_function_args[1] is not None:
        #    cost_function_args[1] = cost_function_args[1][:3] # modify limits to not include hyperparameters 

        result = optimize.minimize(cost_function, new_init_guess, args=cost_function_args, method="BFGS")

        if not result.success:
            warnings.warn("Optimizer did not converge! Estimated uncertainties are likely unreliable. Msg: {0}".format(result.message))
            
        # best fit values, and use the Hessian to approximate the uncertainties in the parameters
        ra_best = result.x[0]
        dec_best = result.x[1]
        flux_best = (result.x[2])
        covar_params_best = [nm_result.x[i] for i in range(3, np.size(nm_result.x))]
        ra_err = np.sqrt(np.abs(result.hess_inv[0,0]))
        dec_err = np.sqrt(np.abs(result.hess_inv[1,1]))
        flux_err = np.sqrt(np.abs(result.hess_inv[2,2]))

        # convert to linear space the flux and covariance parameters
        covar_params_best = np.exp(covar_params_best)
        # flux error is approx symmetric in log space
        flux_err_two_sided = np.array([np.exp(flux_best - flux_err), np.exp(flux_best + flux_err)])
        flux_best = math.exp(flux_best)

        # save best fit values
        # percentiles has shape [ndims, 3]
        self.raw_RA_offset = ParamRange(ra_best, ra_err)
        self.raw_Dec_offset = ParamRange(dec_best, dec_err)
        self.raw_flux =  ParamRange(flux_best, flux_err_two_sided)
        self.covar_params = [ParamRange(param_best, 0) for param_best in covar_params_best]

        self.hess_inv = result.hess_inv
        


    def make_corner_plot(self, fig=None):
        """
        Generate a corner plot of the posteriors from the MCMC
        Args:
            fig: if not None, a matplotlib Figure object

        Returns:
            fig: the Figure object. If input fig is None, function will make a new one

        """
        if not self.isbayesian:
            raise AttributeError("Corner plot is only available if using Bayesian MCMC framework")
        
        import corner

        all_labels = [r"x", r"y", r"$\alpha$"]
        all_labels = np.append(all_labels, self.covar_param_labels)

        fig = corner.corner(self.sampler.flatchain, labels=all_labels, quantiles=[0.16, 0.5, 0.84], fig=fig)

        return fig


    def best_fit_and_residuals(self, fig=None):
        """
        Generate a plot of the best fit FM compared with the data_stamp and also the residuals
        Args:
            fig (matplotlib.Figure): if not None, a matplotlib Figure object

        Returns:
            fig (matplotlib.Figure): the Figure object. If input fig is None, function will make a new one

        """
        import matplotlib
        import matplotlib.pylab as plt

        if fig is None:
            fig = plt.figure(figsize=(12, 4))

        # create best fit FM
        dx = -(self.raw_RA_offset.bestfit - self.data_stamp_RA_offset_center)
        dy = self.raw_Dec_offset.bestfit - self.data_stamp_Dec_offset_center

        fm_bestfit = self.raw_flux.bestfit * sinterp.shift(self.fm_stamp, [dy, dx])
        if self.padding > 0:
            fm_bestfit = fm_bestfit[self.padding:-self.padding, self.padding:-self.padding]

        # make residual map
        residual_map = self.data_stamp - fm_bestfit

        # normalize all images to same scale
        colornorm = matplotlib.colors.Normalize(vmin=np.percentile(self.data_stamp, 0.03),
                                                vmax=np.percentile(self.data_stamp, 99.7))

        # plot the data_stamp
        ax1 = fig.add_subplot(131)
        im1 = ax1.imshow(self.data_stamp, interpolation='nearest', cmap='cubehelix', norm=colornorm)
        ax1.invert_yaxis()
        ax1.set_title("Data")
        ax1.set_xlabel("X (pixels)")
        ax1.set_ylabel("Y (pixels)")

        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(fm_bestfit, interpolation='nearest', cmap='cubehelix', norm=colornorm)
        ax2.invert_yaxis()
        ax2.set_title("Best-fit Model")
        ax2.set_xlabel("X (pixels)")

        ax3 = fig.add_subplot(133)
        im3 = ax3.imshow(residual_map, interpolation='nearest', cmap='cubehelix', norm=colornorm)
        ax3.invert_yaxis()
        ax3.set_title("Residuals")
        ax3.set_xlabel("X (pixels)")

        fig.subplots_adjust(right=0.82)
        fig.subplots_adjust(hspace=0.4)
        ax_pos = ax3.get_position()

        cbar_ax = fig.add_axes([0.84, ax_pos.y0, 0.02, ax_pos.height])
        cb = fig.colorbar(im1, cax=cbar_ax)
        cb.set_label("Counts (DN)")

        return fig

    def propogate_errs(self, star_center_err=None, platescale=None, platescale_err=None, pa_offset=None, pa_uncertainty=None):
        """
        Propogate astrometric error. Stores results in its own fields

        Args:
            star_center_err (float): uncertainity of the star location (pixels)
            platescale (float): mas/pix conversion to angular coordinates 
            platescale_err (float): mas/pix error on the platescale
            pa_offset (float): Offset, in the same direction as position angle, to set North up (degrees)
            pa_uncertainity (float): Error on position angle/true North calibration (Degrees)
        """
        if self.isbayesian:
            # format MCMC chains
            # ensure numpy arrays
            x_fit = self.sampler.chain[:,:,0].flatten()
            y_fit = self.sampler.chain[:,:,1].flatten()

        else:
            x_fit = self.raw_RA_offset.bestfit
            y_fit = self.raw_Dec_offset.bestfit

        # calcualte statistial errors in x and y
        x_best = self.raw_RA_offset.bestfit
        y_best = self.raw_Dec_offset.bestfit
        x_1sigma_raw = self.raw_RA_offset.error_2sided
        y_1sigma_raw = self.raw_Dec_offset.error_2sided

        print("Raw X/Y Centroid = ({0}, {1}) with statistical error of {2} pix in X and {3} pix in Y".format(x_best, y_best, x_1sigma_raw, y_1sigma_raw))

        # calculate sep and pa from x/y separation
        sep_fit = np.sqrt((x_fit)**2 + (y_fit)**2)
        # For PA compute mean using circstats package, find delta_pa between all points and the mean,
        # then compute median/precentiles
        pa_fit = (np.arctan2(y_fit, -x_fit) - (np.pi/2.0)) % (2.0*np.pi) # Radians!

        if self.isbayesian:
            # for Bayesian, convert the chains to sep/pa to get uncertainity
            pa_mean = circstats.circmean(pa_fit - np.pi) + np.pi # Circmean [-pi, pi]
            d_pa = np.arctan2(np.sin(pa_fit-pa_mean), np.cos(pa_fit-pa_mean))
            pa_median = np.median(d_pa) + pa_mean
            pa_percentile = np.nanpercentile(d_pa, [84,16])  - np.median(d_pa) # median of d_pa should be small
            pa_fit = np.degrees(pa_fit) # Convert to degrees

            # calculate sep and pa statistical errors
            sep_best = np.median(sep_fit)
            pa_best = np.degrees(pa_median)
            sep_1sigma_raw = (np.percentile(sep_fit, [84,16]) - sep_best)
            pa_1sigma_raw = np.degrees(pa_percentile)
        else:
            # since we just have point estimates, use analytical error propogration
            sep_best = sep_fit
            pa_fit = np.degrees(pa_fit)
            pa_best = pa_fit

            sep_1sigma_raw = (x_fit/sep_fit)**2 * x_1sigma_raw**2 + (y_fit/sep_fit)**2 * y_1sigma_raw**2
            sep_1sigma_raw = np.sqrt(sep_1sigma_raw)

            pa_1sigma_raw = (y_fit/sep_fit**2)**2 * x_1sigma_raw**2 + (x_fit/sep_fit**2)**2 * y_1sigma_raw**2
            pa_1sigma_raw = np.sqrt(pa_1sigma_raw)
            pa_1sigma_raw = np.degrees(pa_1sigma_raw)

        print("Raw Sep/PA Centroid = ({0}, {1}) with statistical error of {2} pix in Sep and {3} pix in PA".format(sep_best, pa_best, sep_1sigma_raw, pa_1sigma_raw))
        
        # store the raw sep and PA values
        self.raw_sep = ParamRange(sep_best, sep_1sigma_raw)
        self.raw_PA = ParamRange(pa_best, pa_1sigma_raw)

        # Now let's start propogating error terms if they are supplied. 
        # We do them in Sep/PA space first since it's more natural here

        # star center error
        if star_center_err is None:
            print("Skipping star center uncertainity...")
            star_center_err = 0
        else:
            print("Adding in star center uncertainity")

        sep_err_pix = (sep_1sigma_raw**2) + star_center_err**2
        sep_err_pix = np.sqrt(sep_err_pix)

        # plate scale error
        if platescale is not None:
            print("Converting pixels to milliarcseconds")
            if platescale_err is None:
                print("Skipping plate scale uncertainity...")
                platescale_err = 0
            else:
                print("Adding in plate scale error")
            sep_err_mas = np.sqrt((sep_err_pix * platescale)**2 + (platescale_err * sep_best)**2)

        # PA Offset
        if pa_offset is not None:
            print("Adding in a PA/North angle offset")
            # Have to repeat wrapping procedure above in case pa_offset is large
            pa_fit = np.radians((pa_fit + pa_offset) % 360) # Convert back to radians for circstats
            if self.isbayesian:
                pa_mean = circstats.circmean(pa_fit - np.pi) + np.pi # Circmean [-pi, pi]
                d_pa = np.arctan2(np.sin(pa_fit-pa_mean), np.cos(pa_fit-pa_mean))

                pa_median = np.median(d_pa) + pa_mean
                pa_best = np.degrees(pa_median)
            else:
                pa_best = np.degrees(pa_fit)
            pa_fit = np.degrees(pa_fit) # Convert back to degrees
        
        # PA Uncertainity
        if pa_uncertainty is None:
            print("Skipping PA/North uncertainity...")
            pa_uncertainty = 0
        else:
            print("Adding in PA uncertainity")

        pa_err = np.radians(pa_1sigma_raw)**2 + (star_center_err/sep_best)**2 + np.radians(pa_uncertainty)**2
        pa_err = np.sqrt(pa_err)
        pa_err_deg = np.degrees(pa_err)

        sep_err_pix_avg = np.mean(np.abs(sep_err_pix))
        pa_err_deg_avg = np.mean(np.abs(pa_err_deg))

        print("Sep = {0} +/- {1} ({2}) pix, PA = {3} +/- {4} ({5}) degrees".format(sep_best, sep_err_pix_avg, sep_err_pix, pa_best, pa_err_deg_avg, pa_err_deg))

        # Store sep/PA (excluding platescale) values
        self.sep = ParamRange(sep_best, sep_err_pix)
        self.PA = ParamRange(pa_best, pa_err_deg)

        if platescale is not None:
            sep_err_mas_avg = np.mean(np.abs(sep_err_mas))
            print("Sep = {0} +/- {1} ({2}) mas, PA = {3} +/- {4} ({5}) degrees".format(sep_best*platescale, sep_err_mas_avg, sep_err_mas, pa_best, pa_err_deg_avg, pa_err_deg))
            # overwrite sep values with values converted to milliarcseconds
            self.sep = ParamRange(sep_best*platescale, sep_err_mas)

        # convert PA errors back into x y (RA/Dec)
        ra_fit = -sep_fit * np.cos(np.radians(pa_fit+90))
        dec_fit = sep_fit * np.sin(np.radians(pa_fit+90))

        # ra/dec statistical errors. This is after the rotation from pa_offset is applied
        if self.isbayesian:
            ra_best = np.median(ra_fit)
            dec_best = np.median(dec_fit)
            ra_1sigma_raw = np.percentile(ra_fit, [84,16]) - ra_best
            dec_1sigma_raw = np.percentile(dec_fit, [84,16]) - dec_best
        else:
            ra_best = ra_fit
            dec_best = dec_fit

            # this should depend on sine and cosine of pa_offset
            pa_offset_rad = np.radians(pa_offset)
            ra_1sigma_raw = np.sqrt(math.cos(pa_offset_rad)**2 * x_1sigma_raw**2 + math.sin(pa_offset_rad)**2 * y_1sigma_raw**2)
            dec_1sigma_raw = np.sqrt(math.sin(pa_offset_rad)**2 * x_1sigma_raw**2 + math.cos(pa_offset_rad)**2 * y_1sigma_raw**2)

        ra_err_full_pix = np.sqrt((ra_1sigma_raw**2)  + (star_center_err)**2 + (dec_best * np.radians(pa_uncertainty))**2 )
        dec_err_full_pix = np.sqrt((dec_1sigma_raw**2)  + (star_center_err)**2 + (ra_best * np.radians(pa_uncertainty))**2 )

        # Store error propgoated RA/Dec values (excluding platescale)
        self.RA_offset = ParamRange(ra_best, ra_err_full_pix)
        self.Dec_offset = ParamRange(dec_best, dec_err_full_pix)

        print("RA offset = {0} +/- {1} ({2}) pix".format(self.RA_offset.bestfit, self.RA_offset.error, self.RA_offset.error_2sided))
        print("Dec offset = {0} +/- {1} ({2}) pix".format(self.Dec_offset.bestfit, self.Dec_offset.error, self.Dec_offset.error_2sided))

        if platescale is not None:
            ra_err_full_mas = np.sqrt((ra_err_full_pix*platescale)**2 + (platescale_err * ra_best)**2)
            dec_err_full_mas = np.sqrt((dec_err_full_pix*platescale)**2 + (platescale_err * dec_best)**2)
            
            # Overwrite with calibrated RA/Dec converted to milliarcsecs
            self.RA_offset = ParamRange(ra_best*platescale, ra_err_full_mas)
            self.Dec_offset = ParamRange(dec_best*platescale, dec_err_full_mas)

            print("RA offset = {0} +/- {1} ({2}) mas".format(self.RA_offset.bestfit, self.RA_offset.error, self.RA_offset.error_2sided))
            print("Dec offset = {0} +/- {1} ({2}) mas".format(self.Dec_offset.bestfit, self.Dec_offset.error, self.Dec_offset.error_2sided))




def lnprior(fitparams, bounds, readnoise=False, negate=False):
    """
    Bayesian prior

    Args:
        fitparams: array of params (size N)

        bounds: array of (N,2) with corresponding lower and upper bound of params
                bounds[i,0] <= fitparams[i] < bounds[i,1]
        readnoise (bool): If True, the last fitparam fits for diagonal noise
        negate (bool): if True, negatives the probability (used for minimization algos)

    Returns:
        prior: 0 if inside bound ranges, -inf if outside

    """
    prior = 0.0

    for param, bound in zip(fitparams, bounds):
        if (param >= bound[1]) | (param < bound[0]):
            prior *= -np.inf
            break

    if negate:
        prior *= -1

    return prior


def lnlike(fitparams, fma, cov_func, readnoise=False, negate=False):
    """
    Likelihood function
    Args:
        fitparams: array of params (size N). First three are [dRA,dDec,f]. Additional parameters are GP hyperparams
                    dRA,dDec: RA,Dec offsets from star. Also coordianaes in self.data_{RA,Dec}_offset
                    f: flux scale factor to normalizae the flux of the data_stamp to the model
        fma (FMAstrometry): a FMAstrometry object that has been fully set up to run
        cov_func (function): function that given an input [x,y] coordinate array returns the covariance matrix
                  e.g. cov = cov_function(x_indices, y_indices, sigmas, cov_params)
        readnoise (bool): If True, the last fitparam fits for diagonal noise
        negate (bool): if True, negatives the probability (used for minimization algos)

    Returns:
        likeli: log of likelihood function (minus a constant factor)
    """
    dRA_trial = fitparams[0]
    dDec_trial = fitparams[1]
    f_trial = fitparams[2]
    hyperparms_trial = fitparams[3:]

    if readnoise:
        # last hyperparameter is a diagonal noise term. Separate it out
        readnoise_amp = np.exp(hyperparms_trial[-1])
        hyperparms_trial = hyperparms_trial[:-1]

    # get trial parameters out of log space
    #f_trial = math.exp(f_trial)
    #hyperparms_trial = np.exp(hyperparms_trial)

    dx = -(dRA_trial - fma.data_stamp_RA_offset_center)
    dy = dDec_trial - fma.data_stamp_Dec_offset_center

    fm_shifted = sinterp.shift(fma.fm_stamp, [dy, dx])

    if fma.padding > 0:
        fm_shifted = fm_shifted[fma.padding:-fma.padding, fma.padding:-fma.padding]

    diff_ravel = fma.data_stamp.ravel() - f_trial * fm_shifted.ravel()

    cov = cov_func(fma.data_stamp_RA_offset.ravel(), fma.data_stamp_Dec_offset.ravel(), fma.noise_map.ravel(),
                   hyperparms_trial)

    if readnoise:
        # add a diagonal term
        cov = (1 - readnoise_amp) * cov + readnoise_amp * np.diagflat(fma.noise_map.ravel()**2 )

    # solve Cov * x = diff for x = Cov^-1 diff. Numerically more stable than inverse
    # to make it faster, we comptue the Cholesky factor and use it to also compute the determinent
    try:
        (L_cov, lower_cov) = linalg.cho_factor(cov)
        cov_inv_dot_diff = linalg.cho_solve((L_cov, lower_cov), diff_ravel) # solve Cov x = diff for x
        # compute log(det(Cov))
        logdet = 2*np.sum(np.log(np.diag(L_cov)))
    except: 
        cov_inv = np.linalg.inv(cov)
        cov_inv_dot_diff = np.dot(cov_inv, diff_ravel)
        # compute log(det(Cov))
        _, logdet = np.linalg.slogdet(cov)

    residuals = diff_ravel.dot(cov_inv_dot_diff)
    constant = logdet

    loglikelihood = -0.5 * (residuals + constant)

    if negate:
        loglikelihood *= -1
    
    return loglikelihood


def lnprob(fitparams, fma, bounds, cov_func, readnoise=False, negate=False, cov_params=None):
    """
    Function to compute the relative posterior probabiltiy. Product of likelihood and prior
    Args:
        fitparams: array of params (size N). First three are [dRA,dDec,f]. Additional parameters are GP hyperparams
                    dRA,dDec: RA,Dec offsets from star. Also coordianaes in self.data_{RA,Dec}_offset
                    f: flux scale factor to normalizae the flux of the data_stamp to the model
        fma: a FMAstrometry object that has been fully set up to run
        bounds: array of (N,2) with corresponding lower and upper bound of params
                bounds[i,0] <= fitparams[i] < bounds[i,1]
        cov_func: function that given an input [x,y] coordinate array returns the covariance matrix
                  e.g. cov = cov_function(x_indices, y_indices, sigmas, cov_params)
        readnoise (bool): If True, the last fitparam fits for diagonal noise
        negate (bool): if True, negatives the probability (used for minimization algos)
        cov_params (list): if not None, GP hyperparams aren't being fit. Are being passed here as set instead

    Returns:

    """
    if bounds is not None:
        lp = lnprior(fitparams, bounds, readnoise=readnoise, negate=negate)
    else:
        lp = 0

    if not np.isfinite(lp):
        if not negate:
            return -np.inf
        else:
            return np.inf

    if cov_params is not None:
        # need to stil cov_params into the fitparams, even though we are not fitting for them
        # easier to not modify that function
        fitparams = np.append(fitparams, cov_params)
    
    return lp + lnlike(fitparams, fma, cov_func, readnoise=readnoise, negate=negate)


class ParamRange(object):
    """
    Stores the best fit value and uncertainities for a parameter in a neat fasion

    Args:
        bestfit (float): the bestfit value
        err_range: either a float or a 2-element tuple (+val1, -val2) and gives the 1-sigma range

    Attributes:
        bestfit (float): the bestfit value
        error (float): the average 1-sigma error
        error_2sided (np.array): [+error1, -error2] 2-element array with asymmetric errors
    """
    def __init__(self, bestfit, err_range):
        self.bestfit = bestfit

        if isinstance(err_range, (int, float)):
            self.error = err_range
            self.error_2sided = np.array([err_range, -err_range])
        elif len(err_range) == 2:
            self.error_2sided = np.array(err_range)
            self.error = np.mean(np.abs(err_range))

class FMAstrometry(FitPSF):
    """
    FMAstrometry is now obsolete as all its functionality is in FitPSF
    """
    def __init__(self, guess_sep, guess_pa, fitboxsize, method='mcmc'):
        super(FMAstrometry, self).__init__(guess_sep, guess_pa, fitboxsize, method)

    def fit_astrometry(self, nwalkers=100, nburn=200, nsteps=800, save_chain=True, chain_output="bka-chain.pkl",
                       numthreads=None):
        self.fit_psf(nwalkers, nburn, nsteps, save_chain, chain_output, numthreads)

import scipy.ndimage as ndimage
from copy import copy
from scipy.optimize import minimize

def place_model_PSF(PSF_template,x_cen,y_cen,output_shape, x_grid = None, y_grid = None):
    """
    Interpolate PSF_template to recenter it at at x_cen,y_cen

    Args:
        PSF_template: PSF model
        x_cen: x center of the PSF
        y_cen: y center of the PSF
        output_shape: shape of the original image
        x_grid: grid of x coordinates (from meshgrid)
        y_grid: grid of y coordinates (from meshgrid)

    Returns: stamp with PSF template centered at x_cen,y_cen
    """

    ny_template, nx_template = PSF_template.shape
    if x_grid is None and y_grid is None:
        x_grid, y_grid = np.meshgrid(np.arange(0,output_shape[1],1),np.arange(0,output_shape[0],1))

    x_grid = x_grid.astype(np.float)
    y_grid = y_grid.astype(np.float)

    x_grid -= x_cen - nx_template//2
    y_grid -= y_cen - ny_template//2

    return ndimage.map_coordinates(PSF_template, [y_grid,x_grid], mode='constant', cval=0.0)

def LSQ_place_model_PSF(PSF_template,x_cen,y_cen,planet_image, x_grid = None, y_grid = None):
    """
    Calculate the least square value of (planet_image - PSF_template(x_cen,y_cen))

    Args:
        PSF_template: PSF model
        x_cen: x center of the PSF
        y_cen: y center of the PSF
        planet_image: Image to be fitted
        x_grid: grid of x coordinates (from meshgrid)
        y_grid: grid of y coordinates (from meshgrid)

    Returns: least square value
    """
    model = place_model_PSF(PSF_template,x_cen,y_cen,planet_image.shape, x_grid = x_grid, y_grid = y_grid)
    return np.nansum((planet_image-model)**2,axis = (0,1))#/y_model


def simplecentroid(image, PSF, guessx,guessy):
    """
    Fit a PSF array to image and return the centroid.

    Info:
    - model is fitted using ndimage.map_coordinates()
    - optimization is performed with scipy.optimize.minimize().

    Args:
        image: Image to be fitted
        PSF: model of the point source
        guessx: first guess x-axis (ie columns) position
        guessy: first guess y-axis (ie rows) position

    Returns: (x_cen,y_cen)
    """

    ny,nx = image.shape
    row_id,col_id = int(round(guessy)),int(round(guessx))
    ny_PSF,nx_PSF = PSF.shape

    row_m = int(np.floor(ny_PSF/2.0))
    row_p = int(np.ceil(ny_PSF/2.0))
    col_m = int(np.floor(nx_PSF/2.0))
    col_p = int(np.ceil(nx_PSF/2.0))
    stamp = copy(image[np.max([0,(row_id-row_m)]):np.min([ny-1,(row_id+row_p)]), np.max([0,(col_id-col_m)]):np.min([nx-1,(col_id+col_p)])])
    if np.sum(np.isfinite(stamp)) <= 0.1*nx_PSF*ny_PSF:
        return None,None
    stamp_max = np.nanmax(stamp)
    if stamp_max != 0:
        stamp /= stamp_max
    else:
        return None,None
    _PSF= copy(PSF)
    _PSF /= np.nanmax(_PSF)

    nanargmax_flat_stamp= np.nanargmax(stamp)
    max_row_id = np.floor(nanargmax_flat_stamp/nx_PSF)
    max_col_id = nanargmax_flat_stamp-nx_PSF*max_row_id

    param0 = (float(max_col_id),float(max_row_id))

    LSQ_func = lambda para: LSQ_place_model_PSF(_PSF,para[0],para[1],stamp)
    param_fit = minimize(LSQ_func,param0, method="Nelder-Mead").x

    return (param_fit[0]+col_id-col_m),(param_fit[1]+row_id-row_m)

def quick_psf_fit(data, psf, x_guess, y_guess, fitboxsize):
    """
    A wrapper for a quick maximum likelihood fit to a PSF to the data. 

    Args:
        data (np.array): 2-D data frame
        psf (np.array): 2-D PSF template. This should be smaller than the size of data and
                        larger than the fitboxsize
        x_guess (float): approximate x position of the location you are fitting the psf to
        y_guess (float): approximate y position of the location you are fitting the psf to
        fitboxsize (int): fitting region is a square. This is the lenght of one side of the square

    Returns:
        x_fit, y_fit, flux_fit
        x_fit (float): x position
        y_fit (float): y position
        flux_fit (float): multiplicative scale factor for the psf to match the data
    """
