__author__ = 'JB'


import warnings
import itertools

from scipy.optimize import leastsq
from astropy.modeling import models, fitting
from matplotlib import rcParams
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

import numpy as np
from copy import copy

from pyklip.kpp_new.utils.mathfunc import *
from pyklip.kpp_new.utils.multiproc import *
from pyklip.kpp_new.utils.GPIimage import *


def get_image_stat_map_perPixMasking(image,
                                     image_without_planet,
                                     mask_radius = 7,
                                     IOWA = None,
                                     N = None,
                                     centroid = None,
                                     mute = True,
                                     N_threads = None,
                                     Dr = None,
                                     Dth = None,
                                     type = "SNR"):
    """
    Calculate the SNR or probability (tail distribution) of a given image on a per pixel basis.

    :param image: The image or cubes for which one wants the statistic.
    :param image_without_planet: Same as image but where real signal has been masked out. The code will actually use
                                map to calculate the standard deviation or the density function.
    :param mask_radius: Radius of the mask used around the current pixel when use_mask_per_pixel = True.
    :param IOWA: (IWA,OWA) inner working angle, outer working angle. It defines boundary to the zones in which the
                statistic is calculated.
    :param N: Defines the width of the ring by the number of pixels it has to include.
            The width of the annuli will therefore vary with sepration.
    :param centroid: Define the cente rof the image. Default is x_cen = np.ceil((nx-1)/2) ; y_cen = np.ceil((ny-1)/2)
    :param mute: Won't print any logs.
    :param N_threads: Number of threads to be used. If None run sequentially.
    :param Dr: If not None defines the width of the ring as Dr. N is then ignored.
    :param Dth: Define the angular size of a sector in degree (will apply for either Dr or N)
    :param type: Indicate the type of statistic to be calculated.
                If "SNR" (default) simple stddev calculation and returns SNR.
                If "stddev" returns the pure standard deviation map.
                If "proba" triggers proba calculation with pdf fitting.
    :return: The statistic map for image.
    """
    ny,nx = image.shape


    if IOWA is None:
        IWA,OWA,inner_mask,outer_mask = get_occ(image, centroid = centroid)
    else:
        IWA,OWA = IOWA


    if centroid is None :
        x_cen = np.ceil((nx-1)/2) ; y_cen = np.ceil((ny-1)/2)
    else:
        x_cen, y_cen = centroid

    image_without_planet_mask = np.ones((ny,nx))
    image_without_planet_mask[np.where(np.isnan(image_without_planet))] = 0

    # Build the x and y coordinates grids
    x_grid, y_grid = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
    # Calculate the radial distance of each pixel
    r_grid = abs(x_grid +y_grid*1j)
    th_grid = np.arctan2(x_grid,y_grid)

    image_noNans = np.where(np.isfinite(image)*(r_grid>IWA)*(r_grid<OWA))

    if N is not None:
        r_min_firstZone,r_max_firstZone = (IWA,np.sqrt(N/np.pi+IWA**2))
        r_limit_firstZone = (r_min_firstZone + r_max_firstZone)/2.
        r_min_lastZone,r_max_lastZone = (OWA,np.max([ny,nx]))
        r_limit_lastZone = OWA - N/(4*np.pi*OWA)
    else:
        r_min_firstZone,r_max_firstZone,r_limit_firstZone = None,None,None
        r_min_lastZone,r_max_lastZone,r_limit_lastZone = None,None,None

    stat_map = np.zeros(image.shape) + np.nan
    if N_threads is not None:
        pool = NoDaemonPool(processes=N_threads)
        #pool = mp.Pool(processes=N_threads)

        N_pix = image_noNans[0].size
        chunk_size = N_pix/N_threads
        N_chunks = N_pix/chunk_size

        # Shuffle the list of indices such that a thread doesn't end up with only the outer most pixel (where the number
        # of pixels in the pdf is much bigger which make it a lot slower compared to his brothers)
        image_noNans_rows = copy(image_noNans[0])
        image_noNans_cols = copy(image_noNans[1])
        rng_state = np.random.get_state()
        np.random.shuffle(image_noNans_rows)
        np.random.set_state(rng_state)
        np.random.shuffle(image_noNans_cols)

        # Get the chunks
        chunks_row_indices = []
        chunks_col_indices = []
        for k in range(N_chunks-1):
            chunks_row_indices.append(image_noNans_rows[(k*chunk_size):((k+1)*chunk_size)])
            chunks_col_indices.append(image_noNans_cols[(k*chunk_size):((k+1)*chunk_size)])
        chunks_row_indices.append(image_noNans_rows[((N_chunks-1)*chunk_size):N_pix])
        chunks_col_indices.append(image_noNans_cols[((N_chunks-1)*chunk_size):N_pix])

        outputs_list = \
            pool.map(get_image_stat_map_perPixMasking_threadTask_star,
                       itertools.izip(chunks_row_indices,
                       chunks_col_indices,
                       itertools.repeat(image),
                       itertools.repeat(image_without_planet),
                       itertools.repeat(x_grid),
                       itertools.repeat(y_grid),
                       itertools.repeat(N),
                       itertools.repeat(mask_radius),
                       itertools.repeat((r_limit_firstZone,r_min_firstZone,r_max_firstZone)),
                       itertools.repeat((r_limit_lastZone,r_min_lastZone,r_max_lastZone)),
                       itertools.repeat(Dr),
                       itertools.repeat(Dth),
                       itertools.repeat(type)))

        for row_indices,col_indices,out in zip(chunks_row_indices,chunks_col_indices,outputs_list):
            stat_map[(row_indices,col_indices)] = out
        pool.close()

    else:
        stat_map[image_noNans] = \
            get_image_stat_map_perPixMasking_threadTask(image_noNans[0],
                                                               image_noNans[1],
                                                               image,
                                                               image_without_planet,
                                                               x_grid,y_grid,
                                                               N,
                                                               mask_radius,
                                                               (r_limit_firstZone,r_min_firstZone,r_max_firstZone),
                                                               (r_limit_lastZone,r_min_lastZone,r_max_lastZone),
                                                               Dr = Dr,
                                                               Dth = Dth,
                                                               type = type)
    if type == "proba":
        return -np.log10(stat_map)
    else:
        return stat_map

def get_image_stat_map_perPixMasking_threadTask_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call get_image_probability_map_perPixMasking_threadTask() with a tuple of parameters.
    """
    return get_image_stat_map_perPixMasking_threadTask(*params)

def get_image_stat_map_perPixMasking_threadTask(row_indices,
                                               col_indices,
                                               image,
                                               image_without_planet,
                                               x_grid,
                                               y_grid,
                                               N,
                                               mask_radius,
                                               firstZone_radii,
                                               lastZone_radii,
                                               Dr = None,
                                               Dth = None,
                                               type = "SNR"):
    """
    Calculate the SNR or probability (tail distribution) for some pixels in image on a per pixel basis.
    The pixels are defined by row_indices and col_indices.

    This function is used for parallelization

    :param row_indices: The row indices of images for which we want the statistic.
    :param col_indices: The column indices of images for which we want the statistic.
    :param image: The image or cubes for which one wants the statistic.
    :param image_without_planet: Same as image but where real signal has been masked out. The code will actually use
                                map to calculate the standard deviation or the density function.
    :param mask_radius: Radius of the mask used around the current pixel when use_mask_per_pixel = True.
    :param IOWA: (IWA,OWA) inner working angle, outer working angle. It defines boundary to the zones in which the
                statistic is calculated.
    :param N: Defines the width of the ring by the number of pixels it has to include.
            The width of the annuli will therefore vary with sepration.
    :param centroid: Define the cente rof the image. Default is x_cen = np.ceil((nx-1)/2) ; y_cen = np.ceil((ny-1)/2)
    :param mute: Won't print any logs.
    :param N_threads: Number of threads to be used. If None run sequentially.
    :param Dr: If not None defines the width of the ring as Dr. N is then ignored.
    :param Dth: Define the angular size of a sector in degree (will apply for either Dr or N)
    :param type: Indicate the type of statistic to be calculated.
                If "SNR" (default) simple stddev calculation and returns SNR.
                If "stddev" returns the pure standard deviation map.
                If "proba" triggers proba calculation with pdf fitting.
    :return: The statistic map for image.
    """

    ny,nx = image.shape

    #print(row_indices)

    image_without_planet_mask = np.ones((ny,nx))
    image_without_planet_mask[np.where(np.isnan(image_without_planet))] = 0

    if N is not None:
        r_limit_firstZone,r_min_firstZone,r_max_firstZone = firstZone_radii
        r_limit_lastZone,r_min_lastZone,r_max_lastZone = lastZone_radii

    # Calculate the radial distance of each pixel
    r_grid = abs(x_grid +y_grid*1j)
    th_grid = np.arctan2(x_grid,y_grid)
    if Dth != None:
        Dth_rad = Dth/180.*np.pi

    N_it = row_indices.size
    stat_map = np.zeros((N_it)) + np.nan
    #stdout.write("\r%d" % 0)
    for id,k,l in zip(range(N_it),row_indices,col_indices):
        if 1:#k == 109 and l == 135:
            #stdout.write("\r{0}/{1}".format(id,N_it))
            #stdout.flush()

            x = x_grid[(k,l)]
            y = y_grid[(k,l)]
            #print(x,y)
            r = r_grid[(k,l)]
            th = th_grid[(k,l)]

            if Dr is None:
                if r < r_limit_firstZone:
                    #Calculate stat for pixels close to IWA
                    r_min,r_max = r_min_firstZone,r_max_firstZone
                elif r > r_limit_lastZone:
                    r_min,r_max = r_min_lastZone,r_max_lastZone
                else:
                    dr = N/(4*np.pi*r)
                    r_min,r_max = (r-dr, r+dr)

            else:
                r_min,r_max = (r-Dr, r+Dr)

            if Dth is None:
                where_ring = np.where((r_min< r_grid) * (r_grid < r_max) * image_without_planet_mask)
            else:
                delta_th_grid = np.mod(th_grid - th +np.pi,2.*np.pi)-np.pi
                where_ring = np.where((r_min< r_grid) * (r_grid < r_max) * image_without_planet_mask * \
                                    (abs(delta_th_grid)<(+Dth_rad*50./r)))

            where_ring_masked = np.where((((x_grid[where_ring]-x)**2 +(y_grid[where_ring]-y)**2) > mask_radius*mask_radius))
            #print(np.shape(where_ring_masked[0]))

            data = image_without_planet[(where_ring[0][where_ring_masked],where_ring[1][where_ring_masked])]

            if 0:
                import matplotlib.pyplot as plt
                print(image[k,l])
                im_cpy = copy(image)
                im_cpy[(where_ring[0][where_ring_masked],where_ring[1][where_ring_masked])] = np.nan
                plt.figure(1)
                plt.imshow(im_cpy)
                plt.show()

            if type == "proba":
                cdf_model, pdf_model, sampling, im_histo, center_bins  = get_cdf_model(data)

                cdf_fit = interp1d(sampling,cdf_model,kind = "linear",bounds_error = False, fill_value=1.0)
                stat_map[id] = 1-cdf_fit(image[k,l])
            elif type == "SNR":
                stat_map[id] = image[k,l]/np.nanstd(data)
            elif type == "stddev":
                stat_map[id] = np.nanstd(data)
            #print(probability_map[proba_map_k,l])


    return stat_map


def get_cdf_model(data,interupt_plot = False,pure_gauss=False):
    """
    Calculate a model CDF for some data.

    /!\ This function is for some reason still a work in progress. JB could never decide what the best option was.
    But it should work even if the code is a mess.

    :param data: arrays of samples from a random variable
    :param interupt_plot: Plot the histogram and model fit. It
    :param pure_gauss: Assume gaussian statistic. Do not fit exponential tails.
    :return: (cdf_model,new_sampling,im_histo, center_bins) with:
                cdf_model: The cdf model = np.cumsum(pdf_model)
                pdf_model: The pdf model
                sampling: sampling of pdf/cdf_model
                im_histo: histogram from original data
                center_bins: bin centers for im_histo
    """
    pdf_model,sampling,im_histo,center_bins = get_pdf_model(data,interupt_plot=interupt_plot,pure_gauss=pure_gauss)
    return np.cumsum(pdf_model),pdf_model,sampling,im_histo,center_bins


def get_pdf_model(data,interupt_plot = False,pure_gauss = False):
    """
    Calculate a model PDF for some data.

    /!\ This function is for some reason still a work in progress. JB could never decide what the best option was.
    But it should work even if the code is a mess.

    :param data: arrays of samples from a random variable
    :param interupt_plot: Plot the histogram and model fit. It
    :param pure_gauss: Assume gaussian statistic. Do not fit exponential tails.
    :return: (pdf_model,new_sampling,im_histo, center_bins) with:
                pdf_model: The pdf model
                new_sampling: sampling of pdf_model
                im_histo: histogram from original data
                center_bins: bin centers for im_histo
    """
    im_std = np.std(data)
    #print(im_std)
    bins = np.arange(np.min(data),np.max(data),im_std/5.)
    im_histo = np.histogram(data, bins=bins)[0]


    N_bins = bins.size-1
    center_bins = 0.5*(bins[0:N_bins]+bins[1:N_bins+1])

    g_init = models.Gaussian1D(amplitude=np.max(im_histo), mean=0.0, stddev=im_std)
    fit_g = fitting.LevMarLSQFitter()
    warnings.simplefilter('ignore')
    g = fit_g(g_init, center_bins, im_histo)#, weights=1/im_histo)
    g.stddev = abs(g.stddev)

    right_side_noZeros = np.where((center_bins > (g.mean+2*g.stddev))*(im_histo != 0))
    N_right_bins_noZeros = len(right_side_noZeros[0])
    left_side_noZeros = np.where((center_bins < (g.mean-2*g.stddev))*(im_histo != 0))
    N_left_bins_noZeros = len(left_side_noZeros[0])

    right_side = np.where((center_bins > (g.mean+2*g.stddev)))
    left_side = np.where((center_bins < (g.mean-2*g.stddev)))

    if not pure_gauss:
        if N_right_bins_noZeros < 5:
            where_pos_zero = np.where((im_histo == 0) * (center_bins > g.mean))
            if len(where_pos_zero[0]) != 0:
                right_side_noZeros = (range(where_pos_zero[0][0]-5,where_pos_zero[0][0]),)
                right_side = (range(where_pos_zero[0][0]-5,center_bins.size),)
            else:
                right_side_noZeros = (range(center_bins.size-5,center_bins.size),)
                right_side = right_side_noZeros
            N_right_bins_noZeros = 5

        if N_left_bins_noZeros < 5:
            where_neg_zero = np.where((im_histo == 0) * (center_bins < g.mean))
            if len(where_neg_zero[0]) != 0:
                left_side_noZeros = (range(where_neg_zero[0][len(where_neg_zero[0])-1]+1,where_neg_zero[0][len(where_neg_zero[0])-1]+6),)
                left_side = (range(0,where_neg_zero[0][len(where_neg_zero[0])-1]+6),)
            else:
                left_side_noZeros = (range(0,5),)
                left_side = left_side_noZeros
            N_left_bins_noZeros = 5

        #print(left_side,right_side)
        #print(im_histo[left_side],im_histo[right_side])
        #print(right_side_noZeros,left_side_noZeros)
        #print(im_histo[right_side_noZeros],im_histo[left_side_noZeros])



        #print(N_right_bins_noZeros,N_left_bins_noZeros)
        if N_right_bins_noZeros >= 2:
            alpha0 = (np.log(im_histo[right_side_noZeros[0][N_right_bins_noZeros-1]])-np.log(im_histo[right_side_noZeros[0][0]]))/(center_bins[right_side_noZeros[0][0]]-center_bins[right_side_noZeros[0][N_right_bins_noZeros-1]])
            m_alpha0 = -np.log(im_histo[right_side_noZeros[0][0]])-alpha0*center_bins[right_side_noZeros[0][0]]
            param0_rightExp = (m_alpha0,alpha0)

            LSQ_func = lambda para: LSQ_model_exp((bins[0:bins.size-1])[right_side], im_histo[right_side],para[0],para[1])
            param_fit_rightExp = leastsq(LSQ_func,param0_rightExp)
        else:
            param_fit_rightExp = None
        #print(param0_rightExp,param_fit_rightExp)

        if N_left_bins_noZeros >= 2:
            alpha0 = (np.log(im_histo[left_side_noZeros[0][N_left_bins_noZeros-1]])-np.log(im_histo[left_side_noZeros[0][0]]))/(center_bins[left_side_noZeros[0][0]]-center_bins[left_side_noZeros[0][N_left_bins_noZeros-1]])
            m_alpha0 = -np.log(im_histo[left_side_noZeros[0][0]])-alpha0*center_bins[left_side_noZeros[0][0]]
            param0_leftExp = (m_alpha0,alpha0)

            LSQ_func = lambda para: LSQ_model_exp((bins[0:bins.size-1])[left_side], im_histo[left_side],para[0],para[1])
            param_fit_leftExp = leastsq(LSQ_func,param0_leftExp)
        else:
            param_fit_leftExp = None
        #print(param0_leftExp,param_fit_leftExp)


    new_sampling = np.arange(2*np.min(data),4*np.max(data),im_std/100.)

    if pure_gauss:
        pdf_model = g(new_sampling)
        pdf_model_exp = new_sampling*0
    else:
        pdf_model_gaussian = interp1d(center_bins,np.array(im_histo,dtype="double"),kind = "cubic",bounds_error = False, fill_value=0.0)(new_sampling)


    if not pure_gauss:
        right_side2 = np.where((new_sampling >= g.mean))
        left_side2 = np.where((new_sampling < g.mean))

        #print(g.mean+0.0,g.stddev+0.0)
        pdf_model_exp = np.zeros(new_sampling.size)
        weights = np.zeros(new_sampling.size)
        if param_fit_rightExp is not None:
            pdf_model_exp[right_side2] = model_exp(new_sampling[right_side2],*param_fit_rightExp[0])
            weights[right_side2] = np.tanh((new_sampling[right_side2]-(g.mean+2*g.stddev))/(0.1*g.stddev))
        else:
            weights[right_side2] = -1.

        if param_fit_leftExp is not None:
            pdf_model_exp[left_side2] = model_exp(new_sampling[left_side2],*param_fit_leftExp[0])
            weights[left_side2] = np.tanh(-(new_sampling[left_side2]-(g.mean-2*g.stddev))/(0.1*g.stddev))
        else:
            weights[left_side2] = -1.


        weights = 0.5*(weights+1.0)

        #weights[np.where(weights > 1-10^-3)] = 1


        pdf_model = weights*pdf_model_exp + (1-weights)*pdf_model_gaussian
        #pdf_model[np.where(weights > 1-10^-5)] = pdf_model_exp[np.where(pdf_model > 1-10^-5)]

    if 0:
        import matplotlib.pyplot as plt
        fig = 2
        plt.figure(fig,figsize=(8,8))
        plt.plot(new_sampling, weights, "r")
        #plt.plot(new_sampling, (1-weights), "--r")
        #plt.plot(new_sampling, pdf_model_exp, "g")
        #plt.plot(new_sampling, pdf_model_gaussian, "b")
        #plt.plot(new_sampling, pdf_model, "c") #/np.sum(pdf_model)
        #plt.plot(new_sampling, 1-np.cumsum(pdf_model/np.sum(pdf_model)), "--.")
        ax = plt.gca()
        #ax.set_yscale('log')
        plt.grid(True)
        #plt.ylim((10**-15,100000))
        #plt.xlim((1*np.min(data),2*np.max(data)))
        plt.show()

    if interupt_plot:
        import matplotlib.pyplot as plt
        rcParams.update({'font.size': 20})
        fig = 2
        plt.close(2)
        plt.figure(fig,figsize=(16,8))
        plt.subplot(121)
        plt.plot(new_sampling,pdf_model,'r-',linewidth=5)
        plt.plot(center_bins,g(center_bins),'c--',linewidth=3)
        plt.plot(new_sampling,pdf_model_exp,'g--',linewidth=3)
        plt.plot(center_bins,np.array(im_histo,dtype="double"),'b.', markersize=10,linewidth=3)
        #plt.plot(new_sampling,np.cumsum(pdf_model),'g.')
        plt.xlabel('Metric value')
        plt.ylabel('Number per bin')
        plt.xlim((2*np.min(data),2*np.max(data)))
        plt.grid(True)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax = plt.gca()
        ax.tick_params(axis='x')
        ax.tick_params(axis='y')
        ax.legend(['PDF Model Fit','Central Gaussian Fit','Tails Exponential Fit','Histogram'], loc = 'lower left', fontsize=15)
        ax.set_yscale('log')
        plt.ylim((10**-1,10000))

    pdf_model /= np.sum(pdf_model)

    if interupt_plot:
        host = host_subplot(122, axes_class=AA.Axes)
        par1 = host.twinx()
        p1, = host.plot(new_sampling,pdf_model/(new_sampling[1]-new_sampling[0]),'r-',linewidth=5)
        host.tick_params(axis='x', labelsize=20)
        host.tick_params(axis='y', labelsize=20)
        host.set_ylim((10**-3,10**2))
        host.set_yscale('log')
        p2, = par1.plot(new_sampling,1-np.cumsum(pdf_model),'g-',linewidth=5)
        par1.set_ylabel("False positive rate")
        par1.set_yscale('log')
        par1.set_ylim((10**-4,10.))
        host.axis["left"].label.set_color(p1.get_color())
        par1.axis["right"].label.set_color(p2.get_color())
        plt.xlabel('Metric value')
        plt.ylabel('Probability density')
        plt.xlim((2*np.min(data),2*np.max(data)))
        plt.grid(True)
        plt.legend(['PDF model','Tail distribution'], loc = 'lower left', fontsize=15)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.show()

    return pdf_model,new_sampling,np.array(im_histo,dtype="double"), center_bins