__author__ = 'JB'

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import leastsq
import astropy.io.fits as pyfits
from astropy.modeling import models, fitting
from copy import copy
import warnings
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sys import stdout

def model_exp(x,m,alpha):
    return np.exp(-alpha*x-m)

def LSQ_model_exp(x,y,m,alpha):
    y_model = model_exp(x,m,alpha)
    return (y-y_model)**2/y_model


def model_gauss1D(x,a,m,sigma):
    return a*np.exp(-0.5/sigma*(x-m)**2)

def LSQ_model_gauss1D(x,y,a,m,sigma):
    y_model = model_gauss1D(x,a,m,sigma)
    return (y-y_model)**2#/y_model

def get_pdf_model(data):

    im_std = np.std(data)
    bins = np.arange(np.min(data),np.max(data),im_std/10.)
    im_histo = np.histogram(data, bins=bins)[0]


    N_bins = bins.size-1
    center_bins = 0.5*(bins[0:N_bins]+bins[1:N_bins+1])

    g_init = models.Gaussian1D(amplitude=np.max(im_histo), mean=0.0, stddev=im_std)
    fit_g = fitting.LevMarLSQFitter()
    warnings.simplefilter('ignore')
    g = fit_g(g_init, center_bins, im_histo)
    g.stddev = abs(g.stddev)

    right_side_noZeros = np.where((center_bins > (g.mean+2*g.stddev))*(im_histo != 0))
    N_right_bins_noZeros = len(right_side_noZeros[0])
    left_side_noZeros = np.where((center_bins < (g.mean-2*g.stddev))*(im_histo != 0))
    N_left_bins_noZeros = len(left_side_noZeros[0])

    right_side = np.where((center_bins > (g.mean+2*g.stddev))) #*(im_histo != 0))
    N_right_bins = len(right_side[0])
    left_side = np.where((center_bins < (g.mean-2*g.stddev))) #*(im_histo != 0))
    N_left_bins = len(left_side[0])

    if 0:
        param0_gauss1D = (np.max(im_histo),0.0,im_std)
        LSQ_func = lambda para: LSQ_model_gauss1D(bins[0:bins.size-1], im_histo,para[0],para[1],para[2])
        param_fit_gauss1D = leastsq(LSQ_func,param0_gauss1D)

        right_side_noZeros = np.where((center_bins > (param_fit_gauss1D[0][1]+2*param_fit_gauss1D[0][2]))*( im_histo != 0))
        N_right_bins_noZeros = len(right_side_noZeros[0])
        left_side_noZeros = np.where((center_bins < (param_fit_gauss1D[0][1]-2*param_fit_gauss1D[0][2]))*( im_histo != 0))
        N_left_bins_noZeros = len(left_side_noZeros[0])

        right_side = np.where((center_bins > (param_fit_gauss1D[0][1]+2*param_fit_gauss1D[0][2])))#*( im_histo != 0))
        N_right_bins = len(right_side[0])
        left_side = np.where((center_bins < (param_fit_gauss1D[0][1]-2*param_fit_gauss1D[0][2])))#*( im_histo != 0))
        N_left_bins = len(left_side[0])

        print(param_fit_gauss1D[0][0],param_fit_gauss1D[0][1],param_fit_gauss1D[0][2])
        print(g.amplitude+0.0,g.mean+0.0,g.stddev+0.0)

    if 0:
        fig = 1
        plt.figure(fig,figsize=(8,8))
        plt.plot(center_bins,np.array(im_histo,dtype="double"),'bx-', markersize=5,linewidth=3)
        plt.plot(center_bins,g(center_bins),'g.')
        plt.plot((bins[0:bins.size-1])[right_side_noZeros],(bins[0:bins.size-1])[right_side_noZeros]*0+1,'ro')
        plt.plot((bins[0:bins.size-1])[left_side_noZeros],(bins[0:bins.size-1])[left_side_noZeros]*0+1,'ro')
        #plt.plot(center_bins,model_gauss1D(center_bins,*param_fit_gauss1D[0]),'r--')
        #plt.plot(new_sampling,np.cumsum(pdf_model),'g.')
        plt.xlabel('criterion value', fontsize=20)
        plt.ylabel('Probability of the value', fontsize=20)
        #plt.xlim((-30.* im_std,20.*im_std))
        plt.grid(True)
        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.legend(['flat cube histogram','flat cube histogram (Gaussian fit)','planets'], loc = 'upper right', fontsize=12)
        ax.set_yscale('log')
        plt.ylim((10**-5,100000))
        plt.show()


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

    pdf_model_gaussian = g(new_sampling)


    if 0:
        right_side = np.where((new_sampling >= g.mean))
        left_side = np.where((new_sampling < g.mean))
        print(g.mean+0.0,g.stddev+0.0)
        pdf_model_exp = np.zeros(new_sampling.size)
        weights = np.zeros(new_sampling.size)
        if param_fit_rightExp is not None:
            pdf_model_exp[right_side] = model_exp(new_sampling[right_side],*param_fit_rightExp[0])
            weights[right_side] = np.tanh((new_sampling[right_side]-(g.mean+2*g.stddev))/(0.1*g.stddev))
            plt.plot(np.tanh((new_sampling[right_side]-(g.mean+2*g.stddev))/(0.1*g.stddev)))
            plt.show()
        else:
            weights[right_side] = -1.

        if param_fit_leftExp is not None:
            pdf_model_exp[left_side] = model_exp(new_sampling[left_side],*param_fit_leftExp[0])
            weights[left_side] = np.tanh(-(new_sampling[left_side]-(g.mean-2*g.stddev))/(0.1*g.stddev))
        else:
            weights[left_side] = -1.
    else:
        right_side = np.where((new_sampling > (g.mean+2*g.stddev)))
        left_side = np.where((new_sampling < (g.mean-2*g.stddev)))
        pdf_model_exp = np.zeros(new_sampling.size)
        weights = np.zeros(new_sampling.size)
        if param_fit_rightExp is not None:
            pdf_model_exp[right_side] = model_exp(new_sampling[right_side],*param_fit_rightExp[0])
            weights[right_side] = 1.
        else:
            weights[right_side] = -1.

        if param_fit_leftExp is not None:
            pdf_model_exp[left_side] = model_exp(new_sampling[left_side],*param_fit_leftExp[0])
            weights[right_side] = 1.
        else:
            weights[left_side] = -1.

    weights = 0.5*(weights+1.0)

    #weights[np.where(weights > 1-10^-3)] = 1


    pdf_model = weights*pdf_model_exp + (1-weights)*pdf_model_gaussian
    #pdf_model[np.where(weights > 1-10^-5)] = pdf_model_exp[np.where(pdf_model > 1-10^-5)]

    if 0:
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

    if 0:
        fig = 1
        plt.figure(fig,figsize=(8,16))
        plt.subplot(211)
        plt.plot(center_bins,np.array(im_histo,dtype="double"),'bx-', markersize=5,linewidth=3) #/np.sum(im_histo)/(im_std/10.)
        plt.plot(center_bins,g(center_bins),'g.')#/np.sum(g(center_bins))/(im_std/10.)
        plt.plot(new_sampling,pdf_model,'r--')
        plt.plot(new_sampling,pdf_model_exp,'b--')
        #plt.plot(new_sampling,np.cumsum(pdf_model),'g.')
        plt.xlabel('criterion value', fontsize=20)
        plt.ylabel('Probability of the value', fontsize=20)
        plt.xlim((1*np.min(data),2*np.max(data)))
        plt.grid(True)
        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.legend(['flat cube histogram','flat cube histogram (Gaussian fit)','planets'], loc = 'upper right', fontsize=12)
        ax.set_yscale('log')
        plt.ylim((10**-7,1000))

    pdf_model /= np.sum(pdf_model)

    if 0:
        plt.subplot(212)
        plt.plot(center_bins,np.array(im_histo,dtype="double")/np.sum(im_histo)/(im_std/10.),'bx-', markersize=5,linewidth=3) #
        plt.plot(center_bins,g(center_bins)/np.sum(g(center_bins))/(im_std/10.),'g.')#
        plt.plot(new_sampling,pdf_model,'r--')
        plt.plot(new_sampling,1-np.cumsum(pdf_model),'g.')
        plt.xlabel('criterion value', fontsize=20)
        plt.ylabel('Probability of the value', fontsize=20)
        plt.xlim((1*np.min(data),2*np.max(data)))
        plt.grid(True)
        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.legend(['flat cube histogram','flat cube histogram (Gaussian fit)','planets'], loc = 'upper right', fontsize=12)
        ax.set_yscale('log')
        plt.ylim((10**-17,1000))
        plt.show()

    return pdf_model,new_sampling,np.array(im_histo,dtype="double"), center_bins

def get_cdf_model(data):
    pdf_model,sampling,im_histo,center_bins = get_pdf_model(data)
    return np.cumsum(pdf_model),pdf_model,sampling,im_histo,center_bins


def get_image_PDF(image,IOWA,N = 2000,centroid = None, r_step = None):
    IWA,OWA = IOWA
    ny,nx = image.shape

    image_mask = np.ones((ny,nx))
    image_mask[np.where(np.isnan(image))] = 0

    if centroid is None :
        x_cen = np.ceil((nx-1)/2) ; y_cen = np.ceil((ny-1)/2)
    else:
        x_cen, y_cen = centroid

    # Build the x and y coordinates grids
    x, y = np.meshgrid(np.arange(nx)-x_cen, np.arange(ny)-y_cen)
    # Calculate the radial distance of each pixel
    r_grid = abs(x +y*1j)
    th_grid = np.arctan2(x,y)

    # Define the radii intervals for each annulus
    r0 = IWA
    annuli_radii = []
    if r_step is None:
        while np.sqrt(N/np.pi+r0**2) < OWA:
            annuli_radii.append((r0,np.sqrt(N/np.pi+r0**2)))
            r0 = np.sqrt(N/np.pi+r0**2)
    else:
        while np.sqrt(N/np.pi+r0**2) < OWA:
            annuli_radii.append((r0,np.sqrt(N/np.pi+r0**2)))
            r0 += r_step

    annuli_radii.append((r0,np.max([ny,nx])))
    N_annuli = len(annuli_radii)



    pdf_list = []
    cdf_list = []
    sampling_list = []
    annulus_radii_list = []
    for it, rminmax in enumerate(annuli_radii):
        r_min,r_max = rminmax
        #print(rminmax)

        where_ring = np.where((r_min< r_grid) * (r_grid < r_max) * image_mask)
        #print(np.size(where_ring[0]))
        if 0:
            image_tmp = copy(image)
            image_tmp[where_ring] = np.nan
            plt.figure(2)
            plt.imshow(image_tmp,interpolation="nearest")
            plt.show()

        data = image[where_ring]
        cdf_model, pdf_model, sampling, im_histo, center_bins  = get_cdf_model(data)

        pdf_list.append(pdf_model)
        cdf_list.append(cdf_model)
        sampling_list.append(sampling)
        annulus_radii_list.append(((r_min+r_max)/2.,r_min,r_max))
        if 0:
            fig = 1
            plt.figure(fig,figsize=(8,8))
            plt.subplot(np.ceil(np.sqrt(N_annuli)),np.ceil(np.sqrt(N_annuli)),it)
            plt.plot(sampling,pdf_model,'b-',linewidth=3)
            plt.plot(sampling,1.-cdf_model,'r-',linewidth=3)
            plt.xlabel('criterion value', fontsize=20)
            plt.ylabel('Probability of the value', fontsize=20)
            plt.grid(True)
            ax = plt.gca()
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            ax.legend(['flat cube histogram','flat cube histogram (Gaussian fit)','planets'], loc = 'upper right', fontsize=12)
            ax.set_yscale('log')
            plt.ylim((10**-7,10))

    plt.show()



    return pdf_list, cdf_list, sampling_list, annulus_radii_list


def get_image_probability_map(image,image_without_planet,IOWA,N = 2000,centroid = None, r_step = 5):
    IWA,OWA = IOWA
    pdf_list, cdf_list, sampling_list, annulus_radii_list = get_image_PDF(image_without_planet,IOWA,N,centroid,r_step=r_step)

    pdf_radii = np.array(annulus_radii_list)[:,0]

    probability_map = np.zeros(image.shape) + np.nan
    ny,nx = image.shape

    # Build the x and y coordinates grids
    x_grid, y_grid = np.meshgrid(np.arange(nx)-centroid[0], np.arange(ny)-centroid[1])

    # Calculate the radial distance of each pixel
    r_grid = abs(x_grid +y_grid*1j)

    image_finite = np.where(np.isfinite(image))

    #Build the cdf_models from interpolation
    cdf_interp_list = []
    for sampling,cdf_sampled in zip(sampling_list,cdf_list):
        cdf_interp_list.append(interp1d(sampling,cdf_sampled,kind = "linear",bounds_error = False, fill_value=1.0))

        #f = interp1d(sampling,cdf_sampled,kind = "linear",bounds_error = False, fill_value=1.0)
        #plt.plot(np.arange(-10,10,0.1),f(np.arange(-10,10,0.1)))
        #plt.show()

    for k,l in zip(image_finite[0],image_finite[1]):
        #stdout.flush()
        #stdout.write("\r%d" % k)
        r = r_grid[k,l]

        if r < OWA:
            r_closest_id, r_closest = min(enumerate(pdf_radii), key=lambda x: abs(x[1]-r))


            if (r-r_closest) < 0:
                r_closest_id2 = r_closest_id - 1
            else:
                r_closest_id2 = r_closest_id + 1
            r_closest2 = pdf_radii[r_closest_id2]

            if (r_closest_id2 < 0) or (r_closest_id2 > (pdf_radii.size-1)):
                probability_map[k,l] = 1-cdf_interp_list[r_closest_id](image[k,l])
                #plt.plot(np.arange(-10,10,0.1),cdf(np.arange(-10,10,0.1)))
                #plt.show()
            else:
                probability_map[k,l] = 1-(cdf_interp_list[r_closest_id](image[k,l])*abs(r-r_closest2)+cdf_interp_list[r_closest_id2](image[k,l])*abs(r-r_closest))/abs(r_closest-r_closest2)
        else:
                probability_map[k,l] = 1-cdf_interp_list[pdf_radii.size-1](image[k,l])

    if 0:
        plt.figure(1)
        plt.subplot(1,3,1)
        plt.imshow(np.log10(probability_map),interpolation="nearest")
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.imshow(image,interpolation="nearest")
        plt.subplot(1,3,3)
        plt.imshow(image_without_planet,interpolation="nearest")
        plt.show()

    return -np.log10(probability_map)



def pdf_model_gaussAsymExp(x,m,var,var2,m_exp,alpha,beta):
    #m,var,var2,m_exp,alpha,beta = param

    pdf = np.zeros(x.shape)

    where_x_is_greater_than_m = np.where(x >= m)
    where_x_is_lower_than_m = np.where(x < m)

    pdf[where_x_is_greater_than_m] = np.exp(-(x[where_x_is_greater_than_m]-m)**2/(2*var)*np.exp(-x[where_x_is_greater_than_m]**2/(2*var2)) \
                                            - (alpha*(x[where_x_is_greater_than_m]-m)+m_exp)*(1.-np.exp(-x[where_x_is_greater_than_m]**2/(2*var2))) )
    pdf[where_x_is_lower_than_m] = np.exp(-(x[where_x_is_lower_than_m]-m)**2/(2*var)*np.exp(-x[where_x_is_lower_than_m]**2/(2*var2)) \
                                          - (-beta*(x[where_x_is_lower_than_m]-m)+m_exp)*(1.-np.exp(-x[where_x_is_lower_than_m]**2/(2*var2))) )

    pdf /= np.sum(pdf)

    return pdf

def MIN_pdf_model_gaussAsymExp(x,y,m,var,var2,m_exp,alpha,beta):
    pdf_model = pdf_model_gaussAsymExp(x,m,var,var2,m_exp,alpha,beta)
    return np.sum((y-pdf_model)**2/pdf_model)

def LSQ_pdf_model_gaussAsymExp(x,y,m,var,var2,m_exp,alpha,beta):
    pdf_model = pdf_model_gaussAsymExp(x,m,var,var2,m_exp,alpha,beta)
    return (y-pdf_model)**2/pdf_model


"""
        if 1:
            print(rminmax)
            im_histo_max = np.max(im_histo)

            g_init = models.Gaussian1D(amplitude=np.max(im_histo), mean=0.0, stddev=im_std)
            fit_g = fitting.LevMarLSQFitter()
            warnings.simplefilter('ignore')
            g = fit_g(g_init, bins[0:bins.size-1], im_histo)

            #m,var,var2,m_exp,alpha,beta = param
            print(g.amplitude,g.mean,g.stddev)

            #im_histo_tmp = cpy(im_histo)
            right_side = np.where(bins[0:bins.size-1] > (g.mean+2*g.stddev))
            left_side = np.where(bins[0:bins.size-1] < (g.mean-2*g.stddev))
            #im_histo_tmp[)]

            param0_rightExp = (0,0)
            param_fit_rightExp,pcov = curve_fit(pdf_model_exp, (bins[0:bins.size-1])[right_side], im_histo[right_side], p0=param0_rightExp, sigma=None)
            param0_leftExp = (0,0)
            param_fit_leftExp,pcov = curve_fit(pdf_model_exp, (bins[0:bins.size-1])[left_side], im_histo[left_side], p0=param0_leftExp, sigma=None)
            print(param_fit_rightExp,param_fit_leftExp)

            LSQ_func = lambda para: LSQ_pdf_model_exp((bins[0:bins.size-1])[right_side], im_histo[right_side],para[0],para[1])
            param_fit_rightExp2 = leastsq(LSQ_func,param0_rightExp)
            LSQ_func = lambda para: LSQ_pdf_model_exp((bins[0:bins.size-1])[left_side], im_histo[left_side],para[0],para[1])
            param_fit_leftExp2 = leastsq(LSQ_func,param0_leftExp)


            #param0 = (g.mean+0.0,g.stddev+0.0,g.stddev+0.0,0,0,0)
            #param_fit,pcov = curve_fit(pdf_model_gaussAsymExp, bins[0:bins.size-1], im_histo, p0=param0, sigma=None)
            #print(param_fit)

            #param0 = [g.mean+0.0,g.stddev+0.0,g.stddev+0.0,0,0,0]
            #MIN_func = lambda para: MIN_pdf_model_gaussAsymExp(bins[0:bins.size-1],im_histo,para[0],para[1],para[2],para[3],para[4],para[5])
            #param_fit2 = minimize(MIN_func,param0, method='BFGS').x

            param0 = [g.mean+0.0,g.stddev+0.0,g.stddev+0.0,0,0,0]
            LSQ_func = lambda para: LSQ_pdf_model_gaussAsymExp(bins[0:bins.size-1],im_histo,para[0],para[1],para[2],para[3],para[4],para[5])
            param_fit3 = leastsq(LSQ_func,param0)
            print(param_fit3)

            #LSQ_func = lambda para: para1**

            fig = 1
            plt.figure(fig,figsize=(8,8))
            plt.plot(bins[0:bins.size-1],im_histo,'bx-', markersize=5,linewidth=3)
            plt.plot(bins[0:bins.size-1],g(bins[0:bins.size-1]),'c--',linewidth=1)
            #plt.plot(bins[0:bins.size-1],pdf_model_gaussAsymExp(bins[0:bins.size-1],*param_fit),'r-',linewidth=2)
            #plt.plot(bins[0:bins.size-1],pdf_model_gaussAsymExp(bins[0:bins.size-1],*param_fit2),'g-',linewidth=2)
            plt.plot(bins[0:bins.size-1],pdf_model_gaussAsymExp(bins[0:bins.size-1],*param_fit3[0]),'p-',linewidth=2)
            plt.plot((bins[0:bins.size-1])[right_side],im_histo[right_side],'r.', markersize=8)
            plt.plot((bins[0:bins.size-1])[left_side],im_histo[left_side],'g.', markersize=8)
            plt.plot((bins[0:bins.size-1])[right_side],pdf_model_exp((bins[0:bins.size-1])[right_side],*param_fit_rightExp),'r-',linewidth=2)
            plt.plot((bins[0:bins.size-1])[left_side],pdf_model_exp((bins[0:bins.size-1])[left_side],*param_fit_leftExp),'g-',linewidth=2)
            plt.plot((bins[0:bins.size-1])[right_side],pdf_model_exp((bins[0:bins.size-1])[right_side],*param_fit_rightExp2[0]),'r--',linewidth=2)
            plt.plot((bins[0:bins.size-1])[left_side],pdf_model_exp((bins[0:bins.size-1])[left_side],*param_fit_leftExp2[0]),'g--',linewidth=2)

            plt.xlabel('criterion value', fontsize=20)
            plt.ylabel('Probability of the value', fontsize=20)
            plt.xlim((-10.* im_std,10.*im_std))
            plt.grid(True)
            ax = plt.gca()
            #ax.text(10.*im_std, 2.0*im_histo_max/5., str(N_high_SNR_planets),
            #        verticalalignment='bottom', horizontalalignment='right',
            #        color='red', fontsize=50)
            #ax.text(3.*im_std, 2.0*im_histo_max/5., str(N_low_SNR_planets),
            #        verticalalignment='bottom', horizontalalignment='right',
            #        color='red', fontsize=50)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            ax.legend(['flat cube histogram','flat cube histogram (Gaussian fit)','planets'], loc = 'upper right', fontsize=12)
            #plt.savefig(outputDir+"histo_"+filename+".png", bbox_inches='tight')
            #plt.clf()
            #plt.close(fig)
            ax.set_yscale('log')
            plt.ylim((10**-7,1))
            plt.show()


"""