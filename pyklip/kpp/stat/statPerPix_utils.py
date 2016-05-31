__author__ = 'JB'


import itertools

from scipy.optimize import leastsq
from astropy.modeling import models, fitting
from matplotlib import rcParams
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

import numpy as np
from copy import copy

from pyklip.kpp.utils.mathfunc import *
from pyklip.kpp.utils.multiproc import *
from pyklip.kpp.utils.GPIimage import *
from pyklip.kpp.stat.stat_utils import *


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
    :param Dr: If not None defines the width of the ring as Dr. N is then ignored if Dth is defined as well.
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

    # if N is not None:
    #     r_min_firstZone,r_max_firstZone = (IWA,np.sqrt(N/np.pi+IWA**2))
    #     r_limit_firstZone = (r_min_firstZone + r_max_firstZone)/2.
    #     r_min_lastZone,r_max_lastZone = (OWA,np.max([ny,nx]))
    #     r_limit_lastZone = OWA - N/(4*np.pi*OWA)
    # else:
    #     r_min_firstZone,r_max_firstZone,r_limit_firstZone = None,None,None
    #     r_min_lastZone,r_max_lastZone,r_limit_lastZone = None,None,None

    stat_map = np.zeros(image.shape) + np.nan
    if N_threads is None:
        N_threads = mp.cpu_count()

    if N_threads != -1:
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
                       itertools.repeat((None,None,None)),# itertools.repeat((r_limit_firstZone,r_min_firstZone,r_max_firstZone)),
                       itertools.repeat((None,None,None)),# itertools.repeat((r_limit_lastZone,r_min_lastZone,r_max_lastZone)),
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
                                                               (None,None,None),#(r_limit_firstZone,r_min_firstZone,r_max_firstZone),
                                                               (None,None,None),#(r_limit_lastZone,r_min_lastZone,r_max_lastZone),
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
    :param firstZone_radii: (DISABLED) When N is not None it contains the meam_radius, the min radius and the max radius defining
                        the first sector. The first sector in that case has includes roughly N pixels. For pixel too
                        close to the inner edge this sector is taken by default.
    :param lastZone_radii: (DISABLED) Same as firstZone_radii for the outer edge.
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

    # if N is not None:
    #     r_limit_firstZone,r_min_firstZone,r_max_firstZone = firstZone_radii
    #     r_limit_lastZone,r_min_lastZone,r_max_lastZone = lastZone_radii

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
                # if r < r_limit_firstZone:
                #     #Calculate stat for pixels close to IWA
                #     r_min,r_max = r_min_firstZone,r_max_firstZone
                # elif r > r_limit_lastZone:
                #     r_min,r_max = r_min_lastZone,r_max_lastZone
                # else:
                #     dr = N/(4*np.pi*r)
                #     r_min,r_max = (r-dr, r+dr)
                dr = N/(4*np.pi*r)
                r_min,r_max = (r-dr, r+dr)

            else:
                r_min,r_max = (r-Dr, r+Dr)

            if Dth is None:
                if N is None:
                    where_ring = np.where((r_min< r_grid) * (r_grid < r_max) * image_without_planet_mask)
                else:
                    N_ring = np.pi*(r_max**2-r_min**2)
                    Dth_rad = np.pi*(N/N_ring)
                    # print((N/N_ring),Dth_rad)
                    delta_th_grid = np.mod(th_grid - th +np.pi,2.*np.pi)-np.pi
                    where_ring = np.where((r_min< r_grid) * (r_grid < r_max) * image_without_planet_mask * \
                                        (abs(delta_th_grid)<Dth_rad))
                    # import matplotlib.pyplot as plt
                    # im_cpy = copy(image)
                    # im_cpy[where_ring] = 1000
                    # plt.figure(1)
                    # plt.imshow(im_cpy)
                    # plt.show()
                    # print(where_ring)
            else:
                delta_th_grid = np.mod(th_grid - th +np.pi,2.*np.pi)-np.pi
                where_ring = np.where((r_min< r_grid) * (r_grid < r_max) * image_without_planet_mask * \
                                    (abs(delta_th_grid)<(Dth_rad*50./r)))

            where_ring_masked = np.where((((x_grid[where_ring]-x)**2 +(y_grid[where_ring]-y)**2) > mask_radius*mask_radius))

            # print(where_ring_masked)
            data = image_without_planet[(where_ring[0][where_ring_masked],where_ring[1][where_ring_masked])]

            if 0:#(k == 135 and l == 155) or (k == 162 and l == 165) or (k == 139 and l == 133) or (k == 165 and l == 132) :
                import matplotlib.pyplot as plt
                print(image[k,l])
                im_cpy = copy(image)
                im_cpy[(where_ring[0][where_ring_masked],where_ring[1][where_ring_masked])] = 1000
                #im_cpy[where_ring] = 1000
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
