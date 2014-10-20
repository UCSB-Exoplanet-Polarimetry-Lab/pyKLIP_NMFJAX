import numpy as np
import itertools

"""
Most of these are just direct translations Python into of Laurent's Mathematica code
"""

def _create_diaphragm_offset(R1, n, n0, m0):
    """
    Create an n-x-n square with a mask of ones centered at (n0,m0) with radius R1
    """
    arr = np.zeros(shape=(n,n))
    rows,cols = np.indices(arr.shape)
    arr[np.sqrt((rows+1-0.5-n0)**2+(cols+1-0.5-m0)**2)<R1]=1
    return arr

def _remove_hot_pixels(image, neighbor_dist, threshold):
    """
    Calculate the standard deviation of each pixel with its neighbors up to neighbor_dist away
    If a pixel is above some threshold, median-filter it.
    Pseudocode:
    1. Make a series of 2D arrays the same shape as image, but shifted in x and y
    2. Compute the median and std dev along the z-axis
    3. bad pixels are where the image pix is > median + thresh*stdev
    4. apply median filter to all the bad_pix
    """
    
    shift_table=list(itertools.permutations(np.arange(-neighbor_dist,neighbor_dist+1),2))
    for i in np.arange(-neighbor_dist,neighbor_dist+1): shift_table.append((i,i))
    shifted_images = [np.roll(np.roll(image, shift=shift[0], axis=0), shift=shift[1], axis=1) 
                       for shift in shift_table]
    shifted_images.append(image)
    stdev = shifted_images.std(axis=0)
    median = np.median(shifted_images, axis=0)
    test = np.abs(image - median) - threshold*stdev
    bad_pix = np.where(test > 0)
    # replace bad pixels 
    out = image.copy()
    out[bad_pix] = median[bad_pix]
    return out

def _make_scaled_SFT(Hum, Npix, a):
    """
    dxtmp=N[1/Npix];
    d\[Xi]tmp=\[Alpha]//N;
    xstmp=Table[k*dxtmp+dxtmp/2,{k,-Npix/2,Npix/2-1}];
    ystmp=xstmp;
    \[Xi]stmp=Table[k*d\[Xi]tmp+d\[Xi]tmp/2,{k,-Npix/2,Npix/2-1}];
    \[Eta]stmp=\[Xi]stmp;
    x\[Xi]tmp=Transpose[{xstmp}].{\[Xi]stmp};
    y\[Eta]tmp=Transpose[{ystmp}].{\[Eta]stmp};
    expx\[Xi]tmp=Exp[-2\[Pi] I x\[Xi]tmp];
    expy\[Eta]tmp=Exp[-2\[Pi] I y\[Eta]tmp];
    testSFT=Npix*Transpose[expx\[Xi]tmp].(Hum.expy\[Eta]tmp)*dxtmp^2
    """
    pass

def _cost_scale2(a, FTtmpRef, tmpCheck, Npix, MaskCorrelationScale):
    """
    FTtmpCheck=Abs[MakeScaledSFT[tmpCheck,Npix,\[Alpha]]]*MaskCorrelationScale;
    Err=1-Total[Flatten[FTtmpRef*FTtmpCheck]]/Sqrt[Total[Flatten[FTtmpRef^2]]*Total[Flatten[FTtmpCheck^2]]]
    """
    pass

def _get_scaling(cube):
    """
    Input:
        Npix x Npix x Nlambda datacube
    Output:
        array of scaling factors for the wavelengths
    """
    pass

def _get_scaling_tmp(wlsolfile, refwl=22):
    """
    Temporary solution before we get the actual scaling calculation going. 
    Input: 
        wlsolfile: a file containing the centers of the wavelength bins
        refwl: (default 23): unscaled wavelength; all other wavelengths get scaled on top of this one
    Output:
        A dim=(Nlambda,) array of scaling factors wavelength/refwl
    """
    wlsol = np.genfromtxt(wsolfile)
    wavelengths = wsol[:,0]
    scaling = wavelengths/wavelengths[refwl]
    return scaling

def _get_center_offsets(cube, scaling):
    """
    Input:
        Npix x Npix x Nlambda datacube
    Output:
       array of (x,y) tuples giving the center offsets for each wavelength slice
    """
    pass
