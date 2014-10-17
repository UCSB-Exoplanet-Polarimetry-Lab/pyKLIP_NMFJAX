import numpy as np

def _remove_hot_pixels(image, NOneSidedShifts, Thres):
    """
    stuff
    """
    nshifts = 2*NOneSidedShifts+1
    ShiftTable=np.array(list(itertools.permutations(
        np.arange(-NOneSidedShifts,NOneSidedShifts+1))
                         ))
    tmp=np.array([np.roll(np.roll(im,p[0]*im.shape[0]),p[1],axis=1) for p in ShiftTable])

def _get_scaling(cube):
    """
    Input:
        Npix x Npix x Nlambda datacube
    Output:
        array of scaling factors for the wavelengths
    """

def _center_cube(cube, scaling):
    """
    """
    pass
