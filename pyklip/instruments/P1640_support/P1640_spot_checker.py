#!/usr/bin/env python
"""
Given a datacube, find the four corresponding spot files. 
Plot the calculated positions on top of the original cube.

Run from an ipython terminal with:
%run spot_checker.py full/path/to/cubes.fits
"""

from __future__ import division

import sys
import os
import glob
import warnings

from multiprocessing import Pool, Process, Queue

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import PatchCollection
from matplotlib.patches import CirclePolygon

from astropy.io import fits

#plt.ion()

sys.path.append(".")
import P1640spots

spot_directory = '/data/p1640/data/users/spot_positions/jonathan/'

# open a fits file and draw the cube
def draw_cube(cube, cube_name, spots):
    """
    Make a figure and draw cube slices on it
    spots are a list of [row, col] positions for each spot
    """
    # mask center for better image scaling
    cube[:,100:150,100:150] = np.nan #cube[:,100:150,100:150]*1e-3
    chan=0
    nchan = cube.shape[0]
    # get star positions
    star_positions = P1640spots.get_single_cube_star_positions(np.array(spots))
    
    #try:
    fig = plt.figure()
    while True:
        plt.cla()
        chan = chan % nchan

        patches1 = [CirclePolygon(xy=spot[chan][::-1], radius=5,
                                  fill=False, alpha=1, ec='k', lw=2)
                    for spot in spots] # large circles centered on spot
        patches2 = [CirclePolygon(xy=spot[chan][::-1], radius=1,
                                  fill=True, alpha=0.3, ec='k', lw=2)
                    for spot in spots] # dots in location of spot
        starpatches = [CirclePolygon(xy=star_positions[chan][::-1], radius=3,
                                     fill=True, alpha=0.3, ec='k', lw=2)
                       for spot in spots] # star position
        patchcoll = PatchCollection(patches1+patches2, match_original=True)
        
        imax = plt.imshow(cube[chan], norm=LogNorm())
        imax.axes.add_collection(patchcoll)
        plt.title("{0}\nChannel {1:02d}".format(cubefile_name, chan))
        
        plt.pause(0.2)
        chan += 1
    #except KeyboardInterrupt:
    #    pass


if __name__ == "__main__":

    if sys.argv[1] == 'help':
        print """
        Usage:
        python cube_checker.py /path/to/data/cube.fits
        OR
        python cube_checker.py space.fits separated.fits sets.fits of.fits paths.fits to.fits cubes.fits
        If running from IPython:
        #files = glob.glob("all/the/fits/*fits")
        %run cube_checker.py {' '.join(files)}
        """
    
    fitsfiles = sys.argv[1:]
    
    good_cubes = dict(zip(fitsfiles, [None for f in fitsfiles]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #fig = plt.figure()
        for i, ff in enumerate(fitsfiles):
            # check file
            if not os.path.isfile(ff):
                print("File not found: {0}".format(ff))
                sys.exit(0)
                
            # get spots
            cubefile_name = os.path.splitext(os.path.basename(ff))[0]
            spot_files = glob.glob(os.path.join(spot_directory, cubefile_name)+"*")
            if len(spot_files) == 0:
                print("No spot files found for {0}".format(os.path.basename(ff)))
                sys.exit(0)
            spots = [np.genfromtxt(f, delimiter=',') for f in spot_files]

            hdulist = fits.open(ff)
            cube = hdulist[0].data
            cube_name = os.path.splitext(os.path.basename(ff))[0]

            # start drawing subprocess
            p = Process(target=draw_cube, args=(cube, cube_name, spots))
            p.start()

            # print cube information
            print "\n{0}/{1} files".format(i+1, len(fitsfiles))
            print "\nCube: {0}".format(cube_name)
            print "\tExposure time: {0}".format(fits.getval(ff, "EXP_TIME"))
            print "\tSeeing: {0}".format(fits.getval(ff, "SEEING"))
            print "\tAirmass: {0}".format(np.mean([fits.getval(ff, "INIT_AM"),
                                                   fits.getval(ff, "FINL_AM")]))
            # ask if cube is good or not
            keep_cube = None
            while keep_cube not in ['y', 'n']:
                keep_cube = raw_input('\t\tKeep? y/n: ').lower()[0]
            good_cubes[ff] = keep_cube

            # close drawing subprocess
            p.terminate()
            p.join()

        plt.close('all')
    for key, val in good_cubes.iteritems():
        if val == 'y': 
            good_cubes[key] = True
        elif val == 'n': 
            good_cubes[key] = False
        else:
            good_cubes[key] = None
    
    #print good_cubes
    print "Good cubes: "
    for i in sorted([key for key, val in good_cubes.iteritems() if val == True]):
        print i










