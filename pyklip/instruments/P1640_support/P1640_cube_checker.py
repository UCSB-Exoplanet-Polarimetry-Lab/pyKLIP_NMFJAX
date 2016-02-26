#!/usr/bin/env python
"""
Given a datacube, find the four corresponding spot files. 
Plot the calculated positions on top of the original cube.

Run from an ipython terminal with:
%run spot_checker.py full/path/to/cube.fits
"""

from __future__ import division

import sys
import os
import warnings

from multiprocessing import Pool, Process, Queue

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy.io import fits

#plt.ion()


# use multiple threads - one for drawing the figure, and another for handling user input

"""
Pseudocode:
1. Load list of files
2. Create the "good files" dictionary
3. For each file:
3a. Split offt a thread for drawing the cube
3b. Ask for user input
4. When the user provides 'y' or 'n', update the dictionary and kill the drawing thread
5. Move on to the next file
"""




# open a fits file and draw the cube
def draw_cube(cube, cube_name, header):
    """
    Make a figure and draw cube slices on it
    """
    chan = 14
    nchan = cube.shape[0]
    max_val = np.max(cube)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.tick_right()
    fig.subplots_adjust(left=0.3)
    datainfo = """Exp time: {exptime:.3f}
Seeing: {seeing:>9.3f}
Airmass: {airmass:>8.3f}
Max val: {maxval:>8.1f}
Min val: {minval:>8.1f}""".format(exptime=header["EXP_TIME"],
                              seeing=header["SEEING"],
                              airmass=header["INIT_AM"],
                              maxval=np.nanmax(cube),
                              minval=np.nanmin(cube))
    fig.text(0.01,0.7, datainfo, size='large', family='monospace',
             linespacing=2, 
             verticalalignment='top')
    plt.draw()
    while True:
        plt.cla()
        chan = chan % nchan
        # you're gonna think you want a common scale for all the slices but you're wrong, leave LogNorm alone
        imax = ax.imshow(cube[chan], norm=LogNorm())#vmax=max_val))
        plt.title("{name}\nChannel {ch:02d}".format(name=cube_name, ch=chan))
        plt.pause(0.25)
        chan += 1

if __name__ == "__main__":

    if sys.argv[1] == 'help':
        print """Usage:
        python cube_checker.py /path/to/data/cube.fits
        OR
        python cube_checker.py space.fits separated.fits sets.fits of.fits paths.fits to.fits cubes.fits
        If running from IPython:
        files = glob.glob("all/the/fits/*fits")
        %run cube_checker.py {' '.join(files)}
        """
    
    fitsfiles = sys.argv[1:]
    
    good_cubes = dict(zip(fitsfiles, [None for f in fitsfiles]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #fig = plt.figure()
        repeat = True
        while repeat:
            for i, ff in enumerate(fitsfiles):
                # check file
                if not os.path.isfile(ff):
                    print("File not found: {0}".format(ff))
                    sys.exit(0)

                hdulist = fits.open(ff)
                cube = hdulist[0].data
                cube_name = os.path.splitext(os.path.basename(ff))[0]
                header = hdulist[0].header
                # start drawing subprocess
                p = Process(target=draw_cube, args=(cube, cube_name, header))
                p.start()

                # print cube information

                print "\n{0}/{1} files".format(i+1, len(fitsfiles))
                print "Cube: {0}".format(cube_name)

                # ask if cube is good or not
                keep_cube = None
                while keep_cube not in ['y', 'n']:
                    keep_cube = raw_input('\tKeep? Y/n: ').lower()[0]
                good_cubes[ff] = keep_cube
                hdulist.close()
                # close drawing subprocess
                p.terminate()
                p.join()
            plt.close('all')
            repeat = raw_input("Finished viewing cubes. Print list and quit? Y/n: ").lower()[0]
            if repeat == 'y':
                repeat = False
            else:
                continue
    # convert good_cubes dict to Boolean
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


