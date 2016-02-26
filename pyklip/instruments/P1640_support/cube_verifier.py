#!/usr/bin/env python
"""
Interactive program to explore cubes and mark them as good or bad
"""


import sys, os
import glob
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons

#plt.ion()

class CubeVerify(object):
    # plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    imax = plt.imshow(np.ones((250,250)), norm=LogNorm())
    plt.title("No cube loaded yet")
    
    axcolor = 'lightgoldenrodyellow'
    axchan = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    axchan.set_xticklabels(range(31))
    schan = Slider(axchan, 'Channel', valinit=0, valmin=0, valmax=31,
                        valfmt='%d')
    nextax = plt.axes([0.8, 0.025, 0.1, 0.04])
    nbutton = Button(nextax, 'Next', color='r', hovercolor='0.975')
    prevax = plt.axes([0.25, 0.025, 0.1, 0.04])
    pbutton = Button(prevax, 'Prev', color='r', hovercolor='0.975')

    good_files = None

    def __init__(self, fitsfiles):
        self.fitsfiles = fitsfiles
        self.fitsfiles.sort()
        self.good_file_mask = []
        self.cube = np.ones((32,250,250))
        
    def initialize_figure(self):
            pass
        
    def check_user_input(self):
        while True:
            try:
                check = raw_input("Keep cube? y/n: ").lower()
                assert(check in ['y','n'])
                return (True if check == 'y' else False)
            except AssertionError:
                print("Bad input: {0}\nPlease try again".format(check))

    def update(val):
        chan = np.int(self.schan.val)
        plt.sca(self.imax.axes)
        plt.imshow(self.cube[chan], norm=LogNorm())
        #fig.canvas.draw_idle()
        plt.draw()

    def next_chan(event):
        schan.val += 1
        schan.val = np.min([schan.val, 31])
        update(schan.val)
    def prev_chan(event):
        self.schan.val -= 1
        self.schan.val = np.max([self.schan.val, 0])
        self.update(self.schan.val)

    schan.on_changed(update)
    nbutton.on_clicked(next_chan)
    pbutton.on_clicked(prev_chan)

        
    def verify_files(self, fitsfiles=None):
        if fitsfiles == None: fitsfiles = self.fitsfiles
        if isinstance(fitsfiles, str):
            fitsfiles = [fitsfiles]

        for f in fitsfiles:
            plt.sca(self.imax.axes)
            plt.cla()
            print("File: {0}".format(os.path.basename(f)))
            hdulist = fits.open(f)
            self.cube = hdulist[0].data
            self.imax.axes.set_title(os.path.basename(f))
            plt.imshow(self.cube[26], norm=LogNorm())
            plt.gcf().canvas.draw_idle()
            self.good_file_mask.append(self.check_user_input())
            self.schan.reset()
            hdulist.close()
            
    def get_good_files(self):
        """
        Apply the good_file_mask to the list of good files
        """
        good_files = [f for (g,f) in zip(self.good_file_mask, self.fitsfiles)
                      if g]

if __name__ == "__main__":
    fitsfiles = glob.glob("/data/home/jaguilar/scratch/data/HD117376/2014-06-13/CUBES/*Occulted_2014-06-13_[0-9][0-9][0-9].fits")
    
    cv = CubeVerify(fitsfiles)


    cv.verify_files()
    cv.cube_mask
    
