# Project 1640 PyKLIP tutorial

P1640 Instrument class and support code to interface with PyKLIP PSF subtraction.

Author: Jonathan Aguilar

## Overview

The code here defines the instrument class for Project 1640 that interacts with the rest of the PyKLIP module. The Instrument class contains the information that is needed to scale and align the datacubes, and to select the reference slicess. 

### Dependencies
#### Required
* numpy
* scipy
* astropy
* python 2.7 or 3.4
* photutils
#### Recommended
* matplotlib
    
The trickiest part is setting up the grid spot fitting and making sure it succeeds. Once that's done, the grid spot positions can simply be read in from a file. This is described in more detail below.

TODO: Contrast curves and fake injections require unocculted cubes. Currently there is no way to hook these in. Yeah, I want it too. If you want it so bad, do it yourself.

### Steps
The general steps are: 

1. Collect datacubes
1. Vet datacubes
1. Fit grid spots
1. Vet grid spots
1. Run KLIP

A set of tools built into PyKLIP makes this easier to do.

## Tutorial v1: Putting yourself in the hands of fate
The P1640Data class *will* automatically check for the presence of the spot files and, if it doesn't find them, will attempt to do the fitting itself. You're then trusting that the fitting succeeds. It normally does, but generally I like to fit the grid spots first, visually inspect them, and then move on to the KLIP step. 

If you don't think you need to do this - or you already have done the grid spot fitting and vetting - then you can move right on to the Run KLIP step. Otherwise, proceed below to fit the grid spots.

## Collect the datacubes
Easy-peasy.

    :::python
        import glob
        filelist = glob.glob("data/*Occulted*fits")

## Vet the datacubes
This uses the cube checker, a separate command-line tool that lets you quickly decide whether or not you should include a particular cube in your reduction.

Define $PYKLIP_PATH as the path to the root folder of PyKLIP. Then, 

from an IPython terminal, do: (the syntax here is weird because you're mixing python with bash commands)

    :::python
        %run {PYKLIP_PATH}/instruments/P1640_support/P1640_cube_checker.py {" ".join(filelist)}

from a bash terminal, do:
    :::python
        filelist = `ls data/*Occulted*fits`
        python ${PYKLIP_PATH}/instruments/P1640_support/P1640_cube_checker.py ${filelist}

An animation of each cube, along with observing conditions and a comparison to the other cubes in the set, will pop up and the terminal will prompt you Y/N to keep it in the "good cubes" list. If you like the cube, press Y. If you don't, press N. All the Y's will be spit out in a copy-pasteable format at the end. 

## Fit grid spots
First, assemble your handy list of P1640 data. A couple datacubes (with the target information stripped from them) are found in instruments/P1640_support/tutorial/data. I'm going to assume that you are working in the "tutorial" folder.


In order to fit the spots, we need the P1640spots module:

    :::python
        import sys
        sys.path.append(PYKLIP_PATH)
        from pyklip.instruments.P1640 import P1640spots
        for filename in filelist:
            spot_positions = P1640spots.get_single_file_spots(filename, rotated=False)
            spot_filepath = 'shared_spot_folder/'
            spot_filesuffix = '-spot'
            spot_fileext = 'csv'
            P1640spots.write_spots_to_file(filename, spot_positions, spot_filepath, 
                                           overwrite=False, spotid=spot_filesuffix, ext=spot_fileext)
                                           
(For now, only normally-oriented gridspots can be fit, but in the future you should be able to set rotated=True to fit 45deg-rotated grid spots).
The default values for the spot file filenames and directories (on Dnah at AMNH) are found in the P1640.ini config file. I tend to write my own config file specifically for the reduction and define them again there, with a custom directory if I want. An example reduction config file will eventually be added to the repo.

## Vet grid spots
Again, there's a handy-ish command line tool.

From IPython:
    :::python
        filelist = [list of files that you ran grid spot fitting on]
        %run $PYKLIP_PATH/instruments/P1640_support/P1640_spot_checker.py --files {" ".join(filelist)} --spot_path shared_spot_path/

Again, you will be prompted Y/n for each cube. Y = keep it, N = throw it out. At the end, you will be told all the files for which the spot fitting FAILED. You can either try to re-run the fitting, or (more likely) remove that cube from the datacubes that get sent to PyKLIP.

## Run KLIP

Running KLIP on P1640 data is nearly identical to running it on GPI, with the exception that you have to be careful to only use cubes that have corresponding grid spot files. 

Rest of the tutorial to come.