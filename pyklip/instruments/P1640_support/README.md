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
#### Recommended (required to run the cube and spot verifier tools)
* matplotlib

#### Installing photutils ####
Instructions for installing photutils can be found here: http://photutils.readthedocs.io/en/latest/photutils/install.html. Note that the conda instructions are outdated - use `conda install -c https://conda.anaconda.org/astropy photutils`


### Steps
The general steps are: 

1. Collect datacubes
1. Vet datacubes
1. Fit grid spots
1. Vet grid spots
1. Run KLIP

A set of tools built into PyKLIP makes this easier to do.

The trickiest part is setting up the grid spot fitting and making sure it succeeds. Once that's done, the grid spot positions can simply be read in from a file. This is described in more detail below.

TODO: Contrast curves and fake injections require unocculted cubes. Currently there is no way to hook these in. Yeah, I want it too. If you want it so bad, do it yourself.


## Tutorial

**Important** This tutorial assumes you are inside the following directory:
`pyklip/pyklip/instruments/P1640_support/tutorial`

#### Living On The Edge Version
If you trust me, you can do only steps "Collect the datacubes", "Fit the gridspots", and "Run KLIP". This skips visual inspection of the datacubes and spot fitting.

The P1640Data class *will* automatically check for the presence of the spot files and, if it doesn't find them, will attempt to do the fitting itself. You're then trusting that the fitting succeeds. It normally does, but generally I like to fit the grid spots first, visually inspect them, and then move on to the KLIP step.  If you don't think you need to do this - or you already have done the grid spot fitting and vetting - then you can move right on to the Run KLIP step. Otherwise, proceed below to fit the grid spots.

### Collect the datacubes
Easy-peasy.

    :::python
        import glob
        filelist = glob.glob("data/*Occulted*fits")

## Vet the datacubes
This uses the cube checker, a separate command-line tool that lets you quickly decide whether or not you should include a particular cube in your reduction.

From an IPython terminal, do: (the syntax here is weird because telling python to evaluate python variables)

    :::python
        %run ../P1640_cube_checker.py {" ".join(filelist)}
      or
        import sys
        sys.path.append("..")
        import P1640_cube_checker
        good_cubes = P1640_cube_checker.run_checker(filelist)
        
Alternatively, from a bash terminal, do:

    :::bash
        filelist=`ls data/*Occulted*fits`
        python ../P1640_cube_checker.py --files ${filelist}

An animation of each cube, along with observing conditions and a comparison to the other cubes in the set, will pop up and the terminal will prompt you Y/N to keep it in the "good cubes" list. If you like the cube, press Y. If you don't, press N. All the Y's will be spit out in a copy-pasteable format at the end. 

### Fit grid spots
Note: you should only need to do this once, after which you can just read in the grid spot positions from a file.

First, re-assemble your handy list of P1640 data. A couple datacubes (with the target information stripped from them) are found in instruments/P1640_support/tutorial/data. I'm going to assume that you are working in the "tutorial" folder.

Grid spots MUST exist, and (for now) the MUST be in the normal orientation. If this isn't true, then the code will hang. 

In order to fit the spots, we need the P1640spots module:

    :::python
        import sys
        sys.path.append("..")
        import P1640spots
        # if the variables below are not set, default values will be read from P1640.ini
        # for the tutorial, let's set them explicitly
        spot_filepath = 'shared_spot_folder/'
        spot_filesuffix = '-spot'
        spot_fileext = 'csv'
        for test_file in filelist:
            spot_positions = P1640spots.get_single_file_spot_positions(test_file, rotated_spots=False)
            P1640spots.write_spots_to_file(test_file, spot_positions, spot_filepath, 
                                          spotid=spot_filesuffix, ext=spot_fileext,  overwrite=False)
                                           
(For now, only normally-oriented gridspots can be used, but in the future you should be able to set rotated_spots=True to fit 45deg-rotated grid spots).

The default values for the spot file filenames and directories (on Dnah at AMNH) can be found in the P1640.ini config file. I tend to write a separate config file specifically for the reduction and define them again there, with a custom directory if I want. An example reduction config file will eventually be added to the repo.

### Vet grid spots
We can run P1640_cube_checker in "spots" mode to check the spots. Usage is similar to before except now you need to use the --spots flag and specify the location of the spot file folder.

From IPython, there are two ways:

    :::python
        %run ../P1640_cube_checker.py --files {" ".join(filelist)} --spots --spot_path shared_spot_folder/
      or
        import sys
        sys.path.append("..")
        import P1640_cube_checker
        good_cubes = P1640_cube_checker.run_spot_checker(filelist, spot_path='shared_spot_folder/')

From bash, do:
    :::bash
        python ../P1640_cube_checker --files ${filelist} --spots --spot_path shared_spot_folder


Again, you will be prompted Y/n for each cube. Y = keep it, N = throw it out. At the end, you will be told all the files for which the spot fitting FAILED and for which it succeeded. You can either try to re-run the fitting, or (more likely) remove that cube from the datacubes that get sent to PyKLIP.

### Run KLIP

Running KLIP on P1640 data is nearly identical to running it on GPI, with the exception that you have to be careful to only use cubes that have corresponding grid spot files. 

Rest of the tutorial to come. The short version is, replace "GPI" with "P1640" in the tutorial in pyklip/README.md. Some modifications are necessary before this will run anywhere but the server at AMNH. Coming soon!