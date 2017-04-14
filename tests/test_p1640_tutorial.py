import os
import subprocess
import glob
import sys
from time import time

def test_p1640_tutorial():
    """
    Running throught the P1640 tutorial without the interactive parts. 
    """
    directory = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + os.path.join('..','pyklip','instruments','P1640_support','tutorial')
    tarball_get = 'wget https://sites.google.com/site/aguilarja/otherstuff/pyklip-tutorial-data/P1640_tutorial_data.tar.gz'
    tarball_command = 'tar -xvf P1640_tutorial_data.tar.gz'
    #Note: the tarball command on the tutorial is wrong. 

    # time it
    t1 = time()

    status = os.chdir(directory)
    status = os.system(tarball_get)
    status = os.system(tarball_command)
    filelist=glob.glob("*Occulted*fits") #this line is wrong in the docs too. but only because the tarball command is
    assert(len(filelist) == 3) #should have 3 files in the directory after downloading and unzipping the tarball.
    #we are ignoring interactive.
    #vet the datacubes
    # os.system("echo y | python ../P1640_cube_checker.py --files ${filelist}")
    # good_cubes=['pyklip/pyklip/instruments/P1640_support/tutorial/tutorial_Occulted_2001-01-01_001.fits',
    #             'pyklip/pyklip/instruments/P1640_support/tutorial/tutorial_Occulted_2001-01-01_002.fits',
    #             'pyklip/pyklip/instruments/P1640_support/tutorial/tutorial_Occulted_2001-01-01_003.fits']

    good_cubes = []
    for file in filelist:
        good_cubes.append(os.path.abspath(file))
    #Note: Under Fit grid spots there is a typo. "Grid spots MUST exist, and (for now) the[THEY] MUST be in the normal orientation."
    #Fit grid spots
    # fit_grid_spots_path = os.getcwd() + os.path.sep + '..'
    # sys.path.append(fit_grid_spots_path)
    # sys.path.append("..")
    import pyklip.P1640spots as P1640spots
    spot_filepath = 'shared_spot_folder/'
    spot_filesuffix = '-spot'
    spot_fileext = 'csv'
    for test_file in good_cubes:
        spot_positions = P1640spots.get_single_file_spot_positions(test_file, rotated_spots=False)
        P1640spots.write_spots_to_file(test_file, spot_positions, spot_filepath,
                                      spotid=spot_filesuffix, ext=spot_fileext,  overwrite=False)
    
    #Again, ignoring interactive     
    #vet grid spots
    # good_spots = P1640_cube_checker.run_spot_checker(good_cubes, spot_path='shared_spot_folder/')
    # os.system("echo y | python ../P1640_cube_checker --files ${good_cubes} --spots --spot_path shared_spot_folder")
    #capital Y and N. 

    #run KLIP in SDI mode
    # sys.path.append(os.path.join('..','..','..','..'))
    # run_KLIP_path = os.getcwd() + os.path.sep + os.path.join('..','..','..','..')
    # sys.path.append(run_KLIP_path)
    import pyklip.instruments.P1640 as P1640
    import pyklip.parallelized as parallelized
    dataset = P1640.P1640Data(filelist, spot_directory="shared_spot_folder/")
    parallelized.klip_dataset(dataset, outputdir="output/", fileprefix="woohoo", annuli=5, subsections=4, movement=3, numbasis=[1,20,100], calibrate_flux=False, mode="SDI")

    p1640_globbed = glob.glob("output/*")
    assert(len(p1640_globbed) == 4)


    print("{0} seconds to run".format(time()-t1))

if __name__ == "__main__":
    test_p1640_tutorial()