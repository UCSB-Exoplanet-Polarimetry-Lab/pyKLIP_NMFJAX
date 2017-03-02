import os
import subprocess
import glob
import sys
from time import time


directory = os.getcwd()
print(directory)
directory = directory[:-5] + '/pyklip/instruments/P1640_support/tutorial'
tarball_get = 'wget https://sites.google.com/site/aguilarja/otherstuff/pyklip-tutorial-data/P1640_tutorial_data.tar.gz'
tarball_command = 'tar -xvf P1640_tutorial_data.tar.gz'
#Note: the tarball command on the tutorial is wrong. 

def test_p1640_tutorial():

    # time it
    t1 = time()

    status = os.chdir(directory)
    status = os.system(tarball_get)
    status = os.system(tarball_command)
    if(status != 0):
        raise SyntaxError('Something has gone horribly wrong. And all we\'ve done so far is change directory, download the tarball, and unpack it.')
    filelist=glob.glob("*Occulted*fits") #this line is wrong too. but only because the tarball command is

    #we are ignoring interactive.
    #vet the datacubes
    # os.system("echo y | python ../P1640_cube_checker.py --files ${filelist}")
    # good_cubes=['pyklip/pyklip/instruments/P1640_support/tutorial/tutorial_Occulted_2001-01-01_001.fits',
    #             'pyklip/pyklip/instruments/P1640_support/tutorial/tutorial_Occulted_2001-01-01_002.fits',
    #             'pyklip/pyklip/instruments/P1640_support/tutorial/tutorial_Occulted_2001-01-01_003.fits']


    #Note: Under Fit grid spots there is a typo. "Grid spots MUST exist, and (for now) the[THEY] MUST be in the normal orientation."
    #Fit grid spots
    # sys.path.append("..")
    # import P1640spots
    # spot_filepath = 'shared_spot_folder/'
    # spot_filesuffix = '-spot'
    # spot_fileext = 'csv'
    # for test_file in filelist:
    #     spot_positions = P1640spots.get_single_file_spot_positions(test_file, rotated_spots=False)
    #     P1640spots.write_spots_to_file(test_file, spot_positions, spot_filepath,
    #                                   spotid=spot_filesuffix, ext=spot_fileext,  overwrite=False)
    #vet grid spots
    # good_spots = P1640_cube_checker.run_spot_checker(good_cubes, spot_path='shared_spot_folder/')

    #Again, ignoring interactive 
    # os.system("echo y | python ../P1640_cube_checker --files ${good_cubes} --spots --spot_path shared_spot_folder")
    #capital Y and N. 

    #run KLIP
    sys.path.append("../../../../")
    import pyklip.instruments.P1640 as P1640
    import pyklip.parallelized as parallelized
    dataset = P1640.P1640Data(filelist, spot_directory="shared_spot_folder/")
    parallelized.klip_dataset(dataset, outputdir="output/", fileprefix="woohoo", annuli=5, subsections=4, movement=3, numbasis=[1,20,100], calibrate_flux=False, mode="SDI")

    print("{0} seconds to run".format(time()-t1))

if __name__ == "__main__":
    test_p1640_tutorial()