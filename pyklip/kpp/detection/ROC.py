__author__ = 'jruffio'
import os
import astropy.io.fits as pyfits
from glob import glob
import multiprocessing as mp
import numpy as np
from scipy.signal import convolve2d

from pyklip.kpp.utils.kppSuperClass import KPPSuperClass
from pyklip.kpp.stat.stat_utils import *
from pyklip.kpp.utils.GOI import *
import pyklip.kpp.utils.mathfunc as kppmath

class ROC(KPPSuperClass):
    """
    Class for SNR calculation.
    """
    def __init__(self,filename,filename_detec,
                 inputDir = None,
                 outputDir = None,
                 mute=None,
                 N_threads=None,
                 label = None,
                 detec_distance = None,
                 ignore_distance = None,
                 GOI_list_folder = None,
                 threshold_sampling = None,
                 overwrite = False):
        """


        :param filename: Filename of the file on which to calculate the metric. It should be the complete path unless
                        inputDir is defined.
                        It can include wild characters. The file will be selected using the first output of glob.glob().
        :param filename_noSignal: One should be careful with this one since it requires it should find the same number
                            of files with no signal than normal images when calling glob.glob().
                            Besides one has to check that the ordering of glob outputs are matching for both lists.
        :param mute: If True prevent printed log outputs.
        :param N_threads: Number of threads to be used for the metrics and the probability calculations.
                        If None use mp.cpu_count().
                        If -1 do it sequentially.
                        Note that it is not used for this super class.
        :param label: Define the suffix to the output folder when it is not defined. cf outputDir. Default is "default".
        """
        # allocate super class
        super(ROC, self).__init__(filename,
                                     inputDir = inputDir,
                                     outputDir = outputDir,
                                     folderName = None,
                                     mute=mute,
                                     N_threads=N_threads,
                                     label=label,
                                     overwrite = overwrite)

        if detec_distance is None:
            self.detec_distance = 2
        else:
            self.detec_distance = detec_distance

        if ignore_distance is None:
            self.ignore_distance = 10
        else:
            self.ignore_distance = ignore_distance

        if threshold_sampling is None:
            self.threshold_sampling = np.linspace(0.0,20,200)
        else:
            self.threshold_sampling = threshold_sampling

        self.filename_detec = filename_detec
        self.GOI_list_folder = GOI_list_folder


    def initialize(self,inputDir = None,
                         outputDir = None,
                         folderName = None,
                         compact_date = None,
                         label = None):
        """
        Initialize the non general inputs that are needed for the metric calculation and load required files.

        For this super class it simply reads the input file including fits headers and store it in self.image.
        One can also overwrite inputDir, outputDir which is basically the point of this function.
        The file is assumed here to be a fits containing a 2D image or a GPI 3D cube (assumes 37 spectral slice).

        Example for inherited classes:
        It can read the PSF cube or define the hat function.
        It can also read the template spectrum in a 3D scenario.
        It could also overwrite this function in case it needs to read multiple files or non fits file.

        :param inputDir: If defined it allows filename to not include the whole path and just the filename.
                        Files will be read from inputDir.
                        Note tat inputDir might be redefined using initialize at any point.
                        If inputDir is None then filename is assumed to have the absolute path.
        :param outputDir: Directory where to create the folder containing the outputs.
                        Note tat inputDir might be redefined using initialize at any point.
                        If outputDir is None:
                            If inputDir is defined: outputDir = inputDir+os.path.sep+"planet_detec_"
        :param folderName: Name of the folder containing the outputs. It will be located in outputDir.
                        Default folder name is "default_out".
                        The convention is to have one folder per spectral template.
                        If the keyword METFOLDN is available in the fits file header then the keyword value is used no
                        matter the input.
        :param label: Define the suffix to the output folder when it is not defined. cf outputDir. Default is "default".

        :return: None
        """
        if not self.mute:
            print("~~ INITializing "+self.__class__.__name__+" ~~")

        # The super class already read the fits file
        init_out = super(ROC, self).initialize(inputDir = inputDir,
                                         outputDir = outputDir,
                                         folderName = folderName,
                                         label=label)


        # Get center of the image (star position)
        try:
            # Retrieve the center of the image from the fits headers.
            self.center = [self.exthdr['PSFCENTX'], self.exthdr['PSFCENTY']]
        except:
            # If the keywords could not be found the center is defined as the middle of the image
            if not self.mute:
                print("Couldn't find PSFCENTX and PSFCENTY keywords.")
            self.center = [(self.nx-1)/2,(self.ny-1)/2]

        if self.label == "CADI":
            self.center = [140,140]

        try:
            self.folderName = self.exthdr["METFOLDN"]
        except:
            pass



        # Check file existence and define filename_path
        if self.inputDir is None:
            try:
                self.filename_detec = os.path.abspath(glob(self.filename_detec)[self.id_matching_file])
                self.N_matching_files = len(glob(self.filename_detec))
            except:
                raise Exception("File "+self.filename_detec+"doesn't exist.")
        else:
            try:
                self.filename_detec_path = os.path.abspath(glob(self.inputDir+os.path.sep+self.filename_detec)[self.id_matching_file])
                self.N_matching_files = len(glob(self.inputDir+os.path.sep+self.filename_detec))
            except:
                raise Exception("File "+self.inputDir+os.path.sep+self.filename_detec+" doesn't exist.")

        with open(self.filename_detec_path, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            csv_as_list = list(reader)
            self.detec_table_labels = csv_as_list[0]
            self.detec_table = np.array(csv_as_list[1::], dtype='string').astype(np.float)
        if not self.mute:
            print("Opened: "+self.filename_detec_path)


        self.N_detec = self.detec_table.shape[0]
        self.val_id = self.detec_table_labels.index("value")
        self.x_id = self.detec_table_labels.index("x")
        self.y_id = self.detec_table_labels.index("y")

        file_ext_ind = os.path.basename(self.filename_detec_path)[::-1].find(".")
        self.prefix = os.path.basename(self.filename_detec_path)[:-(file_ext_ind+1)]
        self.suffix = "ROC"
        return init_out

    def check_existence(self):
        """

        :return: False
        """

        file_exist = (len(glob(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.csv')) >= 1)

        if file_exist and not self.mute:
            print("Output already exist: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.csv')

        if self.overwrite and not self.mute:
            print("Overwriting is turned ON!")

        return file_exist and not self.overwrite


    def calculate(self):
        """

        :param N: Defines the width of the ring by the number of pixels it has to contain
        :return: self.image the imput fits file.
        """
        if not self.mute:
            print("~~ Calculating "+self.__class__.__name__+" with parameters " + self.suffix+" ~~")

        if self.GOI_list_folder is not None:
            x_real_object_list,y_real_object_list = get_pos_known_objects(self.prihdr,self.exthdr,self.GOI_list_folder,xy = True)

        self.false_detec_proba_vec = []
        self.true_detec_proba_vec = []
        # print(x_real_object_list,y_real_object_list)
        # Loop over all the local maxima stored in the detec csv file
        for k in range(self.N_detec):
            val_criter = self.detec_table[k,self.val_id]
            x_pos = self.detec_table[k,self.x_id]
            y_pos = self.detec_table[k,self.y_id]

            #remove the detection if it is a real object
            if self.GOI_list_folder is not None:
                too_close = False
                for x_real_object,y_real_object  in zip(x_real_object_list,y_real_object_list):
                    #print(np.sqrt((x_pos-x_real_object)**2+(y_pos-y_real_object)**2 ),self.detec_distance**2,self.ignore_distance**2)
                    if (x_pos-x_real_object)**2+(y_pos-y_real_object)**2 < self.detec_distance**2:
                        too_close = True
                        self.true_detec_proba_vec.append(val_criter)
                        if not self.mute:
                            print("Real object detected.")
                        break
                    elif (x_pos-x_real_object)**2+(y_pos-y_real_object)**2 < self.ignore_distance**2:
                        too_close = True
                        if not self.mute:
                            print("Local maxima ignored. Too close to known object")
                        break
                if too_close:
                    continue

            self.false_detec_proba_vec.append(val_criter)

        # print(self.false_detec_proba_vec)
        # print(self.true_detec_proba_vec)

        self.N_false_pos = np.zeros(self.threshold_sampling.shape)
        self.N_true_detec = np.zeros(self.threshold_sampling.shape)
        for id,threshold_it in enumerate(self.threshold_sampling):
            self.N_false_pos[id] = np.sum(self.false_detec_proba_vec >= threshold_it)
            self.N_true_detec[id] = np.sum(self.true_detec_proba_vec >= threshold_it)

        # import matplotlib.pyplot as plt
        # plt.figure(1)
        # plt.plot(self.N_false_pos,self.N_true_detec)
        # plt.show()

        return zip(self.threshold_sampling,self.N_false_pos,self.N_true_detec)


    def save(self):
        """

        :return: None
        """

        if not os.path.exists(self.outputDir+os.path.sep+self.folderName):
            os.makedirs(self.outputDir+os.path.sep+self.folderName)

        if not self.mute:
            print("Saving: "+self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.csv')
        with open(self.outputDir+os.path.sep+self.folderName+os.path.sep+self.prefix+'-'+self.suffix+'.csv', 'w+') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerows([["value","N false pos","N true pos"]])
            csvwriter.writerows(zip(self.threshold_sampling,self.N_false_pos,self.N_true_detec))
        return None

    def load(self):
        """

        :return: None
        """

        return None

def gather_ROC(filename_filter,mute = False):
    """
    Build the combined ROC curve from individual frame ROC curve.
    It looks for all the file matching filename_filter using glob.glob and then add each individual ROC to build the
    master ROC.

    Plot master_N_false_pos vs master_N_true_detec to get a ROC curve.

    :param filename_filter: Filename filter with wild characters indicating which files to pick
    :param mute: If True, mute prints. Default is False.
    :return: threshold_sampling,master_N_false_pos,master_N_true_detec:
        threshold_sampling: The metric sampling. It is the curve parametrization.
        master_N_false_pos: Number of false positives as a function of threshold_sampling
        master_N_true_detec: Number of true positives as a function of threshold_sampling
    """
    file_list = glob(filename_filter)

    with open(file_list[0], 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        csv_as_list = list(reader)
        detec_table_labels = csv_as_list[0]
        detec_table = np.array(csv_as_list[1::], dtype='string').astype(np.float)

    threshold_sampling = detec_table[:,0]
    master_N_false_pos = detec_table[:,1]
    master_N_true_detec = detec_table[:,2]

    for filename in file_list[1::]:
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            csv_as_list = list(reader)
            detec_table_labels = csv_as_list[0]
            detec_table = np.array(csv_as_list[1::], dtype='string').astype(np.float)
        if not mute:
            print("Opened: "+filename)

        master_N_false_pos = master_N_false_pos+detec_table[:,1]
        master_N_true_detec = master_N_true_detec+detec_table[:,2]

    return threshold_sampling,master_N_false_pos,master_N_true_detec


def gather_multiple_ROCs(base_dir,filename_filter_list,mute = False):
    """
    Build the multiple combined ROC curve from individual frame ROC curve while making sure they have the same inputs.
    If the folders are organized following the convention below then it will make sure there is a ROC file for each
    filename_filter in each epoch. Otherwise it skips the epoch.

    The folders need to be organized as:
     base_dir/TARGET/autoreduced/EPOCH_Spec/filename_filter

    In the function TARGET and EPOCH are wild characters.

    It looks for all the file matching filename_filter using glob.glob and then add each individual ROC to build the
    master ROC.

    Plot master_N_false_pos vs master_N_true_detec to get a ROC curve.

    :param base_dir: Base directory from which the file search go.
    :param filename_filter: Filename filter with wild characters indicating which files to pick.
    :param mute: If True, mute prints. Default is False.
    :return: threshold_sampling,master_N_false_pos,master_N_true_detec:
        threshold_sampling: The metric sampling. It is the curve parametrization.
        master_N_false_pos: Number of false positives as a function of threshold_sampling
        master_N_true_detec: Number of true positives as a function of threshold_sampling
    """

    N_ROC = len(filename_filter_list)
    threshold_sampling_list = [[]]*N_ROC
    master_N_false_pos_list = [[]]*N_ROC
    master_N_true_detec_list = [[]]*N_ROC

    dirs_to_reduce = os.listdir(base_dir)
    N=0
    for object in dirs_to_reduce:
        if not object.startswith('.'):
            #print(object)

            epochDir_glob = glob(base_dir+object+os.path.sep+"autoreduced"+os.path.sep+"*_Spec"+os.path.sep)

            for epochDir in epochDir_glob:
                inputDir = os.path.abspath(epochDir)

                file_list = []
                for filename_filter in filename_filter_list:
                    try:
                        file_list.append(glob(inputDir+os.path.sep+filename_filter)[0])
                    except:
                        if not mute:
                            print("ROC: {0} unvailable in {1}. Skipping".format(filename_filter,inputDir))

                if len(file_list) == N_ROC:
                    for index,filename in enumerate(file_list):
                        with open(filename, 'rb') as csvfile:
                            reader = csv.reader(csvfile, delimiter=';')
                            csv_as_list = list(reader)
                            detec_table_labels = csv_as_list[0]
                            detec_table = np.array(csv_as_list[1::], dtype='string').astype(np.float)

                        try:
                            master_N_false_pos_list[index] = master_N_false_pos_list[index]+detec_table[:,1]
                            master_N_true_detec_list[index] = master_N_true_detec_list[index]+detec_table[:,2]
                        except:
                            threshold_sampling_list[index] = detec_table[:,0]
                            master_N_false_pos_list[index] = detec_table[:,1]
                            master_N_true_detec_list[index] = detec_table[:,2]



    return threshold_sampling_list,master_N_false_pos_list,master_N_true_detec_list