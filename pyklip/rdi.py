import numpy as np


class PSFLibrary(object):
    """
    This is an PSF Library to use for reference differential imaging

    Attributes:
        master_library (np.ndarray): aligned library of PSFs. 3-D cube of dim = [N, y, x]. Where N is ALL files
        aligned_center (array-like): a (x,y) coordinate specifying common center the library is aligned to
        master_filenames (np.ndarray): array of N filenames for each frame in the library. Should match with
                                pyklip.instruments.Data.filenames for cross-matching
        master_covar (np.ndarray): N x N array of correlations between each 2 frames
        master_wvs (np.ndarray): N wavelengths for each frame
        dataset (pyklip.instruments.Instrument.Data)
        covar (np.ndarray): N_data x M array of correlations between each 2 frames where M are the selected frames
                            and N_data is the number of files in the dataset. Along the N_data dimension, files are
                            ordered in the same way as the dataset object
        isgoodpsf: array of N indicating which M PSFs are good for this dataset

    """

    def __init__(self, data, aligned_center, filenames, covar_matrix=None, wvs=None, compute_covar=False):
        """

        Args:
            data (np.ndarray): a 3-D cube of PSF library files (dim = [N, y, x]) where N is number of files
                   These files should have already been registered to a common center
            aligned_center (array-like): an (x,y) coordinate specifying the common center all files are registered to
            filenames (np.ndarray): a array of N filenames for each file. These should be in the same format as a
                        pyklip.instruments.Instrument.Data.filenames array so that the two can be cross-matched
            covar_matrix (np.ndarray): an N x N matrix that expresses the correlation between each two frames in library
            wvs (np.ndarray): array of N wavelengths that correspond to the wavelengths of the library
            compute_covar (boolean): if True, compute the covariance matrix. Note that this can potentially take a
                                    long time, so you really should be doing it once and saving it
        """
        # call init() of super class
        super(PSFLibrary, self).__init__()

        # generate master list of files and meta data from inputs
        self.master_library = data
        self.aligned_center = aligned_center
        self.master_filenames = filenames
        self.master_covar = covar_matrix
        self.master_wvs = wvs
        # fields in the context of a specific dataset
        self.dataset = None
        self.covar = None
        self.isgoodpsf = None

        # check if covariance matrix was passed in
        if covar_matrix is None and not compute_covar:
            raise AttributeError("You didn't pass in a covariance matrix, which means it needs to be computed. Are you "
                                 "sure you want to do this? (This may take a while if you have 10,000+ files)")
        elif compute_covar:
            self._compute_covar()

    def _compute_covar(self):
        """
        Computes the covariance matrix and saves it in self.covar

        """
        pass

    def save_covar(self, filename, format="numpy"):
        """
        Saves self.covar to a file specified by filename
        Args:
            filename (str): filepath to store the covariance matrix
            format (str): type of file to store the covariance matrix as. Supports numpy?/fits?/pickle? (TBD)

        """
        pass

    def prepare_library(self, dataset, badfiles=None):
        """
        Prepare the PSF Library for an RDI reduction of a specific dataset by only taking the part of the
        library we need.

        Args:
            dataset (pyklip.instruments.Instrument.Data):
            badfiles (np.ndarray): a list of filenames corresponding to bad files we want to also exclude

        Returns:

        """

        # we need to exclude bad files and files already in the dataset itself (since that'd be ADI/SDI/etc)
        in_dataset = np.in1d(self.master_filenames, dataset.filenames)
        # don't compare directly with None
        if badfiles is None:
            badfiles = []
        are_bad = np.in1d(self.master_filenames, badfiles)
        # good ones are the ones that don't fall in either category
        isgood = ~in_dataset & ~badfiles
        good = np.where(isgood)

        # create a view on the good files

        # figure out how the ordering of dataset files are in the PSF library compared to the dataset
        # we want to match the dataset
        filenames_of_dataset_in_lib = self.master_filenames[np.where(in_dataset)]
        dataset_file_indices_in_lib = []
        for filename in filenames_of_dataset_in_lib:
            index = np.where(filenames_of_dataset_in_lib == self.master_filenames)
            dataset_file_indices_in_lib.append(index)
        dataset_file_indices_in_lib = np.array(dataset_file_indices_in_lib)

        # generate a covariance matrix that's N_dataset x N_goodpsfs
        # the ordering of the covariance matrix also ensures that N_dataset is ordered the same as dataset
        self.covar = self.master_covar[dataset_file_indices_in_lib, good]

        # generate a list indicating which files are good
        self.isgoodpsf = isgood