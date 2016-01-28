import abc

class Data(object):
    """
    Abstract Class with the required fields and methods that need to be implemented

    Attributes:
        input: Array of shape (N,y,x) for N images of shape (y,x)
        centers: Array of shape (N,2) for N centers in the format [x_cent, y_cent]
        filenums: Array of size N for the numerical index to map data to file that was passed in
        filenames: Array of size N for the actual filepath of the file that corresponds to the data
        PAs: Array of N for the parallactic angle rotation of the target (used for ADI) [in degrees]
        wvs: Array of N wavelengths of the images (used for SDI) [in microns]. For polarization data, defaults to "None"
        wcs: Array of N wcs astormetry headers for each image.
        IWA: a floating point scalar (not array). Specifies to inner working angle in pixels
        output: Array of shape (b, len(files), len(uniq_wvs), y, x) where b is the number of different KL basis cutoffs
        creator: string for creator of the data (used to identify pipelines that call pyklip)
        klipparams: a string that saves the most recent KLIP parameters

    Methods:
        readdata(): reread in the dadta
        savedata(): save a specified data in the GPI datacube format (in the 1st extension header)
        calibrate_output(): flux calibrate the output data
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        # set field for the creator of the data (used for pipeline work)
        self.creator = None
        # set field for klip parameters
        self.klipparams = None


    ###################################
    ### Required Instance Variances ###
    ###################################

    #Note that each field has a getter and setter method so by default they are all read/write

    @abc.abstractproperty
    def input(self):
        """
        Input Data. Shape of (N, y, x)
        """
        return
    @input.setter
    def input(self, newval):
        return

    @abc.abstractproperty
    def centers(self):
        """
        Image centers. Shape of (N, 2) where the 2nd dimension is [x,y] pixel coordinate (in that order)
        """
        return
    @centers.setter
    def centers(self, newval):
        return

    @abc.abstractproperty
    def filenums(self):
        """
        Array of size N for the numerical index to map data to file that was passed in
        """
        return
    @filenums.setter
    def filenums(self, newval):
        return

    @abc.abstractproperty
    def filenames(self):
        """
        Array of size N for the actual filepath of the file that corresponds to the data
        """
        return
    @filenames.setter
    def filenames(self, newval):
        return


    @abc.abstractproperty
    def PAs(self):
        """
        Array of N for the parallactic angle rotation of the target (used for ADI) [in degrees]
        """
        return
    @PAs.setter
    def PAs(self, newval):
        return


    @abc.abstractproperty
    def wvs(self):
        """
        Array of N wavelengths (used for SDI) [in microns]. For polarization data, defaults to "None"
        """
        return
    @wvs.setter
    def wvs(self, newval):
        return


    @abc.abstractproperty
    def wcs(self):
        """
        Array of N wcs astormetry headers for each image.
        """
        return
    @wcs.setter
    def wcs(self, newval):
        return


    @abc.abstractproperty
    def IWA(self):
        """
        a floating point scalar (not array). Specifies to inner working angle in pixels
        """
        return
    @IWA.setter
    def IWA(self, newval):
        return


    @abc.abstractproperty
    def output(self):
        """
        Array of shape (b, len(files), len(uniq_wvs), y, x) where b is the number of different KL basis cutoffs
        """
        return
    @output.setter
    def output(self, newval):
        return



    ########################
    ### Required Methods ###
    ########################
    @abc.abstractmethod
    def readdata(self, filepaths):
        """
        Reads in the data from the files in the filelist and writes them to fields
        """
        return NotImplementedError("Subclass needs to implement this!")

    @staticmethod
    @abc.abstractmethod
    def savedata(self, filepath, data, klipparams=None, filetype=None, zaxis=None):
        """
        Saves data for this instrument

        Args:
            filepath: filepath to save to
            data: data to save
            klipparams: a string of KLIP parameters. Write it to the 'PSFPARAM' keyword
            filtype: type of file (e.g. "KL Mode Cube", "PSF Subtracted Spectral Cube"). Wrriten to 'FILETYPE' keyword
            zaxis: a list of values for the zaxis of the datacub (for KL mode cubes currently)
        """
        return NotImplementedError("Subclass needs to implement this!")

    @abc.abstractmethod
    def calibrate_output(self):
        """
        Calibrates the flux of the output klipped data.

        Assumes self.output exists and has shape (b,N,y,x) for N is the number of images and b is
        number of KL modes used.
        """
        return NotImplementedError("Subclass needs to implement this!")