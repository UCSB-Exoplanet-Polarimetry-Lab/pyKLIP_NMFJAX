import astropy.io.fits as pyfits
from astropy import wcs
import numpy as np


def gpi_readdata(filepaths):
    """
    Method to open and read a list of GPI data

    Inputs:
        filespaths: a list of filepaths

    Outputs:
        dataset: a dictionary with the following keys. Here N = (number of datacubes) * (number of datacubes per image)
            data: Array of shape (N,y,x) for N images of shape (y,x)
            centers: Array of shape (N,2) for N centers in the format [x_cent, y_cent]
            filenums: Array of size N for the numerical index to map data to file that was passed in
            filenames: Array of size N for the actual filepath of the file that corresponds to the data
            PAs: parallactic angle rotation of the target (used for ADI) [in degrees]
            wvs: wavelength of the image (used for SDI) [in microns]. For polarization data, defaults to "None"
            wcs: wcs astormetry headers for each image.
    """
    #check to see if user just inputted a single filename string
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    #make some lists for quick appending
    data = []
    filenums = []
    filenames = []
    rot_angles = []
    wvs = []
    centers = []
    wcs_hdrs = []
    #extract data from each file
    for index, filepath in enumerate(filepaths):
        cube, center, pa, wv, astr_hdrs = gpi_process_file(filepath)

        data.append(cube)
        centers.append(center)
        rot_angles.append(pa)
        wvs.append(wv)
        filenums.append(np.ones(pa.shape[0]) * index)
        wcs_hdrs.append(astr_hdrs)

        #filename = np.chararray(pa.shape[0])
        #filename[:] = filepath
        filenames.append([filepath for i in range(pa.shape[0])])

    #convert everything into numpy arrays
    #reshape arrays so that we collapse all the files together (i.e. don't care about distinguishing files)
    data = np.array(data)
    dims = data.shape
    data = data.reshape([dims[0] * dims[1], dims[2], dims[3]])
    filenums = np.array(filenums).reshape([dims[0] * dims[1]])
    filenames = np.array(filenames).reshape([dims[0] * dims[1]])
    rot_angles = -(np.array(rot_angles).reshape([dims[0] * dims[1]])) + (90 - 24.5)  # TODO: read from ini file
    wvs = np.array(wvs).reshape([dims[0] * dims[1]])
    wcs_hdrs = np.array(wcs_hdrs).reshape([dims[0] * dims[1]])
    centers = np.array(centers).reshape([dims[0] * dims[1], 2])

    #return as a dictionary to index all of these numpy arrays
    dataset = {'data': data, 'centers': centers, 'filenums': filenums, 'filenames': filenames,
               'PAs': rot_angles, 'wvs': wvs, 'wcs' : wcs_hdrs}

    return dataset


def gpi_process_file(filepath):
    """
    Method to open and parse a GPI file

    Inputs:
        filepath: the file to open

    Outputs: (using z as size of 3rd dimension, z=37 for spec, z=1 for pol (collapsed to total intensity))
        cube: 3D data cube from the file. Shape is (z,281,281)
        center: array of shape (z,2) giving each datacube slice a [xcenter,ycenter] in that order
        parang: array of z of the parallactic angle of the target (same value just repeated z times)
        wvs: array of z of the wavelength of each datacube slice. (For pol mode, wvs = [None])
        astr_hdrs: array of z of the WCS header for each datacube slice
    """

    try:
        hdulist = pyfits.open(filepath)

        #grab the data and header from the first extension
        cube = hdulist[1].data
        exthdr = hdulist[1].header

        #grab the astro header
        w = wcs.WCS(header=exthdr, naxis=[1,2])

        #for spectral mode we need to treat each wavelegnth slice separately
        if exthdr['CTYPE3'].strip() == 'WAVE':
            channels = exthdr['NAXIS3']
            wvs = exthdr['CRVAL3'] + exthdr['CD3_3'] * np.arange(channels) #get wavelength solution
            center = []
            #calculate centers from satellite spots
            for i in range(channels):
                spot0 = exthdr['SATS{wave}_0'.format(wave=i)].split()
                spot1 = exthdr['SATS{wave}_1'.format(wave=i)].split()
                spot2 = exthdr['SATS{wave}_2'.format(wave=i)].split()
                spot3 = exthdr['SATS{wave}_3'.format(wave=i)].split()
                centx = np.mean([float(spot0[0]), float(spot1[0]), float(spot2[0]), float(spot3[0])])
                centy = np.mean([float(spot0[1]), float(spot1[1]), float(spot2[1]), float(spot3[1])])
                center.append([centx, centy])

            parang = np.repeat(exthdr['AVPARANG'], channels) #populate PA for each wavelength slice (the same)
            astr_hdrs = [w.deepcopy() for i in range(channels)] #repeat astrom header for each wavelength slice
        #for pol mode, we consider only total intensity but want to keep the same array shape to make processing easier
        elif exthdr['CTYPE3'].strip() == 'STOKES':
            wvs = [None]
            cube = np.sum(cube, axis=0)  #sum to total intensity
            cube = cube.reshape([1, cube.shape[0], cube.shape[1]])  #maintain 3d-ness
            center = [[exthdr['PSFCENTX'], exthdr['PSFCENTY']]]
            parang = exthdr['AVPARANG']*np.ones(1)
            astr_hdrs = np.repeat(w, 1)
        else:
            raise AttributeError("Unrecognized GPI Mode: %{mode}".format(mode=exthdr['CTYPE3']))
    finally:
        hdulist.close()

    return (cube, center, parang, wvs, astr_hdrs)

def gpi_savedata(filepath, data, astr_hdr=None):
    if astr_hdr is None:
        pyfits.writeto(filepath, data, clobber=True)
    else:
        hdulist = astr_hdr.to_fits()
        hdulist.append(hdulist[0])
        hdulist[1].data = data
        hdulist.writeto(filepath, clobber=True)
        hdulist.close()
