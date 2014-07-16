import pyfits
import numpy as np


def gpi_read_data(filepaths):
    """
    Method to open and read a list of GPI data

    Inputs:
        filespaths: a list of filepaths (as strings)
    """

    #make some lists for quick appending
    data = []
    filenums = []
    filenames = []
    rot_angles = []
    wvs = []
    centers = []
    #extract data from each file
    for index, filepath in enumerate(filepaths):
        cube, center, pa, wv = gpi_process_file(filepath)

        data.append(cube)
        centers.append(center)
        rot_angles.append(pa)
        wvs.append(wvs)
        filenums.append(np.ones(pa.shape[0]) * index)

        filename = np.chararray(pa.shape[0])
        filename[:] = filepath
        filenames.append(filename)

    #convert everything into numpy arrays
    #reshape arrays so that we collapse all the files together (i.e. don't care about distinguishing files)
    data = np.array(data)
    dims = data.shape
    data = data.reshape([dims[0] * dims[1], dims[2], dims[3]])
    filenums = np.array(filenums).reshape([dims[0] * dims[1]])
    filenames = np.array(filenames).reshape([dims[0] * dims[1]])
    rot_angles = np.array(rot_angles).reshape([dims[0] * dims[1]])
    wvs = np.array(wvs).reshape([dims[0] * dims[1]])
    centers = np.array(centers).reshape([dims[0] * dims[1], 2])

    #return as a dictionary to index all of these numpy arrays
    dataset = {'data': data, 'centers': centers, 'filenums': filenums, 'filenames': filenames,
               'PAs': rot_angles, 'wvs': wvs}

    return dataset


def gpi_process_file(filepath):
    '''
    Method to open and parse a GPI file

    Inputs:
        filepath: the file to open
    '''
    hdulist = pyfits.open(filepath)

    #grab the data and header from the first extension
    cube = hdulist[1].data
    exthdr = hdulist[1].header

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

        parang = np.ones(channels) * exthdr['AVPARANG'] #populate PA for each wavelength slice (the same)
    #for pol mode, we consider only total intensity but want to keep the same array shape to make processing easier
    elif exthdr['CTYPE3'].strip() == 'STOKES':
        wvs = [None]
        cube = np.sum(cube, axis=0)  #sum to total intensity
        cube = cube.reshape([1, cube.shape[0], cube.shape[1]])  #maintain 3d-ness
        center = [[exthdr['PSFCENTX'], exthdr['PSFCENTY']]]
        parang = [exthdr['AVPARANG']]
    else:
        raise AttributeError("Unrecognized GPI Mode: %{mode}".format(mode=exthdr['CTYPE3']))

    return (cube, center, parang, wvs)



