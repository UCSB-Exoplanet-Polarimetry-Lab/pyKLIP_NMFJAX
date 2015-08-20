import multiprocessing as mp
import ctypes

import numpy as np

class NoFM():
    def __init__(self, inputs_shape, numbasis):
        """ Initializes teh NoFM class

        Args:
            inputs_shape: shape of the inputs numpy array. Typically (wv, y, x)
            numbasis: 1d numpy array consisting of the number of basis vectors to use

        Returns:
            None
        """
        self.inputs_shape = numbasis
        self.numbasis = numbasis
        self.outputs_shape = inputs_shape + numbasis.shape
        self.need_aux = False


    def alloc_output(self):
        """Allocates shared memory array for final output

        Only use multiprocessing data structors as we are using the multiprocessing class

        Args:

        Returns:
            outputs: mp.array to store final outputs in (shape of (N*wv, y, x, numbasis))
            outputs_shape: shape of outputs array to convert to numpy arrays
        """

        outputs_size = np.prod(np.array(self.inputs_shape)) * np.size(self.numbasis)

        outputs = mp.Array(ctypes.c_double, outputs_size)
        outputs_shape = self.outputs_shape

        return outputs, outputs_shape


    def alloc_interm(self, sector_size, numsciframes):
        """Allocates shared memory array for intermediate step

        Intermediate step is allocated for a sector by sector basis

        Args:
            sector_size: number of pixels in this sector

        Returns:
            interm: mp.array to store intermediate products from one sector in
            interm_shape:shape of interm array (used to convert to numpy arrays)

        """

        interm_size = sector_size * np.size(self.numbasis) * numsciframes

        interm = mp.Array(ctypes.c_double, interm_size)
        interm_shape = [numsciframes, sector_size, np.size(self.numbasis)] # (numframes, size of sector, b)

        return interm, interm_shape


    def alloc_aux(self):
        """Allocates shared memory of an auxilliary array used in the start

        Note: It might be useful to store the pointer to the aux array into the state of this class if you use it
        for easy access

        Args:

        Returns:
            aux: mp.array to store auxilliary data in
            aux_shape: shape of auxilliary array

        """
        self.aux = None
        self.aux_shape = None

        return self.aux, self.aux_shape