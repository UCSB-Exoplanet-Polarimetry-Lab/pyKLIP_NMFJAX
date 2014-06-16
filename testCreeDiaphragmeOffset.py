import unittest

import numpy
import itertools

def CreeDiaphragmeOffset(R1, n, n0, m0):
    arr = np.array(shape(n,n))
    row,col = np.indices(arr.shape)
    arr[((rows-0.5-n0)**2+(cols-0.5-m0)**2)<R1]=1
    return arr

class CDOtest(unittest.TestCase):
    def test(self):
        self.assert
