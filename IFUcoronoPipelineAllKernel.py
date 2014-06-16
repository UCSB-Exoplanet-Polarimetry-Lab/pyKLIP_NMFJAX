#!/usr/bin/env python

import numpy as np
import itertools
import pandas as pd

def CreeDiaphragmeOffset(R1, n, n0, m0): # checked!
    arr = np.zeros(shape=(n,n))
    rows,cols = np.indices(arr.shape)
    arr[np.sqrt((rows+1-0.5-n0)**2+(cols+1-0.5-m0)**2)<R1]=1
    return arr
    
def RemoveHotPixels(im1, NOneSidedShifts, Thres):
    NShifts=2*NOneSidedShifts+1
    ShiftTable=itertools.permutations(np.arange(-NOneSidedShifts,NOneSidedShifts+1))
    tmp=np.array([RotateRight(im1,ShiftTable[p,0],ShiftTable[p,1]) 
         for p in range(0,size(ShiftTable))])
    devs = np.std(tmp)
    ms = np.median(tmp)
    mms = np.mean(tmp)
    test = np.abs(im1-ms)-Thres*devs
    #Bad=Position[test,_?((#>=0)&)]
    Bad = np.where(test>=0)
    out = im1
    for k in range(len(Bad)):
    	out[Bad[k,1],Bad[k,2]]=ms[Bad[k,1],Bad[k,2]]
	return out