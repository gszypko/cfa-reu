#!/usr/bin/env python3
#psplib.py
#Library for reading in PSP data from .cdf files

import cdflib
import numpy as np
from matplotlib import pyplot as plt
#import datetime
from os import listdir

def unpack_vars(filename, varnamelist):
    "Loads variable arrays listed in varnamelist from filename into a list of arrays"
    outputs = []
    dat = cdflib.CDF(filename)
    for varname in varnamelist:
        outputs.append(dat.varget(varname))
    return outputs

def multi_unpack_vars(directory, varnamelist):
    "Loads variable arrays listed in varnamelist from all files in directory into a list of arrays"
    filenames = sorted(listdir(directory))
    outputs = unpack_vars(directory+filenames[0],varnamelist)
    for i in range(1,len(filenames)):
        newoutputs = unpack_vars(directory+filenames[i],varnamelist)
        for j in range(0,len(newoutputs)):
            outputs[j] = np.append(outputs[j],newoutputs[j],axis=0)
    return outputs

def compute_magnitudes(scpos, project=False):
    "Takes three-dimensional array of vectors, returns corresponding magnitudes"
    xloc = scpos[:,0]
    yloc = scpos[:,1]
    if project:
        zloc = 0 #projecting all positions down into the solar equatorial plane
    else:
        zloc = scpos[:,2]
    dist = np.sqrt(xloc**2 + yloc**2 + zloc**2)
    return dist

def time_lerp(epoch, epoch_b, b_mag):
    "From values in b_mag at times epoch_b, creates interpolated arrays of values at times epoch"
    b_lerp = []
    for i in range(0,len(epoch)):
#         print(str(i)+' of '+str(len(epoch)))
        idx = np.searchsorted(epoch_b,epoch[i],side='right')
        if epoch[i]==epoch_b[idx-1]:
            b_lerp.append(b_mag[idx-1])
        elif idx == 0 or idx == len(epoch_b):
            b_lerp.append(-1e31)
        else: #linear interpolation case
            delta1 = epoch[i]-epoch_b[idx-1]
            delta2 = epoch_b[idx]-epoch[i]
            delta = epoch_b[idx]-epoch_b[idx-1]
            b_lerp.append((delta1/delta*b_mag[idx-1] + delta2/delta*b_mag[idx]))
    b_lerp = np.array(b_lerp)
    return b_lerp
