#!/usr/bin/env python3
#psplib.py
#Library for reading in PSP data from .cdf files

import cdflib
import numpy as np
from matplotlib import pyplot as plt
import datetime
from os import listdir
from bisect import insort
import traces
import pandas
import pickle

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

def time_median_filter(signal,epoch,filter_radius):
    filtered = np.zeros_like(signal)
    for i in range(0,len(signal)):
        curr_window = []
        #right window
        j = 1
        while i + j < len(signal) and epoch[i+j]-epoch[i]<=filter_radius:
            insort(curr_window,signal[i+j])
            j+=1
        #left window
        k = 1
        while (i-k) >= 0 and (epoch[i]-epoch[i-k]) <= filter_radius:
            insort(curr_window,signal[i-k])
            k+=1
        if len(curr_window) % 2 == 0:
            if i + j < len(signal):
                insort(curr_window,signal[i+j])
            elif i - k >= 0:
                insort(curr_window,signal[i-k])
            else:
                insort(curr_window,0)
        filtered[i] = curr_window[len(curr_window)//2]
    return filtered

def uniform_cadence(signal,epoch,sampling_period):
    tseries = traces.TimeSeries(zip(epoch,signal))
    uniform = tseries.moving_average(sampling_period,pandas=True)
    return uniform

def uniform_median_filter(signal,epoch,sampling_period,filter_radius):
    tseries = traces.TimeSeries(zip(epoch,signal))
    #NOTE: traces.TimeSeries.moving_average is implemented such that if sampling_period
    #is given as a number and not a timedelta object, it treats it as seconds
    uniform = tseries.moving_average(sampling_period,pandas=True)
    return uniform.rolling(filter_radius).median(center=True)

def get_min_diff(values):
    curr_min = values[1]-values[0]
    for i in range(1,len(values)-1):
        this_diff = values[i+1]-values[i]
        if this_diff < curr_min:
            curr_min = this_diff
    return curr_min

# time = np.append(np.linspace(0.0,2,600),np.linspace(2,4,200)) + np.random.normal(0,0.001,800)
# val1 = np.cos(2*np.pi*time) 
# val2 = val1 + np.random.normal(0,0.1,800)
# val3 = uniform_median_filter(val2,time,0.001,101)
# # val3 = time_median_filter(val2,time,0.2)
# # plt.plot(time,val1)
# # plt.show()
# # plt.plot(time,val2)
# # plt.show()
# # plt.plot(time,val3)
# # plt.show()
# fig = plt.figure(figsize=(12,9))
# fig.add_subplot(111).plot(time,val1)
# fig.add_subplot(111).plot(time,val2)
# val3.plot()
# plt.show()
































