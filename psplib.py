#!/usr/bin/env python3
"""
psplib.py
Greg Szypko, Summer 2019
A library for reading in and manipulating PSP data from .cdf files
"""

import cdflib
import numpy as np
# from matplotlib import pyplot as plt
import datetime
from os import listdir
# from bisect import insort
import traces
import pandas
from pspconstants import *

def unpack_vars(filename, varnamelist):
    "Loads variable arrays listed in varnamelist from filename into a list of arrays"
    outputs = []
    dat = cdflib.CDF(filename)
    for varname in varnamelist:
        outputs.append(dat.varget(varname))
    return outputs

def multi_unpack_vars(directory, varnamelist,startfile=0,endfile=-1):
    "Loads variable arrays listed in varnamelist from all files in directory into a list of arrays"
    filenames = sorted(listdir(directory))
    outputs = unpack_vars(directory+filenames[startfile],varnamelist)
    if endfile == -1:
        endfile = len(filenames) - 1
    for i in range(startfile+1,endfile+1):
        newoutputs = unpack_vars(directory+filenames[i],varnamelist)
        for j in range(0,len(newoutputs)):
            outputs[j] = np.append(outputs[j],newoutputs[j],axis=0)
    return outputs

# def list_file_bounds(directory):
#     filenames = sorted(listdir(directory))
#     bounds = []
#     for i in range(0,len(filenames)):
#         if i == 0:
#             bounds.append((0,1))
#         elif i == len(filenames)-1:
#             bounds.append((i-1,i))
#         else:
#             bounds.append((i-1,i+1))

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

# DEPRECATED IN FAVOR OF uniform_median_filter()
# def time_median_filter(signal,epoch,filter_radius):
#     filtered = np.zeros_like(signal)
#     for i in range(0,len(signal)):
#         curr_window = []
#         #right window
#         j = 1
#         while i + j < len(signal) and epoch[i+j]-epoch[i]<=filter_radius:
#             insort(curr_window,signal[i+j])
#             j+=1
#         #left window
#         k = 1
#         while (i-k) >= 0 and (epoch[i]-epoch[i-k]) <= filter_radius:
#             insort(curr_window,signal[i-k])
#             k+=1
#         if len(curr_window) % 2 == 0:
#             if i + j < len(signal):
#                 insort(curr_window,signal[i+j])
#             elif i - k >= 0:
#                 insort(curr_window,signal[i-k])
#             else:
#                 insort(curr_window,0)
#         filtered[i] = curr_window[len(curr_window)//2]
#     return filtered

def uniform_cadence(signal,epoch,sampling_period):
    """Converts signal array with corresponding epoch array into a pandas Series, at a 
    constant cadence. Samples to sampling_period, the length of time between uniform cadence
    data, calculated by moving average. NOTE: If not given as a datetime.timedelta,
    sampling_period is treated as seconds."""
    tseries = traces.TimeSeries(zip(epoch,signal))
    uniform = tseries.moving_average(sampling_period,pandas=True)
    return uniform

def uniform_median_filter(signal,epoch,sampling_period,filter_window,output_epoch):
    """Converts signal array with corresponding epoch array into a median-filtered pandas Series, at a 
    constant cadence. Samples to sampling_period, the length of time between uniform cadence
    data, calculated by moving average. Filter_window is in number of array elements.
    output_epoch is the array of datetimes corresponding to the array to be output. If not
    specified, epoch is used.
    NOTE: If not given as a datetime.timedelta, sampling_period is treated as seconds."""
#     if output_epoch == None:
#         output_epoch = epoch
    tseries = traces.TimeSeries(zip(epoch,signal))
    uniform = tseries.moving_average(sampling_period,pandas=True)
    filtered = traces.TimeSeries(uniform.rolling(filter_window).median(center=True))
    return resample_variable(filtered,output_epoch)

def get_min_diff(values):
    "Calculates minimum difference between adjacent values in array values"
    curr_min = values[1]-values[0]
    for i in range(1,len(values)-1):
        this_diff = values[i+1]-values[i]
        if this_diff < curr_min:
            curr_min = this_diff
    return curr_min

def resample_variable(signal, epoch):
    """Takes uniform upsampled traces TimeSeries (signal) back to an array of the original cadence, corresponding 
    to the times in epoch (datetime array). Accomplishes this using linear interpolation."""
    signal_out = np.zeros_like(epoch)
    for i in range(0,len(epoch)):
        signal_out[i] = signal.get(epoch[i],'linear')
    return signal_out

def filter_known_transients(epoch, unfiltered_vars):
    """Removes known transient periods from each variable array in unfiltered_vars based on
    known_transients as defined in pspconstants.py"""
    filtered_vars = []
    for unfiltered in unfiltered_vars:
        curr_unfiltered = unfiltered
        for transient in known_transients:
            normal = np.where(np.logical_or(epoch < transient[0], epoch > transient[1]))
            curr_unfiltered = curr_unfiltered[normal]
        filtered_vars.append(curr_unfiltered)
    return filtered_vars

def find_epoch_idx(epoch,latest_value):
    for i in range(0,len(epoch)):
        if epoch[i] > latest_value: return i-1
    return len(epoch) - 1

# modular_median_filter(mag_path, 'psp_fld_mag_rtn', 'psp_fld_mag_epoch', path, 'epoch', precomp_path, 'b_r')
# def modular_median_filter(varpath, varname, varepochname, targetpath, targetepochname, fileoutpath, fileoutname, filter_time):
def modular_median_filter(var_tofilter,var_epoch,target_epoch,filter_time,out_path,out_filename,num_files=16):
    """Filters data and saves out file by file. Cuts down on memory usage."""
    var_length = len(var_tofilter)
    min_diff = get_min_diff(var_epoch)
    filter_window = int(filter_time/min_diff + 2)
    out_epoch_idxs = [0,0]
    print("filter_window: "+str(filter_window))
    for i in range(0,num_files):
        print("Processing file "+str(i))
        idx_bnds = (int(var_length*i/num_files),int(var_length*(i+1)/num_files))
        if i == num_files - 1:
            out_epoch_idxs[1] = len(target_epoch)
        else:
            out_epoch_idxs[1] = find_epoch_idx(target_epoch,var_epoch[idx_bnds[1]])
        window_bnds = (max(0,idx_bnds[0]-int(filter_window/2)),min(len(var_tofilter)-1,idx_bnds[1]+int(filter_window/2)))
        print("idx_bnds: "+str(idx_bnds))
        print("window_bnds: "+str(window_bnds))
        print("out_epoch_idxs: "+str(out_epoch_idxs))
        this_filtered = uniform_median_filter(var_tofilter[window_bnds[0]:window_bnds[1]],\
            var_epoch[window_bnds[0]:window_bnds[1]],min_diff,filter_window,target_epoch[out_epoch_idxs[0]:out_epoch_idxs[1]])
        print("Filtering complete")
#         filt_bnds = (idx_bnds[0]-window_bnds[0],idx_bnds[1]-window_bnds[0])
#         print("filt_bnds: "+str(filt_bnds))
        np.save(out_path+out_filename+str(i).zfill(3),this_filtered)
        print("File saved")
        out_epoch_idxs[0] = out_epoch_idxs[1]

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
































