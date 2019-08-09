#!/usr/bin/env python3
"""
boxcar.py

"""

"""
dqf_gen.npy

b_mag_spc.npy
n_p_filtered.npy
temp_filtered.npy
v_r_filtered.npy
"""

import numpy as np
import cdflib
from pspconstants import *
import datetime
import psplib
import os

carrlon, scpos = psplib.multi_unpack_vars(path, ['carr_longitude','sc_pos_HCI'])
dist = psplib.compute_magnitudes(scpos/au_km,True)
epoch_ns = psplib.multi_unpack_vars(path, ['epoch'])[0]
epoch = np.array([datetime_t0+datetime.timedelta(seconds=i/1e9) for i in epoch_ns])
dqf = np.load(precomp_path+'dqf_gen.npy')

# carrlon = np.load(precomp_path+"spiral_longitude.npy")

boxcar_path = precomp_path + "boxcar/"
filter_time = datetime.timedelta(hours=1) #1 hour
filenames = ['b_mag_spc.npy','n_p_filtered.npy','temp_filtered.npy','v_r_filtered.npy']

if not os.path.exists(boxcar_path):
    os.makedirs(boxcar_path)

reftime = datetime.datetime(2018, 1, 1) #dummy reference time for datetime averaging
for filename in filenames:
    var = np.load(precomp_path+filename)
    print(filename)
    var_boxcar = []
    time_boxcar = []
    dist_boxcar = []
    long_boxcar = []
    start_time = epoch[0]
    curr_times = []
    curr_vals = []
    curr_dist = []
    curr_long = []
    for i in range(0,len(epoch)):
        if epoch[i] > start_time + filter_time:
            time_boxcar.append(np.mean([time - reftime for time in curr_times])+reftime)
            curr_vals = np.array(curr_vals)
            var_boxcar.append(np.mean(curr_vals[np.where(curr_vals != None)]))
            curr_dist = np.array(curr_dist)
            dist_boxcar.append(np.mean(curr_dist[np.where(curr_dist != None)]))
            curr_long = np.array(curr_long)
            long_boxcar.append(np.mean(curr_long[np.where(curr_long != None)]))
            curr_times = []
            curr_vals = []
            curr_dist = []
            curr_long = []
            start_time = epoch[i]
        curr_dist.append(dist[i])
        curr_times.append(epoch[i])
        curr_vals.append(var[i])
        curr_long.append(carrlon[i])
    np.save(boxcar_path+filename,np.array(var_boxcar))
np.save(boxcar_path+"boxcar_times",np.array(time_boxcar))
np.save(boxcar_path+"boxcar_dists",np.array(dist_boxcar))
np.save(boxcar_path+"boxcar_longs",np.array(long_boxcar))
# np.save(boxcar_path+"boxcar_spiral_longs",np.array(long_boxcar))
