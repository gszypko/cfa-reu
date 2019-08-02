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

carrlon, scpos = psplib.multi_unpack_vars(path, ['carr_longitude','sc_pos_HCI'])
dist = psplib.compute_magnitudes(scpos/au_km,True)
epoch_ns = psplib.multi_unpack_vars(path, ['epoch'])[0]
epoch = np.array([datetime_t0+datetime.timedelta(seconds=i/1e9) for i in epoch_ns])
dqf = np.load(precomp_path+'dqf_gen.npy')

boxcar_path = precomp_path + "boxcar/"
filter_time = datetime.timedelta(hours=1) #1 hour
filenames = ['b_mag_spc.npy','n_p_filtered.npy','temp_filtered.npy','v_r_filtered.npy']

for filename in filenames:
    var = np.load(precomp_path+filename)
    var_boxcar = []
    start_time = epoch[0]
    for time in epoch:
    