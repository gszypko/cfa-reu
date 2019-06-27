#!/usr/bin/env python3
"""
precompute.py
Greg Szypko, Summer 2019

Computes relevant values from PSP data and writes
out to .npy binary files to use for quick plotting
"""

import pickle
import cdflib
import numpy as np
from pspconstants import *
import psplib
import traces
import datetime

mag_filter_time = datetime.timedelta(hours=1) #1 hour

datetime_t0 = datetime.datetime(2000,1,1,12,0,0)
# epoch = np.array([datetime_t0+datetime.timedelta(seconds=i/1e9) for i in epoch_ns])

vp = psplib.multi_unpack_vars(path, ['vp_moment_RTN'])[0]
v_r = vp[:,0]
np.save(precomp_path+'v_r',v_r)
print('v_r computed')

n_p = psplib.multi_unpack_vars(path, ['np_moment'])[0]
np.save(precomp_path+'n_p',n_p)
print('n_p computed')

wp = psplib.multi_unpack_vars(path, ['wp_moment'])[0]
temp = np.square(wp)*(mp_kg/(k_b*1e-6)/3)
np.save(precomp_path+'temp',temp)
print('temp computed')

epoch_ns = psplib.multi_unpack_vars(path, ['epoch'])[0]
epoch = np.array([datetime_t0+datetime.timedelta(seconds=i/1e9) for i in epoch_ns])

b_rtn, epoch_b_ns = psplib.multi_unpack_vars(mag_path, ['psp_fld_mag_rtn','psp_fld_mag_epoch'])
epoch_b = np.array([datetime_t0+datetime.timedelta(seconds=i/1e9) for i in epoch_b_ns])

b_mag = psplib.compute_magnitudes(b_rtn)
b_r = b_rtn[:,0]
b_t = b_rtn[:,1]
b_n = b_rtn[:,2]
print('b components computed')
np.save(precomp_path+'b_mag',b_mag)
np.save(precomp_path+'b_r',b_r)
print('mag cadence b saved')

valid = np.where(abs(b_mag)<1e30)
b_mag, b_r, epoch_b = [b_mag[valid], b_r[valid], epoch_b[valid]]
min_diff = psplib.get_min_diff(epoch_b)

# b_mag_filtered = traces.TimeSeries(psplib.uniform_median_filter(b_mag,epoch_b,min_diff,mag_filter_time / min_diff))
b_mag_filtered = traces.TimeSeries(zip(epoch_b,b_mag))
print('b_mag_filtered created')
b_r_filtered = traces.TimeSeries(psplib.uniform_median_filter(b_r,epoch_b,min_diff,mag_filter_time / min_diff))
print('b_r_filtered computed')
b_mag_spc, b_r_spc = [np.zeros_like(epoch)]*2

print('entering spc-cadence write loop')
for i in range(0,len(epoch)):
    b_mag_spc[i] = b_mag_filtered.get(epoch[i],'linear')
    b_r_spc[i] = b_r_filtered.get(epoch[i],'linear')
print('writing loop complete')
np.save(precomp_path+'b_mag_spc',b_mag_spc)
np.save(precomp_path+'b_r_spc',b_r_spc)
print('spc cadence b saved')
    
# vars = [v_r,n_p,temp,epoch,b_mag_spc,b_r_spc,b_t_spc,b_n_spc]
# varnames = ['v_r','n_p','temp','epoch','b_mag_spc','b_r_spc','b_t_spc','b_n_spc']
# vars = [v_r,n_p,temp,epoch,b_mag_spc,b_r_spc]
# varnames = ['v_r','n_p','temp','epoch','b_mag_spc','b_r_spc']
# for i in range(0,len(varnames)):
#     np.save(precomp_path+varnames[i],vars[i])
