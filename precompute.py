#!/usr/bin/env python3
"""
precompute.py
Greg Szypko, Summer 2019

Computes relevant values from PSP data and writes
out to .npy binary files to use for quick plotting
"""

import cdflib
import numpy as np
from pspconstants import *
import psplib
import traces
import datetime
from os import listdir

filter_time = datetime.timedelta(hours=1) #1 hour

epoch_ns, dqf = psplib.multi_unpack_vars(path, ['epoch','DQF'])
dqf = dqf[:,0]
epoch = np.array([datetime_t0+datetime.timedelta(seconds=i/1e9) for i in epoch_ns])

epoch_min_diff = psplib.get_min_diff(epoch)

# vp = psplib.multi_unpack_vars(path, ['vp_moment_RTN'])[0]
# v_r = vp[:,0]
# np.save(precomp_path+'v_r',v_r)
# print('v_r computed')
v_r_good = np.where(np.logical_and(abs(v_r)<dat_upperbnd,dqf==0))
v_r_filtered = psplib.uniform_median_filter(v_r[v_r_good],epoch[v_r_good],epoch_min_diff,filter_time // epoch_min_diff,epoch)
np.save(precomp_path+'v_r_filtered',v_r)
# 
# n_p = psplib.multi_unpack_vars(path, ['np_moment'])[0]
# np.save(precomp_path+'n_p',n_p)
# print('n_p computed')
n_p_good = np.where(np.logical_and(abs(n_p)<dat_upperbnd,dqf==0))
n_p_filtered = psplib.uniform_median_filter(n_p[n_p_good],epoch,epoch_min_diff,filter_time // epoch_min_diff,epoch)
np.save(precomp_path+'n_p_filtered',n_p)
# 
# wp = psplib.multi_unpack_vars(path, ['wp_moment'])[0]
# temp = np.square(wp)*(mp_kg/(k_b*1e-6)/3)
# np.save(precomp_path+'temp',temp)
# print('temp computed')
temp_good = np.where(np.logical_and(abs(temp)<dat_upperbnd,dqf==0))
temp_filtered = psplib.uniform_median_filter(temp[temp_good],epoch,epoch_min_diff,filter_time // epoch_min_diff,epoch)
np.save(precomp_path+'n_p_filtered',n_p)


# dqf_gen = psplib.multi_unpack_vars(path,['DQF'])[0][:,0]
# np.save(precomp_path+'dqf_gen',dqf_gen)

# b_bounds = psplib.list_file_bounds(mag_path)
filenames = sorted(listdir(mag_path))

for i in range(0,len(filenames)):
    print('\nProcessing file number ' + str(i))
    b_rtn, epoch_b_ns = psplib.unpack_vars(mag_path+filenames[i], ['psp_fld_mag_rtn','psp_fld_mag_epoch'])
    epoch_b = np.array([datetime_t0+datetime.timedelta(seconds=i/1e9) for i in epoch_b_ns])
    
    if i != 0:
        pre_b_rtn, pre_epoch_b_ns = psplib.unpack_vars(mag_path+filenames[i-1], ['psp_fld_mag_rtn','psp_fld_mag_epoch'])
        pre_epoch_b = np.array([datetime_t0+datetime.timedelta(seconds=i/1e9) for i in pre_epoch_b_ns])
        end_pre = pre_epoch_b[-1]
        pre_idx = -2
        while end_pre - pre_epoch_b[pre_idx] < 1.1*filter_time:
            pre_idx -= 1
        b_rtn = np.append(pre_b_rtn[pre_idx-1:],b_rtn,axis=0)
        epoch_b = np.append(pre_epoch_b[pre_idx-1:],epoch_b)
    if i != len(filenames)-1:
        post_b_rtn, post_epoch_b_ns = psplib.unpack_vars(mag_path+filenames[i+1], ['psp_fld_mag_rtn','psp_fld_mag_epoch'])
        post_epoch_b = np.array([datetime_t0+datetime.timedelta(seconds=i/1e9) for i in post_epoch_b_ns])
        start_post = post_epoch_b[0]
        post_idx = 1
        while post_epoch_b[post_idx] - start_post < 1.1*filter_time:
            post_idx += 1
        b_rtn = np.append(b_rtn,post_b_rtn[:post_idx+1],axis=0)
        epoch_b = np.append(epoch_b,post_epoch_b[:post_idx+1])
    
    b_mag = psplib.compute_magnitudes(b_rtn)
    b_r = b_rtn[:,0]
#     b_t = b_rtn[:,1]
#     b_n = b_rtn[:,2]
# print('b components computed')
# np.save(precomp_path+'b_mag',b_mag)
# np.save(precomp_path+'b_r',b_r)
# print('mag cadence b saved')
    
    valid = np.where(abs(b_mag)<1e30)
    b_mag, b_r, epoch_b = [b_mag[valid], b_r[valid], epoch_b[valid]]
    min_diff = psplib.get_min_diff(epoch_b)
    print('Sampling to cadence: ' + str(min_diff))
    
    this_epoch_ns = psplib.unpack_vars(path+sorted(listdir(path))[i], ['epoch'])[0]
    this_epoch = np.array([datetime_t0+datetime.timedelta(seconds=i/1e9) for i in this_epoch_ns])
    
    b_mag_spc = psplib.uniform_median_filter(b_mag,epoch_b,min_diff,filter_time // min_diff,this_epoch)
    print('b_mag_filtered created')
#     b_r_spc = psplib.uniform_median_filter(b_r,epoch_b,min_diff,filter_time // min_diff,this_epoch)
#     print('b_r_filtered computed')
    print('writing loop complete')
    np.save(precomp_path+'b_mag_spc'+str(i).zfill(3),b_mag_spc)
#     np.save(precomp_path+'b_r_spc_'+str(i).zfill(3),b_r_spc)
    print('spc cadence b saved')