#!/usr/bin/env python3
"""
linfit.py
Greg Szypko, Summer 2019

A script for identifying solar wind streams in PSP data using linear regression
"""

import cdflib
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import colors
from pspconstants import *
import psplib
import os

ang_range = (-45,-20)
ang_res = 0.05

output_path = '/home/gszypko/Desktop/streamfind/'+str(ang_res)+'/'
# dump_path = '/home/gszypko/Desktop/dump/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

r_value = np.zeros((int((ang_range[1]-ang_range[0])/ang_res),int((ang_range[1]-ang_range[0])/ang_res)))
slope, intercept, r_sq, p_value, std_err = (np.zeros_like(r_value) for i in range(5))

v_r = np.load(precomp_path+'v_r.npy')
dqf = np.load(precomp_path+'dqf_gen.npy')

carrlon, scpos = psplib.multi_unpack_vars(path, ['carr_longitude','sc_pos_HCI'])
dist = psplib.compute_magnitudes(scpos/au_km,True)

valid = np.where(np.logical_and(abs(v_r)<dat_upperbnd,dqf==0))
data = v_r[valid]
dist = dist[valid]
carrlon = carrlon[valid]

# for ang_start in range(ang_range[0],ang_range[1],ang_res):
for x in range(0,r_value.shape[0]):
    ang_start = ang_range[0] + ang_res*x
#     x = (ang_start - ang_range[0])//ang_res
#     for ang_size in range(ang_res,ang_range[1]-ang_start,ang_res):
    for y in range(0,r_value.shape[1]-x):
#         y = (ang_size)//ang_res - 1
        ang_size = (y+1) * ang_res
        ang_end = ang_start + ang_size
#         print(str(ang_start)+','+str(ang_end))
        print(str(x)+','+str(y))
        ang_slice = np.where(np.logical_and(np.greater(carrlon,ang_start),np.less(carrlon,ang_end)))
        slope[y,x], intercept[y,x], r_value[y,x], p_value[y,x], std_err[y,x] = stats.linregress(dist[ang_slice],data[ang_slice])
        print(slope[y,x])
        
#         fig = plt.figure(figsize=(4,3))
#         ax = fig.add_subplot(111)
#         ax.set_ylim(0,700)
#         ax.set_xlim(0.15,0.3)
#         ax.set_title(str(ang_start)+' to '+str(ang_end)+' deg')
#         ax.plot(dist[ang_slice],data[ang_slice],marker='.',ms=1,ls='')
#         min_dist = 0.16
#         max_dist = 0.29
#         ax.plot([min_dist,max_dist],[min_dist*slope[y,x] + intercept[y,x],max_dist*slope[y,x] + intercept[y,x]],color='black')
#         ax.text(0.1,0.8,'r^2 = '+str(r_value[y,x]**2),transform=ax.transAxes)
#         ax.text(0.1,0.9,'slope = '+str(slope[y,x]),transform=ax.transAxes)
#         ax.text(0.1,0.7,'intercept = '+str(intercept[y,x]),transform=ax.transAxes)
#         fig.savefig(dump_path+str(abs(ang_start))+','+str(abs(ang_end)))
#         plt.close(fig)

r_sq = r_value**2

fig, ax = plt.subplots()

image = ax.imshow(r_sq, cmap='Blues', interpolation='nearest',vmin=1e-2,norm=colors.LogNorm(),origin='lower',extent=[-45,-20,0,25])
color_bar = fig.colorbar(image, ax=ax)
color_bar.set_label('r^2')
ax.set_title('R Squared for Linear Fit of Radial Velocity by Angular Slice')
ax.set_xlabel('Starting Carrington Longitude (degrees)')
ax.set_ylabel('Angular Size (degrees counterclockwise)')
fig.savefig(output_path+'r_sq')

fig, ax = plt.subplots()

image = ax.imshow(slope, cmap='seismic', interpolation='nearest', vmin = -5000, vmax=5000,origin='lower',extent=[-45,-20,0,25])
color_bar = fig.colorbar(image, ax=ax)
color_bar.set_label('Slope (km/s/AU)')
ax.set_title('Slope for Linear Fit of Radial Velocity by Angular Slice')
ax.set_xlabel('Starting Carrington Longitude (degrees)')
ax.set_ylabel('Angular Size (degrees counterclockwise)')
fig.savefig(output_path+'slope')

