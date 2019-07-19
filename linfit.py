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
from pspconstants import *
import psplib
import os

output_path = '/home/gszypko/Desktop/doublechecktest/vr/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

ybounds = (200,700)
ang_res = 5 #degrees, size of angular bins to plot
ang_start = -45 #degrees, heading of first bin to plot
ang_end = -20
plot_title = 'Radial proton velocity '
y_label = 'Proton velocity (km/s)'

v_r = np.load(precomp_path+'v_r.npy')
dqf = np.load(precomp_path+'dqf_gen.npy')
v_r_fit = np.load(precomp_path+'v_r_fit.npy')

carrlon, scpos = psplib.multi_unpack_vars(path, ['carr_longitude','sc_pos_HCI'])
dist = psplib.compute_magnitudes(scpos/au_km,True)

percentdiff = abs((v_r_fit - v_r) / v_r)
maxdiff = 0.05

valid = np.where(np.logical_and(percentdiff < maxdiff,np.logical_and(abs(v_r)<dat_upperbnd,dqf==0)))
data = (v_r[valid] + v_r_fit[valid])/2
dist = dist[valid]
carrlon = carrlon[valid]



for angle in range(ang_start,ang_end,ang_res):
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    ax.set_ylim(ybounds[0],ybounds[1])
    ax.set_xlim(0.15,0.3)
    ax.set_title(plot_title+', Carrington Longitude = '+str(angle)+' to '+str(angle+ang_res)+' deg')
    ax.set_xlabel('Radial distance (AU)')
    ax.set_ylabel(y_label)
    ang_slice = np.where(np.logical_and(np.greater(carrlon,angle),np.less(carrlon,angle+ang_res)))
    ax.plot(dist[ang_slice],data[ang_slice],marker='.',ms=1,ls='')
    slope, intercept, r_value, p_value, std_err = stats.linregress(dist[ang_slice],data[ang_slice])
#     min_dist = min(dist[ang_slice])
#     max_dist = max(dist[ang_slice])
    min_dist = 0.16
    max_dist = 0.29
    ax.plot([min_dist,max_dist],[min_dist*slope + intercept,max_dist*slope + intercept],color='black')
    ax.text(0.1,0.8,'r^2 = '+str(r_value**2),transform=ax.transAxes)
    ax.text(0.1,0.9,'slope = '+str(slope),transform=ax.transAxes)
#     plt.show()
    fig.savefig(output_path+str(abs(angle)))
    plt.close(fig)