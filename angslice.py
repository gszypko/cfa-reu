#!/usr/bin/env python3
#angslice.py
#Plots different PSP measurements as a function of solar radius
#within different angular "bins"

import cdflib
import numpy as np
from matplotlib import pyplot as plt
#import datetime
from os import listdir
import psplib

#temp=temperature, vr=radial vel, np=density, b=magnetic field
#beta=plasma beta, alf=alfven speed
colormode = 'vr'

au_km = 1.496e8

mp_kg = 1.6726219e-27 #proton mass in kg
k_b = 1.38064852e-23 #boltzmann constant in m^2 kg s^-2 K^-1
mu_0 = 4e-7*np.pi #vacuum permeability in T m / A

ang_res = 5 #degrees, size of angular bins to plot
ang_start = -50 #degrees, heading of first bin to plot
ang_end = -10

path = '/data/reu/gszypko/data/loopback/'
carrlon, scpos = psplib.multi_unpack_vars(path, ['carr_longitude','sc_pos_HCI'])
dist = psplib.compute_magnitudes(scpos/au_km,True)

if colormode in {'vr','alfmach'}:
    vp = psplib.multi_unpack_vars(path, ['vp_moment_RTN'])[0]
    v_r = vp[:,0]
if colormode in {'np','beta','alf','alfmach'}:
    n_p = psplib.multi_unpack_vars(path, ['np_moment'])[0]
if colormode in {'temp','beta','alf','alfmach'}:
    wp = psplib.multi_unpack_vars(path, ['wp_moment'])[0]
if colormode in {'b','beta','alf','alfmach'}:
    epoch = psplib.multi_unpack_vars(path, ['epoch'])[0]
    b_rtn, epoch_b = psplib.multi_unpack_vars(mag_path, ['psp_fld_mag_rtn','psp_fld_mag_epoch'])
    b_mag = psplib.compute_magnitudes(b_rtn)
    b_lerp = psplib.time_lerp(epoch,epoch_b,b_mag)

for angle in range(ang_start,ang_end,ang_res):
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    ax.set_title('Radial Proton Velocity (Carrington Longitude = '+str(angle)+' to '+str(angle+ang_res)+' deg)')
    ax.set_xlabel('Radial distance (AU)')
    ax.set_ylabel('Radial proton speed (km/s)')
    ang_slice = np.where(np.logical_and(np.logical_and(np.greater(carrlon,angle),np.less(carrlon,angle+ang_res)),np.less(abs(v_r),1e12)))
    ax.scatter(dist[ang_slice],v_r[ang_slice],marker='.',s=1)

plt.show()