#!/usr/bin/env python3
"""
angslice.py
by Greg Szypko, Summer 2019

Plots data collected by the Parker Solar Probe as a function of heliocentric radius.
Divides data into angular 'bins' corresponding to Carrington longitude.

Usage: Run from the command line. Takes one command line argument corresponding to
the data mode to plot (temp, vr, np, b, beta, alf, alfmach)
Example: $ ./angslice.py vr

Reads in .cdf files from the directory /data/reu/gszypko/data/loopback/ and
/data/reu/gszypko/data/mag/ for proton velocity and magnetic field data respectively.

Writes out .png files of the plots to the directory /home/gszypko/Desktop/norm_plots/datamode
where datamode is replaced by one of the modes described above.
"""

import cdflib
import numpy as np
from matplotlib import pyplot as plt
import os
import psplib
import sys
from pspconstants import *

#temp=temperature, vr=radial vel, np=density, b=magnetic field
#beta=plasma beta, alf=alfven speed, alfmach=alfven mach number
datamode = sys.argv[1]
#show=display plots on screen
#file=save plots to file directory
outputmode = 'file'
br_color = False

ang_res = 5 #degrees, size of angular bins to plot
ang_start = -45 #degrees, heading of first bin to plot
ang_end = -20

carrlon, scpos = psplib.multi_unpack_vars(path, ['carr_longitude','sc_pos_HCI'])
dist = psplib.compute_magnitudes(scpos/au_km,True)

output_path = '/home/gszypko/Desktop/norm_plots/'+datamode+'/'
if br_color:
    output_path = '/home/gszypko/Desktop/bcolor_plots/'+datamode+'/'


if datamode in {'vr','alfmach'}:
    vp = psplib.multi_unpack_vars(path, ['vp_moment_RTN'])[0]
    v_r = vp[:,0]
if datamode in {'np','beta','alf','alfmach'}:
    n_p = psplib.multi_unpack_vars(path, ['np_moment'])[0]
if datamode in {'temp','beta','alf','alfmach'}:
    wp = psplib.multi_unpack_vars(path, ['wp_moment'])[0]
    temp = np.square(wp)*(mp_kg/(k_b*1e-6)/3)
if datamode in {'b','beta','alf','alfmach'} or br_color:
    epoch = psplib.multi_unpack_vars(path, ['epoch'])[0]
    b_rtn, epoch_b = psplib.multi_unpack_vars(mag_path, ['psp_fld_mag_rtn','psp_fld_mag_epoch'])
    b_mag = psplib.compute_magnitudes(b_rtn)
    b_r = b_rtn[:,0]
    if datamode != 'b': #only need to interpolate when combining with normal data set
        b_lerp = psplib.time_lerp(epoch,epoch_b,b_mag)
        b_r_lerp = psplib.time_lerp(epoch,epoch_b,b_r)

if datamode == 'temp':
    plot_title = 'Proton temperature '
    y_label = 'Proton temperature (K)'
    data = temp
    ybounds = (0,1e6)
elif datamode == 'vr':
    plot_title = 'Radial proton velocity '
    y_label = 'Proton velocity (km/s)'
    data = v_r
    ybounds = (200,700)
elif datamode == 'np':
    plot_title = 'Proton density '
    y_label = 'Proton density (cm^-3)'
    data = n_p
    ybounds = (0,1500)
elif datamode == 'b':
    plot_title = 'Magnetic field strength '
    y_label = 'Field strength (nT)'
    data = b_mag
    ybounds = (0,120)
    dist = psplib.time_lerp(epoch_b,epoch,dist)
    carrlon = psplib.time_lerp(epoch_b,epoch,carrlon)
elif datamode == 'beta':
    plot_title = 'Plasma beta '
    y_label = 'Plasma beta'
    ybounds = (0.01,10)
    data = (n_p*1e6) * k_b * temp * 2 * mu_0 / np.square(b_lerp*1e-9)
elif datamode == 'alf':
    plot_title = 'Alfven speed '
    y_label = 'Alfven speed (km/s)'
    ybounds = (0,500)
    data = b_lerp*1e-9/np.sqrt(mp_kg*(n_p*1e6)*mu_0) * 1e-3 #in km/s
elif datamode == 'alfmach':
    plot_title = 'Alfven Mach number '
    y_label = 'Alfven Mach number'
    ybounds = (0,20)
    alf = b_lerp*1e-9/np.sqrt(mp_kg*(n_p*1e6)*mu_0) * 1e-3 #in km/s
    data = v_r/alf

if not os.path.exists(output_path):
    os.makedirs(output_path)

for angle in range(ang_start,ang_end,ang_res):
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    ax.set_ylim(ybounds[0],ybounds[1])
    ax.set_xlim(0.15,0.3)
    ax.set_title(plot_title+', Carrington Longitude = '+str(angle)+' to '+str(angle+ang_res)+' deg')
    ax.set_xlabel('Radial distance (AU)')
    ax.set_ylabel(y_label)
    if datamode == 'beta':
        ax.set_yscale('log')
    ang_slice = np.where(np.logical_and(np.logical_and(np.greater(carrlon,angle),np.less(carrlon,angle+ang_res)),np.less(abs(data),1e30)))
    if br_color:
        if datamode == 'b':
            scatter = ax.scatter(dist[ang_slice],data[ang_slice],marker='.',s=1,c=b_r[ang_slice],cmap='bwr',vmin=-10,vmax=10)
            color_bar = fig.colorbar(scatter, ax=ax)
            color_bar.set_label('Radial magnetic field (nT)')
        else:
            scatter = ax.scatter(dist[ang_slice],data[ang_slice],marker='.',s=1,c=b_r_lerp[ang_slice],cmap='bwr',vmin=-10,vmax=10)
            color_bar = fig.colorbar(scatter, ax=ax)
            color_bar.set_label('Radial magnetic field (nT)')
    else:
        ax.plot(dist[ang_slice],data[ang_slice],marker='.',ms=1,ls='')
    if outputmode == 'file':
        fig.savefig(output_path+str(abs(angle)))
        plt.close(fig)

if outputmode == 'show':
    plt.show()