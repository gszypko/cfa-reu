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
import argparse

parser = argparse.ArgumentParser()

pointcolor = parser.add_mutually_exclusive_group()
pointcolor.add_argument("--bcolor",action="store_true")
pointcolor.add_argument("--longcolor",action="store_true")

parser.add_argument("--tofile")

parser.add_argument("--batch",type=int)

parser.add_argument("--fitsmargin",type=float)

parser.add_argument("datamode",help="data to be plotted on the y-axis",choices=["vr","np","temp","b","beta","alf","alfmach"])
parser.add_argument("startangle",type=int)
parser.add_argument("endangle",type=int)

args = parser.parse_args()

datamode = args.datamode
ang_start = args.startangle
ang_end = args.endangle


br_color = args.bcolor
long_color = args.longcolor

if args.tofile:
    outputmode = 'file'
    output_path = default_output_path + args.tofile + '/'
else:
    outputmode = 'show'

if args.batch:
    ang_res = args.batch
else:
    ang_res = ang_end - ang_start

if args.fitsmargin:
    maxdiff = args.fitsmargin
else:
    maxdiff = 0.1

#temp=temperature, vr=radial vel, np=density, b=magnetic field
#beta=plasma beta, alf=alfven speed, alfmach=alfven mach number
# datamode = sys.argv[1]
#show=display plots on screen
#file=save plots to file directory
# outputmode = 'file'
# br_color = False

# ang_start = -45 #degrees, heading of first bin to plot
# ang_end = -20
# ang_start = int(sys.argv[2]) #degrees, heading of first bin to plot
# ang_end = int(sys.argv[3])

# ang_res = ang_end - ang_start #degrees, size of angular bins to plot


filter_radius = 30*60*1e9 #in nanoseconds

carrlon, scpos = psplib.multi_unpack_vars(path, ['carr_longitude','sc_pos_HCI'])
dist = psplib.compute_magnitudes(scpos/au_km,True)
epoch_ns = psplib.multi_unpack_vars(path, ['epoch'])[0]
epoch = np.array([datetime_t0+datetime.timedelta(seconds=i/1e9) for i in epoch_ns])
dqf = np.load(precomp_path+'dqf_gen.npy')

output_path = '/home/gszypko/Desktop/stream_candidates/'+str(int(abs(ang_start)))+','+str(int(abs(ang_end)))+'/'
# output_path = '/home/gszypko/Desktop/stream_candidates/'
if br_color:
    output_path = '/home/gszypko/Desktop/bcolor_filtered_plots/'+datamode+'/'


#                         if datamode in {'vr','alfmach'}:
#                             vp = psplib.multi_unpack_vars(path, ['vp_moment_RTN'])[0]
#                             v_r = vp[:,0]
#                         if datamode in {'np','beta','alf','alfmach'}:
#                             n_p = psplib.multi_unpack_vars(path, ['np_moment'])[0]
#                         if datamode in {'temp','beta','alf','alfmach'}:
#                             wp = psplib.multi_unpack_vars(path, ['wp_moment'])[0]
#                             temp = np.square(wp)*(mp_kg/(k_b*1e-6)/3)
if datamode == 'b':
    epoch_b_ns = psplib.multi_unpack_vars(mag_path, ['psp_fld_mag_epoch'])[0]
    epoch_b = np.array([datetime_t0+datetime.timedelta(seconds=i/1e9) for i in epoch_b_ns])
#                             b_mag = psplib.compute_magnitudes(b_rtn)
#                             b_r = b_rtn[:,0]
#                             if datamode != 'b': #only need to interpolate when combining with normal data set
#                                 b_lerp = psplib.time_lerp(epoch,epoch_b,b_mag)
#                                 b_r_spc = psplib.time_lerp(epoch,epoch_b,b_r)


dqf_gen = np.load(precomp_path+'dqf_gen.npy')
if datamode == 'temp':
    plot_title = 'Proton temperature '
    y_label = 'Proton temperature (K)'
    data = np.load(precomp_path+'temp.npy')
    data_fit = np.load(precomp_path+'temp_fit.npy')
    ybounds = (0,6e5)
elif datamode == 'vr':
    plot_title = 'Radial proton velocity '
    y_label = 'Proton velocity (km/s)'
    data = np.load(precomp_path+'v_r.npy')
    data_fit = np.load(precomp_path+'v_r_fit.npy')
    ybounds = (200,700)
elif datamode == 'np':
    plot_title = 'Proton density '
    y_label = 'Proton density (cm^-3)'
    data = np.load(precomp_path+'n_p.npy')
    data_fit = np.load(precomp_path+'n_p_fit.npy')
    ybounds = (0,600)
elif datamode == 'b':
    plot_title = 'Magnetic field strength '
    y_label = 'Field strength (nT)'
    data = np.load(precomp_path+'b_mag.npy')
    ybounds = (0,120)
    dist = psplib.time_lerp(epoch_b,epoch,dist)
    carrlon = psplib.time_lerp(epoch_b,epoch,carrlon)
elif datamode == 'beta':
    plot_title = 'Plasma beta '
    y_label = 'Plasma beta'
    ybounds = (0.01,10)
    temp = np.load(precomp_path+'temp.npy')
    b_lerp = np.load(precomp_path+'b_mag_spc.npy')
    n_p = np.load(precomp_path+'n_p.npy')
    data = (n_p*1e6) * k_b * temp * 2 * mu_0 / np.square(b_lerp*1e-9)
elif datamode == 'alf':
    plot_title = 'Alfven speed '
    y_label = 'Alfven speed (km/s)'
    ybounds = (0,500)
    b_lerp = np.load(precomp_path+'b_mag_spc.npy')
    n_p = np.load(precomp_path+'n_p.npy')
    data = b_lerp*1e-9/np.sqrt(mp_kg*(n_p*1e6)*mu_0) * 1e-3 #in km/s
elif datamode == 'alfmach':
    plot_title = 'Alfven Mach number '
    y_label = 'Alfven Mach number'
    ybounds = (0,20)
    b_lerp = np.load(precomp_path+'b_mag_spc.npy')
    n_p = np.load(precomp_path+'n_p.npy')
    alf = b_lerp*1e-9/np.sqrt(mp_kg*(n_p*1e6)*mu_0) * 1e-3 #in km/s
    v_r = np.load(precomp_path+'v_r.npy')
    data = v_r/alf

if not os.path.exists(output_path):
    os.makedirs(output_path)


if datamode in {'temp','vr','np'}:
    #Filter by agreement with the fits version of the data
    percentdiff = abs((data_fit - data) / data)
    if datamode == 'temp':
        maxdiff = 1 - np.square(1-maxdiff)
#         maxdiff = 0.2
    valid = np.where(np.logical_and(percentdiff < maxdiff,np.logical_and(abs(data)<dat_upperbnd,dqf==0)))
    data = (data[valid] + data_fit[valid])/2
else:
    valid = np.where(np.logical_and(abs(data)<dat_upperbnd,dqf==0))
    data = data[valid]
dist = dist[valid]
carrlon = carrlon[valid]

if approach_num == 2:
    print(len(data))
    dqf_fullscan = np.load(precomp_path+'dqf_fullscan.npy')[valid]
    not_fullscan = np.where(dqf_fullscan==0)
    data = data[not_fullscan]
    dist = dist[not_fullscan]
    carrlon = carrlon[not_fullscan]
    print(len(data))

if br_color:
    if datamode == 'b':
        #NOTE: fields data currently lacks quality flags
#         valid = np.where(abs(data) < dat_upperbnd)
        b_r = np.load(precomp_path+'b_r.npy')[valid]
        epoch_b = epoch_b[valid]
    else:
#         valid = np.where(np.logical_and(abs(data) < dat_upperbnd,dqf_gen==0))
        b_r_spc = np.load(precomp_path+'b_r_spc.npy')[valid]
        epoch = epoch[valid]
else:
#     valid = np.where(np.logical_and(abs(data) < dat_upperbnd,dqf_gen==0))
    epoch = epoch[valid]

# data = data[valid]
# dist = dist[valid]
# carrlon = carrlon[valid]

vars_tofilter = [data,dist,carrlon]

if approach_num == 1:
    if datamode == 'b':
        if br_color:
            vars_tofilter.append(b_r)
            data, dist, carrlon, b_r = psplib.filter_known_transients(epoch_b, vars_tofilter)
        else:
            data, dist, carrlon = psplib.filter_known_transients(epoch_b, vars_tofilter)
    else:
        if br_color:
            vars_tofilter.append(b_r_spc)
            data, dist, carrlon, b_r_spc = psplib.filter_known_transients(epoch, vars_tofilter)
        else:
            data, dist, carrlon = psplib.filter_known_transients(epoch, vars_tofilter)


    
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
    ang_slice = np.where(np.logical_and(np.greater(carrlon,angle),np.less(carrlon,angle+ang_res)))
#     print(b_r_spc[ang_slice].shape)
#     print(data[ang_slice].shape)
#     print(dist[ang_slice].shape)
    if br_color:
        if datamode == 'b':
            scatter = ax.scatter(dist[ang_slice],data[ang_slice],marker='.',s=1,c=b_r[ang_slice],cmap='bwr',vmin=-10,vmax=10)
            color_bar = fig.colorbar(scatter, ax=ax)
            color_bar.set_label('Radial magnetic field (nT)')
        else:
            scatter = ax.scatter(dist[ang_slice],data[ang_slice],marker='.',s=1,c=b_r_spc[ang_slice],cmap='bwr',vmin=-10,vmax=10)
            color_bar = fig.colorbar(scatter, ax=ax)
            color_bar.set_label('Radial magnetic field (nT)')
    elif long_color:
        scatter = ax.scatter(dist[ang_slice],data[ang_slice],marker='.',s=1,c=carrlon[ang_slice],cmap='gist_rainbow')
        color_bar = fig.colorbar(scatter, ax=ax)
        color_bar.set_label('Carrington Longitude')
    else:
        scatter = ax.scatter(dist[ang_slice],data[ang_slice],marker='.',s=1)
    if outputmode == 'file':
        fig.savefig(output_path+datamode)
        plt.close(fig)

if outputmode == 'show':
    plt.show()