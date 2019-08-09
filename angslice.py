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
pointcolor.add_argument("--spiralcolor",action="store_true")

parser.add_argument("--spiralslice",action="store_true")

parser.add_argument("--boxcar",action="store_true")

parser.add_argument("--tofile")

parser.add_argument("--batch",type=int)

parser.add_argument("--fitsmargin",type=float)

parser.add_argument("--filtered",action="store_true")

parser.add_argument("datamode",help="data to be plotted on the y-axis",choices=["vr","np","temp","b","beta","alf","alfmach","full"])
parser.add_argument("startangle",type=int)
parser.add_argument("endangle",type=int)

args = parser.parse_args()

datamode = args.datamode
ang_start = args.startangle
ang_end = args.endangle

filtered = args.filtered

br_color = args.bcolor
long_color = args.longcolor
spiral_color = args.spiralcolor
spiral_slice = args.spiralslice
boxcar = args.boxcar

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

print(datamode)
print(ang_start)
print(ang_end)

filter_radius = 30*60*1e9 #in nanoseconds

carrlon, scpos = psplib.multi_unpack_vars(path, ['carr_longitude','sc_pos_HCI'])
dist = psplib.compute_magnitudes(scpos/au_km,True)
epoch_ns = psplib.multi_unpack_vars(path, ['epoch'])[0]
epoch = np.array([datetime_t0+datetime.timedelta(seconds=i/1e9) for i in epoch_ns])
dqf = np.load(precomp_path+'dqf_gen.npy')

if boxcar:
    if spiral_slice or spiral_color:
        spiral_lon_fillval = 500
        print("loading spiral boxcar longs")
        carrlon = np.load(precomp_path+"boxcar/boxcar_spiral_longs.npy")
    else:
        print("loading normal boxcar longs")
        carrlon = np.load(precomp_path+"boxcar/boxcar_longs.npy")
    dist = np.load(precomp_path+"boxcar/boxcar_dists.npy")
    epoch = np.load(precomp_path+"boxcar/boxcar_times.npy")
    print(len(carrlon))
    print(len(dist))
    print(len(epoch))

if (spiral_color or spiral_slice) and not boxcar:
    try:
        carrlon = np.load(precomp_path+"spiral_longitude.npy")
    except:
        ang_bin_vels = {}
        ang_bin_rads = {}
        streamline_outer_r = 0.2
        spiral_lon_fillval = 500
        if approach_num == 1:
            for i in range(-41,-23):
                ang_bin_rads[i]=[0,0]
                ang_bin_vels[i]=[0,0]

        if approach_num == 2:
            for i in range(-6,12):
                ang_bin_rads[i]=[0,0]
                ang_bin_vels[i]=[0,0]
    
        color = np.load(precomp_path+'v_r_filtered.npy')
        for i in range(0,len(carrlon)):
            ang_bin = round(carrlon[i])
            if dist[i] < streamline_outer_r and ang_bin in ang_bin_vels:
                ang_bin_vels[ang_bin][0] = (ang_bin_vels[ang_bin][0]*ang_bin_vels[ang_bin][1] + color[i])/(ang_bin_vels[ang_bin][1]+1)
                ang_bin_vels[ang_bin][1] += 1
                ang_bin_rads[ang_bin][0] = (ang_bin_rads[ang_bin][0]*ang_bin_rads[ang_bin][1] + dist[i])/(ang_bin_rads[ang_bin][1]+1)
                ang_bin_rads[ang_bin][1] += 1

        ang_bin_rads[max(ang_bin_rads.keys())+1]=[ang_bin_rads[max(ang_bin_rads.keys())][0],ang_bin_rads[max(ang_bin_rads.keys())][1]]
        ang_bin_vels[max(ang_bin_vels.keys())+1]=[ang_bin_vels[max(ang_bin_vels.keys())][0],ang_bin_vels[max(ang_bin_vels.keys())][1]]
    
        print(ang_bin_rads.keys())
        print(ang_bin_vels.keys())
        spiral_lon = np.ones_like(carrlon)*spiral_lon_fillval
        w = 1.54e-4 #deg/s
        for i in range(0,len(carrlon)):
            r = dist[i]
            phi = carrlon[i]
            print("r: "+str(r))
            print("phi: "+str(phi))
            prev_phi = min(ang_bin_rads.keys())
    #         print("sweaping (heh) through streams...")
            for ang_bin in sorted(ang_bin_rads.keys()):
    #             print("ang_bin: "+str(ang_bin))
                r_0 = ang_bin_rads[ang_bin][0]
                u = ang_bin_vels[ang_bin][0]/au_km
                this_phi = ang_bin+w/u*(r_0-r)
                if this_phi > phi and prev_phi < phi:
    #                 print("this_phi: "+str(this_phi))
    #                 print("prev_phi: "+str(prev_phi))
                    spiral_lon[i]=((phi-prev_phi)*(ang_bin)+(this_phi-phi)*(ang_bin-1))/(this_phi-prev_phi)
                    print("spiral_lon: "+str(spiral_lon[i]))
                    break
                else:
    #                 if ang_bin == max(ang_bin_rads.keys()):
    #                     spiral_lon[i] = phi
                    prev_phi = this_phi
        np.save(precomp_path+"spiral_longitude",spiral_lon)
        carrlon = spiral_lon

#                         if datamode in {'vr','alfmach'}:
#                             vp = psplib.multi_unpack_vars(path, ['vp_moment_RTN'])[0]
#                             v_r = vp[:,0]
#                         if datamode in {'np','beta','alf','alfmach'}:
#                             n_p = psplib.multi_unpack_vars(path, ['np_moment'])[0]
#                         if datamode in {'temp','beta','alf','alfmach'}:
#                             wp = psplib.multi_unpack_vars(path, ['wp_moment'])[0]
#                             temp = np.square(wp)*(mp_kg/(k_b*1e-6)/3)
# if datamode == 'b':
#     print("unpacking mag cadence epoch data")
#     epoch_b_ns = psplib.multi_unpack_vars(mag_path, ['psp_fld_mag_epoch'])[0]
#     epoch_b = np.array([datetime_t0+datetime.timedelta(seconds=i/1e9) for i in epoch_b_ns])
#                             b_mag = psplib.compute_magnitudes(b_rtn)
#                             b_r = b_rtn[:,0]
#                             if datamode != 'b': #only need to interpolate when combining with normal data set
#                                 b_lerp = psplib.time_lerp(epoch,epoch_b,b_mag)
#                                 b_r_spc = psplib.time_lerp(epoch,epoch_b,b_r)


dqf_gen = np.load(precomp_path+'dqf_gen.npy')
print("loading relevant variables")
if boxcar:
    precomp_path = precomp_path + "boxcar/"
    filtered = True
if datamode == 'full':
    data_list = []
    y_label_list = []
    ybounds_list = []
if datamode == 'temp' or datamode == 'full':
    plot_title = 'Proton temperature '
    y_label = 'Proton temperature (K)'
    if filtered:
        print("filtered loading")
        data = np.load(precomp_path+'temp_filtered.npy')
    else:
        print("unfiltered loading")
        data = np.load(precomp_path+'temp.npy')
        data_fit = np.load(precomp_path+'temp_fit.npy')
    ybounds = (0,6e5)
    if datamode == 'full':
        data_list.append(data)
        y_label_list.append(y_label)
        ybounds_list.append(ybounds)
if datamode == 'vr' or datamode == 'full':
    plot_title = 'Radial proton velocity '
    y_label = 'Proton velocity (km/s)'
    if filtered:
        print("filtered loading")
        data = np.load(precomp_path+'v_r_filtered.npy')
    else:
        print("unfiltered loading")
        data = np.load(precomp_path+'v_r.npy')
        data_fit = np.load(precomp_path+'v_r_fit.npy')
    ybounds = (200,700)
    if datamode == 'full':
        data_list.append(data)
        y_label_list.append(y_label)
        ybounds_list.append(ybounds)
if datamode == 'np' or datamode == 'full':
    plot_title = 'Proton density '
    y_label = 'Proton density (cm^-3)'
    if filtered:
        print("filtered loading")
        data = np.load(precomp_path+'n_p_filtered.npy')
        print(len(data))
    else:
        print("unfiltered loading")
        data = np.load(precomp_path+'n_p.npy')
        data_fit = np.load(precomp_path+'n_p_fit.npy')
    ybounds = (0,600)
    if datamode == 'full':
        data_list.append(data)
        y_label_list.append(y_label)
        ybounds_list.append(ybounds)
if datamode == 'b' or datamode == 'full':
    plot_title = 'Magnetic field strength '
    y_label = 'Field strength (nT)'
    data = np.load(precomp_path+'b_mag_spc.npy')
    ybounds = (0,120)
    if datamode == 'full':
        data_list.append(data)
        y_label_list.append(y_label)
        ybounds_list.append(ybounds)
#     dist = psplib.time_lerp(epoch_b,epoch,dist)
#     carrlon = psplib.time_lerp(epoch_b,epoch,carrlon)
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
    
print("variable loading complete")

if outputmode == 'file' and not os.path.exists(output_path):
    os.makedirs(output_path)

if datamode in {'temp','vr','np'} and not filtered and not boxcar:
    #Filter by agreement with the fits version of the data
    percentdiff = abs((data_fit - data) / data)
    if datamode == 'temp':
        maxdiff = 1 - np.square(1-maxdiff)
#         maxdiff = 0.2
    valid = np.where(np.logical_and(percentdiff < maxdiff,np.logical_and(abs(data)<dat_upperbnd,dqf==0)))
    data = (data[valid] + data_fit[valid])/2
else:
    if filtered:
        valid = np.where(dist == dist)
    else:
        valid = np.where(np.logical_and(abs(data)<dat_upperbnd,dqf==0))
    data = data[valid]
dist = dist[valid]
carrlon = carrlon[valid]

# print(data)
# print(dist)

if approach_num == 2 and not boxcar:
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

if approach_num == 1 and not datamode == 'full':
#     if datamode == 'b':
#         if br_color:
#             vars_tofilter.append(b_r)
#             data, dist, carrlon, b_r = psplib.filter_known_transients(epoch_b, vars_tofilter)
#         else:
#             data, dist, carrlon = psplib.filter_known_transients(epoch_b, vars_tofilter)
#     else:
    if br_color:
        vars_tofilter.append(b_r_spc)
        data, dist, carrlon, b_r_spc = psplib.filter_known_transients(epoch, vars_tofilter)
    else:
        data, dist, carrlon = psplib.filter_known_transients(epoch, vars_tofilter)

print("entering plotting subroutine")
print(ybounds_list)
print(y_label_list)
print(data_list)
if boxcar:
    pointsize = 30
else:
    pointsize = 1
for angle in range(ang_start,ang_end,ang_res):
    if datamode == 'full':
        axs = [None, None, None, None]
        fig, (axs[0], axs[1], axs[2], axs[3]) = plt.subplots(4, 1, sharex=True,figsize=(8,14))
#         fig, axs_array = plt.subplots(2, 2, figsize=(14,10))
#         axs[0] = axs_array[0,0]
#         axs[1] = axs_array[0,1]
#         axs[2] = axs_array[1,0]
#         axs[3] = axs_array[1,1]
        for i in range(0,len(axs)):
            axs[i].set_ylim(ybounds_list[i][0],ybounds_list[i][1])
            axs[i].set_xlim(0.16,0.25)
            axs[i].set_ylabel(y_label_list[i])
#         axs[2].set_ylim(ybounds_list[2][0],300)
#         axs[0].set_ylim(ybounds_list[0][0],300000)
#         axs[1].set_ylim(ybounds_list[1][0],400)
#         axs[2].set_ylim(ybounds_list[2][0],400)
        fig.suptitle('Stream Candidate, Carrington Longitude = '+str(angle)+' to '+str(angle+ang_res)+' deg')
        axs[3].set_xlabel('Radial distance (AU)')
        axs[2].set_xlabel('Radial distance (AU)')
    else:
        fig = plt.figure(figsize=(12,9))
        ax = fig.add_subplot(111)
        ax.set_ylim(ybounds[0],ybounds[1])
        ax.set_xlim(0.15,0.3)
        ax.set_ylabel(y_label)
        ax.set_xlabel('Radial distance (AU)')
        if spiral_slice:
            ax.set_title(plot_title+', Parker Spiral Longitude = '+str(angle)+' to '+str(angle+ang_res)+' deg')
        else:
            ax.set_title(plot_title+', Carrington Longitude = '+str(angle)+' to '+str(angle+ang_res)+' deg')
    if spiral_color or spiral_slice:
        ang_slice = np.where(np.logical_and(carrlon < spiral_lon_fillval-1,np.logical_and(np.greater(carrlon,angle),np.less(carrlon,angle+ang_res))))
    else:
        ang_slice = np.where(np.logical_and(np.greater(carrlon,angle),np.less(carrlon,angle+ang_res)))
#     print(angle)
#     print(ang_res)
#     print(ang_slice)
    if datamode == 'beta':
        ax.set_yscale('log')
    print("plotting")
    if datamode == 'full':
        for i in range(0,len(axs)):
            ax = axs[i]
            data = data_list[i]
            if br_color:
                if datamode == 'b':
                    scatter = ax.scatter(dist[ang_slice],data[ang_slice],marker='.',s=pointsize,c=b_r[ang_slice],cmap='bwr',vmin=-10,vmax=10)
                    color_bar = fig.colorbar(scatter, ax=ax)
                    color_bar.set_label('Radial magnetic field (nT)')
                else:
                    scatter = ax.scatter(dist[ang_slice],data[ang_slice],marker='.',s=pointsize,c=b_r_spc[ang_slice],cmap='bwr',vmin=-10,vmax=10)
                    color_bar = fig.colorbar(scatter, ax=ax)
                    color_bar.set_label('Radial magnetic field (nT)')
            elif long_color:
                scatter = ax.scatter(dist[ang_slice],data[ang_slice],marker='.',s=pointsize,c=carrlon[ang_slice],cmap='gist_rainbow',zorder=1)
                color_bar = fig.colorbar(scatter, ax=ax)
                color_bar.set_label('Carrington Longitude')
            elif spiral_color:
                scatter = ax.scatter(dist[ang_slice],data[ang_slice],marker='.',s=pointsize,c=carrlon[ang_slice],cmap='gist_rainbow')
                color_bar = fig.colorbar(scatter, ax=ax)
                color_bar.set_label('Spiral Longitude at Closest Approach')
            else:
                scatter = ax.scatter(dist[ang_slice],data[ang_slice],marker='.',s=pointsize,c='black')
            if boxcar:
                ax.plot(dist[ang_slice],data[ang_slice],c='#e8e8e8',ls='-',zorder=-1)
    else:
        if br_color:
            if datamode == 'b':
                scatter = ax.scatter(dist[ang_slice],data[ang_slice],marker='.',s=pointsize,c=b_r[ang_slice],cmap='bwr',vmin=-10,vmax=10)
                color_bar = fig.colorbar(scatter, ax=ax)
                color_bar.set_label('Radial magnetic field (nT)')
            else:
                scatter = ax.scatter(dist[ang_slice],data[ang_slice],marker='.',s=pointsize,c=b_r_spc[ang_slice],cmap='bwr',vmin=-10,vmax=10)
                color_bar = fig.colorbar(scatter, ax=ax)
                color_bar.set_label('Radial magnetic field (nT)')
        elif long_color:
            scatter = ax.scatter(dist[ang_slice],data[ang_slice],marker='.',s=pointsize,c=carrlon[ang_slice],cmap='gist_rainbow',zorder=1)
            color_bar = fig.colorbar(scatter, ax=ax)
            color_bar.set_label('Carrington Longitude')
        elif spiral_color:
            scatter = ax.scatter(dist[ang_slice],data[ang_slice],marker='.',s=pointsize,c=carrlon[ang_slice],cmap='gist_rainbow')
            color_bar = fig.colorbar(scatter, ax=ax)
            color_bar.set_label('Spiral Longitude at Closest Approach')
        else:
            scatter = ax.scatter(dist[ang_slice],data[ang_slice],marker='.',s=pointsize)
        if boxcar:
            ax.plot(dist[ang_slice],data[ang_slice],c='#e8e8e8',ls='-',zorder=-1)
    
    if outputmode == 'file':
        if filtered:
            fig.savefig(output_path+datamode+'_filtered')
        else:
            fig.savefig(output_path+datamode)
        plt.close(fig)

if outputmode == 'show':
    plt.show()