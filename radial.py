#!/usr/bin/env python3
#radial.py
#Plots different measurements from PSP along its first close approach
#to the sun (Oct/Nov 2018) in the corotating heliocentric frame

import cdflib
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
#import datetime
from os import listdir
#from scipy.optimize import curve_fit
import psplib
from pspconstants import *
import os
import sys

#temp=temperature, vr=radial vel, np=density, b=magnetic field
#beta=plasma beta, alf=alfven speed, alfmach=alfven mach number
colormode = 'vr'
# colormode = sys.argv[1]

output_path = '/home/gszypko/Desktop/radial2/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

file_names = sorted(listdir(path))
mag_files = sorted(listdir(mag_path))

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111,projection='polar')

ang_bin_vels = {}
ang_bin_rads = {}
streamline_outer_r = 0.2
if approach_num == 1:
    for i in range(-41,-23):
        ang_bin_rads[i]=[0,0]
        ang_bin_vels[i]=[0,0]

if approach_num == 2:
    for i in range(-6,12):
        ang_bin_rads[i]=[0,0]
        ang_bin_vels[i]=[0,0]


for i in range(0,len(file_names)):
# for i in range(0,1):
    file_name = path + file_names[i]
    carrlon, carrlat, scpos = psplib.unpack_vars(file_name, ['carr_longitude','carr_latitude','sc_pos_HCI'])
#     carrlon, carrlat, scpos = psplib.multi_unpack_vars(path, ['carr_longitude','carr_latitude','sc_pos_HCI'])
    dist = psplib.compute_magnitudes(scpos/au_km, True)
#     print(file_names[i])
#     print("r="+str(round(np.amin(dist),2))+" to "+str(round(np.amax(dist),2)))
#     print("theta="+str(round(np.amin(carrlon)+360,1))+" to "+str(round(np.amax(carrlon)+360,1))+'\n')
#     continue
    if colormode in {'vr','alfmach'}:
        vp = psplib.unpack_vars(file_name, ['vp_moment_RTN'])[0]
#         vp = psplib.multi_unpack_vars(path, ['vp_moment_RTN'])[0]
    if colormode in {'np','beta','alf','alfmach'}:
        n_p = psplib.unpack_vars(file_name, ['np_moment'])[0]
#         n_p = psplib.unpack_vars(path, ['np_moment'])[0]
    if colormode in {'temp','beta','alf','alfmach'}:
        wp = psplib.unpack_vars(file_name, ['wp_moment'])[0]
    if colormode in {'b','beta','alf','alfmach','br'}:
#         foo = 1
        epoch = psplib.unpack_vars(file_name, ['epoch'])[0]
        mag_file = mag_path + mag_files[i]
        b_rtn, epoch_b = psplib.unpack_vars(mag_file, ['psp_fld_mag_rtn','psp_fld_mag_epoch'])
        b_mag = psplib.compute_magnitudes(b_rtn)
        b_lerp = psplib.time_lerp(epoch,epoch_b,b_mag)
    dist = psplib.compute_magnitudes(scpos/au_km, True)
    n=1 #step between good data points to plot
    if colormode == 'temp':
        temp = np.square(wp)*(mp_kg/(k_b*1e-6)/3)
        good = np.where(abs(wp) < dat_upperbnd)
        color = temp[good][::n]
    elif colormode == 'vr':
        v_r = vp[:,0]
        good = np.where(abs(v_r)<dat_upperbnd)
        color = v_r[good][::n]
    elif colormode == 'np':
        good = np.where(abs(n_p) < dat_upperbnd)
        color = n_p[good][::n]
    elif colormode == 'b':
        good = np.where(abs(b_lerp) < dat_upperbnd)
        color = b_lerp[good][::n]
    elif colormode == 'beta':
        temp = np.square(wp)*(mp_kg/(k_b*1e-6)/3)
        beta = (n_p*1e6) * k_b * temp * 2 * mu_0 / np.square(b_lerp*1e-9)
        good = np.where(abs(beta) < dat_upperbnd)
        color = beta[good][::n]
    elif colormode == 'alf':
        alf = b_lerp*1e-9/np.sqrt(mp_kg*(n_p*1e6)*mu_0) * 1e-3 #in km/s
        good = np.where(abs(alf) < dat_upperbnd)
        color = alf[good][::n]
    elif colormode == 'alfmach':
        alf = b_lerp*1e-9/np.sqrt(mp_kg*(n_p*1e6)*mu_0) * 1e-3 #in km/s
        alfmach = vp[:,0]/alf
        good = np.where(abs(alfmach) < dat_upperbnd)
        color = alfmach[good][::n]
    elif colormode == 'br':
        b_r_lerp = psplib.time_lerp(epoch,epoch_b,b_rtn[:,0])
        good = np.where(abs(b_r_lerp)<dat_upperbnd)
        color = b_r_lerp[good][::n]
#         b_r_lerp = np.load(precomp_path+'b_r_spc.npy')
#         dqf_gen = np.load(precomp_path+'dqf_gen.npy')
#         good = np.where(np.logical_and(abs(b_r_lerp) < dat_upperbnd,dqf_gen==0))
#         color = b_r_lerp[good]

    else:
        good = np.where(True)

    theta = (carrlon[good]*np.pi/180)[::n]
    radius = dist[good][::n]

    size = carrlat[good][::n]*30 + 150

#Uncomment for logarithmic color scaling
#     plt.scatter(theta,radius,cmap='jet',c=color,s=300,norm=matplotlib.colors.LogNorm())
#     plt.scatter(theta,radius,cmap='bwr',c=color,s=300,vmin=-20,vmax=20)
    plt.scatter(theta,radius,cmap='jet',c=color,s=300)

    for i in range(0,len(color)):
        ang_bin = round(theta[i]*180/np.pi)
        if radius[i] < streamline_outer_r and ang_bin in ang_bin_vels:
            ang_bin_vels[ang_bin][0] = (ang_bin_vels[ang_bin][0]*ang_bin_vels[ang_bin][1] + color[i])/(ang_bin_vels[ang_bin][1]+1)
            ang_bin_vels[ang_bin][1] += 1
            ang_bin_rads[ang_bin][0] = (ang_bin_rads[ang_bin][0]*ang_bin_rads[ang_bin][1] + radius[i])/(ang_bin_rads[ang_bin][1]+1)
            ang_bin_rads[ang_bin][1] += 1


#     num_spirals = 2
#     bin_size = len(color)//2
#     for i in range(0,num_spirals):
#         this_color = color[bin_size*i:bin_size*(i+1)]
#         r_0 = np.average(radius[bin_size*i:bin_size*(i+1)])
#         if r_0 < 0.25:
#             phi_0 = np.average(theta[bin_size*i:bin_size*(i+1)])
#             u = np.average(this_color)/au_km
#             r = np.linspace(r_0,0.4,30)
#             w = 2.69e-6 #deg/s
#             plt.plot(phi_0+w/u*(r_0-r),r,color='black')

print(ang_bin_vels)
print(ang_bin_rads)
for ang_bin in ang_bin_vels:
    r_0 = ang_bin_rads[ang_bin][0]
    u = ang_bin_vels[ang_bin][0]/au_km
    r = np.linspace(r_0,0.4,30)
    w = 2.69e-6 #deg/s
    plt.plot(ang_bin*np.pi/180+w/u*(r_0-r),r,color='black')
    if ang_bin == max(ang_bin_vels.keys()):
        plt.plot(ang_bin*np.pi/180+w/u*(r_0-r)+np.pi/180,r,color='black')

ax.set_rmax(0.4)
if approach_num == 2:
    ax.set_thetamin(-45)
    ax.set_thetamax(45)
    ax.set_theta_zero_location("SE")
else:
    ax.set_thetamin(-90)
    ax.set_thetamax(0)
ax.set_rlabel_position(135-45/2)
color_bar = plt.colorbar()

if colormode == 'temp':
    ax.set_title("Proton temperature in heliocentric corotating frame (Approach "+str(approach_num)+")")
    color_bar.set_label("Proton temperature (K)")
elif colormode == 'vr':
    ax.set_title("Radial particle velocity in heliocentric corotating frame (Approach "+str(approach_num)+")")
    color_bar.set_label("Radial particle velocity (km/s)")
elif colormode == 'np':
    ax.set_title("Estimated proton density in heliocentric corotating frame (Approach "+str(approach_num)+")")
    color_bar.set_label("Proton density (cm^-3)")
elif colormode == 'b':
    ax.set_title("Magnetic field strength (interpolated) in heliocentric corotating frame (Approach "+str(approach_num)+")")
    color_bar.set_label('Magnetic field magnitude (nT)')
elif colormode == 'beta':
    ax.set_yscale('log')
    ax.set_title("Plasma beta in heliocentric corotating frame (Approach "+str(approach_num)+")")
    color_bar.set_label("Plasma beta")
elif colormode == 'alf':
    ax.set_title("Alfven speed in heliocentric corotating frame (Approach "+str(approach_num)+")")
    color_bar.set_label("Alfven speed (km/s)")
elif colormode == 'alfmach':
    ax.set_title("Alfven Mach number in heliocentric corotating frame (Approach "+str(approach_num)+")")
    color_bar.set_label("Alfven Mach number")
elif colormode == 'br':
    ax.set_title("Radial magnetic field in heliocentric corotating frame (Approach "+str(approach_num)+")")
    color_bar.set_label("Radial magnetic field (nT)")

# fig.savefig(output_path+colormode)
plt.show()
