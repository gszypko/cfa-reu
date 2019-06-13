#!/usr/bin/env python3

import cdflib
import numpy as np
from matplotlib import pyplot as plt
#import datetime
from os import listdir
#from scipy.optimize import curve_fit
from bisect import bisect

colormode = 'b' #temp=temperature, vr=radial vel, np=density, b=magnetic field

au_km = 1.496e8

mp_kg = 1.6726219e-27 #proton mass in kg
k_b = 1.38064852e-23*1e-6 #boltzmann constant in km^2 kg s^-2 K^-1

path = '/data/reu/gszypko/data/loopback/'
file_names = sorted(listdir(path), reverse=True)

mag_path = '/data/reu/gszypko/data/mag/'
mag_file_names = sorted(listdir(mag_path), reverse=True)

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111,projection='polar')

for i in range(0,len(file_names)):
    dat = cdflib.CDF(path + file_names[i])
    carrlon = dat.varget('carr_longitude')
    carrlat = dat.varget('carr_latitude')
    scpos = dat.varget('sc_pos_HCI')
    vp = dat.varget('vp_moment_RTN')
    n_p = dat.varget('np_moment')
    wp = dat.varget('wp_moment')
    epoch = dat.varget('epoch')
    
#     print('\n\nfile number ' + str(i))
#     for j in range(0,len(mag_file_names)):
#         dat_b = cdflib.CDF(mag_path + mag_file_names[j])
#         epoch_b = dat_b.varget('psp_fld_mag_epoch')
#         print('mag file number '+str(j))
#         print(np.amin(epoch) >= np.amin(epoch_b) and np.amin(epoch) <= np.amax(epoch_b))
#         print(np.amax(epoch) >= np.amin(epoch_b) and np.amax(epoch) <= np.amax(epoch_b))
#     
#     continue
    
    dat_b = cdflib.CDF(mag_path + mag_file_names[i])
    b_rtn = dat_b.varget('psp_fld_mag_rtn')
    epoch_b = dat_b.varget('psp_fld_mag_epoch')
    b_r = b_rtn[:,0]
    b_t = b_rtn[:,1]
    b_n = b_rtn[:,2]
    b_mag = np.sqrt(b_r**2 + b_t**2 + b_n**2)
    print(b_mag)
    
    b_lerp = []
#     print('\n\n\n')
#     print(epoch)
#     print(epoch_b)
#     
#     continue
    
    for i in range(0,len(epoch)):
        idx = np.searchsorted(epoch_b,epoch[i],side='right')
#         print(epoch[i])
#         print(np.max(epoch_b))
        if epoch[i]==epoch_b[idx-1]:
            b_lerp.append(b_mag[idx-1])
#             np.append(b_lerp,b_mag[idx-1])
        elif idx == 0 or idx == len(epoch_b):
#             np.append(b_lerp,-1e31) #remove edge points from consideration
            b_lerp.append(-1e31)
        else: #linear interpolation case
            delta1 = epoch[i]-epoch_b[idx-1]
            delta2 = epoch_b[idx]-epoch[i]
            delta = epoch_b[idx]-epoch_b[idx-1]
#             np.append(b_lerp,(delta1/delta*epoch_b[idx-1] + delta2/delta*epoch_b[idx]))
            b_lerp.append((delta1/delta*b_mag[idx-1] + delta2/delta*b_mag[idx]))
#     print(b_lerp)
    b_lerp = np.array(b_lerp)
    xloc = scpos[:,0]/au_km
    yloc = scpos[:,1]/au_km
    #zloc = scpos[:,2]/au_km
    zloc = 0 #projecting all positions down into the solar equatorial plane
    dist = np.sqrt(xloc**2 + yloc**2 + zloc**2)

    v_r = vp[:,0]

    temp = np.square(wp)*(mp_kg/k_b/3)

    n=50 #step between good data points to plot
    if colormode == 'temp':
        good = np.where(abs(wp) < 1e20)
        color = temp[good][::n]
    elif colormode == 'vr':
        good = np.where(abs(v_r)<1e12)
        color = v_r[good][::n]
    elif colormode == 'np':
        good = np.where(abs(n_p) < 1e12)
        color = n_p[good][::n]
    elif colormode == 'b':
        good = np.where(abs(b_lerp) < 1e12)
        color = b_lerp[good][::n]
    else:
        good = np.where(True)

    theta = (carrlon[good]*np.pi/180)[::n]
    radius = dist[good][::n]

    size = carrlat[good][::n]*30 + 150

    plt.scatter(theta,radius,cmap='jet',c=color,s=300)

ax.set_rmax(0.4)
ax.set_thetamin(270)
ax.set_thetamax(360)
ax.set_rlabel_position(135-45/2)
color_bar = plt.colorbar()
if colormode == 'temp':
    ax.set_title("Proton temperature in heliocentric corotating frame")
    color_bar.set_label("Proton temperature (K)")
elif colormode == 'vr':
    ax.set_title("Radial particle velocity in heliocentric corotating frame")
    color_bar.set_label("Radial particle velocity (km/s)")
elif colormode == 'np':
    ax.set_title("Estimated proton density in heliocentric corotating frame")
    color_bar.set_label("Proton density (cm^-3)")
elif colormode == 'b':
    ax.set_title("Magnetic field strength (interpolated) in heliocentric corotating frame")
    color_bar.set_label('Magnetic field magnitude (nT)')

plt.show()
