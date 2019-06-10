#!/usr/bin/env python3

import cdflib
import numpy as np
from matplotlib import pyplot as plt
import datetime
from os import listdir
from scipy.optimize import curve_fit

dataloc = '/data/reu/gszypko/data/loopback/'
file_names = listdir(dataloc)
dat = cdflib.CDF(dataloc+file_names[0])

carrlon = dat.varget('carr_longitude')
scpos = dat.varget('sc_pos_HCI')
vp = dat.varget('vp_moment_RTN')
#n_p = dat.varget('np1_fit')
dqf = dat.varget('DQF')

for i in range(1,len(file_names)):
	this_dat = cdflib.CDF(dataloc + file_names[i])
	this_carrlon = this_dat.varget('carr_longitude')
	this_scpos = this_dat.varget('sc_pos_HCI')
	this_vp = this_dat.varget('vp_moment_RTN')
	#this_n_p = this_dat.varget('np1_fit')
	this_dqf = this_dat.varget('DQF')
	carrlon = np.append(carrlon, this_carrlon)
	scpos = np.append(scpos, this_scpos, axis=0)
	vp = np.append(vp, this_vp, axis=0)
	#n_p = np.append(n_p, this_n_p)
	dqf = np.append(dqf, this_dqf,axis=0)

au_km = 1.496e8
xloc = scpos[:,0]/au_km
yloc = scpos[:,1]/au_km
#zloc = scpos[:,2]/au_km
zloc = 0 #projecting all positions down into the solar equatorial plane
dist = np.sqrt(xloc**2 + yloc**2 + zloc**2)
#n_p.sort()
#print(n_p)
#bad = np.where(n_p<=dat.varattsget('np1_fit')['FILLVAL'])
#n_p[bad]=np.nan
#print(n_p)
#n_p[np.where(n_p<0)]=np.nan
#print(n_p)

#carrlon[bad]=np.nan
#dist[bad]=np.nan

v_r = vp[:,0]
#v_t = vp[:,1]
#v_n = vp[:,2]
#v_norm = np.sqrt(v_r**2 + v_t**2 + v_n**2)

good = np.where(abs(v_r)<1e12)

fig = plt.figure()
ax = fig.add_subplot(111,projection='polar')

plt.scatter(carrlon[good]*np.pi/180,dist[good],c=v_r[good],cmap='rainbow',s=0.2)
ax.set_rmax(0.5)
plt.colorbar()
plt.show()
