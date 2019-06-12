#!/usr/bin/env python3

import cdflib
import numpy as np
from matplotlib import pyplot as plt
#import datetime
from os import listdir
#from scipy.optimize import curve_fit

au_km = 1.496e8

dataloc = '/data/reu/gszypko/data/loopback/'
file_names = listdir(dataloc)

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111,projection='polar')

for i in range(0,len(file_names)):
	dat = cdflib.CDF(dataloc + file_names[i])
	carrlon = dat.varget('carr_longitude')
	carrlat = dat.varget('carr_latitude')
	scpos = dat.varget('sc_pos_HCI')
	vp = dat.varget('vp_moment_RTN')
	n_p = dat.varget('np_moment')	

	xloc = scpos[:,0]/au_km
	yloc = scpos[:,1]/au_km
	#zloc = scpos[:,2]/au_km
	zloc = 0 #projecting all positions down into the solar equatorial plane
	dist = np.sqrt(xloc**2 + yloc**2 + zloc**2)	

	# plotting radial velocity with reasonable values
	v_r = vp[:,0]	
	
	#good = np.where(abs(v_r)<1e12)
	good = np.where(abs(n_p) < 1e12)
	
	n=50 #step between good data points to plot
	theta = (carrlon[good]*np.pi/180)[::n]
	radius = dist[good][::n]
	color = n_p[good][::n]
	#color = v_r[good][::n]
	size = carrlat[good][::n]*30 + 150

	plt.scatter(theta,radius,cmap='jet',c=color,s=size)
	
ax.set_rmax(0.5)
ax.set_rlabel_position(135-45/2)
ax.set_title("Estimated proton density in heliocentric corotating frame")
#ax.set_title("Radial particle velocity in heliocentric corotating frame")
color_bar = plt.colorbar()
color_bar.set_label("Proton density (cm^-3)")
#color_bar.set_label("Radial particle velocity (km/s)")
plt.show()
