#!/usr/bin/env python3

import cdflib
import numpy as np
from matplotlib import pyplot as plt
import datetime
from scipy.optimize import curve_fit

#datapath = '/data/reu/gszypko/data/l2/spp_swp_spc_l2_20180830_v17.cdf'
datapath = '/data/reu/gszypko/data/l2/spp_swp_spc_l2_20181023_v11.cdf'
dat = cdflib.CDF(datapath)

timeslc = 250

epoch_ns = dat.varget('Epoch')
datetime_t0 = datetime.datetime(2000,1,1,12,0,0)
epoch = np.array([datetime_t0+datetime.timedelta(seconds=i/1e9) for i in epoch_ns])

flux = dat.varget('diff_charge_flux_density')
flux_atts = dat.varattsget('diff_charge_flux_density')

bad = np.where(flux==flux_atts['FILLVAL'])
flux[bad] = np.nan

flux_error_varname = flux_atts['DELTA_PLUS_VAR']
flux_error = dat.varget(flux_error_varname)

mvhi = dat.varget('mv_hi')
mvhi[np.where(mvhi==dat.varattsget('mv_hi')['FILLVAL'])]=np.nan
mvlo = dat.varget('mv_lo')
mvlo[np.where(mvlo==dat.varattsget('mv_lo')['FILLVAL'])]=np.nan
mvmid = 0.5*(mvlo+mvhi)

fig,ax = plt.subplots(1,1)
ax.errorbar(mvmid[timeslc],flux[timeslc],yerr=flux_error[timeslc],marker='.',linestyle='-')
ax.set_xlabel(dat.varattsget('mv_lo')['LABLAXIS']+' ('+dat.varattsget('mv_lo')['UNITS']+')')
ax.set_ylabel(dat.varattsget('diff_charge_flux_density')['LABLAXIS']+' ('+dat.varattsget('diff_charge_flux_density')['UNITS']+')')
ax.set_title('Date/Time='+str(epoch[timeslc]))

qe = 1.60217662e-19
mp = 1.6726219e-27
velocity_low_kms = 1e-3 * np.sqrt(2*qe*mvlo/mp)
velocity_hi_kms = 1e-3 * np.sqrt(2*qe*mvhi/mp)
velocity_mid = 1e-3 * np.sqrt(2*qe*mvmid/mp)
dv = velocity_hi_kms - velocity_low_kms
rdf = 1e-5 * 1e-12 * flux / (qe*velocity_mid*dv)

fig,ax = plt.subplots(1,1)
ax.plot(velocity_mid[timeslc],rdf[timeslc],marker='.',linestyle='-')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('RDF (cm^-3 (km/s)^-1')
ax.set_title('Date/Time'+str(epoch[timeslc]))

velocity_guess = velocity_mid[timeslc,[np.nanargmax(rdf[timeslc,:])]]
def maxwellian(velocity_mid, denp, vp, v_th_kms): return(denp/((v_th_kms*np.sqrt(np.pi)))*np.exp(-(velocity_mid-vp)**2 / v_th_kms**2))
gpix = np.where(rdf[timeslc,:]==rdf[timeslc,:]) #where not NaN
popt, pcurve = curve_fit(maxwellian, velocity_mid[timeslc][gpix], rdf[timeslc,gpix].flatten(), p0=[10, velocity_guess, 50])

fig,ax = plt.subplots(1,1)
ax.plot(velocity_mid[timeslc],rdf[timeslc],marker='.',linestyle='-')
velocity_values = np.arange(200,800,1)
fitlabel = 'np={:6.2f} cm^-3\nvp={:10.2f} km/s'.format(popt[0],popt[1],popt[2])
ax.plot(velocity_values, maxwellian(velocity_values, *popt), '-r', label=fitlabel)
ax.set_xlabel('Velocity km/s')
ax.set_ylabel(dat.varattsget('diff_charge_flux_density')['LABLAXIS']+' ('+dat.varattsget('diff_charge_flux_density')['UNITS']+')')
ax.set_title('Date/Time='+str(epoch[timeslc]))
ax.legend(loc='upper right', prop={'size':8})

fig,ax = plt.subplots(1,1)
for i in range(300): foo=ax.scatter([epoch[i]]*len(mvmid[i]), mvmid[i], c=flux[i,:], marker='s',s=20)
ax.set_yscale('log')
ax.set_xlim((epoch[0],epoch[300]))

l3filename = '/data/reu/gszypko/data/l3/spp_swp_spc_l3i_20181023_v05.cdf'
l3dat = cdflib.CDF(l3filename)

l3epoch = l3dat.varget('Epoch')
l3epoch_ns = dat.varget('Epoch')
l3epoch = np.array([datetime_t0+datetime.timedelta(seconds=i/1e9) for i in l3epoch_ns])

vpmom = l3dat.varget('vp_moment_RTN')
bpix = np.where(vpmom==l3dat.varattsget('vp_moment_RTN')['FILLVAL'])
vpmom[bpix] = np.nan
vr = vpmom[:,0]
vt = vpmom[:,1]
vn = vpmom[:,2]

dqf = l3dat.varget('DQF')
gpix = np.where(dqf[:,0]==0)

fig,ax = plt.subplots(1,1)
ax.plot(l3epoch[gpix], vr[gpix], color='r', label='Vr')
ax.plot(l3epoch[gpix], vt[gpix], color='g', label='Vt')
ax.plot(l3epoch[gpix], vn[gpix], color='b', label='Vn')
ax.set_ylabel(l3dat.varattsget('vp_moment_RTN')['LABLAXIS']+' ('+l3dat.varattsget('vp_moment_RTN')['UNIT_PTR'][0]+')')
ax.legend(loc='upper left', prop={'size':7})
fig.autofmt_xdate()

carrlat = l3dat.varget('carr_latitude')
carrlon = l3dat.varget('carr_longitude')
scpos = l3dat.varget('sc_pos_HCI')
au_km = 1.496e8
xloc = scpos[:,0]/au_km
yloc = scpos[:,1]/au_km
zloc = scpos[:,2]/au_km
dist = np.sqrt(xloc**2 + yloc**2 + zloc**2)

fig,axes = plt.subplots(2,1)
axes[0].plot(l3epoch,carrlat,color='r',label='Lat')
axes[0].plot(l3epoch,carrlon,color='b',label='Lon')
axes[0].legend(loc='upper right',prop={'size':8})
axes[0].set_ylabel('Carrington Position (degrees)')
axes[1].plot(l3epoch,xloc,color='r',label='x')
axes[1].plot(l3epoch,yloc,color='g',label='y')
axes[1].plot(l3epoch,zloc,color='b',label='z')
axes[1].set_ylabel('HCI Position (AU)')
axes[1].legend(loc='upper right',prop={'size':8})
fig.autofmt_xdate()

vth = l3dat.varget('wp_moment')
vth[np.where(vth==l3dat.varattsget('wp_moment')['FILLVAL'])]=np.nan
fig,ax = plt.subplots(1,1)
ax.plot(dist,vth,color='r',label='v_th')
ax.set_xlabel('Heliocentric Distance (AU)')
ax.set_ylabel(l3dat.varattsget('wp_moment')['LABLAXIS'] + ' (' + l3dat.varattsget('wp_moment')['UNITS']+')')

plt.show()
