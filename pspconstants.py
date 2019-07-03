#!/usr/bin/env python3
"""
pspconstants.py
Greg Szypko, Summer 2019

Defines physical constants as well as relevant path names for use in plotting scripts
"""

import numpy as np
import datetime

au_km = 1.496e8 #1 AU in kilometers

mp_kg = 1.6726219e-27 #proton mass in kg
k_b = 1.38064852e-23 #boltzmann constant in m^2 kg s^-2 K^-1
mu_0 = 4e-7*np.pi #vacuum permeability in T m / A

# Conservative upper bound for data, to filter out fill values
dat_upperbnd = 1e30

# Datetime for noon on 1 Jan 2000
datetime_t0 = datetime.datetime(2000,1,1,12,0,0)

# Path where relevant spc data is located
path = '/data/reu/gszypko/data/approach1/'
# Path where relevant fields data is located
mag_path = '/data/reu/gszypko/data/approach1mag/'
# Path where precomputed data arrays are saved to and loaded from
precomp_path = '/data/reu/gszypko/data/precomp/approach1/'

known_transients = [(datetime.datetime(2018,11,11,23,53,0),datetime.datetime(2018,11,12,6,0,0))]
