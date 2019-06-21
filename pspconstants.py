#!/usr/bin/env python3
"""
pspconstants.py
Greg Szypko, Summer 2019

Defines physical constants as well as relevant path names for use in plotting scripts
"""

import numpy as np

au_km = 1.496e8

mp_kg = 1.6726219e-27 #proton mass in kg
k_b = 1.38064852e-23 #boltzmann constant in m^2 kg s^-2 K^-1
mu_0 = 4e-7*np.pi #vacuum permeability in T m / A

path = '/data/reu/gszypko/data/loopback/'
mag_path = '/data/reu/gszypko/data/mag/'
