#!/usr/bin/env python3

import numpy as np
from pspconstants import *
import sys

endidx = 15

array_files = []

filename = sys.argv[1]

for i in range(0,endidx+1):
    array_files.append(np.load(precomp_path+filename+str(i).zfill(3)+'.npy'))

full_array = np.array(array_files[0])

for i in range(1,endidx+1):
    full_array = np.append(full_array,array_files[i])

np.save(precomp_path+filename+'.npy',full_array)