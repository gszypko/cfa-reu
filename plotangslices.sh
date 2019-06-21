#!/bin/tcsh
#plotangslices.sh
#Automates production of radial angular slice plots for different variables

echo "starting"
./angslice.py temp
echo "temp complete"
./angslice.py vr
echo "vr complete"
./angslice.py np
echo "np complete"
./angslice.py beta
echo "beta complete"
./angslice.py alf
echo "alf complete"
./angslice.py alfmach
echo "alfmach complete"
./angslice.py b
echo "b complete"
echo "all done"

