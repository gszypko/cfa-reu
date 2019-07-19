#!/bin/tcsh

#temp=temperature, vr=radial vel, np=density, b=magnetic field
#beta=plasma beta, alf=alfven speed, alfmach=alfven machn number

echo "starting"
./radial.py vr
echo "vr plotted"
./radial.py np
echo "np plotted"
./radial.py temp
echo "temp plotted"
./radial.py beta
echo "beta plotted"
./radial.py alf
echo "alf plotted"
./radial.py alfmach
echo "alfmach plotted"
./radial.py b
echo "b plotted"
echo "complete!"
