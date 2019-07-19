#!/bin/tcsh
#plotangslices.sh
#Automates production of radial angular slice plots for different variables

./angslice.py vr 8 12
./angslice.py np 8 12
./angslice.py temp 8 12

./angslice.py vr -6 1
./angslice.py np -6 1
./angslice.py temp -6 1

# ./angslice.py vr -43 -39
# ./angslice.py np -43 -39
# ./angslice.py temp -43 -39

# ./angslice.py vr -36 -29
# ./angslice.py np -36 -29
# ./angslice.py temp -36 -29
# 
# ./angslice.py vr -26 -23
# ./angslice.py np -26 -23
# ./angslice.py temp -26 -23

# echo "starting"
# ./angslice.py temp
# echo "temp complete"
# ./angslice.py vr
# echo "vr complete"
# ./angslice.py np
# echo "np complete"
# ./angslice.py beta
# echo "beta complete"
# ./angslice.py alf
# echo "alf complete"
# ./angslice.py alfmach
# echo "alfmach complete"
# ./angslice.py b
# echo "b complete"
# echo "all done"

