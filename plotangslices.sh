#!/bin/tcsh
#plotangslices.sh
#Automates production of radial angular slice plots for different variables

# foreach n ( "-43 -39" "-36 -29" "-26 -23" )
# foreach n ( "8 12" "-6 1" )
#     set filename=`echo $n | tr ' ' ',' | tr -d '-' `
#     set foldername="stream_candidates_spiral_equatorrot"
#     ./angslice.py np $n --spiralcolor --spiralslice --filtered --tofile $foldername/$filename/
#     ./angslice.py temp $n --spiralcolor --spiralslice --filtered --tofile $foldername/$filename/
#     ./angslice.py vr $n --spiralcolor --spiralslice --filtered --tofile $foldername/$filename/
#     ./angslice.py np $n --spiralcolor --spiralslice --tofile $foldername/$filename/
#     ./angslice.py temp $n --spiralcolor --spiralslice --tofile $foldername/$filename/
#     ./angslice.py vr $n --spiralcolor --spiralslice --tofile $foldername/$filename/
# end

./angslice.py np -45 -20 --spiralcolor --spiralslice --filtered --tofile spiralcolor_test/approach1/
./angslice.py temp -45 -20 --spiralcolor --spiralslice --filtered --tofile spiralcolor_test/approach1/
# ./angslice.py np -10 15 --spiralcolor --spiralslice --filtered --tofile spiralcolor_test/approach2/
# ./angslice.py temp -10 15 --spiralcolor --spiralslice --filtered --tofile spiralcolor_test/approach2/

# ./angslice.py b 8 12 --longcolor --tofile stream_candidates/8,12/
# ./angslice.py b -6 1 --longcolor --tofile stream_candidates/6,1/

# ./angslice.py b -43 -39 --longcolor --tofile stream_candidates/43,39/
# ./angslice.py b -36 -29 --longcolor --tofile stream_candidates/36,29/
# ./angslice.py b -26 -23 --longcolor --tofile stream_candidates/26,23/

# ./angslice.py vr 8 12
# ./angslice.py np 8 12
# ./angslice.py temp 8 12
# 
# ./angslice.py vr -6 1
# ./angslice.py np -6 1
# ./angslice.py temp -6 1

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

