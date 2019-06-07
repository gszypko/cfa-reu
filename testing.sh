#!/bin/tcsh

echo "$1"
if ( "$1" == "" ) then
	echo "empty variable"
else
	echo "full variable"
endif
