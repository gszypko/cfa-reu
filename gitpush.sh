#!/bin/tcsh
#gitpush.sh
#A script to automate pushing my program directory to git
#Greg Szypko

if ( "$1" == "" ) then
        set message = `date +"%F %T"`
else
        set message = "$1"
endif
git add .
git commit -m "$message"
git push
