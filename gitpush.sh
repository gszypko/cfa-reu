#!/bin/tcsh
#gitpush.sh
#A script to automate pushing my program directory to git
#Greg Szypko

set datetime = `date +"%F %T"`
git add .
git commit -m "$datetime"
git push
