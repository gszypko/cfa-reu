#!/bin/tcsh

#UNCOMMENT FOR VERBOSE COPYING
#rsync -avzh /data/reu/gszypko/data/* /data/reu/gszypko/backup
rsync -azh /data/reu/gszypko/data/* /data/reu/gszypko/backup
zip -r /data/reu/gszypko/backup.zip /data/reu/gszypko/backup/
#UNCOMMENT TO DELETE COPIED FILE
#rm -r /data/reu/gszypko/backup/
