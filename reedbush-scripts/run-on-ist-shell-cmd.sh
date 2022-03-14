#!/bin/sh
echo $1
#cat ./$1
while [ "$(squeue -u u00222| wc -l)" -gt 10 ]
do
  sleep 1
done
#echo "$@"
sh -c "$@"
#cat ./$1 | grep -o "singularity exec (.)*"

