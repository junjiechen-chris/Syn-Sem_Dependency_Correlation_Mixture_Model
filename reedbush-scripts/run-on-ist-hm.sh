#!/bin/sh
echo "USAGE: <rb_script> <max running instance> <running partition>"
#echo $1
#cat ./$1
while [ "$(squeue -u u00222| wc -l)" -gt $2 ]
do
  sleep 1
done
#cat ./$1
SAVEDIR=$(cat ./$1 | grep -o "SAVEDIR=.*"| sed -E  's/SAVEDIR=([\.a-zA-Z\/\-]+)/\1/')
CONF=$(cat ./$1 | grep -o "CONF=.*"| sed -E  's/CONF=([\.a-zA-Z\/\-]+)/\1/')
SINGULARITY_IMAGE=/home/u00222/singularity/images/LISA.simg
JOB_NAME=$(cat ./$1 | grep -o "#PBS -N .*"| sed -E  's/#PBS\ -N\ (.*)/\1/')
ADDITIONAL_PARAMETERS=$(cat ./$1 | grep -o "ADDITIONAL_PARAMETERS=.*"| sed -E  's/ADDITIONAL_PARAMETERS=([\.a-zA-Z\/\-\"]+)/\1/')
echo $JOB_NAME
echo $SAVEDIR $CONF $SINGULARITY_IMAGE

#cat ./$1 | grep -o "singularity exec .*" | sed 's/\/lustre\/gk77\/k77015\/.Singularity\/imgs\/LISA.simg/\/home\/u00222\/singularity\/images\/LISA.simg/' | while read line; do echo "-p p --gres=gpu:1 --mem=24GB" $line ; done | xargs -n 16 -I '{}' echo "srun {} guard"
mkdir -p $SAVEDIR/run-1
mkdir -p $SAVEDIR/run-2
cat ./$1 | grep -o "singularity exec .*" |sed "s#\$CONF#$CONF#" | sed "s#\$SAVEDIR#$SAVEDIR#g" | sed "s#\$SINGULARITY_IMG#$SINGULARITY_IMAGE#" | sed "s#\$ADDITIONAL_PARAMETERS#$ADDITIONAL_PARAMETERS#g"  | while read line; do echo "-p $3 --gres=gpu:1 -J $JOB_NAME --cpus-per-task=9 --mem-per-cpu=10240MB " $line ; done |  xargs -n 16 -P 2 -L 1 -I '{}' bash -c "echo {}"#"srun {}"
#cat ./$1 | grep -o "singularity exec .*" | sed 's/\/lustre\/gk77\/k77015\/.Singularity\/imgs\/LISA.simg/\/home\/u00222\/singularity\/images\/LISA.simg/' | while read line; do echo "-p p --gres=gpu:1 --mem=24GB" $line ; done | xargs -n 16 -P 4 -L 1 srun

#cat ./$1 | grep -o "singularity exec (.)*"

