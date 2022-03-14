#!/bin/bash
echo $1
while [ "$(squeue -u u00222| wc -l)" -gt $4 ]
do
  sleep 1
done

SAVEDIR=$(cat ./$1 | grep -o "SAVEDIR=.*"| sed -E  's/SAVEDIR=([\.a-zA-Z\/\-]+)/\1/')
CONF=$(cat ./$1 | grep -o "CONF=.*"| sed -E  's/CONF=([\.a-zA-Z\/\-]+)/\1/')
SINGULARITY_IMAGE=/home/u00222/singularity/images/LISA.simg
target_config=$(echo $2 |sed 's/\//\\\//g' )
echo $target_config
config_name=$(basename -- "$2" .conf)
echo $config_name
JOB_NAME=$(cat ./$1 | grep -o "#PBS -N .*"| sed -E  's/#PBS\ -N\ (.*)/\1/')-$config_name
#echo $JOB_NAME-

#cat ./$1
#dropper() {
#   echo "${@:1:$#-2}";
#}

#cat ./$1 | grep -o "singularity exec .*" | sed 's/\/lustre\/gk77\/k77015\/.Singularity\/imgs\/LISA.simg/\/home\/u00222\/singularity\/images\/LISA.simg/' | while read line; do echo "-p p --gres=gpu:1 --mem=24GB" $line ; done | xargs -n 16 -I '{}' echo "srun {} guard"
cat ./$1 | grep -o "singularity exec .*" | sed 's/bin\/train.sh/bin\/evaluate-exported.sh/' |sed "s#\$CONF#$CONF#" |sed "s#\$SAVEDIR#$SAVEDIR#g" | sed -E "s#run-([0-9]{1}) #run-\1/best_checkpoint #" | sed "s#\$SINGULARITY_IMG#$SINGULARITY_IMAGE#" | sed 's/train.log/evaluation./' |sed -E "s/config\/.+conf/$target_config/" |  while read line; do echo "$line$config_name"; done| while read line; do echo "-p $3 --gres=gpu:1 -J $JOB_NAME --cpus-per-task=8 --mem-per-cpu=4096MB " $line ; done |xargs  -n 17 -P 4 -L 1 -I '{}' bash -c "srun {}"
#  while read line; do echo "$line-$config_name"; done| while read line; do echo "-p $3 --gres=gpu:1 --time=1-06:00:00 --mem=24GB" $line ; done | xargs -n 17 -P1 -I '{}' sh -c "sleep 2; echo \"{}\"" | # |# | while read line; do echo "$line-${$2##*/}"; done | xargs -n 13 -P 4 -L 1  -I '{}' echo '{}' #xargs -n 16 -P 4 -L 1 -I '{}' bash -c "{}"
#cat ./$1 | grep -o "singularity exec .*" | sed 's/\/lustre\/gk77\/k77015\/.Singularity\/imgs\/LISA.simg/\/home\/u00222\/singularity\/images\/LISA.simg/' | while read line; do echo "-p p --gres=gpu:1 --mem=24GB" $line ; done | xargs -n 16 -P 4 -L 1 srun

#cat ./$1 | grep -o "singularity exec (.)*"

