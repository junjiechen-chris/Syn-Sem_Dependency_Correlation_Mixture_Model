#!/bin/sh
echo $1
while [ "$(ps -fA | grep evaluate_exported.py| wc -l)" -gt 2 ]
do
  sleep 1
done

#cat ./$1
#dropper() {
#   echo "${@:1:$#-2}";
#}
target_config=$(echo $2 |sed 's/\//\\\//g' )
echo $target_config
config_name=$(basename -- "$2" .conf)
#cat ./$1 | grep -o "singularity exec .*" | sed 's/\/lustre\/gk77\/k77015\/.Singularity\/imgs\/LISA.simg/\/home\/u00222\/singularity\/images\/LISA.simg/' | while read line; do echo "-p p --gres=gpu:1 --mem=24GB" $line ; done | xargs -n 16 -I '{}' echo "srun {} guard"
cat ./$1 | grep -o "CUDA_VISIBLE_DEVICES.*" | sed 's/bin\/train.sh/bin\/evaluate-exported.sh/'   |sed 's/.log/.result/' |sed -E "s/config\/.+conf/$target_config/" | while read line; do echo "$line-$config_name"; done|xargs -n 16 -P 4 -L 1 -I '{}' bash -c "{}" # |# | while read line; do echo "$line-${$2##*/}"; done | xargs -n 13 -P 4 -L 1  -I '{}' echo '{}' #xargs -n 16 -P 4 -L 1 -I '{}' bash -c "{}"
#cat ./$1 | grep -o "singularity exec .*" | sed 's/\/lustre\/gk77\/k77015\/.Singularity\/imgs\/LISA.simg/\/home\/u00222\/singularity\/images\/LISA.simg/' | while read line; do echo "-p p --gres=gpu:1 --mem=24GB" $line ; done | xargs -n 16 -P 4 -L 1 srun

#cat ./$1 | grep -o "singularity exec (.)*"

