#!/usr/bin/env bash

config_file=$1

source ${config_file}

params=${@:2}

echo "Using CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES

#transition_stats=$data_dir/transition_probs.tsv

echo $data_dir

#echo "python3 src/train.py --train_files $train_files   --dev_files $dev_files   --transition_stats $transition_stats   --data_config $data_config --attention_configs $attention_configs  --model_configs $model_configs   --task_configs $task_configs   --layer_configs $layer_configs   --best_eval_key $best_eval_key   $params"
echo $attention_configs
if ! [ -z "$attention_configs" ]
then
  additional_params="$additional_params --attention_configs $attention_configs"
  echo $additional_params
fi
if ! [ -z "$okazaki_discounting" ]
then
  additional_params="$additional_params --okazaki_discounting"
  echo $additional_params
fi
echo "python3 src/train.py --train_files $train_files   --dev_files $dev_files   --transition_stats $transition_stats   --data_config $data_config --attention_configs $attention_configs  --model_configs $model_configs   --task_configs $task_configs   --layer_configs $layer_configs   --best_eval_key $best_eval_key   $params $additional_params "
#save_dir=$(grep -e "save_dir [^\ ]" $params)
#echo $save_dir


python3 src/train.py \
--train_files $train_files \
--dev_files $dev_files \
--transition_stats $transition_stats \
--data_config $data_config \
--model_configs $model_configs \
--task_configs $task_configs \
--layer_configs $layer_configs \
--best_eval_key $best_eval_key \
$params \
$additional_params


#echo "find  -regex '.*best_exporter/[0-9]+' | xargs -n 1 -I {} singularity exec --nv /lustre/gk77/k77015/.Singularity/imgs/LISA.simg bin/evaluate-exported.sh config/lisa-discounting/conll05-lisa.conf --save_dir {} 2> mypipe | grep srl_f1 mypipe"

#if [ -z "$attention_configs" ]
#then
#  python3 src/train.py \
#  --train_files $train_files \
#  --dev_files $dev_files \
#  --transition_stats $transition_stats \
#  --data_config $data_config \
#  --model_configs $model_configs \
#  --task_configs $task_configs \
#  --layer_configs $layer_configs \
#  --best_eval_key $best_eval_key \
#  $params
#else
#  python3 src/train.py \
#  --train_files $train_files \
#  --dev_files $dev_files \
#  --transition_stats $transition_stats \
#  --data_config $data_config \
#  --model_configs $model_configs \
#  --task_configs $task_configs \
#  --layer_configs $layer_configs \
#  --attention_configs $attention_configs \
#  --best_eval_key $best_eval_key \
#  $params
#fi
#--num_gpus 2\


