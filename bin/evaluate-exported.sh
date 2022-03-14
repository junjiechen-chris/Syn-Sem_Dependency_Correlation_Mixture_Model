#!/usr/bin/env bash
echo $1
config_file=$1

source ${config_file}

params=${@:2}

echo "Using CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES

transition_stats=$data_dir/transition_probs.tsv

echo "python3 src/evaluate_exported.py \
--test_files $test_files \
--dev_files $dev_files \
--transition_stats $transition_stats \
--data_config $data_config \
--model_configs $model_configs \
--task_configs $task_configs \
--layer_configs $layer_configs \
--attention_configs "$attention_configs" \
--hparam "mode=evaluate" \
$params"

python3 src/evaluate_exported.py \
--test_files $test_files \
--dev_files $dev_files \
--transition_stats $transition_stats \
--data_config $data_config \
--model_configs $model_configs \
--task_configs $task_configs \
--layer_configs $layer_configs \
--attention_configs "$attention_configs" \
--hparam "mode=evaluate" \
$params

