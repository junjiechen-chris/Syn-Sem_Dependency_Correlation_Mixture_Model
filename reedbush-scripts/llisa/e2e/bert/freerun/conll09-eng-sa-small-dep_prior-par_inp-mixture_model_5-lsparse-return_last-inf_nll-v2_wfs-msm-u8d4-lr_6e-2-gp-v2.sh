#!/bin/sh
#PBS -q h-regular
#PBS -l select=1
#PBS -W group_list=gk77
#PBS -l walltime=18:00:00
#PBS -N LISA-HI-ELMO
#PBS -j oe
#PBS -M christopher@orudo.cc
#PBS -m abe

JOB_NAME="conll09-eng-sa-small-dep_prior-par_inp-mixture_model_5-lsparse-return_last-inf_nll-v2_wfs-msm-u8d4-lr_6e-2-gp-v2"
SAVEDIR=.model/llisa/e2e/bert/freerun/conll09-eng-sa-small-dep_prior-par_inp-mixture_model_5-lsparse-return_last-inf_nll-v2_wfs-msm-u8d4-lr_6e-2-gp-v2
CONF=config/llisa/e2e/bert/conll09-eng-sa-small-dep_prior-par_inp-bilinear-gp.conf
#SINGULARITY_IMG=/lustre/gk77/k77015/.Singularity/imgs/LISA.simg
SINGULARITY_IMG=/home/u00222/singularity/images/LISA-tfp.simg
HPARAMS_STR=""
source reedbush-scripts/llisa/hparam_str/conll09.sh
source reedbush-scripts/llisa/hparam_str/dep_prior_trainable.sh
source reedbush-scripts/llisa/hparam_str/memory_efficient_prior_implementation.sh
source reedbush-scripts/llisa/hparam_str/use_dependency_type_on_bilinear.sh
source reedbush-scripts/llisa/hparam_str/aggregation_weight_one_init.sh
source reedbush-scripts/llisa/hparam_str/learn_mask_per_srl_label.sh
source reedbush-scripts/llisa/hparam_str/exclude_specific_path.sh
source reedbush-scripts/llisa/hparam_str/apply_mean_weight.sh
source reedbush-scripts/llisa/hparam_str/mixture_model_5.sh
source reedbush-scripts/llisa/hparam_str/return_last.sh
#source reedbush-scripts/llisa/hparam_str/share_pred_role_mlp.sh
source reedbush-scripts/llisa/hparam_str/new_updown_search.sh
source reedbush-scripts/llisa/hparam_str/mixture_model_inference_nll.sh
source reedbush-scripts/llisa/hparam_str/lr_6e-2.sh
source reedbush-scripts/llisa/hparam_str/search_u8d4.sh
#source reedbush-scripts/llisa/hparam_str/exact_est_of_prob.sh
source reedbush-scripts/llisa/hparam_str/correct_vi_objective.sh
echo $HPARAMS_STR
ADDITIONAL_PARAMETERS="--hparams  "$HPARAMS_STR"$3 --use_llisa_proj"
echo $ADDITIONAL_PARAMETERS
source reedbush-scripts/llisa/run_experiment.sh
