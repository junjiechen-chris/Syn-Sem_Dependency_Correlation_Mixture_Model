#!/bin/sh
#PBS -q h-regular
#PBS -l select=1
#PBS -W group_list=gk77
#PBS -l walltime=18:00:00
#PBS -N LISA-HI-ELMO
#PBS -j oe
#PBS -M christopher@orudo.cc
#PBS -m abe

JOB_NAME="CONLL09-ger-SA-SMALL-bert-PX3-srl_only-GP"
SAVEDIR=.model/baselines/bert/freerun/conll09-ger-sa-small-srl_only-lr_6e-2-gp
CONF=config/baselines/bert/conll09-ger-sa-small-srl_only-gp.conf
#SINGULARITY_IMG=/lustre/gk77/k77015/.Singularity/imgs/LISA.simg
SINGULARITY_IMG=/home/u00222/singularity/images/LISA-tfp.simg
HPARAMS_STR=""
source reedbush-scripts/llisa/hparam_str/conll09.sh
source reedbush-scripts/llisa/hparam_str/lr_6e-2.sh
echo $HPARAMS_STR
ADDITIONAL_PARAMETERS="--hparams  "$HPARAMS_STR"  --use_llisa_proj"
echo $ADDITIONAL_PARAMETERS
source reedbush-scripts/llisa/run_experiment.sh
