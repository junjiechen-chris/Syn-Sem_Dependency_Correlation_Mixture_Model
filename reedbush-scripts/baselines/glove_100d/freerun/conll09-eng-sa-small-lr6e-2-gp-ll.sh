#!/bin/sh
#PBS -q h-regular
#PBS -l select=1
#PBS -W group_list=gk77
#PBS -l walltime=18:00:00
#PBS -N LISA-HI-ELMO
#PBS -j oe
#PBS -M christopher@orudo.cc
#PBS -m abe

JOB_NAME="CONLL09-eng-SA-SMALL-glove_100d-GP-ll"
SAVEDIR=.model/baselines/glove_100d/freerun/conll09-eng-sa-small-lr_6e-2-gp-ll
CONF=config/baselines/glove_100d/conll09-eng-sa-small-gp-ll.conf
#SINGULARITY_IMG=/lustre/gk77/k77015/.Singularity/imgs/LISA.simg
SINGULARITY_IMG=/home/u00222/singularity/images/LISA-tfp.simg
HPARAMS_STR=""
source reedbush-scripts/llisa/hparam_str/conll09.sh
source reedbush-scripts/llisa/hparam_str/lr_6e-2.sh
echo $HPARAMS_STR
ADDITIONAL_PARAMETERS="--hparams  "$HPARAMS_STR"  --use_llisa_proj"
echo $ADDITIONAL_PARAMETERS
source reedbush-scripts/llisa/run_experiment.sh
