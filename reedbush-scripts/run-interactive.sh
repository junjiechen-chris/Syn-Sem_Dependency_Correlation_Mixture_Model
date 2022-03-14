#!/bin/sh
#PBS -q h-interactive
#PBS -l select=1
#PBS -W group_list=gk77
#PBS -l walltime=02:00:00
#PBS -N SA
#PBS -j oe
#PBS -M christopher@orudo.cc

export PATH=$PBS_O_PATH:$PATH

cd $PBS_O_WORKDIR

module add cuda10 singularity/2.5.1 
rm SA.o*
singularity exec --nv /lustre/gk77/k77015/.Singularity/imgs/LISA.simg bin/train.sh config/conll05-sa.conf --save_dir .model-sa

