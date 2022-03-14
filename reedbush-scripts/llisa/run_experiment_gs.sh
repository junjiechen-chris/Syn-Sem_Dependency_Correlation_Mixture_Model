#for RUN_COUNT in 1
#do
#  mkdir -p $SAVEDIR/run-$RUN_COUNT
#  echo "srun -p $1 --gres gpu:1 -t 1:00:00 --cpus-per-task=5 --mem-per-cpu=10240MB -J $JOB_NAME singularity exec --nv $SINGULARITY_IMG bin/train.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT --num_gpus 1 $ADDITIONAL_PARAMETERS 2>&1 $SAVEDIR/run-$RUN_COUNT/train.log &" > $SAVEDIR/run-$RUN_COUNT/cmd.log
#  echo "srun -p $1 --gres gpu:1 -t 1:00:00 --cpus-per-task=5 --mem-per-cpu=10240MB -J $JOB_NAME singularity exec --nv $SINGULARITY_IMG bin/evaluate-exported.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT/best_checkpoint --num_gpus 1 $ADDITIONAL_PARAMETERS &> $SAVEDIR/run-$RUN_COUNT/eval.log &" > $SAVEDIR/run-$RUN_COUNT/eval.cmd
#  echo "srun -p $1 --gres gpu:1 -t 46:00:00 -J $JOB_NAME --cpus-per-task=4 --mem-per-cpu=10240MB singularity exec --nv $SINGULARITY_IMG bin/train.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT --num_gpus 1 $ADDITIONAL_PARAMETERS --neptune_job_name $JOB_NAME  &> $SAVEDIR/run-$RUN_COUNT/train.log &"
#  srun -p $1 --gres gpu:1 -t 46:00:00 -J $JOB_NAME --cpus-per-task=4 --mem-per-cpu=10240MB singularity exec --nv $SINGULARITY_IMG bin/train.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT --num_gpus 1 $ADDITIONAL_PARAMETERS --neptune_job_name $JOB_NAME  &> $SAVEDIR/run-$RUN_COUNT/train.log &
#done
RUN_COUNT=$2
#ADDITIONAL_PARAMS=$3
mkdir -p $SAVEDIR/run-$RUN_COUNT
echo "srun -p $1 --gres gpu:1 -t 1:00:00 --cpus-per-task=5 --mem-per-cpu=10240MB -J $JOB_NAME singularity exec --nv $SINGULARITY_IMG bin/train.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT --num_gpus 1 $ADDITIONAL_PARAMETERS 2>&1 $SAVEDIR/run-$RUN_COUNT/train.log &" > $SAVEDIR/run-$RUN_COUNT/cmd.log
echo "srun -p $1 --gres gpu:1 -t 1:00:00 --cpus-per-task=5 --mem-per-cpu=10240MB -J $JOB_NAME singularity exec --nv $SINGULARITY_IMG bin/evaluate-exported.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT/best_checkpoint --num_gpus 1 $ADDITIONAL_PARAMETERS &> $SAVEDIR/run-$RUN_COUNT/eval.log &" > $SAVEDIR/run-$RUN_COUNT/eval.cmd
echo "srun -p $1 --gres gpu:1 -t 46:00:00 -J $JOB_NAME --cpus-per-task=4 --mem-per-cpu=10240MB singularity exec --nv $SINGULARITY_IMG bin/train.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT --num_gpus 1 $ADDITIONAL_PARAMETERS --neptune_job_name $JOB_NAME  &> $SAVEDIR/run-$RUN_COUNT/train.log &"
srun -p $1 --gres gpu:1 -t 14:00:00 -J $JOB_NAME --cpus-per-task=4 --mem-per-cpu=10240MB singularity exec --nv $SINGULARITY_IMG bin/train.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT --num_gpus 1 $ADDITIONAL_PARAMETERS --neptune_job_name $JOB_NAME  &> $SAVEDIR/run-$RUN_COUNT/train.log &