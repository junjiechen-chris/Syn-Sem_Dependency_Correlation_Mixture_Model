#for RUN_COUNT in 1
#do
#  mkdir -p $SAVEDIR/run-$RUN_COUNT
#  echo "srun -p $1 --gres gpu:1 -t 1:00:00 -J $JOB_NAME singularity exec --nv $SINGULARITY_IMG bin/train.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT --num_gpus 1 $ADDITIONAL_PARAMETERS 2>&1 $SAVEDIR/run-$RUN_COUNT/train.log &" > $SAVEDIR/run-$RUN_COUNT/cmd.log
#  echo "srun -p $1 --cpus-per-task=9 -t 1:00:00 --mem-per-cpu=10240MB --gres gpu:1 -J $JOB_NAME singularity exec --nv $SINGULARITY_IMG bin/evaluate-exported.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT/best_checkpoint --num_gpus 1 $ADDITIONAL_PARAMETERS &> $SAVEDIR/run-$RUN_COUNT/eval.log &" > $SAVEDIR/run-$RUN_COUNT/eval.cmd
#  srun -p $1 --gres gpu:1 -J $JOB_NAME -t 46:00:00 --cpus-per-task=6 --mem-per-cpu=13240MB singularity exec --nv $SINGULARITY_IMG bin/train.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT --num_gpus 1 $ADDITIONAL_PARAMETERS --neptune_job_name $JOB_NAME  &> $SAVEDIR/run-$RUN_COUNT/train.log &
#done

RUN_COUNT=$2
AO_CONF=${CONF:0:-5}-attachment_only.conf
mkdir -p $SAVEDIR/run-$RUN_COUNT
echo "srun -p $1 --gres gpu:1 -t 1:00:00 --cpus-per-task=5 --mem-per-cpu=10240MB -J $JOB_NAME singularity exec --nv $SINGULARITY_IMG bin/train.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT --num_gpus 1 $ADDITIONAL_PARAMETERS 2>&1 $SAVEDIR/run-$RUN_COUNT/train.log &" > $SAVEDIR/run-$RUN_COUNT/cmd.log
echo "srun -p $1 --gres gpu:1 -t 1:00:00 --cpus-per-task=5 --mem-per-cpu=10240MB -J $JOB_NAME singularity exec --nv $SINGULARITY_IMG bin/evaluate-exported.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT/best_checkpoint --num_gpus 1 $ADDITIONAL_PARAMETERS &> $SAVEDIR/run-$RUN_COUNT/eval.log &" > $SAVEDIR/run-$RUN_COUNT/eval.cmd
echo "srun -p $1 --gres gpu:1 -t 1:00:00 --cpus-per-task=5 --mem-per-cpu=10240MB -J $JOB_NAME singularity exec --nv $SINGULARITY_IMG bin/evaluate-exported.sh $AO_CONF --save_dir $SAVEDIR/run-$RUN_COUNT/best_checkpoint --num_gpus 1 $ADDITIONAL_PARAMETERS &> $SAVEDIR/run-$RUN_COUNT/eval-attachment_only.log &" >> $SAVEDIR/run-$RUN_COUNT/eval.cmd
echo "srun -p $1 --gres gpu:1 -t 25:00:00 -J $JOB_NAME --cpus-per-task=4 --mem-per-cpu=10240MB singularity exec --nv $SINGULARITY_IMG bin/train.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT --num_gpus 1 $ADDITIONAL_PARAMETERS --neptune_job_name $JOB_NAME  &> $SAVEDIR/run-$RUN_COUNT/train.log &"
srun -p $1 --gres gpu:1 -t 40:00:00 -J $JOB_NAME --cpus-per-task=6 --mem-per-cpu=13240MB singularity exec --nv $SINGULARITY_IMG bin/train.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT --num_gpus 1 $ADDITIONAL_PARAMETERS --neptune_job_name $JOB_NAME  &> $SAVEDIR/run-$RUN_COUNT/train.log &