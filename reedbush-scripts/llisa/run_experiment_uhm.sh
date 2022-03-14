for RUN_COUNT in 1 2
do
  mkdir -p $SAVEDIR/run-$RUN_COUNT
  echo "srun -p $1 --gres gpu:1 -J $JOB_NAME singularity exec --nv $SINGULARITY_IMG bin/train.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT --num_gpus 1 $ADDITIONAL_PARAMETERS 2>&1 $SAVEDIR/run-$RUN_COUNT/train.log" > $SAVEDIR/run-$RUN_COUNT/cmd.log
  echo "srun -p $1 --cpus-per-task=9 --mem-per-cpu=12360MB --gres gpu:1 -J $JOB_NAME singularity exec --nv $SINGULARITY_IMG bin/evaluate-exported.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT/best_checkpoint --num_gpus 1 $ADDITIONAL_PARAMETERS &> $SAVEDIR/run-$RUN_COUNT/eval.log" > $SAVEDIR/run-$RUN_COUNT/eval.cmd
  srun -p $1 --gres gpu:1 -J $JOB_NAME --cpus-per-task=9 --mem-per-cpu=15360MB singularity exec --nv $SINGULARITY_IMG bin/train.sh $CONF --save_dir $SAVEDIR/run-$RUN_COUNT --num_gpus 1 $ADDITIONAL_PARAMETERS --neptune_job_name $JOB_NAME  &> $SAVEDIR/run-$RUN_COUNT/train.log &
  sleep 10
done