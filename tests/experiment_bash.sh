#!/bin/bash -l
#SBATCH --array=0-683

# Each job will get an index from the list of experiment dicts
srun python batch_experiments.py exp_settings.json ${SLURM_ARRAY_TASK_ID}