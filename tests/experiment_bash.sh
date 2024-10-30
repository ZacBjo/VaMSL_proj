#!/bin/bash -l
#SBATCH --array=0-3
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/log_%A_%a.out
#SBATCH --error=logs/log_%A_%a.err

# for aalto sci-comp env
#module load scicomp-python-env
#export PYTHONPATH=$WRKDIR/PML_2024/VaMSL_proj:$PYTHONPATH

source activate ../env/

export PYTHONUNBUFFERED=1 # force python to output print pronto
# Each job will get an index from the list of experiment dicts
python batch_experiments.py mixing_exp_settings.json ${SLURM_ARRAY_TASK_ID}
