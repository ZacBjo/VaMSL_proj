#!/bin/bash -l
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gpus=1
#SBATCH --output=logs/log_%A_%a.out
#SBATCH --error=logs/log_%A_%a.err

# for aalto sci-comp env
#module load scicomp-python-env
#export PYTHONPATH=$WRKDIR/PML_2024/VaMSL_proj:$PYTHONPATH

source activate ../env/

export PYTHONUNBUFFERED=1 # force python to output print pronto
# Each job will get an index from the list of experiment dicts
python batch_experiment_results.py experiment_settings/1_source_2_components_d20_N200_S9000_U100_sf.json