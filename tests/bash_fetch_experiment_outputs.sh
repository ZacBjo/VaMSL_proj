#!/bin/bash -l
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gpus=1
#SBATCH --output=logs/log_%A_%a.out
#SBATCH --error=logs/log_%A_%a.err

# for aalto sci-comp env
#module load scicomp-python-env
#export PYTHONPATH=$WRKDIR/PML_2024/VaMSL_proj:$PYTHONPATH

source activate ../vamsl_env_2/

export PYTHONUNBUFFERED=1 # force python to output print pronto
# Each job will get an index from the list of experiment dicts
python fetch_experiment_outputs.py experiment_settings/dibs_recreation_d20_mixing_rates_N200_S3000_U10_sf.json
