#!/bin/bash
#SBATCH --job-name=metaData
#SBATCH --output=logs/metaData_%A_%a.out
#SBATCH --error=logs/metaData_%A_%a.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=25G
#SBATCH --array=0-15

source "/global/home/hpc6129/optGME/.venv/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

ng_index=$((SLURM_ARRAY_TASK_ID / 4))
ndbp_index=$((SLURM_ARRAY_TASK_ID % 4))

export NG_INDEX="${ng_index}"
export NDBP_INDEX="${ndbp_index}"

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} -> ng${ng_index} ndbp${ndbp_index}"
python metaDataRun.py
