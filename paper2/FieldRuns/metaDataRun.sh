#!/bin/bash
#SBATCH --job-name=metaData
#SBATCH --output=logs/metaData_%A_%a.out
#SBATCH --error=logs/metaData_%A_%a.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=25G
#SBATCH --array=0-8

source "/global/home/hpc6129/optGME/.venv/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# This will assign ng_index for 3x10s, 3x20s, then 3x30s as task IDs 0-2, 3-5, 6-8
# Note: make sure to set --array=0-8 in the SLURM directive above to match
case $SLURM_ARRAY_TASK_ID in
    0|1|2)
        ng_index=10
        ;;
    3|4|5)
        ng_index=20
        ;;
    6|7|8)
        ng_index=30
        ;;
    *)
        echo "Invalid SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac
ndbp_index=$((SLURM_ARRAY_TASK_ID % 3))

export NG_INDEX="${ng_index}"
export NDBP_INDEX="${ndbp_index}"

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} -> ng${ng_index} ndbp${ndbp_index}"
python -u metaDataRun.py
