#!/bin/bash
#SBATCH --job-name=metaData
#SBATCH --output=logs/metaData_%j.out
#SBATCH --error=logs/metaData_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=25G

source "/global/home/hpc6129/optGME/.venv/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python metaDataRun.py
