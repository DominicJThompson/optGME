#!/bin/bash
#SBATCH --job-name=NDBP_lossTest
#SBATCH --mem=10G
#SBATCH --time=13-23:50:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---- Safety checks ----
if [ -z "$LOSS_INDEX" ]; then
    echo "ERROR: LOSS_INDEX environment variable not set"
    exit 1
fi

if [ -z "$NGS_INDEX" ]; then
    echo "ERROR: NGS_INDEX environment variable not set"
    exit 1
fi
echo "Running the loss index $LOSS_INDEX and NGS index $NGS_INDEX on host $(hostname)"

# ---- Activate your virtual environment ----
source "/global/home/hpc6129/optGME/.venv/bin/activate"

# ---- Limit NumPy/SciPy threads ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Run and time, append to CSV ----
TIME_CMD=$(command -v time)

/usr/bin/env "$TIME_CMD" \
    -f "$LOSS_INDEX,%e,%U,%S,%M" \
    -o "$SLURM_SUBMIT_DIR/media/results.csv" -a \
    python3 slurmruns.py
