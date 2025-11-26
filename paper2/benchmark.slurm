#!/bin/bash
#SBATCH --job-name=cpu_scaling_test
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---- Safety checks ----
if [ -z "$CPUS" ]; then
    echo "ERROR: CPUS environment variable not set"
    exit 1
fi

echo "Running with $CPUS CPUs on host $(hostname)"

# ---- Activate your virtual environment ----
source "/global/home/hpc6129/optGME/.venv/bin/activate"

# ---- Limit NumPy/SciPy threads ----
export OMP_NUM_THREADS=$CPUS
export MKL_NUM_THREADS=$CPUS
export OPENBLAS_NUM_THREADS=$CPUS
export NUMEXPR_NUM_THREADS=$CPUS

# ---- Run and time, append to CSV ----
TIME_CMD=$(command -v time)

/usr/bin/env "$TIME_CMD" \
    -f "$CPUS,%e,%U,%S,%M" \
    -o "$SLURM_SUBMIT_DIR/media/results.csv" -a \
    python3 slurmruns.py
