#!/bin/bash
#SBATCH --job-name=cpu_scaling_test
#SBATCH --mem=35G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---- Safety checks ----
if [ -z "$CPUS" ]; then
    echo "ERROR: CPUS environment variable not set"
    exit 1
fi

echo "Running with $CPUS CPUs on host $(hostname)"

# ---- Activate your virtual environment ----
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

# ---- Limit NumPy/SciPy threads ----
export OMP_NUM_THREADS=$CPUS
export MKL_NUM_THREADS=$CPUS
export OPENBLAS_NUM_THREADS=$CPUS
export NUMEXPR_NUM_THREADS=$CPUS

# ---- Run and time, append to CSV ----
/usr/bin/time -f "$CPUS,%e,%U,%S,%M" \
    -o results.csv -a \
    python3 slurmruns.py
