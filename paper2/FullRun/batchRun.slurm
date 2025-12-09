#!/bin/bash
#SBATCH --job-name=NDBP_FullRun
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --array=1-16            # 1650 jobs
#SBATCH --time=13-23:50:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G

# ---- Activate your virtual environment ----
source "/global/home/hpc6129/optGME/.venv/bin/activate"

# Path to your CSV
CSV_FILE="arrayInputs.csv"

# Extract the row corresponding to this array index.
# NR==SLURM_ARRAY_TASK_ID+1 to skip header row.
INPUT_LINE=$(awk -v idx="$SLURM_ARRAY_TASK_ID" 'NR==idx+1' $CSV_FILE)

echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "CSV Line: $INPUT_LINE"

# Parse CSV columns into variables
# Works even if values contain text
IFS=',' read -r LOSS_INDEX NDBP_INDEX NG_INDEX SEED <<< "$INPUT_LINE"

# ---- Limit NumPy/SciPy threads ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "LOSS_INDEX = $LOSS_INDEX"
echo "NDBP_INDEX = $NDBP_INDEX"
echo "NG_INDEX  = $NG_INDEX"
echo "SEED = $SEED"

# Run your code
python worker.py \
    --LOSS_INDEX "$LOSS_INDEX" \
    --NDBP_INDEX "$NDBP_INDEX"\
    --NG_INDEX "$NG_INDEX" \
    --SEED "$SEED"
