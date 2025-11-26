#!/bin/bash

CPUS_LIST=(4 8 12 16 20 24 28 32)

for CPUS in "${CPUS_LIST[@]}"; do
    echo "Submitting job with $CPUS CPUs"

    sbatch \
      --cpus-per-task=$CPUS \
      --export=ALL,CPUS=$CPUS \
      benchmark.slurm
done
