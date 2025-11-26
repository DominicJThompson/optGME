#!/bin/bash

CPUS_LIST=(1 2 3 4 5 6)

for CPUS in "${CPUS_LIST[@]}"; do
    echo "Submitting job with $CPUS CPUs"

    sbatch \
      --cpus-per-task=$CPUS \
      --export=ALL,CPUS=$CPUS \
      benchmark.slurm
done
