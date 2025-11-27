#!/bin/bash

LOSS_INDEX_LIST=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

for LOSS_INDEX in "${LOSS_INDEX_LIST[@]}"; do
    echo "Submitting job with $LOSS_INDEX"

    sbatch \
      --export=ALL,LOSS_INDEX=$LOSS_INDEX \
      benchmark.slurm
done
