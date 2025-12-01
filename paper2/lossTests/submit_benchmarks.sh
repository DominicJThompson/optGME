#!/bin/bash

LOSS_INDEX_LIST=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)

NGS_INDEX_LIST=(0 1 2 3 4 5 6 7 8 9)

for LOSS_INDEX in "${LOSS_INDEX_LIST[@]}"; do
    for NGS_INDEX in "${NGS_INDEX_LIST[@]}"; do
      echo "Submitting job with $LOSS_INDEX and $NGS_INDEX"

      sbatch \
        --export=ALL,LOSS_INDEX=$LOSS_INDEX,NGS_INDEX=$NGS_INDEX \
        benchmark.slurm
    done
done
