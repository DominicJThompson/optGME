import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--loss_index", type=int)
parser.add_argument("--ndbp_index", type=int)
parser.add_argument("--ngs_index", type=int)
parser.add_argument("--seed", type=int)
args = parser.parse_args()

loss_index = args.loss_index
ndbp_index = args.ndbp_index
ngs_index = args.ngs_index
seed = args.seed

print(f"Running with loss index {loss_index}, ndbp index {ndbp_index}, ngs index {ngs_index}, and seed {seed}")
