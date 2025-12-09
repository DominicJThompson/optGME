import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--LOSS_INDEX", type=int)
parser.add_argument("--NDBP_INDEX", type=int)
parser.add_argument("--NG_INDEX", type=int)
parser.add_argument("--SEED", type=int)
args = parser.parse_args()

loss_index = args.LOSS_INDEX
ndbp_index = args.NDBP_INDEX
ngs_index = args.NG_INDEX
seed = args.SEED

print(f"Running with loss index {loss_index}, ndbp index {ndbp_index}, ngs index {ngs_index}, and seed {seed}")
