import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
import optomization as opt
import json 
import numpy as np 
import glob

ng_index = int(os.environ.get("NG_INDEX", "0"))
ndbp_index = int(os.environ.get("NDBP_INDEX", "0"))

print(f"running metaDataRun.py for ng{ng_index} ndbp{ndbp_index}")

# Search for all raw_data.json files in the directory and any subdirectories under media/QDs
qd_files = glob.glob(os.path.join('media', 'QDs', '**', 'raw_data.json'), recursive=True)

matching_files = [
    file for file in qd_files
    if f"ng{ng_index}" in file and f"ndbp{ndbp_index}" in file
]

print(f"Found {len(matching_files)} matching raw_data.json files")

for file in matching_files:

    with open(file, 'r') as f:
        data = json.load(f)
    try:
        _ = data[-1]['result']['x']
    except KeyError:
        print(f"KeyError for file: {file}")
        continue

    output_dir = file.rsplit('/', 1)[0]
    output_png = output_dir + '/meta_data.png'
    print(output_png)

    opt.dispLossPlot(np.array(data[-1]['result']['x']),
                            opt.W1,
                            data[-1]['gmeParams']['kpoints'][0],
                            output_png,
                            gmax=4.01,
                            phcParams=data[-1]['phcParams'],
                            mode=14,
                            a=455,
                            final_cost=data[-1]['result']['fun'],
                            execution_time=data[-1]['result']['execution_time'],
                            niter=data[-1]['result']['niter'],
                            field='QDs')