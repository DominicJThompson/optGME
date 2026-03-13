import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
import optomization as opt
import json 
import numpy as np 
import glob

# Search for all raw_data.json files in the directory and any subdirectories under media/media/QDs
qd_files = glob.glob(os.path.join('media', 'QDs', '**', 'raw_data.json'), recursive=True)
print("Found raw_data.json files:", qd_files)

for file in qd_files:

    with open(file, 'r') as f:
        data = json.load(f)
    try:
        _ = data[-1]['result']['x']
    except KeyError:
        print(f"KeyError for file: {file}")
        continue

    opt.dispLossPlot(np.array(data[-1]['result']['x']),
                            opt.W1,
                            data[-1]['gmeParams']['kpoints'][0],
                            file.rsplit('/', 1)[0]+'/meta_data.png',
                            gmax=4.01,
                            phcParams=data[-1]['phcParams'],
                            mode=14,
                            a=data[-1]['cost']['a'],
                            final_cost=data[-1]['result']['fun'],
                            execution_time=data[-1]['result']['execution_time'],
                            niter=data[-1]['result']['niter'],
                            field='QDs')