#%%
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import legume
import autograd.numpy as npa
import optomization
import json
from optomization.utils import NG
from optomization.crystals import W1
from optomization.cost import Backscatter

#%%
# Initialize storage for all results
# Initialize storage for all results
all_results = {}

# Base path for input files
base_path = '/home/dominic/Desktop/optGME/optGME/tests/media/w1/sweepNG'

# Get all files in the directory
files = [f for f in os.listdir(base_path) if f.endswith('.json')]

# Initialize storage for each file
for file_name in files:
    file_key = file_name.replace('.json', '')
    all_results[file_key] = {
        'bs_values': {str(i): [] for i in range(1, 6)},
        'ng_values': {str(i): [] for i in range(1, 6)},
        'freq_values': {str(i): [] for i in range(1, 6)},
        'original': {}
    }

# Process each file
for fileNum,file_name in enumerate(files):
    print(f"\nProcessing {file_name}...")
    file_key = file_name.replace('.json', '')
    
    # Load data from current file
    with open(f'{base_path}/{file_name}', 'r') as file:
        data = json.load(file)

    if data[-2]['constraint_violation']>0:
        continue

    try:
        # Get parameters from the data
        W1holes_original = np.array(data[-1]['result']['x'])
        costParams = data[-1]['cost']
        gmeParams = data[-1]['gmeParams']
        gmeParams['kpoints'] = np.array(gmeParams['kpoints'])
    except KeyError:
        W1holes_original = np.array(data[-1]['x_values'])
    
    # Create and run original PHC
    W1PHC = W1(vars=W1holes_original)
    gme = legume.GuidedModeExp(W1PHC, 4.01)
    gme.run(**gmeParams)
    
    # Calculate original values
    backscatter = Backscatter(**costParams)
    bs_original = backscatter.cost(gme, W1PHC, 20)
    ng_original = np.abs(NG(gme, 0, 20))
    freq_original = gme.freqs[0][20]  # Get frequency at k=0, mode=20
    
    # Store original values
    all_results[file_key]['original'] = {
        'bs': float(bs_original),
        'ng': float(ng_original),
        'freq': float(freq_original)
    }
    
    print(f"Original design for {file_key}: {fileNum}: Backscatter = {bs_original}, NG = {ng_original}, Freq = {freq_original}")
    
    # Run noise iterations for each noise level
    num_iterations = 50
    noise_levels = [i for i in range(1, 6)]
    
    for noise_level in noise_levels:
        print(f"Processing noise level {noise_level}...")
        for i in range(num_iterations):
            
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level/266, size=6)
            W1holes_noisy = W1holes_original.copy()
            W1holes_noisy[-6:] += noise
            
            # Create and run noisy PHC
            W1PHC = W1(vars=W1holes_noisy)
            gme = legume.GuidedModeExp(W1PHC, 4.01)
            gme.run(**gmeParams)
            
            # Calculate values
            backscatter = Backscatter(**costParams)
            bs = backscatter.cost(gme, W1PHC, 20)
            ng = np.abs(NG(gme, 0, 20))
            freq = gme.freqs[0][20]  # Get frequency at k=0, mode=20
            
            # Store results
            noise_key = str(noise_level)
            all_results[file_key]['bs_values'][noise_key].append(float(bs))
            all_results[file_key]['ng_values'][noise_key].append(float(ng))
            all_results[file_key]['freq_values'][noise_key].append(float(freq))
        
        # Save results after each noise level is completed, overwriting previous data
        output_path = '/home/dominic/Desktop/optGME/optGME/tests/media/w1/noise_analysis.json'
        # Clear file and write new data
        with open(output_path, 'w') as file:
            # Truncate file to clear it
            file.truncate(0)
            # Write new data
            json.dump(all_results, file, indent=4)
        print(f"Saved results for noise level {noise_level}")

print(f"\nAll results saved to {output_path}")

# %%
