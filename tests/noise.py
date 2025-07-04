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
# Load W1Best.json data for analysis
with open('/Users/dominic/Desktop/optGME/tests/media/ginds3/W1Best.json', 'r') as file:
    w1_data = json.load(file)

# Get the original W1holes from the data
W1holes_original = np.array(w1_data[-1]['result']['x'])
costParams = w1_data[-1]['cost']
gmeParams = w1_data[-1]['gmeParams']
gmeParams['kpoints'] = np.array(gmeParams['kpoints'])
# Create the PHC with the original hole parameters
W1PHC = W1(vars=W1holes_original)

# Run the GME simulation
gme = legume.GuidedModeExp(W1PHC, 4.01)
gme.run(**gmeParams)

# Calculate backscatter and group index for the original design
backscatter = Backscatter(**costParams)
bs_original = backscatter.cost(gme, W1PHC, 20)
ng_original = np.abs(NG(gme, 0, 20))

print(f"Original design: Backscatter = {bs_original}, NG = {ng_original}")

#%%

# Number of noise iterations to run
num_iterations = 100
bs_values = []
ng_values = []

# Loop over noise iterations
for i in range(num_iterations):
    print(f"Iteration {i+1}/{num_iterations}")
    # Add Gaussian noise with mean 0 and std 1 to each value in W1holes
    noise = np.random.normal(0, 1/266, size=W1holes_original.shape)
    W1holes_noisy = W1holes_original + noise
    
    # Create the PHC with the noisy hole parameters
    W1PHC = W1(vars=W1holes_noisy)
    
    # Run the GME simulation
    gme = legume.GuidedModeExp(W1PHC, 4.01)
    gme.run(**gmeParams)
    
    # Calculate backscatter and group index
    backscatter = Backscatter(**costParams)
    bs = backscatter.cost(gme, W1PHC, 20)
    ng = np.abs(NG(gme, 0, 20))
    
    # Store the results
    bs_values.append(bs)
    ng_values.append(ng)
    
    print(f"Iteration {i+1}/{num_iterations}: Backscatter = {bs}, NG = {ng}")

# Convert results to numpy arrays
bs_values = np.array(bs_values)
ng_values = np.array(ng_values)

# %%
# Create histograms for backscatter and group index values
import matplotlib.pyplot as plt

# Create a figure for backscatter values
plt.figure(figsize=(8, 6))
plt.hist(bs_values, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Backscatter Values', fontsize=14)
plt.xlabel('Backscatter Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(x=bs_original, color='r', linestyle='--', label='Original Design')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Create a figure for group index values
plt.figure(figsize=(8, 6))
plt.hist(ng_values, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('Distribution of Group Index Values', fontsize=14)
plt.xlabel('Group Index Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(x=ng_original, color='r', linestyle='--', label='Original Design')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
# %%
