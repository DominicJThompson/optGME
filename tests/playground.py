#%%
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import legume
legume.set_backend('autograd')
import autograd.numpy as npa
import optomization
import json

#%%
ngs = [5, 10, 30]
fun_values = {ng: [] for ng in ngs}
constr_violations = {ng: [] for ng in ngs}

# Go through each ng value
for ng in ngs:
    # Look for files matching the pattern ng{ng}_*.json
    for i in range(10):  # Assuming max 10 files per ng
        try:
            with open(f"./media/w1/sweepNG/ng{ng}_{i}.json", 'r') as f:
                data = json.load(f)
            fun_values[ng].append(data[-1]['result']['fun'])
            constr_violations[ng].append(data[-1]['result']['constr_violation'])
        except:
            continue

# Create the plot
plt.figure(figsize=(10, 6))  # Increased figure size

# Plot points for each ng value
for ng in ngs:
    for fun, constr in zip(fun_values[ng], constr_violations[ng]):
        if constr == 0:
            plt.scatter(ng, fun, color='green', s=150)  # Increased marker size
        else:
            plt.scatter(ng, fun, c=[constr], cmap='plasma', s=150)  # Increased marker size

plt.hlines(-7.1,0,40,'r',linestyle='--')
plt.colorbar(label='Constraint Violation')  # Increased font size
plt.xlabel('ng value', fontsize=14)  # Increased font size
plt.ylabel('fun value', fontsize=14)  # Increased font size
plt.title('Optimization Results by ng Value', fontsize=16)  # Increased font size
plt.grid(True)
plt.xticks(fontsize=12)  # Increased tick label size
plt.yticks(fontsize=12)  # Increased tick label size
plt.xlim(4,31)
plt.show()
# %%
