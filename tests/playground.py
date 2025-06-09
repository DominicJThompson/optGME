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
from optomization.utils import NG

#%%
ks = np.linspace(np.pi/2,np.pi,150)
phc = optomization.W1(NyChange=0,ra=.3)
gme = legume.GuidedModeExp(phc,4.01)
gme.run(gmode_inds=[0],numeig=21,compute_im=False,kpoints=np.vstack((ks,[0]*len(ks))))
# %%
ngs = []
for i in range(len(ks)):
    ng = NG(gme,i,20)
    ngs.append(ng)
# %%
ngsnp = -np.array(ngs)
plt.plot(ngsnp)

# Find indices where ng equals 5, 10, and 30 (or is closest to these values)
target_ngs = [5, 10, 30]
indices = []
for target in target_ngs:
    idx = np.argmin(np.abs(ngsnp - target))
    indices.append(idx)
    plt.plot(idx, ngsnp[idx], 'o', markersize=8, label=f'ng={target} at index {idx}')

plt.legend()
plt.yscale('log')

plt.show()

# %%
legume.viz.eps_xy(phc)
# %%
print(gme.kpoints[0,indices])
# %%
# Get all files in the sweepNG directory
sweep_dir = './media/w1/sweepNG'
files = os.listdir(sweep_dir)

# Initialize lists to store results
ng_values = []
objective_values = []
constraint_violations = []
colors = []
initial_values = []

# Process each file
for file in files:
    if file.endswith('.json'):
        # Extract ng value from filename
        ng = int(file.split('_')[0][2:])  # Extract number after 'ng'
        
        # Read and parse JSON file
        with open(os.path.join(sweep_dir, file), 'r') as f:
            data = json.load(f)
            
        # Get final results
        final_result = data[-2]['objective_value']
        constraint_violation = data[-2]['constraint_violation']
        initial_value = data[0]['objective_value']  # Get initial value
        
        # Store results
        ng_values.append(ng)
        objective_values.append(final_result)
        constraint_violations.append(constraint_violation)
        colors.append('green' if constraint_violation == 0 else 'red')
        initial_values.append(initial_value)

# Create plot
plt.figure(figsize=(10, 6))

# Plot dashed line at the initial value
plt.axhline(y=initial_values[0], color='gray', linestyle='--', alpha=0.7, 
           label='Initial Value')

# Separate points by color and offset them slightly
red_mask = np.array(colors) == 'red'
green_mask = np.array(colors) == 'green'

# Plot red points slightly to the left
plt.scatter(np.array(ng_values)[red_mask] - 0.5, 
           np.array(objective_values)[red_mask], 
           c='red', s=250, edgecolor='black', linewidth=1)

# Plot green points slightly to the right
plt.scatter(np.array(ng_values)[green_mask] + 0.5, 
           np.array(objective_values)[green_mask], 
           c='green', s=250, edgecolor='black', linewidth=1)

# Add labels and title
plt.xlabel('Group Index (ng)')
plt.ylabel(r'$\log(\text{Loss}/n_g^2)$ [a]')
plt.title('Final Results vs Group Index')

# Add legend for colors and initial value
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
           markersize=10, markeredgecolor='black', label='Constraint Violation = 0'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
           markersize=10, markeredgecolor='black', label='Constraint Violation > 0'),
    Line2D([0], [0], color='gray', linestyle='--', label='Initial Value')
]
plt.legend(handles=legend_elements)

plt.grid(True, alpha=0.3)
plt.show()

# %%

# %%
