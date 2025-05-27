#%%
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import legume
legume.set_backend('autograd')
import autograd.numpy as npa
from optomization import W1, ZIW, W1Vars, ZIWVars, NG
from optomization import Backscatter
import json
import time
# %%

ks = np.linspace(np.pi*.5,np.pi,25)
phc = ZIW(NyChange=0)
legume.viz.eps_xy(phc)
#%%
gmeParams = {'gmode_inds':[0,2,4,6],'verbose':False,'numeig':21,'compute_im':False,'kpoints':np.vstack((ks,np.zeros_like(ks)))}
gme = legume.GuidedModeExp(phc,2.01)
gme.run(**gmeParams)

# %%
gmodes = [[0],[0,2],[0,2,4],[0,2,4,6],[0,2,4,6,8]]
gmaxs = [3.01,4.01,5.01]
Nxs = [60,100,300,700]
Nys = [125,200,600,1000]
output = {}
# Create a list of all combinations of gmodes and gmaxs
combinations = []
for gmode in gmodes:
    for gmax in gmaxs:
        combinations.append((gmode,gmax))

combinations2 = []
for Nx in Nxs:
    for Ny in Nys:
        combinations2.append((Nx,Ny))

for i, (gmode,gmax) in enumerate(combinations):
    start = time.time()
    phc = W1(NyChange=0,ra=.3)
    gmeParams = {'gmode_inds':gmode,'verbose':False,'numeig':21,'compute_im':False,'kpoints':np.vstack(([ks[8],ks[20]],[0,0]))}
    gme = legume.GuidedModeExp(phc,gmax)
    gme.run(**gmeParams)
    gme_time = time.time()-start
    for j, (Nx,Ny) in enumerate(combinations2):
        print(f'{gmode} {gmax} {Nx} {Ny}')
        start = time.time()
        ng8 = NG(gme,0,20,Nx=Nx,Ny=Ny)
        ng20 = NG(gme,1,20,Nx=Nx,Ny=Ny)
        end = time.time()
        output[f'{i}{j}'] = {'gmode':gmode,'gmax':gmax,'Nx':Nx,'Ny':Ny,'ng8':ng8,'ng20':ng20,'time':end-start+gme_time}
    
with open('output.json', 'w') as f:
    json.dump(output, f)

# %%
# Load the data if not already in memory
try:
    with open('output.json', 'r') as f:
        output = json.load(f)
except NameError:
    pass  # output is already defined

# Extract the unique values for each parameter
unique_gmodes = set(tuple(output[k]['gmode']) for k in output)
unique_gmaxs = sorted(set(output[k]['gmax'] for k in output))  # Sort for consistent plotting
unique_Nxs = sorted(set(output[k]['Nx'] for k in output))
unique_Nys = sorted(set(output[k]['Ny'] for k in output))

# Create mapping for markers and colors - adjusted for actual number of parameters
gmode_markers = {gmode: marker for gmode, marker in zip(unique_gmodes, ['o', 's', '^', 'D', 'v'][:len(unique_gmodes)])}
gmax_colors = {gmax: color for gmax, color in zip(unique_gmaxs, ['blue', 'red', 'green'][:len(unique_gmaxs)])}
Nx_linestyles = {Nx: style for Nx, style in zip(unique_Nxs, ['-', '--', ':', '-.'][:len(unique_Nxs)])}
Ny_alphas = {Ny: alpha for Ny, alpha in zip(unique_Nys, [1.0, 0.7, 0.4, 0.2][:len(unique_Nys)])}

# Find the baseline values (lowest convergence parameters)
min_gmode = min(unique_gmodes, key=lambda x: len(x))
min_gmax = min(unique_gmaxs)
min_Nx = min(unique_Nxs)
min_Ny = min(unique_Nys)

# Find the baseline ng8 and ng20 values
baseline_ng8 = None
baseline_ng20 = None
for k in output:
    if (tuple(output[k]['gmode']) == min_gmode and 
        output[k]['gmax'] == min_gmax and
        output[k]['Nx'] == min_Nx and 
        output[k]['Ny'] == min_Ny):
        baseline_ng8 = output[k]['ng8']
        baseline_ng20 = output[k]['ng20']
        break

# Calculate percent change from baseline
def calc_percent_change(values, baseline):
    if not values or baseline == 0:
        return []
    return [(val - baseline) / baseline * 100 for val in values]

# Increase default font sizes
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
})

# Function to create convergence plots
def create_convergence_plots(ng_type, baseline_value):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ng_type} Convergence Analysis', fontsize=22)
    
    # Plot vs gmax for each gmode (top left)
    for gmode in unique_gmodes:
        marker = gmode_markers[gmode]
        gmaxs = []
        values = []
        for k in output:
            if (tuple(output[k]['gmode']) == gmode and 
                output[k]['Nx'] == min_Nx and 
                output[k]['Ny'] == min_Ny):
                gmaxs.append(output[k]['gmax'])
                values.append(output[k][ng_type])
        if gmaxs:
            pct_change = calc_percent_change(values, baseline_value)
            axes[0, 0].plot(gmaxs, pct_change, marker=marker, linestyle='-', 
                          markersize=10, linewidth=2, label=f'gmode={list(gmode)}')

    # Plot vs Nx for each Ny (top right)
    for Ny in unique_Nys:
        Nxs = []
        values = []
        for k in output:
            if (tuple(output[k]['gmode']) == min_gmode and 
                output[k]['gmax'] == min_gmax and 
                output[k]['Ny'] == Ny):
                Nxs.append(output[k]['Nx'])
                values.append(output[k][ng_type])
        if Nxs:
            pct_change = calc_percent_change(values, baseline_value)
            axes[0, 1].plot(Nxs, pct_change, marker='o', linestyle='-', 
                          markersize=10, linewidth=2, label=f'Ny={Ny}')

    # Plot computation time vs gmax (bottom left)
    for gmode in unique_gmodes:
        marker = gmode_markers[gmode]
        gmaxs = []
        times = []
        for k in output:
            if (tuple(output[k]['gmode']) == gmode and 
                output[k]['Nx'] == min_Nx and 
                output[k]['Ny'] == min_Ny):
                gmaxs.append(output[k]['gmax'])
                times.append(output[k]['time'])
        if gmaxs:
            axes[1, 0].plot(gmaxs, times, marker=marker, linestyle='-', 
                          markersize=10, linewidth=2, label=f'gmode={list(gmode)}')

    # Plot computation time vs Nx (bottom right)
    for Ny in unique_Nys:
        Nxs = []
        times = []
        for k in output:
            if (tuple(output[k]['gmode']) == min_gmode and 
                output[k]['gmax'] == min_gmax and 
                output[k]['Ny'] == Ny):
                Nxs.append(output[k]['Nx'])
                times.append(output[k]['time'])
        if Nxs:
            axes[1, 1].plot(Nxs, times, marker='o', linestyle='-', 
                          markersize=10, linewidth=2, label=f'Ny={Ny}')

    # Add annotations and labels
    axes[0, 0].set_xlabel('gmax')
    axes[0, 0].set_ylabel(f'{ng_type} percent change (%)')
    axes[0, 0].set_title(f'{ng_type} vs gmax')
    axes[0, 0].legend(loc='best')

    axes[0, 1].set_xlabel('Nx')
    axes[0, 1].set_ylabel(f'{ng_type} percent change (%)')
    axes[0, 1].set_title(f'{ng_type} vs Nx')
    axes[0, 1].legend(loc='best')

    axes[1, 0].set_xlabel('gmax')
    axes[1, 0].set_ylabel('Computation Time (s)')
    axes[1, 0].set_title('Computation Time vs gmax')
    axes[1, 0].legend(loc='best')

    axes[1, 1].set_xlabel('Nx')
    axes[1, 1].set_ylabel('Computation Time (s)')
    axes[1, 1].set_title('Computation Time vs Nx')
    axes[1, 1].legend(loc='best')

    #axes[0,0].set_yscale('log')
    #axes[0,1].set_yscale('log')
    axes[1,0].set_yscale('log')
    axes[1,1].set_yscale('log')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Create plots for both ng8 and ng20
create_convergence_plots('ng8', baseline_ng8)
create_convergence_plots('ng20', baseline_ng20)

# %%
print(unique_Nys)
# %%
