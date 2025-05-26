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
    for Nx,Ny in combinations2:
        print(f'{gmode} {gmax} {Nx} {Ny}')
        start = time.time()
        ng8 = NG(gme,0,20,Nx=Nx,Ny=Ny)
        ng20 = NG(gme,1,20,Nx=Nx,Ny=Ny)
        end = time.time()
        output[f'{i}'] = {'gmode':gmode,'gmax':gmax,'Nx':Nx,'Ny':Ny,'ng8':ng8,'ng20':ng20,'time':end-start+gme_time}
    
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
unique_gmaxs = set(output[k]['gmax'] for k in output)
unique_Nxs = set(output[k]['Nx'] for k in output)
unique_Nys = set(output[k]['Ny'] for k in output)

# Create mapping for markers and colors
gmode_markers = {gmode: marker for gmode, marker in zip(unique_gmodes, ['o', 's', '^', 'D', 'v'])}
gmax_colors = {gmax: color for gmax, color in zip(unique_gmaxs, ['blue', 'red', 'green', 'purple', 'orange'])}
Nx_linestyles = {Nx: style for Nx, style in zip(unique_Nxs, ['-', '--', ':', '-.'])}
Ny_alphas = {Ny: alpha for Ny, alpha in zip(unique_Nys, [1.0, 0.7, 0.4, 0.2])}

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

# Convergence plots for ng8 vs different parameters
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Percent Change in ng8 with Different Parameters', fontsize=22)

# Plot ng8 vs gmax for each gmode
for gmode in unique_gmodes:
    marker = gmode_markers[gmode]
    for Nx in unique_Nxs:
        linestyle = Nx_linestyles[Nx]
        for Ny in unique_Nys:
            gmaxs = []
            ng8s = []
            for k in output:
                if (tuple(output[k]['gmode']) == gmode and 
                    output[k]['Nx'] == Nx and 
                    output[k]['Ny'] == Ny):
                    gmaxs.append(output[k]['gmax'])
                    ng8s.append(output[k]['ng8'])
            if gmaxs:
                pct_change = calc_percent_change(ng8s, baseline_ng8)
                axes[0, 0].plot(gmaxs, pct_change, marker=marker, linestyle=linestyle, 
                               alpha=Ny_alphas[Ny], markersize=12, linewidth=2)

# Plot ng8 vs Nx for each gmode and gmax
for gmode in unique_gmodes:
    marker = gmode_markers[gmode]
    for gmax in unique_gmaxs:
        color = gmax_colors[gmax]
        for Ny in unique_Nys:
            Nxs = []
            ng8s = []
            for k in output:
                if (tuple(output[k]['gmode']) == gmode and 
                    output[k]['gmax'] == gmax and 
                    output[k]['Ny'] == Ny):
                    Nxs.append(output[k]['Nx'])
                    ng8s.append(output[k]['ng8'])
            if Nxs:
                pct_change = calc_percent_change(ng8s, baseline_ng8)
                axes[0, 1].plot(Nxs, pct_change, marker=marker, color=color, 
                               alpha=Ny_alphas[Ny], markersize=12, linewidth=2)

# Plot ng8 vs Ny for each gmode and gmax
for gmode in unique_gmodes:
    marker = gmode_markers[gmode]
    for gmax in unique_gmaxs:
        color = gmax_colors[gmax]
        for Nx in unique_Nxs:
            Nys = []
            ng8s = []
            for k in output:
                if (tuple(output[k]['gmode']) == gmode and 
                    output[k]['gmax'] == gmax and 
                    output[k]['Nx'] == Nx):
                    Nys.append(output[k]['Ny'])
                    ng8s.append(output[k]['ng8'])
            if Nys:
                pct_change = calc_percent_change(ng8s, baseline_ng8)
                axes[1, 0].plot(Nys, pct_change, marker=marker, color=color, 
                               linestyle=Nx_linestyles[Nx], markersize=12, linewidth=2)

# Plot ng8 vs computation time
times = []
ng8s = []
for k in output:
    times.append(output[k]['time'])
    ng8s.append(output[k]['ng8'])
axes[1, 1].scatter(times, ng8s, c='black', alpha=0.7, s=100)

# Add compact legend for plot identifiers
gmode_legend_elements = [plt.Line2D([0], [0], marker=marker, color='black', label=f'gmode={list(gmode)}', 
                        linestyle='', markersize=12) for gmode, marker in gmode_markers.items()]
gmax_legend_elements = [plt.Line2D([0], [0], color=color, label=f'gmax={gmax}', 
                       linestyle='-', linewidth=2, markersize=12) for gmax, color in gmax_colors.items()]
Nx_legend_elements = [plt.Line2D([0], [0], color='black', label=f'Nx={Nx}', 
                     linestyle=style, linewidth=2, markersize=12) for Nx, style in Nx_linestyles.items()]
Ny_legend_elements = [plt.Line2D([0], [0], color='black', label=f'Ny={Ny}', 
                     linestyle='-', linewidth=2, alpha=alpha, markersize=12) for Ny, alpha in Ny_alphas.items()]

# Add legends to each subplot
axes[0, 0].legend(handles=gmode_legend_elements + Nx_legend_elements + Ny_legend_elements, 
                 loc='best', fontsize='medium', ncol=2)
axes[0, 1].legend(handles=gmode_legend_elements + gmax_legend_elements + Ny_legend_elements, 
                 loc='best', fontsize='medium', ncol=2)
axes[1, 0].legend(handles=gmode_legend_elements + gmax_legend_elements + Nx_legend_elements, 
                 loc='best', fontsize='medium', ncol=2)

axes[0, 0].set_xlabel('gmax')
axes[0, 0].set_ylabel('ng8 percent change (%)')
axes[0, 0].set_title('ng8 vs gmax')

axes[0, 1].set_xlabel('Nx')
axes[0, 1].set_ylabel('ng8 percent change (%)')
axes[0, 1].set_title('ng8 vs Nx')

axes[1, 0].set_xlabel('Ny')
axes[1, 0].set_ylabel('ng8 percent change (%)')
axes[1, 0].set_title('ng8 vs Ny')

axes[1, 1].set_xlabel('Computation Time (s)')
axes[1, 1].set_ylabel('ng8')
axes[1, 1].set_title('ng8 vs Computation Time')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# Similar plots for ng20
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Percent Change in ng20 with Different Parameters', fontsize=22)

# Plot ng20 vs gmax for each gmode
for gmode in unique_gmodes:
    marker = gmode_markers[gmode]
    for Nx in unique_Nxs:
        linestyle = Nx_linestyles[Nx]
        for Ny in unique_Nys:
            gmaxs = []
            ng20s = []
            for k in output:
                if (tuple(output[k]['gmode']) == gmode and 
                    output[k]['Nx'] == Nx and 
                    output[k]['Ny'] == Ny):
                    gmaxs.append(output[k]['gmax'])
                    ng20s.append(output[k]['ng20'])
            if gmaxs:
                pct_change = calc_percent_change(ng20s, baseline_ng20)
                axes[0, 0].plot(gmaxs, pct_change, marker=marker, linestyle=linestyle, 
                               alpha=Ny_alphas[Ny], markersize=12, linewidth=2)

# Plot ng20 vs Nx for each gmode and gmax
for gmode in unique_gmodes:
    marker = gmode_markers[gmode]
    for gmax in unique_gmaxs:
        color = gmax_colors[gmax]
        for Ny in unique_Nys:
            Nxs = []
            ng20s = []
            for k in output:
                if (tuple(output[k]['gmode']) == gmode and 
                    output[k]['gmax'] == gmax and 
                    output[k]['Ny'] == Ny):
                    Nxs.append(output[k]['Nx'])
                    ng20s.append(output[k]['ng20'])
            if Nxs:
                pct_change = calc_percent_change(ng20s, baseline_ng20)
                axes[0, 1].plot(Nxs, pct_change, marker=marker, color=color, 
                               alpha=Ny_alphas[Ny], markersize=12, linewidth=2)

# Plot ng20 vs Ny for each gmode and gmax
for gmode in unique_gmodes:
    marker = gmode_markers[gmode]
    for gmax in unique_gmaxs:
        color = gmax_colors[gmax]
        for Nx in unique_Nxs:
            Nys = []
            ng20s = []
            for k in output:
                if (tuple(output[k]['gmode']) == gmode and 
                    output[k]['gmax'] == gmax and 
                    output[k]['Nx'] == Nx):
                    Nys.append(output[k]['Ny'])
                    ng20s.append(output[k]['ng20'])
            if Nys:
                pct_change = calc_percent_change(ng20s, baseline_ng20)
                axes[1, 0].plot(Nys, pct_change, marker=marker, color=color, 
                               linestyle=Nx_linestyles[Nx], markersize=12, linewidth=2)

# Plot ng20 vs computation time
times = []
ng20s = []
for k in output:
    times.append(output[k]['time'])
    ng20s.append(output[k]['ng20'])
axes[1, 1].scatter(times, ng20s, c='black', alpha=0.7, s=100)

# Add legends to each subplot
axes[0, 0].legend(handles=gmode_legend_elements + Nx_legend_elements + Ny_legend_elements, 
                 loc='best', fontsize='medium', ncol=2)
axes[0, 1].legend(handles=gmode_legend_elements + gmax_legend_elements + Ny_legend_elements, 
                 loc='best', fontsize='medium', ncol=2)
axes[1, 0].legend(handles=gmode_legend_elements + gmax_legend_elements + Nx_legend_elements, 
                 loc='best', fontsize='medium', ncol=2)

axes[0, 0].set_xlabel('gmax')
axes[0, 0].set_ylabel('ng20 percent change (%)')
axes[0, 0].set_title('ng20 vs gmax')

axes[0, 1].set_xlabel('Nx')
axes[0, 1].set_ylabel('ng20 percent change (%)')
axes[0, 1].set_title('ng20 vs Nx')

axes[1, 0].set_xlabel('Ny')
axes[1, 0].set_ylabel('ng20 percent change (%)')
axes[1, 0].set_title('ng20 vs Ny')

axes[1, 1].set_xlabel('Computation Time (s)')
axes[1, 1].set_ylabel('ng20')
axes[1, 1].set_title('ng20 vs Computation Time')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# Create a heatmap to show the combined effects of parameters
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Parameter Influence on Group Index Calculation', fontsize=22)

# Select a specific gmode and gmax to visualize Nx vs Ny
selected_gmode = tuple(list(unique_gmodes)[0])  # First gmode
selected_gmax = list(unique_gmaxs)[0]  # First gmax

# Create meshgrid for Nx and Ny
Nx_grid, Ny_grid = np.meshgrid(sorted(list(unique_Nxs)), sorted(list(unique_Nys)))
ng8_grid = np.zeros_like(Nx_grid, dtype=float)
ng20_grid = np.zeros_like(Nx_grid, dtype=float)
ng8_pct_grid = np.zeros_like(Nx_grid, dtype=float)
ng20_pct_grid = np.zeros_like(Nx_grid, dtype=float)

# Fill in the values
for i, Ny in enumerate(sorted(list(unique_Nys))):
    for j, Nx in enumerate(sorted(list(unique_Nxs))):
        for k in output:
            if (tuple(output[k]['gmode']) == selected_gmode and 
                output[k]['gmax'] == selected_gmax and
                output[k]['Nx'] == Nx and 
                output[k]['Ny'] == Ny):
                ng8_grid[i, j] = output[k]['ng8']
                ng20_grid[i, j] = output[k]['ng20']
                # Calculate percent change from baseline
                if baseline_ng8 is not None and baseline_ng8 != 0:
                    ng8_pct_grid[i, j] = (output[k]['ng8'] - baseline_ng8) / baseline_ng8 * 100
                if baseline_ng20 is not None and baseline_ng20 != 0:
                    ng20_pct_grid[i, j] = (output[k]['ng20'] - baseline_ng20) / baseline_ng20 * 100

# Plot heatmaps for percent change
im1 = axes[0].pcolormesh(Nx_grid, Ny_grid, ng8_pct_grid, shading='auto', cmap='viridis')
axes[0].set_title(f'ng8 % change for gmode={list(selected_gmode)}, gmax={selected_gmax}', fontsize=18)
axes[0].set_xlabel('Nx', fontsize=16)
axes[0].set_ylabel('Ny', fontsize=16)
cbar1 = fig.colorbar(im1, ax=axes[0], label='ng8 % change')
cbar1.ax.tick_params(labelsize=14)
cbar1.set_label('ng8 % change', size=16)

im2 = axes[1].pcolormesh(Nx_grid, Ny_grid, ng20_pct_grid, shading='auto', cmap='viridis')
axes[1].set_title(f'ng20 % change for gmode={list(selected_gmode)}, gmax={selected_gmax}', fontsize=18)
axes[1].set_xlabel('Nx', fontsize=16)
axes[1].set_ylabel('Ny', fontsize=16)
cbar2 = fig.colorbar(im2, ax=axes[1], label='ng20 % change')
cbar2.ax.tick_params(labelsize=14)
cbar2.set_label('ng20 % change', size=16)

for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()
plt.show()

# Relative convergence - how much parameters affect the result (in percent)
param_effects = {
    'gmode': {'ng8': [], 'ng20': []},
    'gmax': {'ng8': [], 'ng20': []},
    'Nx': {'ng8': [], 'ng20': []},
    'Ny': {'ng8': [], 'ng20': []}
}

# Calculate relative percent effects for each parameter from baseline
for param in ['gmode', 'gmax', 'Nx', 'Ny']:
    for metric in ['ng8', 'ng20']:
        baseline = baseline_ng8 if metric == 'ng8' else baseline_ng20
        max_value = baseline
        for k in output:
            if output[k][metric] > max_value:
                max_value = output[k][metric]
        if baseline and baseline != 0:
            param_effects[param][metric] = (max_value - baseline) / baseline * 100  # percent change

# Plot parameter effects
fig, ax = plt.subplots(figsize=(12, 8))
bar_width = 0.35
index = np.arange(len(param_effects))

ng8_effects = [param_effects[param]['ng8'] for param in param_effects]
ng20_effects = [param_effects[param]['ng20'] for param in param_effects]

bar1 = ax.bar(index - bar_width/2, ng8_effects, bar_width, label='ng8', linewidth=2, edgecolor='black')
bar2 = ax.bar(index + bar_width/2, ng20_effects, bar_width, label='ng20', linewidth=2, edgecolor='black')

ax.set_xlabel('Parameter', fontsize=18)
ax.set_ylabel('Relative Effect on Convergence (%)', fontsize=18)
ax.set_title('Parameter Influence on Group Index Calculation', fontsize=22)
ax.set_xticks(index)
ax.set_xticklabels(list(param_effects.keys()), fontsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.legend(fontsize=16)

# Add value labels on top of the bars
for i, v in enumerate(ng8_effects):
    ax.text(index[i] - bar_width/2, v + 0.5, f'{v:.1f}%', 
            ha='center', va='bottom', fontsize=14, fontweight='bold')
for i, v in enumerate(ng20_effects):
    ax.text(index[i] + bar_width/2, v + 0.5, f'{v:.1f}%', 
            ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# %%
