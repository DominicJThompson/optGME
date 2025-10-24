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

#read the json files in the review folder
review_folder = 'media/review'
json_files = [f for f in os.listdir(review_folder) if f.endswith('.json')]
print(json_files)
data_dict = {}
for file in json_files:
    with open(os.path.join(review_folder, file), 'r') as f:
        data = json.load(f)
    data_dict[file] = data
# %%
ngs = [5,10,15,20]
original_yvals = [-7.117821770134741, -6.969215491328398, -6.916122787532995, -6.888998758781426] #original values for each ng
# Map ngs to evenly spaced x positions
ngs_sorted = sorted(ngs)
ng_to_x = {ng: i for i, ng in enumerate(ngs_sorted)}

fig, ax = plt.subplots(figsize=(6,3.5))  # Smaller plot
x_vals = []
y_vals = []
colors = []
shapes = []
for k, v in data_dict.items():
    nv = int(k.split('_')[2].split('_')[0])
    keep_feasible = int(k.split('_')[3].split('_')[0])
    try:
        if v[-1]['result']["constr_violation"] > 0:
            color = 'red'
        else:
            color = 'green'  # Blue -> Green

        print(k,v[-1]['result']['fun'],nv,color)
        x_vals.append(nv)
        y_vals.append((10**v[-1]['result']['fun'])/(10**original_yvals[ng_to_x[nv]]))
        colors.append(color)
    except KeyError:
        continue
    if keep_feasible == 0: #keep feasible -> circle, not feasible -> square
        shapes.append('o')
    else:
        shapes.append('s')

# Larger dots, larger text
# Plot each point individually with its corresponding marker
for x, y, color, shape in zip(x_plot, y_vals, colors, shapes):
    ax.scatter(x, y, c=color, s=120, marker=shape, edgecolor='black')
ax.set_xlabel("initial ng value", fontsize=18)
ax.set_ylabel(r'$\tilde L^{\text{W1}}/\tilde L^{\text{W1}}_0$', fontsize=18)
ax.set_xticks(range(len(ngs_sorted)))
ax.set_xticklabels([str(ng) for ng in ngs_sorted], fontsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_yscale('log')

# Create custom legend handles
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

legend_handles = [
    Line2D([0], [0], color='red', lw=3, label='Failed'),
    Line2D([0], [0], color='green', lw=3, label='Success'),
    Line2D([0], [0], marker='s', color='k', label='Keep Feasible', markersize=12, linestyle='None'),
    Line2D([0], [0], marker='o', color='k', label='Allow Unfeasible', markersize=12, linestyle='None')
]

# Place legend off to the right
ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=14, frameon=False)

plt.tight_layout()
plt.show()
# %%

# %%
nk=50
from optomization import Backscatter
from optomization import W1

def runSims(xs,crystal,params):
    phc = crystal(xs)

    kmin,kmax = .5*np.pi,np.pi
    gmeParams = {}
    gmeParams['kpoints']=np.vstack((np.linspace(kmin,kmax,nk),np.zeros(nk)))
    gmeParams['verbose']=True
    gmeParams['numeig']=25
    gmeParams['gmode_inds']=[0,2,4]
    gmeParams['compute_im']=False
    gme = legume.GuidedModeExp(phc,gmax=2.5)
    gme.run(**gmeParams)

    print('running alpha')
    cost = Backscatter(a=266,phidiv=45,lp=40,sig=3)
    alphas = []
    ngs = []
    gmeParams['numeig']-=1
    for i in range(nk):
        if i%10==0:
            print(i)
        gmeParams['kpoints']=np.vstack(([np.linspace(kmin,kmax,nk)[i]],[0]))
        gmeParams['verbose']=False
        gmeAlphacalc = legume.GuidedModeExp(phc,gmax=2.5)
        gmeAlphacalc.run(**gmeParams)
        alphas.append(10**cost.cost(gmeAlphacalc,phc,20))
        ngs.append(np.abs(NG(gmeAlphacalc,0,20,Nx=100,Ny=125)))
    return(phc,gme,alphas,ngs)



#%%
with open('/home/dominic/Desktop/optGME/optGME/tests/media/review/W1_review_5_0_0.json','r') as file:
    out = json.load(file)

#phcW15,gmeW15,alphasW15,ngW15 = runSims(np.array(out[-1]['result']['x']),W1,out[-1])
phcW15,gmeW15,alphasW15,ngW15 = runSims(np.array(out[-1]['x_values']),W1,out[-1])
#%%
with open('/home/dominic/Desktop/optGME/optGME/tests/media/review/W1_review_10_0_0.json','r') as file:
    out = json.load(file)

phcW110,gmeW110,alphasW110,ngW110 = runSims(np.array(out[-1]['result']['x']),W1,out[-1])
#%%
with open('/home/dominic/Desktop/optGME/optGME/tests/media/review/W1_review_15_0_0.json','r') as file:
    out = json.load(file)

phcW115,gmeW115,alphasW115,ngW115 = runSims(np.array(out[-1]['result']['x']),W1,out[-1])
#%%
with open('/home/dominic/Desktop/optGME/optGME/tests/media/review/W1_review_20_0_0.json','r') as file:
    out = json.load(file)

phcW120,gmeW120,alphasW120,ngW120 = runSims(np.array(out[-1]['result']['x']),W1,out[-1])

#%%
from optomization import W1Vars
phcW1OG,gmeW1OG,alphasW1OG,ngW1OG = runSims(W1Vars(),W1,out[-1]) 
# %%

def plot_things(phc,gme,alphas,ngs,index):
    fig,ax = plt.subplots(1,2,figsize=(6,3.5))

    

    #frequency plot
    ks = np.linspace(0.25,0.5,50)
    ax[0].plot(ks,gme.freqs[:,20],color='darkviolet',linewidth=2,zorder=2)
    ax[0].plot(ks,gme.freqs[:,21],color='darkviolet',linewidth=2,linestyle='--')
    ax[0].fill_between(ks,gme.freqs[:,1],gme.freqs[:,19],color='darkviolet',alpha=.7)
    ax[0].fill_between(ks,ks,np.ones_like(ks),color='darkgray',alpha=.3)

    ax[1].plot(ngs,alphas/266/1E-7,color='darkviolet',linewidth=2,zorder=2)

    if index != -1:
        ks_interest = [16,45//2,51//2,55//2]
        ngs_interest = [5,10,50,100]
        print(ngs[ks_interest[index]])
        ax[0].scatter(ks[ks_interest[index]],gme.freqs[ks_interest[index],20],color='red',s=200,zorder=3)
        ax[1].scatter(ngs[ks_interest[index]],alphas[ks_interest[index]]/266/1E-7,color='red',s=200,zorder=3)
        fig.suptitle(f"Results for Target Group Index: {ngs_interest[index]}", fontsize=16)
    else:
        ks_interest = [ks[45//2],ks[68//2],ks[79//2]]
        ngs_interest = [10,50,100]
        fig.suptitle(f"Original", fontsize=16)

    ax[0].set_ylim(0.24,0.27)
    ax[1].set_ylim(1E-4,1E-2)
    ax[1].set_xlim(1,10)

    #ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    ax[1].set_ylabel(r'$\langle\alpha_{back}\rangle/n_g^2$ [cm$^{-1}$]', fontsize=18)
    ax[1].set_xlabel(r"Group Index $n_g$", fontsize=18)
    ax[0].set_ylabel(r"Frequency $\omega a / 2\pi c$", fontsize=18)
    ax[0].set_xlabel(r"Wavevector $\tilde k$", fontsize=18)

    plt.tight_layout()
    plt.show()

plot_things(phcW15,gmeW15,np.array(alphasW15),np.array(ngW15),index=0)
#plot_things(phcW110,gmeW110,np.array(alphasW110),np.array(ngW110),index=1)
#plot_things(phcW150,gmeW150,np.array(alphasW150),np.array(ngW150),index=2)
#plot_things(phcW1100,gmeW1100,np.array(alphasW1100),np.array(ngW1100),index=3)
plot_things(phcW1OG,gmeW1OG,np.array(alphasW1OG),np.array(ngW1OG),index=0)
# %%
print(legume.__version__)
# %%
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Helper to extract n and m from the filename, e.g. W1Opt_ng10_3.json -> n=10, m=3
def get_nm_indices(filename):
    # e.g. W1Opt_ng5_0.json -> n=5, m=0
    #      W1Opt_ng10_1.json -> n=10, m=1
    #      W1Opt_ng15_9.json -> n=15, m=9
    #      W1Opt_ng20_10.json -> n=20, m=10
    base = os.path.basename(filename)
    name = base.replace('.json', '')
    parts = name.split('_')
    try:
        n_str = parts[-2].replace('ng', '')
        m_str = parts[-1]
        n = int(n_str)
        m = int(m_str)
        # Map n to index: 5->0, 10->1, 15->2, 20->3
        n_map = {5: 0, 10: 1, 15: 2, 20: 3}
        n_idx = n_map.get(n, None)
        return n_idx, m
    except Exception:
        return None, None

# Read all of the files in the media/constr/constr folder and store in a 2D array
constr_folder = '/home/dominic/Desktop/optGME/optGME/tests/media/constr/constr'
constr_files = [f for f in os.listdir(constr_folder) if f.endswith('.json')]

# First, determine the maximum m for each n_idx to size the arrays
max_m = [0, 0, 0, 0]
for file in constr_files:
    n_idx, m = get_nm_indices(file)
    if n_idx is not None and m is not None:
        if m > max_m[n_idx]:
            max_m[n_idx] = m
max_m = [m+1 for m in max_m]  # +1 because m is zero-based

# Now, for each n_idx, allocate arrays of shape (max_m[n_idx], max_constr_len)
# We'll first scan all files to find the max length for padding
max_constr_len = 0
for file in constr_files:
    file_path = os.path.join(constr_folder, file)
    n_idx, m = get_nm_indices(file)
    if n_idx is None or m is None:
        continue
    if os.path.getsize(file_path) == 0:
        continue
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if len(data) > max_constr_len:
            max_constr_len = len(data)
    except Exception:
        continue

# Similarly for data files
data_folder = '/home/dominic/Desktop/optGME/optGME/tests/media/constr/data'
data_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]
max_data_len = 0
for file in data_files:
    file_path = os.path.join(data_folder, file)
    n_idx, m = get_nm_indices(file)
    if n_idx is None or m is None:
        continue
    if os.path.getsize(file_path) == 0:
        continue
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if len(data) > max_data_len:
            max_data_len = len(data)
    except Exception:
        continue

# Now allocate arrays
freqs = [np.full((max_m[n_idx], max_constr_len), np.nan) for n_idx in range(4)]
ngs   = [np.full((max_m[n_idx], max_constr_len), np.nan) for n_idx in range(4)]
objectives = [np.full((max_m[n_idx], max_data_len), np.nan) for n_idx in range(4)]

def constrs(constr_dict):
    freqs = [constr['freq'] for constr in constr_dict]
    ngs = [constr['ng'] for constr in constr_dict]
    return(freqs,ngs)

# Fill arrays for constr files
for file in constr_files:
    file_path = os.path.join(constr_folder, file)
    n_idx, m = get_nm_indices(file)
    if n_idx is None or m is None:
        continue
    if os.path.getsize(file_path) == 0:
        print(f"Warning: Skipping empty file {file}")
        continue
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        fvals, nvals = constrs(data)
        freqs[n_idx][m, :len(fvals)] = fvals
        ngs[n_idx][m, :len(nvals)] = nvals
    except json.JSONDecodeError as e:
        print(f"Warning: Skipping corrupted JSON file {file}: {e}")
        continue

# Fill arrays for data files
for file in data_files:
    file_path = os.path.join(data_folder, file)
    n_idx, m = get_nm_indices(file)
    if n_idx is None or m is None:
        continue
    if os.path.getsize(file_path) == 0:
        print(f"Warning: Skipping empty file {file}")
        continue
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        obj_vals = []
        for entry in data[:-1]:
            obj = entry.get('objective_value', np.nan)
            obj_vals.append(obj)
        objectives[n_idx][m, :len(obj_vals)] = obj_vals
    except json.JSONDecodeError as e:
        print(f"Warning: Skipping corrupted JSON file {file}: {e}")
        continue

# Plotting: for each n_idx (0: ng=5, 1: ng=10, 2: ng=15, 3: ng=20)
ng_labels = [5, 10, 15, 20]
for n_idx in range(4):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    lines0 = []
    lines1 = []
    lines2 = []
    legend_labels = []
    for m in range(freqs[n_idx].shape[0]):
        line0, = ax[0].plot(freqs[n_idx][m], label=f"run {m}")
        line1, = ax[1].plot(ngs[n_idx][m], label=f"run {m}")
        line2, = ax[2].plot(objectives[n_idx][m], label=f"run {m}")
        lines0.append(line0)
        lines1.append(line1)
        lines2.append(line2)
        legend_labels.append(f"run {m}")

    ax[0].hlines(y=0.258,xmin=-10,xmax=500,color="red",linestyle="--",linewidth=2)
    ax[0].set_xlim(-10,500)

    ax[0].set_ylabel(r"Frequency $\omega a / 2\pi c$", fontsize=14)
    ax[0].set_xlabel(r"Evaluation", fontsize=14)
    ax[1].set_ylabel(r"Group Index $n_g$", fontsize=14)
    ax[1].set_xlabel(r"Evaluation", fontsize=14)
    ax[2].set_ylabel(r"Objective Value", fontsize=14)
    ax[2].set_xlabel(r"Iteration", fontsize=14)
    ax[1].set_ylim(ng_labels[n_idx]*.75,ng_labels[n_idx]*1.25)
    fig.suptitle(f"Runs for ng={ng_labels[n_idx]}", fontsize=16)

    # Put the legend in the last plot
    ax[2].legend(lines2, legend_labels, loc='best', fontsize=10, frameon=False)

    plt.tight_layout()
    plt.show()
# %%
final_freqs = freqs[:,:,-10]
plt.imshow(final_freqs,aspect='auto')
plt.colorbar()
plt.show()
# %%
print(final_freqs)
# %%

plt.plot(ngs[1].T)
# %%
