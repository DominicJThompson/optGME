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
ks = np.linspace(np.pi/2,np.pi,100)
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
plt.plot(gme.freqs[:,20])
# %%
