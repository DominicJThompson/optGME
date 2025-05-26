#%%
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import legume
legume.set_backend('autograd')
import autograd.numpy as npa
from optomization import W1, ZIW, W1Vars, ZIWVars
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
gmodes = [[0],[0,2,4],[0,2,4,6,8],[0,2,4,6,8,10,12]]
gmaxs = [2.01,3.01,4.01,5.01]
output = {}
# Create a list of all combinations of gmodes and gmaxs
combinations = []
for gmode in gmodes:
    for gmax in gmaxs:
        combinations.append((gmode, gmax))

for i, (gmode,gmax) in enumerate(combinations):
    print(f'{gmode} {gmax}')
    start = time.time()
    phc = W1(NyChange=0,ra=.3)
    gmeParams = {'gmode_inds':gmode,'verbose':False,'numeig':21,'compute_im':False,'kpoints':np.vstack((ks[8],0))}
    gme = legume.GuidedModeExp(phc,gmax)
    gme.run(**gmeParams)
    
    backscatter = Backscatter()
    cost = backscatter.cost(gme,phc,0)
    end = time.time()
    output[f'{i}'] = {'gmode':gmode,'gmax':gmax,'cost':cost,'time':end-start}
    
with open('output.json', 'w') as f:
    json.dump(output, f)

# %%
#outs = [[c['cost'] for c in output.values() if c['gmode'] == gmode] for gmode in gmodes]
outs = np.array([[c['cost'] for c in output.values() if c['gmax'] == gmax] for gmax in gmaxs])
plt.imshow(10**outs,origin='lower')
plt.colorbar()
plt.xlabel('gmodes')
plt.ylabel('gmax')
plt.show()
# %%
diff = outs[:,1:] - outs[:,:-1]
plt.imshow(diff,origin='lower')
plt.colorbar()
plt.xlabel('gmodes')
plt.ylabel('gmax')
plt.show()
# %%
times = np.array([[c['time'] for c in output.values() if c['gmax'] == gmax] for gmax in gmaxs])
plt.imshow(np.log10(times),origin='lower')
plt.colorbar()
plt.xlabel('gmodes')
plt.ylabel('gmax')
plt.show()
# %%
print(times)
# %%
