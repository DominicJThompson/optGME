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
import matplotlib as mpl

# Disable LaTeX but use Computer Modern fonts
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern math fonts
mpl.rcParams['font.family'] = 'STIXGeneral'  # Use STIX fonts (similar to Computer Modern)

# %%
#backscatter cost object 
backscatter = optomization.Backscatter()

#create PHC
phc = optomization.W1(NyChange=0,Ny=7)

#lets quickly simulate GME at the point of interest 
ks = npa.linspace(npa.pi*.5,npa.pi,25)
gmeParams = {'verbose':True,'numeig':21,'compute_im':False,'kpoints':npa.array([[float(ks[8])],[0]])}
gme = legume.GuidedModeExp(phc,4.01)
gme.run(**gmeParams)
# %%

#now lets get the hole locations
borders,phis,radii = backscatter.hole_borders(phc)

#now the field at the edges of the holesE
EHoles = backscatter.get_xyfield(gme,14,borders,phc.layers[0].d/2,field='E',components='xyz')

#now get the fild across the whole phc
E,_,_ = gme.get_field_xy('E',0,14,phc.layers[0].d/2)
# %%
Eamp = np.sqrt(np.abs(E['x'])**2 + np.abs(E['y'])**2 + np.abs(E['z'])**2)
EHolesamp = np.sqrt(np.abs(EHoles['x'])**2 + np.abs(EHoles['y'])**2 + np.abs(EHoles['z'])**2)
norm = plt.Normalize(vmin=0, vmax=np.max(Eamp))
cmap = plt.cm.viridis

plt.imshow(Eamp,cmap=cmap,norm=norm,extent=[-.5,.5,-np.sqrt(3)/2*7,np.sqrt(3)/2*7])
sc = plt.scatter(borders[:,:,0], borders[:,:,1], c=EHolesamp[:], cmap=cmap, norm=norm)
plt.colorbar(sc, label='|E| amplitude')
plt.ylim(-np.sqrt(3)/2*3,np.sqrt(3)/2*3)
plt.show()
# %%
# %%
