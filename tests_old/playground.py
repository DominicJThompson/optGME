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

#%%
#some variables to match the paper
a=240 #nm
dslab=150/a
eps_slab = np.sqrt(12.11)
ra=70.8/a

#%%

def W1(Nx=1,Ny=10,dslab=dslab,eps_slab=eps_slab,ra=ra,noise=0):

    lattice = legume.Lattice(npa.array([Nx,0]),npa.array([0,Ny*npa.sqrt(3)]))

    #now we define a photonic crystal that goes over our lattice and add the one layer of our W1 waveguide
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=dslab,eps_b=eps_slab**2)

    #now we want to add the air holes to our photonic crystal to make it the W1 waveguide we do this in a loop
    for i in range(Ny*2-1):
        for j in range(Nx):
            iy = i-Ny
            if i>=Ny:
                iy+=1
        
            #move the hole over by half a unit cell if they are on odd rows
            if iy%2==1:
                x = .5
            else:
                x = 0

            #the y component should be scaled by the factor of np.sqrt(3)/2
            y = iy*npa.sqrt(3)/2

            if noise:
                sample = np.random.normal(0,noise)
                sampleangle = np.random.uniform(0,2*np.pi)
                x += sample*npa.cos(sampleangle)
                y += sample*npa.sin(sampleangle)

            #now we can add a circle with the given positions
            phc.add_shape(legume.Circle(x_cent=x+j,y_cent=y,r=ra))

    return(phc)

#%%
phc1 = W1(Nx=1,Ny=7,ra=.3)
phc2 = W1(Nx=2,Ny=7,ra=.3,noise=0.01)
phc3 = W1(Nx=3,Ny=7,ra=.3,noise=0.01)
phc5 = W1(Nx=5,Ny=7,ra=.3,noise=0.01)

legume.viz.eps_xy(phc1)
legume.viz.eps_xy(phc2)
legume.viz.eps_xy(phc3)
legume.viz.eps_xy(phc5)
# %%
gme1 = legume.GuidedModeExp(phc1,2.01)
gme2 = legume.GuidedModeExp(phc2,2.01)
gme3 = legume.GuidedModeExp(phc3,2.01)
gme5 = legume.GuidedModeExp(phc5,2.01)
ks = np.linspace(0,np.pi,90)
gme1.run(gmode_inds=[0],numeig=100,compute_im=False,kpoints=np.vstack((ks,[0]*len(ks))))
gme2.run(gmode_inds=[0],numeig=200,compute_im=False,kpoints=np.vstack((ks,[0]*len(ks))))
gme3.run(gmode_inds=[0],numeig=200,compute_im=False,kpoints=np.vstack((ks,[0]*len(ks))))
gme5.run(gmode_inds=[0],numeig=200,compute_im=False,kpoints=np.vstack((ks,[0]*len(ks))))
#%%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(ks, gme1.freqs)
ax[0].vlines(ks[half],ymin,ymax,color='r')
# For odd i: plot ks up to halfway, then back down; for even i: plot ks from max down to halfway, then back up
n = len(ks)
half = n // 2
for i in range(gme2.freqs.shape[1]):
    if i % 2 == 1:
        # Odd: go up to halfway, then count down
        ks_odd = np.concatenate([ks[:half], ks[:half][::-1]])
        #freqs_odd = np.concatenate([gme2.freqs[:half+1, i], gme2.freqs[half::-1, i]])
        ax[1].plot(ks_odd, gme2.freqs[:, i])
    else:
        # Even: go from max down to halfway, then back up
        ks_even = np.concatenate([ks[half:][::-1], ks[half:]])
        #freqs_even = np.concatenate([gme2.freqs[::-1][:half+1, i], gme2.freqs[half::-1, i]])
        ax[1].plot(ks_even, gme2.freqs[:, i])
# Set both subplots to the same y scale
# Find the min and max y values across both plots
ymin = .2
ymax = .3
ax[0].set_ylim(ymin, ymax)
ax[1].set_ylim(ymin, ymax)
plt.show()
# %%
# Hyperparameters for line thickness
solid_linewidth = 2.0   # thickness for solid lines
dashed_linewidth = 1.0  # thickness for dashed lines

ymin = .255
ymax = .3

fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax[0,0].plot(ks, gme1.freqs, linewidth=solid_linewidth)

#for the 2s we have even odd
half = len(ks) // 2
ks_even = np.concatenate([ks[half:][::-1], ks[half:]])
ks_odd = np.concatenate([ks[:half], ks[:half][::-1]])

import itertools
color_cycle_2 = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
current_color_2 = next(color_cycle_2)
for i in range(gme2.freqs.shape[1]):
    if i % 2 == 0:
        current_color_2 = next(color_cycle_2)
        ax[0,1].plot(ks_even, gme2.freqs[:, i], color=current_color_2, linewidth=solid_linewidth)
    else:
        ax[0,1].plot(ks_odd, gme2.freqs[:, i], color=current_color_2, linewidth=solid_linewidth)
ax[0,1].plot(ks, gme1.freqs[:,14], color='black', linestyle='--', linewidth=dashed_linewidth)
ax[0,1].plot(ks, gme1.freqs[:,15], color='black', linestyle='--', linewidth=dashed_linewidth)

#for the 3s we have divide by 3
third = len(ks) // 3
ks_first = np.concatenate([ks[:third], ks[:third][::-1],ks[:third]])
ks_second = np.concatenate([ks[third:2*third], ks[third:2*third][::-1],ks[third:2*third]])
ks_third = np.concatenate([ks[2*third:], ks[2*third:][::-1],ks[2*third:]])


color_cycle_3 = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
current_color_3 = next(color_cycle_3)
for i in range(gme3.freqs.shape[1]):
    if i % 3 == 0:
        current_color_3 = next(color_cycle_3)
        ax[1,0].plot(ks_third, gme3.freqs[:, i], color=current_color_3, linewidth=solid_linewidth)
    elif i % 3 == 1:
        ax[1,0].plot(ks_second[::-1], gme3.freqs[:, i], color=current_color_3, linewidth=solid_linewidth)
    else:
        ax[1,0].plot(ks_first, gme3.freqs[:, i], color=current_color_3, linewidth=solid_linewidth)
ax[1,0].plot(ks, gme1.freqs[:,14], color='black', linestyle='--', linewidth=dashed_linewidth)
ax[1,0].plot(ks, gme1.freqs[:,15], color='black', linestyle='--', linewidth=dashed_linewidth)

#for the 5s we have divide by 5
fifth = len(ks) // 5
ks_first = np.concatenate([ks[:fifth], ks[:fifth][::-1],ks[:fifth],ks[:fifth][::-1],ks[:fifth]])
ks_second = np.concatenate([ks[fifth:2*fifth], ks[fifth:2*fifth][::-1],ks[fifth:2*fifth],ks[fifth:2*fifth][::-1],ks[fifth:2*fifth]])
ks_third = np.concatenate([ks[2*fifth:3*fifth], ks[2*fifth:3*fifth][::-1],ks[2*fifth:3*fifth],ks[2*fifth:3*fifth][::-1],ks[2*fifth:3*fifth]])
ks_fourth = np.concatenate([ks[3*fifth:4*fifth], ks[3*fifth:4*fifth][::-1],ks[3*fifth:4*fifth],ks[3*fifth:4*fifth][::-1],ks[3*fifth:4*fifth]])
ks_fifth = np.concatenate([ks[4*fifth:], ks[4*fifth:][::-1],ks[4*fifth:],ks[4*fifth:][::-1],ks[4*fifth:]])

color_cycle_5 = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
current_color_5 = next(color_cycle_5)
for i in range(gme5.freqs.shape[1]):
    if i % 5 == 0:
        current_color_5 = next(color_cycle_5)
        ax[1,1].plot(ks_fifth, gme5.freqs[:, i], color=current_color_5, linewidth=solid_linewidth)
    elif i % 5 == 1:
        ax[1,1].plot(ks_fourth[::-1], gme5.freqs[:, i], color=current_color_5, linewidth=solid_linewidth)
    elif i % 5 == 2:
        ax[1,1].plot(ks_third, gme5.freqs[:, i], color=current_color_5, linewidth=solid_linewidth)
    elif i % 5 == 3:
        ax[1,1].plot(ks_second[::-1], gme5.freqs[:, i], color=current_color_5, linewidth=solid_linewidth)
    elif i % 5 == 4:
        ax[1,1].plot(ks_first, gme5.freqs[:, i], color=current_color_5, linewidth=solid_linewidth)
ax[1,1].plot(ks, gme1.freqs[:,14], color='black', linestyle='--', linewidth=dashed_linewidth)
ax[1,1].plot(ks, gme1.freqs[:,15], color='black', linestyle='--', linewidth=dashed_linewidth)

ax[0,0].fill_between(ks, ks/np.pi/2, 1, color='black', alpha=0.2)
ax[0,1].fill_between(ks, ks/np.pi/2, 1, color='black', alpha=0.2)
ax[1,0].fill_between(ks, ks/np.pi/2, 1, color='black', alpha=0.2)
ax[1,1].fill_between(ks, ks/np.pi/2, 1, color='black', alpha=0.2)

ax[0,0].set_ylim(ymin, ymax)
ax[0,1].set_ylim(ymin, ymax)
ax[1,0].set_ylim(ymin, ymax)
ax[1,1].set_ylim(ymin, ymax)

ax[0,0].set_xlim(0,np.pi)
ax[0,1].set_xlim(0,np.pi)
ax[1,0].set_xlim(0,np.pi)
ax[1,1].set_xlim(0,np.pi)

#add titles to each subplot
ax[0,0].set_title('One unit cell',fontsize=20)
ax[0,1].set_title('Two unit cells',fontsize=20)
ax[1,0].set_title('Three unit cells',fontsize=20)
ax[1,1].set_title('Five unit cells',fontsize=20)

#add x and y labels
ax[0,0].set_ylabel('Frequency',fontsize=20)
ax[1,0].set_xlabel('k',fontsize=20)
ax[1,0].set_ylabel('Frequency',fontsize=20)
ax[1,1].set_xlabel('k',fontsize=20)

#add legend to each subplot
plt.show()
# %%
# Hyperparameters for colors and alpha
color_less_noise = 'blue'
color_more_noise = 'green'
color_min_noise = 'red'
alpha_less_noise = 0.7
alpha_more_noise = 0.7
alpha_min_noise = 0.7
fig, ax = plt.subplots(1, 1, figsize=(7, 4),dpi=300)

# for the 5s we have divide by 5
fifth = len(ks) // 5
ks_first = np.concatenate([ks[:fifth], ks[:fifth][::-1], ks[:fifth], ks[:fifth][::-1], ks[:fifth]])
ks_second = np.concatenate([ks[fifth:2*fifth], ks[fifth:2*fifth][::-1], ks[fifth:2*fifth], ks[fifth:2*fifth][::-1], ks[fifth:2*fifth]])
ks_third = np.concatenate([ks[2*fifth:3*fifth], ks[2*fifth:3*fifth][::-1], ks[2*fifth:3*fifth], ks[2*fifth:3*fifth][::-1], ks[2*fifth:3*fifth]])
ks_fourth = np.concatenate([ks[3*fifth:4*fifth], ks[3*fifth:4*fifth][::-1], ks[3*fifth:4*fifth], ks[3*fifth:4*fifth][::-1], ks[3*fifth:4*fifth]])
ks_fifth = np.concatenate([ks[4*fifth:], ks[4*fifth:][::-1], ks[4*fifth:], ks[4*fifth:][::-1], ks[4*fifth:]])

for i in range(gme5.freqs.shape[1]):
    if i % 5 == 0:
        ax.plot(ks_fifth, gme5.freqs[:, i], color=color_less_noise, alpha=alpha_less_noise, linewidth=solid_linewidth, label='less noise' if i == 0 else "")
        ax.plot(ks_fifth, gme5_more.freqs[:, i], color=color_more_noise, alpha=alpha_more_noise, linewidth=solid_linewidth, label='more noise' if i == 0 else "")
        ax.plot(ks_fifth, gme5_min.freqs[:, i], color=color_min_noise, alpha=alpha_min_noise, linewidth=solid_linewidth, label='medium noise' if i == 0 else "")
    elif i % 5 == 1:
        ax.plot(ks_fourth[::-1], gme5.freqs[:, i], color=color_less_noise, alpha=alpha_less_noise, linewidth=solid_linewidth)
        ax.plot(ks_fourth[::-1], gme5_more.freqs[:, i], color=color_more_noise, alpha=alpha_more_noise, linewidth=solid_linewidth)
        ax.plot(ks_fourth[::-1], gme5_min.freqs[:, i], color=color_min_noise, alpha=alpha_min_noise, linewidth=solid_linewidth)
    elif i % 5 == 2:
        ax.plot(ks_third, gme5.freqs[:, i], color=color_less_noise, alpha=alpha_less_noise, linewidth=solid_linewidth)
        ax.plot(ks_third, gme5_more.freqs[:, i], color=color_more_noise, alpha=alpha_more_noise, linewidth=solid_linewidth)
        ax.plot(ks_third, gme5_min.freqs[:, i], color=color_min_noise, alpha=alpha_min_noise, linewidth=solid_linewidth)
    elif i % 5 == 3:
        ax.plot(ks_second[::-1], gme5.freqs[:, i], color=color_less_noise, alpha=alpha_less_noise, linewidth=solid_linewidth)
        ax.plot(ks_second[::-1], gme5_more.freqs[:, i], color=color_more_noise, alpha=alpha_more_noise, linewidth=solid_linewidth)
        ax.plot(ks_second[::-1], gme5_min.freqs[:, i], color=color_min_noise, alpha=alpha_min_noise, linewidth=solid_linewidth)
    elif i % 5 == 4:
        ax.plot(ks_first, gme5.freqs[:, i], color=color_less_noise, alpha=alpha_less_noise, linewidth=solid_linewidth)
        ax.plot(ks_first, gme5_more.freqs[:, i], color=color_more_noise, alpha=alpha_more_noise, linewidth=solid_linewidth)
        ax.plot(ks_first, gme5_min.freqs[:, i], color=color_min_noise, alpha=alpha_min_noise, linewidth=solid_linewidth)

# Plot the reference lines
ax.plot(ks, gme1.freqs[:, 14], color='black', linestyle='--', linewidth=dashed_linewidth, label='no noise')
ax.plot(ks, gme1.freqs[:, 15], color='black', linestyle='--', linewidth=dashed_linewidth)

ax.set_ylim(.255, .27)
ax.set_xlim(.5 * np.pi, np.pi)
ax.set_xlabel('k', fontsize=20)
ax.set_ylabel('Frequency', fontsize=20)
ax.set_title('Noise Levels', fontsize=20)
#make axis numbers bigger
ax.tick_params(axis='both', which='major', labelsize=20)

# Only show unique labels in legend, and move legend outside plot to the left
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
# %%
phc5_more = W1(Nx=5,Ny=10,ra=.3,noise=0.1)
gme5_more = legume.GuidedModeExp(phc5_more,2.01)
gme5_more.run(gmode_inds=[0],numeig=200,compute_im=False,kpoints=np.vstack((ks,[0]*len(ks))))

#%%
phc5_min = W1(Nx=5,Ny=10,ra=.3,noise=0.01)
gme5_min = legume.GuidedModeExp(phc5_min,2.01)
gme5_min.run(gmode_inds=[0],numeig=200,compute_im=False,kpoints=np.vstack((ks,[0]*len(ks))))

#%%
phc5 = W1(Nx=5,Ny=7,ra=.3,noise=0.001)
gme5 = legume.GuidedModeExp(phc5,2.01)
gme5.run(gmode_inds=[0],numeig=200,compute_im=False,kpoints=np.vstack((ks,[0]*len(ks))))

#%%


# %%
def bandCollaps(gme,ks,numfolds=5,plot=False):

    #some checks
    if numfolds%2==0:
        raise ValueError('numfolds must be odd')
    if len(ks)%numfolds!=0:
        raise ValueError('ks must be divisible by numfolds')

    fold = len(ks) // numfolds
    
    #store the different ks that are looped through 
    ks_list = np.zeros((numfolds,len(ks)))
    for i in range(numfolds):
        for j in range(numfolds):
            if j%2==0:
                ks_list[i,j*fold:(j+1)*fold] = ks[fold*i:fold*(i+1)]
            else:
                ks_list[i,j*fold:(j+1)*fold] = ks[fold*i:fold*(i+1)][::-1]

    bands = np.zeros((gme.freqs.shape[1]//numfolds,len(ks)*numfolds))
    ksOut = np.zeros((len(ks)*numfolds))

    color_cycle_numfolds = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    current_color_numfolds = next(color_cycle_numfolds)
    for i in range(gme.freqs.shape[1]):
        # Generalize for any odd numfolds
        idx = i % numfolds
        bands[i//numfolds,idx*len(ks):(idx+1)*len(ks)] = gme.freqs[:,i]
        if idx == 0:
            current_color_numfolds = next(color_cycle_numfolds)
        if idx%2==1:
            if i//numfolds==0:
                ksOut[idx*len(ks):(idx+1)*len(ks)] = ks_list[numfolds-idx-1][::-1]
            if plot:
                plt.plot(ks_list[numfolds-idx-1][::-1], gme.freqs[:,i], color=current_color_numfolds, linewidth=solid_linewidth)
        else:
            if i//numfolds==0:
                ksOut[idx*len(ks):(idx+1)*len(ks)] = ks_list[numfolds-idx-1]
            if plot:
                plt.plot(ks_list[numfolds-idx-1], gme.freqs[:,i], color=current_color_numfolds, linewidth=solid_linewidth)
    return(ksOut,bands.T)

ksList,bands = bandCollaps(gme5,ks,numfolds=5)
plt.ylim(.255, .3)
#%%
plt.plot(ksList,bands,'o')
plt.ylim(.255, .3)
plt.show()
# %%
def bandsToDistribution(ksList,ks,bands):
    #for each band, i want to return the mean and std for each k 
    mean = np.zeros((len(ks),bands.shape[1]))
    std = np.zeros((len(ks),bands.shape[1]))
    for i in range(len(ks)):
        idxs = np.where(ksList==ks[i])[0]
        mean[i,:] = np.mean(bands[idxs],axis=0)
        std[i,:] = np.std(bands[idxs],axis=0)
    return(mean,std)

mean,std = bandsToDistribution(ksList,ks,bands)
# %%
def plotDistribution(ks,mean,std,c):
    for i in range(mean.shape[1]):
        line, = plt.plot(ks, mean[:,i],color=c)
        plt.fill_between(ks, mean[:,i] - std[:,i], mean[:,i] + std[:,i], 
                        color=line.get_color(), alpha=0.3)
# %%
#phc5 = W1(Nx=5,Ny=7,ra=.3,noise=0.01)
#gme5_2_1 = legume.GuidedModeExp(phc5,2.01)
#gme5_2_2 = legume.GuidedModeExp(phc5,2.01)
#gme5_2_3 = legume.GuidedModeExp(phc5,2.01)
#gme5_3 = legume.GuidedModeExp(phc5,3.01)
#gme5_4 = legume.GuidedModeExp(phc5,4.01)
#gme5_5 = legume.GuidedModeExp(phc5,5.01)

#gme5_2_1.run(gmode_inds=[0],numeig=200,compute_im=False,kpoints=np.vstack((ks,[0]*len(ks))))
#gme5_2_2.run(gmode_inds=[0,2],numeig=200,compute_im=False,kpoints=np.vstack((ks,[0]*len(ks))))
#gme5_2_3.run(gmode_inds=[0,2,4],numeig=200,compute_im=False,kpoints=np.vstack((ks,[0]*len(ks))))
#gme5_3.run(gmode_inds=[0],numeig=200,compute_im=False,kpoints=np.vstack((ks,[0]*len(ks))))
#gme5_4.run(gmode_inds=[0],numeig=200,compute_im=False,kpoints=np.vstack((ks,[0]*len(ks))))
#gme5_5.run(gmode_inds=[0],numeig=200,compute_im=False,kpoints=np.vstack((ks,[0]*len(ks))))
#%%
for g,c in zip([gme5_2_1,gme5_2_2,gme5_2_3,gme5_3,gme5_4,gme5_5],['blue','red','green','purple','orange','brown']):
    ksList,bands = bandCollaps(g,ks,numfolds=5)
    mean,std = bandsToDistribution(ksList,ks,bands)
    plotDistribution(ks,mean,std,c)
plt.ylim(.255, .3)
plt.show()

# %%
