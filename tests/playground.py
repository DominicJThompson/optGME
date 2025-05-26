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
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size':14})

with open('/Users/dominic/Desktop/optGME/tests/media/holeSize/W1Sweep.json', "r") as f:
    data = [json.loads(line) for line in f]

rs = np.array([d['val'] for d in data])
cost = np.array([d['cost'] for d in data])
cs = np.array([d['cs'] for d in data])


#%%
colors = np.where(np.all(cs, axis=1), 'green', 'red')
plt.scatter(rs,10**cost/266/1e-9*1e-2,c=colors)
#plt.yscale('log')
plt.ylabel(r'$\langle\alpha_{\text{back}}\rangle/n_g^2$ [cm$^{-1}$]')
plt.xlabel('radius [a]')
plt.ylim(10**-3,10**-2)
plt.xlim(.27,.35)
plt.show()

# %%
cmap = ListedColormap(["red", "green"])
cs_int = cs.astype(int)
plt.imshow(cs_int, cmap=cmap, aspect='auto',origin='lower')
plt.xticks(range(7),labels=['Freq','Mono','Mono','Mono','Mono','Above BW','Below BW'],rotation=35)
plt.yticks(np.arange(5)*20+10, labels=np.round(rs[10::20],2))
plt.xlabel('Constraints')
plt.ylabel('Radius [a]')
plt.show()
# %%
valid_indices = np.where(np.all(cs, axis=1))[0]

# Get the corresponding values
vrs = rs[valid_indices]
print(vrs)
print(cost[valid_indices])
# %%
plt.rcParams.update({'font.size':20})
# Load the updated JSON file
file_path = "/Users/dominic/Desktop/optGME/tests/media/holeSize/ZIWSweep2D_big.json"
with open(file_path, "r") as file:
    data = json.load(file)
    
# Extract values
x_vals = [entry["val"][0] for entry in data]
y_vals = [entry["val"][1] for entry in data]
costs = (10**np.array([entry["cost"] for entry in data]))/266/1e-9*1e-2

# Create arrays for contour finding and interpolation
x_unique = np.sort(list(set(x_vals)))
y_unique = np.sort(list(set(y_vals)))
X, Y = np.meshgrid(x_unique, y_unique)

# Create constraint mask array
cs_array = np.zeros((len(y_unique), len(x_unique)))
cost_array = np.zeros((len(y_unique), len(x_unique)))
for x, y, cs, cost in zip(x_vals, y_vals, [entry["cs"] for entry in data], costs):
    i = np.where(y_unique == y)[0][0]
    j = np.where(x_unique == x)[0][0]
    cs_array[i,j] = all(cs)
    cost_array[i,j] = cost

# Create figure
#plt.figure(figsize=(8, 6))

# Create a masked array for costs where constraints are not met
masked_costs = np.ma.array(cost_array, mask=np.logical_not(cs_array))

vmin = np.round(np.min(cost_array),2)
vmax = np.round(np.max(cost_array),2)

# Plot base heatmap with hatched pattern for invalid regions
plt.pcolormesh(X, Y, cost_array, cmap="terrain", vmin=vmin, vmax=vmax, alpha=1)
# Create masked array for invalid regions
invalid_mask = np.ma.array(X, mask=cs_array)
invalid_mask_T = np.ma.array(X.T, mask=cs_array.T)

# Plot grid lines only in invalid regions
plt.plot(invalid_mask_T, Y.T, 'r--', linewidth=1)
plt.plot(invalid_mask, Y, 'r--', linewidth=1)

# Plot valid regions with full opacity
plt.pcolormesh(X, Y, masked_costs, cmap="terrain", vmin=vmin, vmax=vmax)

# Add contour around valid region
plt.contour(X, Y, cs_array, levels=[0.5], colors='red', linewidths=2)

# Add colorbar
cbar = plt.colorbar(plt.pcolormesh(X, Y, masked_costs, cmap="terrain", vmin=vmin, vmax=vmax))
cbar.set_label(r"$\langle\alpha_{\text{back}}\rangle/n_g^2$ [cm$^{-1}$]",rotation=270,labelpad=40)

#place a red cross at specific coordinates
plt.scatter(0.105, npa.sqrt(3)/12, color='limegreen', s=200, marker='o',linewidths=2)
#plt.scatter(0.3, npa.sqrt(3)/2, color='red', s=200, marker='o',linewidths=2)
#plt.scatter(0.23, 1, color='cyan', s=200, marker='o',linewidths=2,zorder=5)
#plt.scatter(0.4, 1, color='black', s=200, marker='o',linewidths=2,zorder=5)

#set the x and y ticks  
plt.xticks([0.11,0.13,0.15,0.17])
plt.yticks([0.14,.17,.2,.23,.26,.29])

#set the x and y ticks  
#plt.xticks([0.22,.33,.44])
#plt.yticks([.6,.7,.8,.9,1])

# Labels and title
plt.xlabel("Radius [a]")
plt.ylabel("Y Position [a]")

plt.show()

# %%

def W1VarH(vars=np.array([]),Ny=10,dslab=170/266,eps_slab=3.4638,ra=.3):

    lattice = legume.Lattice(npa.array([1,0]),npa.array([0,Ny*npa.sqrt(3)]))

    #now we define a photonic crystal that goes over our lattice and add the one layer of our W1 waveguide
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=dslab,eps_b=eps_slab**2)

    
    #now we want to add the air holes to our photonic crystal to make it the W1 waveguide we do this in a loop
    for i in range(Ny*2-1):
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

        if npa.abs(iy)==1:
            phc.add_shape(legume.Circle(x_cent=x,y_cent=npa.sign(iy)*vars[1],r=vars[0]))
            continue

        #now we can add a circle with the given positions
        phc.add_shape(legume.Circle(x_cent=x,y_cent=y,r=ra))

    return(phc)

def ZIWVarH(vars=np.array([.105,.2]),Ny=10,dslab=170/266,eps_slab=3.4638,r0=.105,r1=.235):

    lattice = legume.Lattice(npa.array([1,0]),npa.array([0,Ny*npa.sqrt(3)-.25]))

    #now we define a photonic crystal that goes over our lattice and add the one layer of our W1 waveguide
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=dslab,eps_b=eps_slab**2)

    #now we want to add the air holes to our photonic crystal to make it the W1 waveguide we do this in a loop
    flip = False
    for i in range(Ny*2):
        iy = i-Ny+1
        y = iy*npa.sqrt(3)/2-npa.sqrt(3)/12

        if iy>0 and not flip:
            r0, r1 = r1, r0
            flip = True

        if iy==0:
            phc.add_shape(legume.Circle(x_cent=.5*((iy+1)%2),y_cent=y-npa.sqrt(3)/6,r=r1))
            phc.add_shape(legume.Circle(x_cent=.5*(iy%2),y_cent=vars[1]+npa.sqrt(3)/6,r=vars[0]))
            continue

        if iy==1:
            phc.add_shape(legume.Circle(x_cent=.5*((iy+1)%2),y_cent=-vars[1],r=vars[0]))
            phc.add_shape(legume.Circle(x_cent=.5*(iy%2),y_cent=y,r=r0))
            continue
        
        #add hole below and above
        phc.add_shape(legume.Circle(x_cent=.5*((iy+1)%2),y_cent=y-npa.sqrt(3)/6,r=r1))
        phc.add_shape(legume.Circle(x_cent=.5*(iy%2),y_cent=y,r=r0))
    
    return(phc)
                   
# %%
phc = W1VarH(vars=np.array([.3,npa.sqrt(3)/2]),Ny=10,dslab=170/266,eps_slab=3.4638,ra=.3)
ks = npa.linspace(npa.pi*.5,npa.pi,100)
gmeParams = {'verbose':True,'numeig':25,'compute_im':False,'kpoints':npa.vstack((ks,npa.zeros(len(ks))))}
gme = legume.GuidedModeExp(phc,4.01)
gme.run(**gmeParams)

#%%
phc = ZIWVarH(vars=np.array([.105,npa.sqrt(3)/12]),Ny=10,dslab=170/266,eps_slab=3.4638,r0=.105,r1=.235)
ks = npa.linspace(npa.pi*.5,npa.pi,100)
gmeParams = {'verbose':True,'numeig':25,'compute_im':False,'kpoints':npa.vstack((ks,npa.zeros(len(ks))))}
gme = legume.GuidedModeExp(phc,4.01)
gme.run(**gmeParams)

# %%


#freqs = gme.freqs.copy()
#freqs[31:,19], freqs[31:,20] = freqs[31:,20], freqs[31:,19]
#temp = freqs[:6,20].copy()
#freqs[:6,20] = freqs[:6,21]
#freqs[:6,21] = temp

freqs = gme.freqs.copy()

# Shade region below mode 19 (bottom layer)
plt.fill_between(np.linalg.norm(gme.kpoints,axis=0)/np.pi/2, 0, freqs[:,19], alpha=.7, color='navy', zorder=1)

# Shade regoin above the 22 mode(top layer)
#plt.fill_between(np.linalg.norm(gme.kpoints,axis=0)/np.pi/2, 100, freqs[:,22], alpha=.7, color='navy', zorder=1)
# Get axis limits
xmin, xmax = plt.xlim()

# Create coordinates for shaded region above x=y
x = np.linspace(xmin, xmax, 100)
y = x.copy()

plt.fill_between(x, y, 100, color='darkGray', alpha=1, zorder=0)

# Add mode 20 and 21 in purple
plt.plot(np.linalg.norm(gme.kpoints,axis=0)/np.pi/2,freqs[:,20], color='purple', linewidth=2, zorder=4)
plt.plot(np.linalg.norm(gme.kpoints,axis=0)/np.pi/2,freqs[:,21], color='purple', linestyle='--', linewidth=2, zorder=4)

#add a red dot on mode 20 and the 8th k point that apears above the mode line
plt.scatter(np.linalg.norm(gme.kpoints,axis=0)[8*4]/np.pi/2,freqs[8*4,20], color='limegreen', s=200, zorder=5)

#add x and y labels 
plt.xlabel(r'Wavevector $k_x 2\pi/a$', fontsize=18)
plt.ylabel(r'Frequency $\omega a/2\pi c$', fontsize=18)

plt.ylim(.245,.295)
plt.xlim(.25,.5)

#add a second y axis with the frequency in THz
ax2 = plt.gca().twinx()
conFac = 1e-12*299792458/266/1e-9
ax2.set_ylim(.245*conFac,.295*conFac)
ax2.set_ylabel("Frequency [THz]", fontsize=18)
plt.show()
# %%
from matplotlib.patches import Circle
ylim= 10*np.sqrt(3)/2
ys = np.linspace(-ylim/2,ylim/2,300)
z=phc.layers[0].d/2
fields,_,_ = gme.get_field_xy('E',8*4,20,z,ygrid=ys,component='xyz')
eabs = np.abs(np.conj(fields['x'])*fields['x']+np.conj(fields['y'])*fields['y']+np.conj(fields['z'])*fields['z'])
maxF = np.max(eabs)

fix,ax = plt.subplots()
cax = ax.imshow(eabs.T,extent=[-ylim/2,ylim/2,.5,-.5],cmap='plasma',vmax=maxF,vmin=0)
circles = [Circle((-s.y_cent,s.x_cent),s.r,edgecolor='white',facecolor='none',linewidth=3) for s in phc.layers[0].shapes]
cirlcesArround = [Circle((0,0),s.r,edgecolor='white',facecolor='none',linewidth=3) for s in phc.layers[0].shapes]
for c,ca in zip(circles,cirlcesArround):
    plt.gca().add_patch(c)
    ca.center = (c.center[0],c.center[1]-np.sign(c.center[1]))
    plt.gca().add_patch(ca)
plt.ylim(-.5,.5)
plt.xlim(-ylim/2,ylim/2)
ax.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
plt.show()


# %%
def W1Spaced(vars=npa.zeros((0,0)),NyChange=3,Ny=10,dslab=170/266,eps_slab=3.4638,ra=.3,space=0):

    vars = vars.reshape((3,NyChange*2))

    lattice = legume.Lattice(npa.array([1,0]),npa.array([0,Ny*npa.sqrt(3)+space]))

    #now we define a photonic crystal that goes over our lattice and add the one layer of our W1 waveguide
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=dslab,eps_b=eps_slab**2)

    for i in range(vars.shape[1]):
        phc.add_shape(legume.Circle(x_cent=vars[0,i],y_cent=vars[1,i],r=vars[2,i]))

    #add space in the middle 
    phc.add_shape(legume.Poly(x_edges=[-.5,.5,.5,-.5],y_edges=[-space/2,-space/2,space/2,space/2]))

    #now we want to add the air holes to our photonic crystal to make it the W1 waveguide we do this in a loop
    for i in range(Ny*2-1):
        iy = i-Ny
        if i>=Ny:
            iy+=1

        if npa.abs(iy)<=NyChange:
            continue
        
        #move the hole over by half a unit cell if they are on odd rows
        if iy%2==1:
            x = .5
        else:
            x = 0

        #the y component should be scaled by the factor of np.sqrt(3)/2
        y = iy*npa.sqrt(3)/2

        #now we can add a circle with the given positions
        phc.add_shape(legume.Circle(x_cent=x,y_cent=y+np.sign(y)*space/2,r=ra))

    return(phc)

phc = W1Spaced(NyChange=0,space=20/266)
_ =legume.viz.eps_xy(phc,Ny=1000)
# %%
ks = npa.linspace(npa.pi*.5,npa.pi,100)
gmeParams = {'verbose':False,'numeig':25,'compute_im':False,'kpoints':npa.vstack((ks,npa.zeros(len(ks)))),'gmode_inds':[0,2]}
freqs_results = []
ng_results = []

gme_results = []  # List to store the gme objects

for i in range(5):
    space = (i+1)*10/266  # Different space for each iteration: 10/266, 20/266, 30/266, 40/266, 50/266
    phc = W1Spaced(NyChange=0, space=space)
    gme = legume.GuidedModeExp(phc, 3.01)
    gme.run(**gmeParams)
    freqs_results.append(gme.freqs.copy())
    gme_results.append(gme)  # Save the gme object
    
    # Calculate group index (ng) for the 20th band (index 19)
    # ng = c/vg = c/(dω/dk)
    band_index = 20  # 20th band (0-indexed)
    freqs_band = gme.freqs[:, band_index]
    # Calculate numerical derivative
    dw_dk = np.gradient(freqs_band, ks)
    ng = 1 / dw_dk /2/np.pi # Group index is inverse of group velocity
    ng_results.append(ng)
    print(f"Completed simulation with space = {space}")
    print(f"Group index (ng) range for band 20: min={np.min(ng):.2f}, max={np.max(ng):.2f}")

# %%
# Plot the band structure for different spacing values
plt.figure(figsize=(10, 6))

# Define colors for different spacing values
colors = ['blue', 'green', 'red', 'purple', 'orange']
labels = []

# Normalize k-points for x-axis
k_normalized = np.linalg.norm(gme.kpoints, axis=0)/np.pi/2
plt.hlines(0.275,0,1,color='black',linestyle='--',linewidth=2)

# Plot each set of results
for i, freqs in enumerate(freqs_results):
    space = (i+1)*10/266
    space_nm = space * 266  # Convert to nm
    plt.plot(k_normalized, freqs[:,20], color=colors[i], linewidth=2,label=f'{space_nm:.1f} nm')
    plt.plot(k_normalized, freqs[:,21], color=colors[i], linewidth=2)

# Add light cone
x = np.linspace(min(k_normalized), max(k_normalized), 100)
y = x.copy()
plt.fill_between(x, y, 1, color='darkGray', alpha=0.3, zorder=0)

# Add labels and title
plt.xlabel(r'Wavevector $k_x 2\pi/a$', fontsize=24)
plt.ylabel(r'Frequency $\omega a/2\pi c$', fontsize=24)
plt.title('Band Structure for Different Spacing Values', fontsize=24)
plt.legend()

# Set axis limits
plt.ylim(0.245, 0.305)
plt.xlim(0.25, 0.5)

# Add second y-axis with frequency in THz
ax2 = plt.gca().twinx()
conFac = 1e-12*299792458/266/1e-9
ax2.set_ylim(0.245*conFac, 0.295*conFac)
ax2.set_ylabel("Frequency [THz]", fontsize=24)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Plot group index (ng) vs normalized frequency
plt.figure(figsize=(10, 6))

# Define colors for different spacing values
colors = ['blue', 'green', 'red', 'purple', 'orange']
markers = ['o', 's', '^', 'd', 'x']

for i, (ng_values, freqs) in enumerate(zip(ng_results, freqs_results)):
    space = (i+1)*10/266
    space_nm = space * 266  # Convert to nm
    
    # Get the frequencies for band 20
    band_freqs = freqs[:, 20]
    
    # Get the k-points
    k_values = np.linalg.norm(gme.kpoints, axis=0)
    
    # Filter points below the light cone (where frequency < k)
    below_light_cone = band_freqs < k_values/(2*np.pi)
    
    # Plot ng vs normalized frequency only for points below the light cone
    plt.plot(band_freqs[below_light_cone], np.abs(ng_values[below_light_cone]), 
             color=colors[i], marker=markers[i], 
             linestyle='-', linewidth=2, markersize=6, 
             label=f'{space_nm:.1f} nm')

# Add labels and title
plt.vlines(0.275,0,100,color='black',linestyle='--',linewidth=2)
plt.xlabel(r'Normalized Frequency $\omega a/2\pi c$', fontsize=24)
plt.ylabel(r'Group Index ($n_g$)', fontsize=24)
plt.title('Group Index vs Frequency for Different Spacing Values', fontsize=24)
plt.legend(loc='upper left')
#plt.yscale('log')
plt.ylim(1,30)

# Add grid and improve layout
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Visualize the field distribution for each spacing value at a specific frequency
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np

# Define the target frequency we want to visualize
target_freq = 0.275  # Target normalized frequency

# Font size variables for consistent styling
TITLE_SIZE = 24
LABEL_SIZE = 22
TICK_SIZE = 18

# Set up variables
ylim = 8*np.sqrt(3)/2
ys = np.linspace(-ylim/2, ylim/2, 300)
z = phc.layers[0].d/2  # Middle of the slab

# Create a figure with subplots for each spacing value
fig, axes = plt.subplots(len(gme_results), 1, figsize=(6, len(gme_results)))

for i, gme in enumerate(gme_results):
    space = (i+1)*10/266
    space_nm = space * 266  # Convert to nm
    
    # Use fixed mode index 21 and find the k-point closest to our target frequency
    mode_idx = 21
    freq_diff = np.abs(gme.freqs[:, mode_idx] - target_freq)
    k_idx = np.argmin(freq_diff)
    actual_freq = gme.freqs[k_idx, mode_idx]
    
    # Get the field at this k-point and mode
    fields, _, _ = gme.get_field_xy('E', k_idx, mode_idx, z, ygrid=ys, component='xyz')
    eabs = np.abs(np.conj(fields['x'])*fields['x'] + np.conj(fields['y'])*fields['y'] + np.conj(fields['z'])*fields['z'])
    
    # Get the current axis
    ax = axes[i] if len(gme_results) > 1 else axes
    
    # Plot the field
    cax = ax.imshow(eabs.T, extent=[-ylim/2, ylim/2, -.5, .5], cmap='plasma', aspect='auto')
    
    # Add the holes as circles
    phc = W1Spaced(NyChange=0, space=space)
    circles = [Circle((-s.y_cent, s.x_cent), s.r, edgecolor='white', facecolor='none', linewidth=2) 
               for s in phc.layers[0].shapes[1:]]
    for c in circles:
        ax.add_patch(c)
        ax.add_patch(Circle((c.center[0],c.center[1]-1),c.radius,edgecolor='white',facecolor='none',linewidth=2))
    
    # Add the rectangle representing the space in the middle of the waveguide
    from matplotlib.patches import Rectangle
    rect = Rectangle((-space/2,-.5), space, 1, edgecolor='white', facecolor='none', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    
    # Add spacing value on the side instead of a full title
    ax.text(-ylim/2 + 0.1, 0, f'{space_nm:.1f} nm', fontsize=LABEL_SIZE, color='white', 
            verticalalignment='center', horizontalalignment='left', fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.75, edgecolor='none', pad=3))
    # Set limits
    ax.set_ylim(-.5, .5)
    ax.set_xlim(-ylim/2, ylim/2)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                  labelbottom=False, labelleft=False)
plt.show()
# %%
# %%
