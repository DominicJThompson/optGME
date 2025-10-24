#%%
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import legume
import autograd.numpy as npa
import optomization
import json
from optomization.utils import NG
from optomization.crystals import W1
from optomization.cost import Backscatter
from matplotlib.patches import Circle
# Disable LaTeX but use Computer Modern fonts
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern math fonts
mpl.rcParams['font.family'] = 'STIXGeneral'  # Use STIX fonts (similar to Computer Modern)

#%%
# Load W1Best.json data for analysis
with open('/Users/dominic/Desktop/optGME/tests/media/ginds3/W1Best.json', 'r') as file:
    w1_data = json.load(file)

# Get the original W1holes from the data
W1holes_original = np.array(w1_data[-1]['result']['x'])
costParams = w1_data[-1]['cost']
gmeParams = w1_data[-1]['gmeParams']
gmeParams['numeig'] = gmeParams['numeig']+10
gmeParams['kpoints'] = np.vstack((np.linspace(.5*np.pi,np.pi,50),np.zeros((50))))
# Create the PHC with the original hole parameters
W1PHC = W1(NyChange=0)

# Run the GME simulation
gme = legume.GuidedModeExp(W1PHC, 4.01)
gme.run(**gmeParams)

#%%
W1PHCOPT = W1(vars=W1holes_original)
gmeOPT = legume.GuidedModeExp(W1PHCOPT, 4.01)
gmeOPT.run(**gmeParams)

#%%
# Calculate group index for each k-point
ng = []
k_points = gme.kpoints[0, :]

for i in range(len(k_points)):
    # Calculate group index at each k-point for the mode of interest
    # We'll use the NG function from optomization.utils
    current_ng = NG(gme, i, 20)
    ng.append(current_ng)

# Convert to numpy array for easier manipulation
ng = np.array(ng)

# Print some information about the calculated group indices
print(f"Mean group index: {np.mean(ng)}")
print(f"Max group index: {np.max(ng)}")
print(f"Min group index: {np.min(ng)}")

# Optional: Plot the group index vs k-point
plt.figure(figsize=(10, 6))
plt.plot(k_points, ng)
plt.xlabel('k-point')
plt.ylabel('Group Index (ng)')
plt.title('Group Index vs k-point')
plt.grid(True)
plt.show()

# %%
def plotBands(gme,params,color='red',plotback=True,index=0):

    # Font size parameters
    TITLE_FONT_SIZE = 34
    LABEL_FONT_SIZE = 34
    ANNOTATION_FONT_SIZE = 34
    TICK_FONT_SIZE = 28
    
    #missilanius variables needed
    conFac = 1e-12*299792458/params['cost']['a']/1e-9
    if index==0 or index==1:
        freqmin, freqmax = .245, .282
    elif index==2 or index==3:
        freqmin, freqmax = .238, .298
    mode = params['mode']
    kindex = np.abs(gme.kpoints[0,:] - params['gmeParams']['kpoints'][0][0]).argmin()
    print(kindex)
    # Generate some sample data
    ks = np.linspace(0.25, 0.5, 50)
    
    # Find the index of the point closest to k=0.3333*2*np.pi
    target_k = 0.3333
    closest_k_idx = np.abs(ks - target_k).argmin()

    # Create figure and define gridspec layout
    fig = plt.figure(figsize=(6.4, 4.8))

    # Main subplot (dispersion curve)
    ax2 = plt.gca()
    ax2.set_xlabel(r"Wavevector $k$",fontsize=LABEL_FONT_SIZE)
    ax2.set_ylabel(r"Frequency $\omega$",fontsize=LABEL_FONT_SIZE)
    ax2.set_xlim(0.25, 0.5)
    ax2.set_ylim(freqmin,freqmax)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)

    #add twin axis
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel("Frequency [THz]",fontsize=LABEL_FONT_SIZE)
    ax2_twin.set_ylim(freqmin*conFac,freqmax*conFac)
    ax2_twin.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)

    #plot frequency plot 
    ax2.fill_between(ks,ks,np.max(ks),color='darkGray',alpha=1) #light line
    ax2.fill_between(ks,gme.freqs[:,mode-1],np.zeros_like(ks),color='navy',alpha=.7) #continums
    ax2.plot(ks,gme.freqs[:,mode],color='darkviolet',linewidth=2,zorder=2) #band of interest
    otherband = gme.freqs[:,mode+1].copy()
    otherband[25:]-=(ks[25:]-0.375)**2
    ax2.plot(ks,otherband,color='darkviolet',linestyle='--',linewidth=2,zorder=2)
    # Add a red dot at k=0.3333*2*np.pi on the band of interest
    ax2.scatter(ks[closest_k_idx], gme.freqs[closest_k_idx, mode], color='red', s=200, zorder=4,edgecolor='black',linewidth=2)
    
    #if plotback:
        #ax2.plot(ks,gme.freqs[:,mode+1],color='darkviolet',linewidth=2,linestyle='--') #other band
    #else:
        #ax2.plot(ks[15:],gme.freqs[15:,mode+1],color='darkviolet',linewidth=2,linestyle='--') #other band
    #ax2.scatter(ks[kindex],gme.freqs[kindex,mode],s=200,color=color,zorder=3) #optomized point

    #find the bandwidth in frequncy
    intersect = np.where(np.sign((gme.freqs[:,mode]-ks)[:-1]) != np.sign((gme.freqs[:,mode]-ks))[1:])[0]
    bandMax = min(np.max(gme.freqs[:,mode]),np.min(gme.freqs[15:,mode+1]),gme.freqs[intersect,mode])-.001
    bandMin = np.max(np.hstack((gme.freqs[:,mode-1],np.min(gme.freqs[:,mode]))))
    #if color=='red':
    #    bandMin+=.0015
    #ax2.fill_between(ks,np.ones_like(ks)*bandMax,np.ones_like(ks)*bandMin,color='cyan',alpha=.5,zorder=0)

    # Show plot
    plt.show()

plotBands(gme,w1_data[-1],color='#EE7733',plotback=True,index=2)
# %%
def filedPlots(phc, gme, params):
    # Font size variables for consistent styling
    TITLE_SIZE = 28
    LABEL_SIZE = 26
    TICK_SIZE = 20
    LEGEND_SIZE = 24
    COLORBAR_LABEL_SIZE = 24
    COLORBAR_TICK_SIZE = 20
    
    # Set up variables
    ylim = 8*np.sqrt(3)/2
    ys = np.linspace(-ylim/2, ylim/2, 300)  # Changed from ys to xs for rotation
    mode = params['mode']
    kindex = np.abs(gme.kpoints[0,:] - params['gmeParams']['kpoints'][0][0]).argmin()
    z = phc.layers[0].d/2

    # Get field of crystal - swapped grid for rotation
    fields, _, _ = gme.get_field_xy('E', kindex, mode, z, ygrid=ys, component='xyz')
    eabs = np.abs(np.conj(fields['x'])*fields['x'] + np.conj(fields['y'])*fields['y'] + np.conj(fields['z'])*fields['z'])
    maxF = np.max(eabs)

    eps = legume.viz.eps_xy(phc, Nx=100, Ny=100, plot=False)
    f, _, _ = gme.get_field_xy('E', kindex, mode, z, Ny=200, Nx=100, component='xyz')
    e = np.abs(np.conj(f['x'])*f['x'] + np.conj(f['y'])*f['y'] + np.conj(f['z'])*f['z'])
    
    # Create a single figure with proper aspect ratio - switched dimensions
    fig = plt.figure(figsize=(15, 6))  # Switched dimensions back
    ax = plt.gca()
    
    # Repeat the data twice in the x direction (horizontally) and transpose for axis swap
    eabs_repeated = np.hstack((eabs, eabs, eabs))
    
    # Plot the repeated field data with swapped axes
    cax = ax.imshow(eabs_repeated, extent=[-1.5, 1.5, -ylim/2, ylim/2], 
                   cmap='plasma', vmax=maxF, vmin=0, aspect='auto', origin='lower')
    
    # Add circles representing holes - swapped coordinates back
    circles = [Circle((s.x_cent, -s.y_cent), s.r, edgecolor='white', 
                     facecolor='none', linewidth=3) for s in phc.layers[0].shapes]
    
    for c in circles:
        ax.add_patch(c)
        # Add circles 1 unit left and right
        c_left = Circle((c.center[0] - 1, c.center[1]), c.radius, edgecolor='white', 
                       facecolor='none', linewidth=3)
        c_leftleft = Circle((c.center[0] - 2, c.center[1]), c.radius, edgecolor='white', 
                           facecolor='none', linewidth=3)
        c_right = Circle((c.center[0] + 1, c.center[1]), c.radius, edgecolor='white', 
                        facecolor='none', linewidth=3)
        c_rightright = Circle((c.center[0] + 2, c.center[1]), c.radius, edgecolor='white', 
                             facecolor='none', linewidth=3)
        ax.add_patch(c_left)
        ax.add_patch(c_right)
        ax.add_patch(c_leftleft)
        ax.add_patch(c_rightright)
    
    # Swapped xlim and ylim
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-ylim/2, ylim/2)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                  labelbottom=False, labelleft=False)
    
    # Removed colorbar code
    plt.tight_layout()
    plt.show()

filedPlots(W1PHC, gme, w1_data[-1])
# %%
def plotBandsWithNG(gme, params, ng_values, color='red', plotback=True, index=0):
    """Plot both group index and band structure of a photonic crystal waveguide."""
    
    # Font size parameters
    TITLE_FONT_SIZE = 34
    LABEL_FONT_SIZE = 34
    ANNOTATION_FONT_SIZE = 34
    TICK_FONT_SIZE = 28
    
    # Conversion factor and frequency ranges
    conFac = 1e-12*299792458/params['cost']['a']/1e-9
    if index==0 or index==1:
        freqmin, freqmax = .245, .282
    elif index==2 or index==3:
        freqmin, freqmax = .238, .298
    mode = params['mode']
    kindex = np.abs(gme.kpoints[0,:] - params['gmeParams']['kpoints'][0][0]).argmin()
    print(kindex)
    
    # Generate k-points array
    ks = np.linspace(0.25, 0.5, 50)
    
    # Create figure with two subplots (2 rows, 1 column) sharing x axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [.66, 1]}, sharex=True)
    
    # Top subplot (group index)
    ax1.set_ylabel(r"$n_g$", fontsize=LABEL_FONT_SIZE+8)
    ax1.set_xlim(0.25, 0.5)
    ax1.set_yscale('log')
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    # Hide x-axis labels for top plot since they're shared
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    # Plot group index
    ax1.plot(ks, ng_values, color='#FF3333', linewidth=4)
    
    # Bottom subplot (dispersion curve)
    ax2.set_xlabel(r"Wavevector $k$", fontsize=LABEL_FONT_SIZE)
    ax2.set_ylabel(r"Frequency $\omega$", fontsize=LABEL_FONT_SIZE)
    ax2.set_ylim(freqmin, freqmax)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    # Add twin axis for frequency in THz
    
    # Plot frequency plot
    ax2.fill_between(ks, ks, np.max(ks), color='darkGray', alpha=1)  # light line
    ax2.fill_between(ks, gme.freqs[:,mode-1], np.zeros_like(ks), color='navy', alpha=.7)  # continuum
    
    # Band of interest - thin line
    ax2.plot(ks, gme.freqs[:,mode], color='darkviolet', linewidth=2, zorder=2)
    
    # Band of interest - thick, transparent red line
    ax2.plot(ks, gme.freqs[:,mode], color='red', linewidth=14, alpha=0.3, zorder=1)
    
    # Other band
    if plotback:
        ax2.plot(ks, gme.freqs[:,mode+1], color='darkviolet', linewidth=2, linestyle='--')
    else:
        ax2.plot(ks[15:], gme.freqs[15:,mode+1], color='darkviolet', linewidth=2, linestyle='--')
    
    # Find the bandwidth in frequency
    intersect = np.where(np.sign((gme.freqs[:,mode]-ks)[:-1]) != np.sign((gme.freqs[:,mode]-ks))[1:])[0]
    bandMax = min(np.max(gme.freqs[:,mode]), np.min(gme.freqs[15:,mode+1]), gme.freqs[intersect,mode])-.001
    bandMin = np.max(np.hstack((gme.freqs[:,mode-1], np.min(gme.freqs[:,mode]))))
    ax2.fill_between(ks, np.ones_like(ks)*bandMax, np.ones_like(ks)*bandMin, color='cyan', alpha=.5, zorder=0)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

plotBandsWithNG(gme, w1_data[-1], np.abs(ng), color='#EE7733', plotback=True, index=2)
# %%
# Create a schematic of an optical buffer with gaussian pulses
def plot_optical_buffer_schematic(compression_factor=4, num_pulses=2, figsize=(10, 5), 
                                 text_size=24, input_width=0.6, output_width=0.6,
                                 input_center_offset=1, output_center_offset=1,
                                 input_cutoff=4.0, output_cutoff=4.0,
                                 buffer_region_start=3, buffer_region_end=7):
    """
    Plot a schematic of an optical buffer showing:
    - Input gaussian pulse coming from left
    - Compression region in the middle (in red)
    - Output pulse on the right
    
    Parameters are the same as the original function but all pulses are shown in a single plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define the x-axis range and buffer regions
    x = np.linspace(0, 10, 1000)
    input_region = (0, buffer_region_start)
    buffer_region = (buffer_region_start, buffer_region_end)
    output_region = (buffer_region_end, 10)
    
    # Color scheme
    input_color = '#3388FF'  # Blue
    buffer_color = '#FF3333'  # Red
    output_color = '#3388FF'  # Blue
    lightgray = '#D3D3D3'
    
    # Draw the buffer region (in red instead of gray)
    ax.axvspan(buffer_region[0], buffer_region[1], alpha=0.2, color=lightgray)
    
    # Add black box around the red region
    rect = plt.Rectangle((buffer_region[0], -0.2), 
                        buffer_region[1] - buffer_region[0], 
                        1.5, 
                        fill=False, 
                        edgecolor='black', 
                        linewidth=2)
    ax.add_patch(rect)
    
    # Add PHC label under the middle region
    ax.text(buffer_region[0] + (buffer_region[1]-buffer_region[0])/2, -0.15, "PHC", 
            ha='center', fontsize=text_size)
    
    # Generate the complete pulses for the entire range
    input_center = input_region[1] - input_center_offset
    input_pulse = np.exp(-((x - input_center) / input_width) ** 2)
    
    # Generate the complete output pulse
    output_center = output_region[0] + output_center_offset
    output_pulse = np.exp(-((x - output_center) / output_width) ** 2)
    
    # Add compressed pulses in the buffer region
    buffer_width = input_width / compression_factor
    
    # Create a combined pulse for the buffer region
    buffer_pulse = np.zeros_like(x)
    for i in range(int(compression_factor * 2)):
        pulse_center = buffer_region[0] + (i + 0.5) * (buffer_region[1] - buffer_region[0]) / (compression_factor * 2)
        buffer_pulse += np.exp(-((x - pulse_center) / buffer_width) ** 2)

    # Plot input pulse
    ax.plot(x, input_pulse+output_pulse+buffer_pulse, color=input_color, linewidth=4)
    
    # Mask to only show buffer pulse within buffer region
    buffer_mask = (x >= buffer_region[0]) & (x <= buffer_region[1])
    ax.plot(x[buffer_mask], (input_pulse+output_pulse+buffer_pulse)[buffer_mask], color=buffer_color, linewidth=4)
    
    # Set axis properties
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-0.2, 1.3)
    
    # Remove all axis elements
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
# Create the optical buffer schematic
plot_optical_buffer_schematic(compression_factor=4, 
                             text_size=46)

# %%
def plotFieldDistribution(phc, gme, params):
    # Font size variables for consistent styling
    TITLE_SIZE = 28
    LABEL_SIZE = 26
    TICK_SIZE = 20
    LEGEND_SIZE = 24
    
    # Set up variables
    ylim = 6*np.sqrt(3)/2
    ys = np.linspace(-ylim/2, ylim/2, 300)
    mode = params['mode']
    kindex = np.abs(gme.kpoints[0,:] - params['gmeParams']['kpoints'][0][0]).argmin()
    z = phc.layers[0].d/2

    # Get field of crystal
    fields, _, _ = gme.get_field_xy('E', kindex, mode, z, ygrid=ys, component='xyz')
    eabs = np.abs(np.conj(fields['x'])*fields['x'] + np.conj(fields['y'])*fields['y'] + np.conj(fields['z'])*fields['z'])
    maxF = np.max(eabs)

    # Create a figure with proper aspect ratio
    fig = plt.figure(figsize=(6, 10))
    ax = plt.gca()
    
    # Plot just one set of the field data
    cax = ax.imshow(eabs.T, extent=[-ylim/2, ylim/2, -0.5, 0.5], 
                   cmap='plasma', vmax=maxF, vmin=0, aspect='auto', origin='lower')
    
    # Add filled circles representing holes (filled with white)
    circles = [Circle((-s.y_cent, s.x_cent), s.r, edgecolor='white', 
                     facecolor='white', linewidth=1.5) for s in phc.layers[0].shapes]
    
    # Add circles for the main row
    for c in circles:
        ax.add_patch(c)
    
    # Add a second row of circles below
    for s in phc.layers[0].shapes:
        circle_below = Circle((-s.y_cent, s.x_cent-1), s.r, 
                             edgecolor='black', facecolor='white', linewidth=1.5)
        ax.add_patch(circle_below)
    
    # Add a large red X at the center point
    center_x, center_y = 0, 0
    marker_size = 500
    line_width = 6
    #ax.scatter(center_x, center_y, s=marker_size, marker='x', color='red', linewidth=line_width)
    
    # Find where the field exists and set limits accordingly
    field_threshold = 0.05 * maxF  # 5% of maximum field value
    field_exists = eabs.T > field_threshold
    y_indices, x_indices = np.where(field_exists)
    
    # Get coordinates from indices
    x_coords = np.linspace(-ylim/2, ylim/2, eabs.T.shape[1])[x_indices]
    y_coords = np.linspace(-0.5, 0.5, eabs.T.shape[0])[y_indices]
    
    # Set limits to the field existence area with some padding
    x_padding = (max(x_coords) - min(x_coords)) * 0.1
    y_padding = (max(y_coords) - min(y_coords)) * 0.1
    
    ax.set_xlim(-ylim/2, ylim/2)
    ax.set_ylim(-0.5, 0.5)  # Extend lower bound to show the second row
    
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                  labelbottom=False, labelleft=False)
    
    plt.tight_layout()
    plt.show()

plotFieldDistribution(W1PHC, gme, w1_data[-1])
# %%
def plotFieldDistribution(phc, gme, params):
    # Font size variables for consistent styling
    TITLE_SIZE = 28
    LABEL_SIZE = 26
    TICK_SIZE = 20
    LEGEND_SIZE = 24
    
    # Set up variables
    ylim = 12*np.sqrt(3)/2
    ys = np.linspace(-ylim/2, ylim/2, 1000)
    mode = params['mode']
    kindex = np.abs(gme.kpoints[0,:] - params['gmeParams']['kpoints'][0][0]).argmin()
    z = phc.layers[0].d/2

    # Get field of crystal
    fields, _, _ = gme.get_field_xy('E', kindex, mode, z, ygrid=ys, component='xyz')
    eabs = np.abs(np.conj(fields['x'])*fields['x'] + np.conj(fields['y'])*fields['y'] + np.conj(fields['z'])*fields['z'])
    maxF = np.max(eabs)

    # Create a figure with proper aspect ratio - wider for repeating in x direction
    fig = plt.figure(figsize=(20, 6))
    ax = plt.gca()
    
    # Number of repetitions in x direction
    num_repeats = 3
    
    # Create repeated field data by stacking horizontally
    repeated_eabs = np.hstack((eabs,eabs,eabs))
    
    # Plot the field data with swapped axes (x and y swapped)
    # Note: we're transposing the data differently to swap axes
    cax = ax.imshow(repeated_eabs, extent=[-0.5, 0.5+num_repeats-1, -ylim/2, ylim/2], 
                   cmap='plasma', vmax=maxF, vmin=0, aspect='auto', origin='lower')
    
    # Add filled circles representing holes (filled with white) for each repetition
    for rep in range(num_repeats):
        x_offset = rep
        
        # Add circles for the main row in each repetition
        for s in phc.layers[0].shapes:
            # Swap x and y coordinates and add offset for repetition
            circle = Circle((s.x_cent + x_offset, -s.y_cent), s.r, 
                           edgecolor='white', facecolor='white', linewidth=1.5)
            ax.add_patch(circle)
        
        # Add a row of circles to the right for each repetition
        for s in phc.layers[0].shapes:
            # Swap x and y coordinates, add offset for repetition, and shift for the second row
            circle_right = Circle((s.x_cent + x_offset + 1, -s.y_cent), s.r, 
                                 edgecolor='black', facecolor='white', linewidth=1.5)
            ax.add_patch(circle_right)
            
        # Add another row of circles to the left for each repetition
        for s in phc.layers[0].shapes:
            # Swap x and y coordinates, add offset for repetition, and shift for the third row
            circle_left = Circle((s.x_cent + x_offset - 1, -s.y_cent), s.r, 
                                edgecolor='black', facecolor='white', linewidth=1.5)
            ax.add_patch(circle_left)
    
    # Add a single large red X at the center point (only one X in total)
    center_x, center_y = num_repeats/2 - 0.5, 0  # Center of the entire plot
    marker_size = 500
    line_width = 6
    #ax.scatter(center_x, center_y, s=marker_size, marker='x', color='red', linewidth=line_width)
    
    # Set limits to show all repetitions
    ax.set_xlim(-0.5, 0.5*num_repeats)
    ax.set_ylim(-ylim/2, ylim/2)
    
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                  labelbottom=False, labelleft=False)
    
    plt.tight_layout()
        # Save the figure with high quality
    filename = f"field_distribution_mode{mode}_k{kindex}.jpg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', format='jpg')
    plt.show()

# Plot the field distribution with swapped axes and repetitions
#plotFieldDistribution(W1PHC, gme, w1_data[-1])
plotFieldDistribution(W1PHCOPT, gmeOPT, w1_data[-1])
# %%\
from matplotlib.patches import Circle
ylim= 6*np.sqrt(3)/2
ys = np.linspace(-ylim/2,ylim/2,300)
z=W1PHC.layers[0].d/2
fields,_,_ = gme.get_field_xy('E',0,20,z,ygrid=ys,component='xyz')
eabs = np.abs(np.conj(fields['x'])*fields['x']+np.conj(fields['y'])*fields['y']+np.conj(fields['z'])*fields['z'])
maxF = np.max(eabs)

#%%
fix,ax = plt.subplots()

# Plot the field pattern multiple times along x
for x_offset in [-1.5, -0.5, 0.5, 1.5]:
    cax = ax.imshow(eabs,extent=[x_offset-0.5,x_offset+0.5,-ylim/2,ylim/2],cmap='plasma',vmax=maxF,vmin=0)
    
    # Add circles for each x offset
    circles = [Circle((s.x_cent+x_offset,-s.y_cent),s.r,edgecolor='white',facecolor='none',linewidth=3) for s in W1PHC.layers[0].shapes]
    cirlcesArround = [Circle((x_offset,0),s.r,edgecolor='white',facecolor='none',linewidth=3) for s in W1PHC.layers[0].shapes]
    
    for c,ca in zip(circles,cirlcesArround):
        plt.gca().add_patch(c)
        ca.center = (c.center[0]-np.sign(c.center[0]-x_offset),c.center[1])
        plt.gca().add_patch(ca)

# Add a box around one specific circle (centered at x=0.5)
from matplotlib.patches import Rectangle
scale = 0
box_x = -.5+scale/2  # Slightly left of x=0.5 to center the box
box_y = np.sqrt(3)/2-.2-.3+scale/2 # Center the box vertically
box_width = 1-scale
box_height = 1-scale
box = Rectangle((box_x, box_y), box_width, box_height, 
                fill=False, color='red', linewidth=2, linestyle='--')
plt.gca().add_patch(box)

# Add zoom box showing expanded view - moved to upper right corner outside main plot
zoom_scale = 5
zoom_x = 2.25  # Position zoom box further right
zoom_y = -ylim/2+.1  # Position zoom box higher up
zoom_width = zoom_scale
zoom_height = zoom_scale
zoom_box = Rectangle((zoom_x, zoom_y), zoom_width, zoom_height,
                    fill=True, facecolor='white', edgecolor='none', zorder=1)
plt.gca().add_patch(zoom_box)

# Plot zoomed field in zoom box
zoom_field = eabs[int((box_y+ylim/2)*300/ylim-10):int((box_y+box_height+ylim/2)*300/ylim+10),:]  # Extract portion around box with padding
zoom_field = np.concatenate((zoom_field[:,int((box_x)*100):],zoom_field[:,:int((box_x+box_width)*100)]),axis=1)
zoom_extent = [zoom_x, zoom_x+zoom_width, zoom_y, zoom_y+zoom_height]
ax.imshow(zoom_field, extent=zoom_extent, cmap='plasma', vmax=maxF, vmin=0,origin='lower', zorder=2)

# Add perturbed circle in zoom box
N = 101  # Number of points around circle (including endpoint that matches start)
theta = np.linspace(0, 2*np.pi, N)
dtheta = np.minimum(np.abs(theta[:, None] - theta[None, :]), 
                   2*np.pi - np.abs(theta[:, None] - theta[None, :]))  # Periodic distance

# Parameters for perturbation
r = 0.3*zoom_scale
sigma = 8/266  # Perturbation amplitude
lp = 40/266     # Persistence length
r_decay = 1.0 # Decay rate

# Compute correlation matrix with periodic boundary conditions
C = sigma**2 * np.exp(-r_decay * dtheta / lp)

# Generate correlated perturbations
L = np.linalg.cholesky(C + 1e-6 * np.eye(N))
xi = np.random.randn(N)
delta_r = L @ xi
delta_r[-1] = delta_r[0]  # Make sure start and end match

# Calculate perturbed radius
rad = r * (1 + delta_r)

# Define regions with and without noise by splitting circle into quarters
theta_q1 = np.linspace(0, np.pi/2, N//4)  # First quadrant (noisy)
theta_q2 = np.linspace(np.pi/2, np.pi, N//4)  # Second quadrant (smooth) 
theta_q3 = np.linspace(np.pi, 3*np.pi/2, N//4)  # Third quadrant (noisy)
theta_q4 = np.linspace(3*np.pi/2, 2*np.pi, N//4)  # Fourth quadrant (smooth)

# Generate coordinates for noisy regions (Q1 and Q3)
px_noisy1 = zoom_x + zoom_width/2 + rad[:N//4]*np.cos(theta_q1)
py_noisy1 = zoom_y + zoom_height/2 + rad[:N//4]*np.sin(theta_q1)
px_noisy3 = zoom_x + zoom_width/2 + rad[N//2:3*N//4]*np.cos(theta_q3)
py_noisy3 = zoom_y + zoom_height/2 + rad[N//2:3*N//4]*np.sin(theta_q3)

# Generate coordinates for smooth regions (Q2 and Q4)
px_smooth2 = zoom_x + zoom_width/2 + r*np.cos(theta_q2)
py_smooth2 = zoom_y + zoom_height/2 + r*np.sin(theta_q2)
px_smooth4 = zoom_x + zoom_width/2 + r*np.cos(theta_q4)
py_smooth4 = zoom_y + zoom_height/2 + r*np.sin(theta_q4)

# Plot all regions
plt.plot(px_noisy1, py_noisy1, 'white', linewidth=3, zorder=3)
plt.plot(px_smooth2, py_smooth2, 'white', linewidth=3, zorder=3)
plt.plot(px_noisy3, py_noisy3, 'white', linewidth=3, zorder=3)
plt.plot(px_smooth4, py_smooth4, 'white', linewidth=3, zorder=3)

# Add connecting lines between box and zoom - adjusted for new zoom position
plt.plot([box_x+box_width, zoom_x], [box_y, zoom_y], 'r--', linewidth=2)
plt.plot([box_x+box_width, zoom_x], [box_y+box_height, zoom_y+zoom_height], 'r--', linewidth=2)

#add a red dashed line vertically and horizontally through the zoom box
plt.plot([zoom_x+zoom_width/2,zoom_x+zoom_width/2],[zoom_y,zoom_y+zoom_height],'r--',linewidth=2,zorder=4)
plt.plot([zoom_x,zoom_x+zoom_width],[zoom_y+zoom_height/2,zoom_y+zoom_height/2],'r--',linewidth=2,zorder=4)

#show cyan arrows points off the edge of the circle in upper right and lower left quadrants
# Upper right arrows
plt.arrow(zoom_x+zoom_width/2+1.2*r*np.cos(np.pi/8), zoom_y+zoom_height/2+1.2*r*np.sin(np.pi/8), 
         0.1*np.cos(np.pi/8), 0.15*np.sin(np.pi/8), color='cyan', head_width=0.15, head_length=0.15, zorder=4)
plt.arrow(zoom_x+zoom_width/2+1.2*r*np.cos(np.pi/4), zoom_y+zoom_height/2+1.2*r*np.sin(np.pi/4),
         0.1*np.cos(np.pi/4), 0.15*np.sin(np.pi/4), color='cyan', head_width=0.15, head_length=0.15, zorder=4)
plt.arrow(zoom_x+zoom_width/2+1.2*r*np.cos(3*np.pi/8), zoom_y+zoom_height/2+1.2*r*np.sin(3*np.pi/8),
         0.1*np.cos(3*np.pi/8), 0.15*np.sin(3*np.pi/8), color='cyan', head_width=0.15, head_length=0.15, zorder=4)
# Lower left arrows
plt.arrow(zoom_x+zoom_width/2-1.2*r*np.cos(np.pi/8), zoom_y+zoom_height/2-1.2*r*np.sin(np.pi/8),
         -0.3*np.cos(np.pi/8), -0.3*np.sin(np.pi/8), color='cyan', head_width=0.15, head_length=0.15, zorder=4)
plt.arrow(zoom_x+zoom_width/2-1.2*r*np.cos(np.pi/4), zoom_y+zoom_height/2-1.2*r*np.sin(np.pi/4),
         -0.4*np.cos(np.pi/4), -0.4*np.sin(np.pi/4), color='cyan', head_width=0.15, head_length=0.15, zorder=4)
plt.arrow(zoom_x+zoom_width/2-1.2*r*np.cos(3*np.pi/8), zoom_y+zoom_height/2-1.2*r*np.sin(3*np.pi/8),
         -0.5*np.cos(3*np.pi/8), -0.5*np.sin(3*np.pi/8), color='cyan', head_width=0.15, head_length=0.15, zorder=4)

# Expand plot limits to show zoom box
plt.xlim(-2,7.5)
plt.ylim(-ylim/2,ylim/2)
ax.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
#remove the box around the plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.show()

# %%
lattice = legume.Lattice(npa.array([1,0]),npa.array([0,20*npa.sqrt(3)]))
phc = legume.PhotCryst(lattice)
phc.add_layer(d=170/266,eps_b=3.4638**2)

gme = legume.GuidedModeExp(phc,2.01)
gme.run(**gmeParams)


# %%
# Create a visualization of the electromagnetic field from GME
z = phc.layers[0].d/2  # Get the middle of the slab
x_grid = np.linspace(0, 1, 100)  # Create x-grid spanning one unit cell
y_grid = np.linspace(-4*np.sqrt(3)/2, 4*np.sqrt(3)/2, 200)  # Create y-grid for sufficient vertical range

# Get the electromagnetic field components at a specific k-point and mode
k_idx = 0  # First k-point
mode_idx = 15  # Adjust this value to select the mode of interest (mode 10 is often in the bandgap)

# Extract field data
fields, x_mesh, y_mesh = gme.get_field_xy('E', k_idx, mode_idx, z, xgrid=x_grid, ygrid=y_grid, component='xyz')

# Calculate the total field intensity
field_intensity = np.abs(fields['x'])**2 + np.abs(fields['y'])**2 + np.abs(fields['z'])**2

# Create a figure with appropriate size
plt.figure(figsize=(10, 8))

fi = np.hstack((field_intensity,field_intensity))

# Plot the field intensity
plt.imshow(field_intensity, extent=[0, 2, -4*np.sqrt(3)/2, 4*np.sqrt(3)/2], cmap='plasma')

# Remove all labels, titles, and colorbars
plt.axis('off')

# Use tight layout for better appearance
plt.tight_layout()
plt.show()


# %%
with open('/Users/dominic/Desktop/optGME/tests/media/ginds3/ziwBest.json','r') as file:
    outZIW = json.load(file)

with open('/Users/dominic/Desktop/optGME/tests/media/ginds3/W1Best.json','r') as file:
    outW1 = json.load(file)

W1opt = []
W1CV = []
ZIWopt = []
ZIWCV = []
for p in outW1[:-1]:
    W1opt.append(p['objective_value'])
    W1CV.append(p['constraint_violation'])
for p in outZIW[:-1]:
    ZIWopt.append(p['objective_value'])
    ZIWCV.append(p['constraint_violation'])

W1opt = 10**(np.array(W1opt))/266/1E-7
W1optPlot = np.array(W1opt)/W1opt[0]
ZIWopt = 10**(np.array(ZIWopt))/266/1E-7
ZIWoptPlot = np.array(ZIWopt)/ZIWopt[0]
# %%
# Create separate figures for W1 and ZIW optimization plots
# Control font size, figure dimensions and DPI at the top level
FONT_SIZE_TITLE = 18
FONT_SIZE_AXIS_LABEL = 18
FONT_SIZE_TICK_LABELS = 15
FONT_SIZE_ANNOTATION = 16
FIGURE_WIDTH = 7
FIGURE_HEIGHT = 4
FIGURE_DPI = 300
FIGURE_ASPECT_RATIO = 0.65  # height/width ratio
LINEWIDTH = 2.5
MARKER_SIZE = 6

# First figure: W1 Optimization
fig1, ax1 = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=FIGURE_DPI)

# Set log scales for W1 plot
ax1.set_yscale('log')
ax1.set_xscale('log')

# Highlight regions where constraint violation is non-zero for W1
violation_regions_w1 = []
start = None

# Find continuous regions of constraint violations for W1
for i in range(len(W1optPlot)):
    if W1CV[i] > 0 and start is None:
        start = i
    elif W1CV[i] <= 0 and start is not None:
        violation_regions_w1.append((start, i))
        start = None

# Add the last region if it extends to the end
if start is not None:
    violation_regions_w1.append((start, len(W1optPlot)))

# Add dashed horizontal lines for initial and final values for W1
initial_value_w1 = W1optPlot[0]
final_value_w1 = W1optPlot[-1]
ax1.axhline(y=initial_value_w1, color='k', linestyle='--', alpha=0.7, zorder=0)
ax1.axhline(y=final_value_w1, color='k', linestyle='--', alpha=0.7, zorder=0)

# Plot W1 optimization data
x_values = np.arange(1, len(W1optPlot) + 1)  # Start from 1 for log scale

# Plot base line first
ax1.plot(x_values, W1optPlot, 'g-', linewidth=LINEWIDTH, zorder=2)

# Add markers with different colors based on constraint violation
for i, val in enumerate(W1optPlot):
    idx = i + 1  # Adjust for 1-based indexing in plot
    in_violation = any(start <= i < end for start, end in violation_regions_w1)
    if in_violation:
        ax1.plot(idx, val, 'ro', markersize=MARKER_SIZE, zorder=3, markeredgecolor='darkred', fillstyle='none')
    else:
        ax1.plot(idx, val, 'o', markersize=MARKER_SIZE, markeredgecolor='darkgreen', zorder=3, fillstyle='none')

# Add violation regions
for start, end in violation_regions_w1:
    ax1.axvspan(start + 1, end + 1, alpha=0.2, color='#FF000080', edgecolor=None)

ax1.set_title('W1 Optimization', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.set_xlabel('Iteration', fontsize=FONT_SIZE_AXIS_LABEL)
ax1.set_ylabel(r'$\left[\tilde{L}^{\text{W1}}/\tilde{L}^{\text{W1}}_0\right]_{\tilde{k}=.33}$', fontsize=FONT_SIZE_AXIS_LABEL)
ax1.set_xlim(1, len(W1optPlot))  # Start from 1 for log scale
ax1.grid(True, linestyle='--', alpha=0.3, which='both')
ax1.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK_LABELS)
ax1.set_ylim(.05, 5)

# Make the plot wider
ax1.set_box_aspect(FIGURE_ASPECT_RATIO)

plt.tight_layout()
plt.show()
#%%
# Second figure: ZIW Optimization (in a separate plot) - First 6 points
fig2, ax2 = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=FIGURE_DPI)

# Set log scales for ZIW plot
ax2.set_yscale('log')
ax2.set_xscale('log')

# Highlight regions where constraint violation is non-zero for ZIW
violation_regions_W1 = []
start = None

# Find continuous regions of constraint violations for ZIW (but only consider the first 6 points)
for i in range(min(6, len(W1optPlot))):
    if W1CV[i] > 0 and start is None:
        start = i
    elif W1CV[i] <= 0 and start is not None:
        violation_regions_W1.append((start, i))
        start = None

# Add the last region if it extends to the end
if start is not None:
    violation_regions_W1.append((start, min(6, len(W1optPlot))))

# Add dashed horizontal line for initial value for ZIW (only initial, not final)
initial_value_W1 = W1optPlot[0]
ax2.axhline(y=initial_value_W1, color='k', linestyle='--', alpha=0.7, zorder=0)

# Plot ZIW optimization data for the first 6 points
x_values = np.arange(1, min(6, len(W1optPlot)) + 1)  # Start from 1 for log scale

# Plot base line first
ax2.plot(x_values, W1optPlot[:6], 'g-', linewidth=LINEWIDTH, zorder=2)

# Add markers with different colors based on constraint violation
for i, val in enumerate(W1optPlot[:6]):
    idx = i + 1  # Adjust for 1-based indexing in plot
    in_violation = any(start <= i < end for start, end in violation_regions_W1)
    if in_violation or i == 5:
        ax2.plot(idx, val, 'ro', markersize=MARKER_SIZE, zorder=3, markeredgecolor='darkred', fillstyle='none')
    else:
        ax2.plot(idx, val, 'o', markersize=MARKER_SIZE, markeredgecolor='darkgreen', zorder=3, fillstyle='none')

# Add violation regions
for start, end in violation_regions_W1:
    ax2.axvspan(start + 1, end, alpha=0.2, color='#FF000080', edgecolor=None)

ax2.set_title('W1 Optimization', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.set_xlabel('Iteration', fontsize=FONT_SIZE_AXIS_LABEL)
ax2.set_ylabel(r'$\left[\tilde{L}^{\text{W1}}/\tilde{L}^{\text{W1}}_0\right]_{\tilde{k}=.33}$', fontsize=FONT_SIZE_AXIS_LABEL)
ax2.set_xlim(1, len(W1optPlot))  # Keep full range for x-axis
ax2.grid(True, linestyle='--', alpha=0.3, which='both')
ax2.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK_LABELS)
ax2.set_ylim(.05, 5)

# Make the plot wider
ax2.set_box_aspect(FIGURE_ASPECT_RATIO)

plt.tight_layout()
plt.show()
#%%
# Second figure: ZIW Optimization (full plot)
fig3, ax3 = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=FIGURE_DPI)

# Set log scales for ZIW plot
ax3.set_yscale('log')
ax3.set_xscale('log')

# Highlight regions where constraint violation is non-zero for ZIW (full dataset)
violation_regions_W1_full = []
start = None

# Find continuous regions of constraint violations for ZIW
for i in range(len(ZIWoptPlot)):
    if ZIWCV[i] > 0 and start is None:
        start = i
    elif W1CV[i] <= 0 and start is not None:
        violation_regions_W1_full.append((start, i))
        start = None

# Add the last region if it extends to the end
if start is not None:
    violation_regions_ziw_full.append((start, len(W1optPlot)))

# Add dashed horizontal lines for initial and final values for ZIW
initial_value_W1 = W1optPlot[0]
final_value_W1 = W1optPlot[-1]
ax3.axhline(y=initial_value_W1, color='k', linestyle='--', alpha=0.7, zorder=0)
#ax3.axhline(y=final_value_ziw, color='k', linestyle='--', alpha=0.7, zorder=0)

# Plot ZIW optimization data (full dataset)
x_values_full = np.arange(1, len(W1optPlot) + 1)  # Start from 1 for log scale

# Plot base line first
ax3.plot(x_values_full, W1optPlot, 'g-', linewidth=LINEWIDTH, zorder=2)

# Add markers with different colors based on constraint violation
for i, val in enumerate(W1optPlot):
    idx = i + 1  # Adjust for 1-based indexing in plot
    in_violation = any(start <= i < end for start, end in violation_regions_ziw_full)
    if in_violation:
        ax3.plot(idx, val, 'ro', markersize=MARKER_SIZE, zorder=3, markeredgecolor='darkred', fillstyle='none')
    else:
        ax3.plot(idx, val, 'o', markersize=MARKER_SIZE, markeredgecolor='darkgreen', zorder=3, fillstyle='none')

# Add violation regions
for start, end in violation_regions_ziw_full:
    ax3.axvspan(start + 1, end + 1, alpha=0.2, color='#FF000080', edgecolor=None)

ax3.set_title('ZIW Optimization', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3.set_xlabel('Iteration', fontsize=FONT_SIZE_AXIS_LABEL)
ax3.set_ylabel(r'$\left[\tilde{L}^{\text{ZIW}}/\tilde{L}^{\text{ZIW}}_0\right]_{\tilde{k}=.33}$', fontsize=FONT_SIZE_AXIS_LABEL)
ax3.set_xlim(1, len(ZIWoptPlot))  # Start from 1 for log scale
ax3.grid(True, linestyle='--', alpha=0.3, which='both')
ax3.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK_LABELS)
ax3.set_ylim(.05, 5)

# Make the plot wider
ax3.set_box_aspect(FIGURE_ASPECT_RATIO)

plt.tight_layout()
plt.show()

#%%# Second figure: ZIW Optimization (full plot)
fig4, ax3 = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=FIGURE_DPI)

# Set log scales for ZIW plot
ax3.set_yscale('log')
ax3.set_xscale('log')

# Highlight regions where constraint violation is non-zero for ZIW (full dataset)
violation_regions_W1_full = []
start = None

# Find continuous regions of constraint violations for ZIW
for i in range(len(W1optPlot)):
    if W1CV[i] > 0 and start is None:
        start = i
    elif W1CV[i] <= 0 and start is not None:
        violation_regions_ziw_full.append((start, i))
        start = None

# Add the last region if it extends to the end
if start is not None:
    violation_regions_ziw_full.append((start, len(W1optPlot)))

# Add dashed horizontal lines for initial and final values for ZIW
initial_value_W1 = W1optPlot[0]
final_value_W1 = W1optPlot[-1]
ax3.axhline(y=initial_value_W1, color='k', linestyle='--', alpha=0.7, zorder=0)
#ax3.axhline(y=final_value_ziw, color='k', linestyle='--', alpha=0.7, zorder=0)

# Plot ZIW optimization data (full dataset)
x_values_full = np.arange(1, len(W1optPlot) + 1)  # Start from 1 for log scale

# Plot base line first
#ax3.plot(x_values_full, ZIWoptPlot, 'g-', linewidth=LINEWIDTH, zorder=2)

# Plot only the first point with appropriate color based on constraint violation
i = 0
val = W1optPlot[i]
idx = i + 1  # Adjust for 1-based indexing in plot
in_violation = any(start <= i < end for start, end in violation_regions_W1_full)
if in_violation:
    ax3.plot(idx, val, 'ro', markersize=MARKER_SIZE, zorder=3, markeredgecolor='darkred', fillstyle='none')
else:
    ax3.plot(idx, val, 'o', markersize=MARKER_SIZE, markeredgecolor='darkgreen', zorder=3, fillstyle='none')

# Add violation regions
#for start, end in violation_regions_ziw_full:
    #ax3.axvspan(start + 1, end + 1, alpha=0.2, color='#FF000080', edgecolor=None)

ax3.set_title('W1 Optimization', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3.set_xlabel('Iteration', fontsize=FONT_SIZE_AXIS_LABEL)
ax3.set_ylabel(r'$\left[\tilde{L}^{\text{W1}}/\tilde{L}^{\text{W1}}_0\right]_{\tilde{k}=.33}$', fontsize=FONT_SIZE_AXIS_LABEL)
ax3.set_xlim(1, len(W1optPlot))  # Start from 1 for log scale
ax3.grid(True, linestyle='--', alpha=0.3, which='both')
ax3.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK_LABELS)
ax3.set_ylim(.05, 5)

# Make the plot wider
ax3.set_box_aspect(FIGURE_ASPECT_RATIO)

plt.tight_layout()
plt.show()
# %%
from optomization.crystals import ZIW
W1PHC = W1(vars=np.array(outW1[-2]['x_values']))
#ZIWPHC = ZIW(NyChange=0)
gmeW1 = legume.GuidedModeExp(W1PHC,4.01)
w1gmeParams = outW1[-1]['gmeParams']
w1gmeParams['numeig'] = w1gmeParams['numeig']+10
w1gmeParams['kpoints'] = np.array(w1gmeParams['kpoints'])
gmeW1.run(**w1gmeParams)

# %%
def plotFieldDistribution(phc, gme, params):
    # Font size variables for consistent styling
    TITLE_SIZE = 28
    LABEL_SIZE = 26
    TICK_SIZE = 20
    LEGEND_SIZE = 24
    
    # Set up variables
    ylim = 10*np.sqrt(3)/2
    ys = np.linspace(-ylim/2, ylim/2, 300)
    mode = 20
    kindex = np.abs(gme.kpoints[0,:] - params['gmeParams']['kpoints'][0][0]).argmin()
    z = phc.layers[0].d/2

    # Get field of crystal
    fields, _, _ = gme.get_field_xy('E', kindex, mode, z, ygrid=ys, component='xyz')
    eabs = np.abs(np.conj(fields['x'])*fields['x'] + np.conj(fields['y'])*fields['y'] + np.conj(fields['z'])*fields['z'])
    maxF = np.max(eabs)

    # Create a figure with proper aspect ratio
    fig = plt.figure(figsize=(6, 10))
    ax = plt.gca()
    
    # Plot just one set of the field data
    cax = ax.imshow(eabs.T, extent=[-ylim/2, ylim/2, -0.5, 0.5], 
                   cmap='plasma', vmax=maxF, vmin=0, aspect='auto', origin='lower')
    
    # Add filled circles representing holes (filled with white)
    circles = [Circle((-s.y_cent, -1*s.x_cent), s.r, edgecolor='white', 
                     facecolor='white', linewidth=1.5) for s in phc.layers[0].shapes]
    
    # Add circles for the main row
    for c in circles:
        ax.add_patch(c)
    
    # Add a second row of circles below
    for s in phc.layers[0].shapes:
        circle_below = Circle((-s.y_cent, -1*(s.x_cent-1)), s.r, 
                             edgecolor='black', facecolor='white', linewidth=1.5)
        ax.add_patch(circle_below)
    
    # Add a large red X at the center point
    center_x, center_y = 0, 0
    marker_size = 500
    line_width = 6
    #ax.scatter(center_x, center_y, s=marker_size, marker='x', color='red', linewidth=line_width)
    
    # Find where the field exists and set limits accordingly
    field_threshold = 0.05 * maxF  # 5% of maximum field value
    field_exists = eabs.T > field_threshold
    y_indices, x_indices = np.where(field_exists)
    
    # Get coordinates from indices
    x_coords = np.linspace(-ylim/2, ylim/2, eabs.T.shape[1])[x_indices]
    y_coords = np.linspace(-0.5, 0.5, eabs.T.shape[0])[y_indices]
    
    # Set limits to the field existence area with some padding
    x_padding = (max(x_coords) - min(x_coords)) * 0.1
    y_padding = (max(y_coords) - min(y_coords)) * 0.1
    
    ax.set_xlim(-ylim/2, ylim/2)
    ax.set_ylim(-0.5, 0.5)  # Extend lower bound to show the second row
    
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                  labelbottom=False, labelleft=False)
    
    plt.tight_layout()
    plt.show()

plotFieldDistribution(W1PHC, gmeW1, outW1[-1])


# %%
from optomization import W1,ZIW

phcW1opt = W1(vars=np.array(outW1[-1]['result']['x']))
phcZIWopt = ZIW(vars=np.array(outZIW[-1]['result']['x']))
phcW1OG = W1(NyChange=0)
phcZIWOG = ZIW(NyChange=0)

shapesW1 = phcW1OG.layers[0].shapes
shapesZIW = phcZIWOG.layers[0].shapes
shapesW1Opt = phcW1opt.layers[0].shapes
shapesZIWOpt = phcZIWopt.layers[0].shapes
# %%
# Create a figure for plotting the W1 structure with flipped axes and grey background
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')  # Set figure background to white

# Create a grey background using imshow with uniform values
# Value 0.75 gives a medium grey (can be adjusted between 0-1)
grey_level = 0.75  # Adjust this value between 0 (black) and 1 (white) for different grey shades
ax.imshow(np.ones((100, 100)) * grey_level, cmap='gray', 
          extent=[-5, 5, -0.5, 2.5], alpha=1.0, aspect='auto', zorder=0,vmin=0,vmax=1)

# Define the zigzag boundaries for the darker region
x_left = -2.75
x_right = 2.75
# Create a darker grey polygon that follows the zigzag pattern
vertices = []
# Add points for left zigzag boundary (bottom to top)
for i in range(-1, 5):
    vertices.append((x_left, i))
    vertices.append((x_left-0.5, i+0.5))
# Add top-right corner
vertices.append((x_right+0.5, 4.5))
# Add points for right zigzag boundary (top to bottom)
for i in range(4, -2, -1):
    vertices.append((x_right+0.5, i+0.5))
    vertices.append((x_right, i))
# Close the polygon
vertices.append(vertices[0])

# Create polygon for darker region
darker_poly = plt.Polygon(vertices, closed=True, facecolor='grey', alpha=0.5, zorder=0.5)
ax.add_patch(darker_poly)

# Plot the circles from the original W1 structure as white holes
for c in shapesW1:
    # Create a circle with flipped x,y coordinates (y as horizontal axis, x as vertical axis)
    circle = plt.Circle((c.y_cent, c.x_cent), c.r, facecolor='white', edgecolor='black', alpha=1.0, zorder=1)
    ax.add_patch(circle)
    
    # Repeat circles in negative x direction (now vertical direction after flip)
    circle = plt.Circle((c.y_cent, c.x_cent - 1), c.r, facecolor='white', edgecolor='black', alpha=1.0, zorder=1)
    ax.add_patch(circle)

# Add dashed white lines for Voronoi cell walls between 3rd and 4th circles
# Left side (negative x)
# Draw zigzag lines along the hexagon edges on the left side
x_left = -2.75
for i in range(-1, 5):
    # Draw segments of the zigzag pattern - each segment is 1/2 of hexagon height
    ax.plot([x_left, x_left-0.5], [i, i+0.5], color='white', linestyle='--', linewidth=2, alpha=0.8, zorder=1.5)
    ax.plot([x_left-0.5, x_left], [i+0.5, i+1], color='white', linestyle='--', linewidth=2, alpha=0.8, zorder=1.5)

# Draw zigzag lines along the hexagon edges on the right side
x_right = 2.75
for i in range(-1, 5):
    # Draw segments of the zigzag pattern - each segment is 1/2 of hexagon height
    ax.plot([x_right, x_right+0.5], [i, i+0.5], color='white', linestyle='--', linewidth=2, alpha=0.8, zorder=1.5)
    ax.plot([x_right+0.5, x_right], [i+0.5, i+1], color='white', linestyle='--', linewidth=2, alpha=0.8, zorder=1.5)

# Set axis limits to show 5 holes on either side of center (expanded y range)
ax.set_xlim(-5*np.sqrt(3)/2, 5*np.sqrt(3)/2)  # More of the y-axis (now horizontal)
ax.set_ylim(-0.5, 0.5)  # From x=-0.5 to x=3.5 after adding 3 more unit cells
ax.set_aspect('equal')
ax.grid(False)

# Turn on the axis but turn off the ticks and labels
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
               labelbottom=False, labeltop=False, labelleft=False, labelright=False)

# Remove all padding
plt.tight_layout(pad=0)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.show()

# %%
# Create a figure for plotting the ZIW structure with flipped axes and grey background
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')  # Set figure background to white

# Add horizontal offset parameter to shift all circles left or right
x_offset = -np.sqrt(3)/2/6  # Change this value to shift circles horizontally (negative = left, positive = right)

# Define region boundaries for different shadings
x_left = -3.25*np.sqrt(3)/2 - x_offset
x_right = 3.25*np.sqrt(3)/2 + x_offset

# Create a grey background using imshow with uniform values
# Value 0.75 gives a medium grey (can be adjusted between 0-1)
grey_level = 0.75  # Adjust this value between 0 (black) and 1 (white) for different grey shades
ax.imshow(np.ones((100, 100)) * grey_level, cmap='gray', 
          extent=[-5 + x_offset, 5 + x_offset, -0.5, 2.5], alpha=1.0, aspect='auto', zorder=0,vmin=0,vmax=1)

# Add a darker grey region between the dashed white lines
darker_grey_level = 0.65  # Darker than the background grey (0.75)
ax.imshow(np.ones((100, 100)) * darker_grey_level, cmap='gray',
          extent=[x_left, x_right, -0.5, 2.5], alpha=1.0, aspect='auto', zorder=0.5, vmin=0, vmax=1)

# Plot the circles from the original ZIW structure as white holes
for c in shapesZIW:
    # Create a circle with flipped x,y coordinates (y as horizontal axis, x as vertical axis)
    # Apply the x_offset to the y coordinate (since axes are flipped)
    circle = plt.Circle((c.y_cent + x_offset, c.x_cent), c.r, facecolor='white', edgecolor='black', alpha=1.0, zorder=1)
    ax.add_patch(circle)
    
    # Repeat circles in negative x direction (now vertical direction after flip)
    circle = plt.Circle((c.y_cent + x_offset, c.x_cent - 1), c.r, facecolor='white', edgecolor='black', alpha=1.0, zorder=1)
    ax.add_patch(circle)

# Add dashed white vertical lines for Voronoi cell walls between 3rd and 4th circles
# Left side (negative x)
ax.plot([x_left, x_left], [-1, 5], color='white', linestyle='--', linewidth=2, alpha=0.8, zorder=1.5)

# Right side (positive x)
ax.plot([x_right, x_right], [-1, 5], color='white', linestyle='--', linewidth=2, alpha=0.8, zorder=1.5)

# Set axis limits to show 5 holes on either side of center (expanded y range)
ax.set_xlim(-5*np.sqrt(3)/2, 5*np.sqrt(3)/2)  # More of the y-axis (now horizontal)
ax.set_ylim(-0.5, 0.5)  # From x=-0.5 to x=3.5 after adding 3 more unit cells
ax.set_aspect('equal')
ax.grid(False)

# Turn on the axis but turn off the ticks and labels
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
               labelbottom=False, labeltop=False, labelleft=False, labelright=False)

# Remove all padding
plt.tight_layout(pad=0)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
# %%
def filedPlots(phc, gme, params):
    # Font size variables for consistent styling
    TITLE_SIZE = 28
    LABEL_SIZE = 26
    TICK_SIZE = 20
    LEGEND_SIZE = 24
    COLORBAR_LABEL_SIZE = 24
    COLORBAR_TICK_SIZE = 20
    
    # Set up variables
    ylim = 4*np.sqrt(3)/2
    ys = np.linspace(-ylim/2, ylim/2, 300)  # Changed from ys to xs for rotation
    mode = params['mode']
    kindex = np.abs(gme.kpoints[0,:] - params['gmeParams']['kpoints'][0][0]).argmin()
    z = phc.layers[0].d/2

    # Get field of crystal - swapped grid for rotation
    fields, _, _ = gme.get_field_xy('E', kindex, mode, z, ygrid=ys, component='xyz')
    eabs = np.abs(np.conj(fields['x'])*fields['x'] + np.conj(fields['y'])*fields['y'] + np.conj(fields['z'])*fields['z'])
    maxF = np.max(eabs)

    eps = legume.viz.eps_xy(phc, Nx=100, Ny=100, plot=False)
    f, _, _ = gme.get_field_xy('E', kindex, mode, z, Ny=200, Nx=100, component='xyz')
    e = np.abs(np.conj(f['x'])*f['x'] + np.conj(f['y'])*f['y'] + np.conj(f['z'])*f['z'])
    
    # Create a single figure with proper aspect ratio - switched dimensions
    fig = plt.figure(figsize=(15, 6))  # Switched dimensions back
    ax = plt.gca()
    
    # Repeat the data twice in the x direction (horizontally) and transpose for axis swap
    eabs_repeated = np.hstack((eabs, eabs, eabs,eabs,eabs))
    
    # Plot the repeated field data with swapped axes
    cax = ax.imshow(eabs_repeated, extent=[-2.5, 2.5, -ylim/2, ylim/2], 
                   cmap='plasma', vmax=maxF, vmin=0, aspect='auto', origin='lower')
    
    # Add circles representing holes - swapped coordinates back
    circles = [Circle((s.x_cent, -s.y_cent), s.r, edgecolor='white', 
                     facecolor='none', linewidth=3) for s in phc.layers[0].shapes]
    
    for c in circles:
        ax.add_patch(c)
        # Add circles 1 unit left and right
        c_left = Circle((c.center[0] - 1, c.center[1]), c.radius, edgecolor='white', 
                       facecolor='none', linewidth=5)
        c_leftleft = Circle((c.center[0] - 2, c.center[1]), c.radius, edgecolor='white', 
                           facecolor='none', linewidth=5)
        c_right = Circle((c.center[0] + 1, c.center[1]), c.radius, edgecolor='white', 
                        facecolor='none', linewidth=5)
        c_rightright = Circle((c.center[0] + 2, c.center[1]), c.radius, edgecolor='white', 
                             facecolor='none', linewidth=5)
        c_leftleftleft = Circle((c.center[0] -3, c.center[1]), c.radius, edgecolor='white', 
                             facecolor='none', linewidth=5)
        c_rightrightright = Circle((c.center[0] + 3, c.center[1]), c.radius, edgecolor='white', 
                             facecolor='none', linewidth=5)
        ax.add_patch(c_left)
        ax.add_patch(c_right)
        ax.add_patch(c_leftleft)
        ax.add_patch(c_rightright)  
        ax.add_patch(c_leftleftleft)
        ax.add_patch(c_rightrightright)
    
    # Swapped xlim and ylim
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-ylim/2, ylim/2)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                  labelbottom=False, labelleft=False)
    
    # Removed colorbar code
    plt.tight_layout()
    plt.show()

filedPlots(W1PHC, gme, w1_data[-1])
# %%
