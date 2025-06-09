#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import matplotlib as mpl

# Disable LaTeX but use Computer Modern fonts
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern math fonts
mpl.rcParams['font.family'] = 'STIXGeneral'  # Use STIX fonts (similar to Computer Modern)

# %%
#--------------------------------Hole Noise--------------------------------


# Create a new figure for the grid of holes
fig_closeup, ax_closeup = plt.subplots(figsize=(10,10))

# Grid parameters
n_grid = 2  # 5x5 grid
std_range = np.linspace(4/266, 0.07, n_grid)  # Range of standard deviations
lp_range = np.linspace(40/266, 1, n_grid)  # Range of correlation lengths

# Base parameters
N = 500  # Number of points around each circle
theta = np.linspace(0, 2*np.pi, N, endpoint=True)
r_decay = 1    # Decay rate parameter
r=.3

# Compute periodic distances once
theta_i, theta_j = np.meshgrid(theta, theta, indexing='ij')
dtheta = np.minimum(np.abs(theta_i - theta_j), 2*np.pi - np.abs(theta_i - theta_j))

# Create grid of holes
for i, sigma in enumerate(std_range):
    for j, lp in enumerate(lp_range):
        # Calculate center position
        x_center = (i - n_grid/2 + 0.5) * 3*r
        y_center = (j - n_grid/2 + 0.5) * 3*r
        
        # Compute correlation matrix
        C = sigma**2 * np.exp(-r_decay * dtheta / lp)
        
        # Generate correlated perturbations
        L = np.linalg.cholesky(C + 1e-6 * np.eye(N))
        xi = np.random.randn(N)
        delta_r = L @ xi
        
        # Calculate perturbed radius
        rad = r * (1 + delta_r)
        
        # Generate shape coordinates
        px = x_center + rad*np.cos(theta)
        py = y_center + rad*np.sin(theta)
        
        # Plot the perturbed circle
        ax_closeup.plot(px, py, 'k-', linewidth=3)
        
        # Plot the unperturbed circle as a dashed orange circle
        unperturbed_circle = plt.Circle((x_center, y_center), r, color='red', fill=False, linestyle='--', linewidth=3)
        ax_closeup.add_patch(unperturbed_circle)

# Set equal aspect ratio and limits
ax_closeup.set_aspect('equal')
margin = n_grid * 2*r
ax_closeup.set_xlim(-margin, margin*.85)
ax_closeup.set_ylim(-margin, margin*.85)

# Add labels
ax_closeup.text(0, -margin*.8, r'Increasing σ', ha='center', va='top', fontsize=35)
ax_closeup.text(-margin*.8, 0, r'Increasing $l_p$', ha='right', va='center', rotation=90, fontsize=35)
ax_closeup.annotate('', xy=(margin*.7, -margin*.75), xytext=(-margin*.75, -margin*.75),
                   arrowprops=dict(arrowstyle='->',lw=5))
ax_closeup.annotate('', xy=(-margin*.75, margin*.7), xytext=(-margin*.75, -margin*.75),
                   arrowprops=dict(arrowstyle='->',lw=5))

# Remove axes for cleaner look
ax_closeup.set_xticks([])
ax_closeup.set_yticks([])
ax_closeup.spines['top'].set_visible(False)
ax_closeup.spines['right'].set_visible(False)
ax_closeup.spines['bottom'].set_visible(False)
ax_closeup.spines['left'].set_visible(False)

# Set transparent background
fig_closeup.patch.set_alpha(0.0)

plt.show()
#%%
#--------------------------------Constraints Diagram--------------------------------
import legume

def W1(Ny=10,dslab=170/266,eps_slab=3.4638,ra=.3):

    lattice = legume.Lattice(np.array([1,0]),np.array([0,Ny*np.sqrt(3)]))

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
        y = iy*np.sqrt(3)/2

        #now we can add a circle with the given positions
        phc.add_shape(legume.Circle(x_cent=x,y_cent=y,r=ra))

    return(phc)

phc = W1()
_ = legume.viz.eps_xy(phc,Ny=300)

#%%
ks = np.linspace(np.pi*.5,np.pi,25)

gmeParams = {'verbose':False,'numeig':50,'compute_im':False,'kpoints':np.vstack((ks,np.zeros(len(ks))))}
gme2 = legume.GuidedModeExp(phc,4.01)
gme2.run(**gmeParams)


#%%
# Define hyperparameters for font sizes and styling
TITLE_SIZE = 34
LABEL_SIZE = 34
LEGEND_SIZE = 24
ANNOTATION_SIZE = 30
TICK_SIZE = 24
LINE_WIDTH = 2
SCATTER_SIZE = 200
ARROW_HEAD_WIDTH = 0.007
ARROW_HEAD_LENGTH = 0.001

# Colors
orange = '#EE7733'
purple = 'purple'
navy = 'navy'
teal = 'teal'
red = 'red'
dark_gray = 'darkGray'

# Set default font size
plt.rcParams.update({'font.size': TICK_SIZE})

# Shade region below mode 19 (bottom layer)
plt.fill_between(np.linalg.norm(gme2.kpoints,axis=0)/np.pi/2, 0, gme2.freqs[:,19], alpha=.7, color=navy, zorder=1)

# Get axis limits
xmin, xmax = plt.xlim()

# Add horizontal dashed lines at min of mode 21 and max of mode 19
plt.axhline(y=.276, color=teal, linestyle='--', linewidth=LINE_WIDTH, zorder=2)
plt.axhline(y=np.min(gme2.freqs[:,20])-.0005, color=teal, linestyle='--', linewidth=LINE_WIDTH, zorder=2)

# Create coordinates for shaded region above x=y
x = np.linspace(xmin, xmax, 100)
y = x.copy()

plt.fill_between(x, y, 100, color=dark_gray, alpha=1, zorder=0)

# Add mode 20 and 21 in purple
plt.plot(np.linalg.norm(gme2.kpoints,axis=0)/np.pi/2,gme2.freqs[:,20], color=purple, linewidth=LINE_WIDTH, zorder=4)
plt.plot(np.linalg.norm(gme2.kpoints,axis=0)/np.pi/2,gme2.freqs[:,21], color=purple, linestyle='--', linewidth=LINE_WIDTH, zorder=4)

# Add a red dot on mode 20 and the 8th k point that appears above the mode line
plt.scatter(np.linalg.norm(gme2.kpoints,axis=0)[8]/np.pi/2,gme2.freqs[8,20], color=red, s=SCATTER_SIZE, zorder=5)

# Add an orange tangent line to the mode 20 at the 8th k point
# Calculate slope at k-point 8 using central difference
dk = np.linalg.norm(gme2.kpoints,axis=0)[9] - np.linalg.norm(gme2.kpoints,axis=0)[8]
df = gme2.freqs[9,20] - gme2.freqs[8,20]
slope = df/dk*np.pi*2-.02

# Get x coordinates for tangent line
k8 = np.linalg.norm(gme2.kpoints,axis=0)[8]/np.pi/2
x = np.linspace(k8-0.05, k8+0.05, 100)
f8 = gme2.freqs[8,20]

# Add teal arrows that go from right above and below the mode 20 line at the 8th k point
plt.arrow(k8, f8+0.002, 0, .276-.002-(f8+0.003), color=teal, 
          head_width=ARROW_HEAD_WIDTH, head_length=ARROW_HEAD_LENGTH)
plt.arrow(k8, f8-0.002, 0, np.min(gme2.freqs[:,20])+.0005-(f8-0.003), color=teal, 
          head_width=ARROW_HEAD_WIDTH, head_length=ARROW_HEAD_LENGTH)

# Plot tangent line y = m(x-x0) + y0
plt.plot(x, slope*(x*np.pi*2-np.linalg.norm(gme2.kpoints,axis=0)[8])/np.pi/2 + f8, color=orange, linewidth=LINE_WIDTH, zorder=0)

# Add "bandwidth" text between the teal arrows
plt.text(k8+0.03, f8+0.005, 'Bandwidth', color=teal, fontsize=ANNOTATION_SIZE, verticalalignment='center')

# Add "group velocity" text near the orange tangent line
plt.text(k8+0.03, f8-0.001, 'Group Velocity', color=orange, fontsize=ANNOTATION_SIZE)

# Add x and y labels 
plt.xlabel(r'Wavevector $k_x 2\pi/a$', fontsize=LABEL_SIZE)
plt.ylabel(r'Frequency $\omega a/2\pi c$', fontsize=LABEL_SIZE)

plt.ylim(.245,.295)
plt.xlim(.25,.5)

# Add a second y axis with the frequency in THz
ax2 = plt.gca().twinx()
conFac = 1e-12*299792458/266/1e-9
ax2.set_ylim(.245*conFac,.295*conFac)
ax2.set_ylabel("Frequency [THz]", fontsize=LABEL_SIZE)
plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
plt.show()

# %%
#--------------------------------backscattering schematic--------------------------------
def W1(Ny=10,dslab=170/266,eps_slab=3.4638,ra=.3):

    lattice = legume.Lattice(np.array([1,0]),np.array([0,Ny*np.sqrt(3)]))

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
        y = iy*np.sqrt(3)/2

        #now we can add a circle with the given positions
        phc.add_shape(legume.Circle(x_cent=x,y_cent=y,r=ra))

    return(phc)

phc = W1()
ks = np.linspace(np.pi*.5,np.pi,25)
gmeParams = {'verbose':False,'numeig':21,'compute_im':False,'kpoints':np.vstack((ks[8],0))}
gme = legume.GuidedModeExp(phc,4.01)
gme.run(**gmeParams)

# %%
from matplotlib.patches import Circle
ylim= 6*np.sqrt(3)/2
ys = np.linspace(-ylim/2,ylim/2,300)
z=phc.layers[0].d/2
fields,_,_ = gme.get_field_xy('E',0,20,z,ygrid=ys,component='xyz')
eabs = np.abs(np.conj(fields['x'])*fields['x']+np.conj(fields['y'])*fields['y']+np.conj(fields['z'])*fields['z'])
maxF = np.max(eabs)

#%%
fix,ax = plt.subplots()

# Plot the field pattern multiple times along x
for x_offset in [-1.5, -0.5, 0.5, 1.5]:
    cax = ax.imshow(eabs,extent=[x_offset-0.5,x_offset+0.5,-ylim/2,ylim/2],cmap='plasma',vmax=maxF,vmin=0)
    
    # Add circles for each x offset
    circles = [Circle((s.x_cent+x_offset,-s.y_cent),s.r,edgecolor='white',facecolor='none',linewidth=3) for s in phc.layers[0].shapes]
    cirlcesArround = [Circle((x_offset,0),s.r,edgecolor='white',facecolor='none',linewidth=3) for s in phc.layers[0].shapes]
    
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

#%%
#--------------------------------backscattering schematic ZIW--------------------------------
def ZIW(Ny=10,dslab=170/266,eps_slab=3.4638,r0=.105,r1=.235):

    lattice = legume.Lattice(np.array([1,0]),np.array([0,Ny*np.sqrt(3)-.25]))

    #now we define a photonic crystal that goes over our lattice and add the one layer of our W1 waveguide
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=dslab,eps_b=eps_slab**2)

    #now we want to add the air holes to our photonic crystal to make it the W1 waveguide we do this in a loop
    flip = False
    for i in range(Ny*2):
        iy = i-Ny+1
        y = iy*np.sqrt(3)/2-np.sqrt(3)/12

        if iy>0 and not flip:
            r0, r1 = r1, r0
            flip = True

        #add hole below and above
        phc.add_shape(legume.Circle(x_cent=.5*((iy+1)%2),y_cent=y-np.sqrt(3)/6,r=r1))
        phc.add_shape(legume.Circle(x_cent=.5*(iy%2),y_cent=y,r=r0))
    
    return(phc)


phc = ZIW()
ks = np.linspace(np.pi*.5,np.pi,25)
gmeParams = {'verbose':False,'numeig':21,'compute_im':False,'kpoints':np.vstack((ks[8],0))}
gme = legume.GuidedModeExp(phc,4.01)
gme.run(**gmeParams)

# %%
from matplotlib.patches import Circle
ylim= 6*np.sqrt(3)/2
ys = np.linspace(-ylim/2-np.sqrt(3)/6,ylim/2-np.sqrt(3)/6,300)
z=phc.layers[0].d/2
fields,_,_ = gme.get_field_xy('E',0,20,z,ygrid=ys,component='xyz')
eabs = np.abs(np.conj(fields['x'])*fields['x']+np.conj(fields['y'])*fields['y']+np.conj(fields['z'])*fields['z'])
maxF = np.max(eabs)

#%%
fix,ax = plt.subplots()

# Plot the field pattern multiple times along x
for x_offset in [-1.5, -0.5, 0.5, 1.5]:
    cax = ax.imshow(eabs,extent=[x_offset-0.5,x_offset+0.5,-ylim/2,ylim/2],cmap='plasma',vmax=maxF,vmin=0)
    
    # Add circles for each x offset
    circles = [Circle((s.x_cent+x_offset,-s.y_cent),s.r,edgecolor='white',facecolor='none',linewidth=3) for s in phc.layers[0].shapes]
    cirlcesArround = [Circle((x_offset,0),s.r,edgecolor='white',facecolor='none',linewidth=3) for s in phc.layers[0].shapes]
    
    for c,ca in zip(circles,cirlcesArround):
        plt.gca().add_patch(c)
        ca.center = (c.center[0]-np.sign(c.center[0]-x_offset),c.center[1])
        plt.gca().add_patch(ca)

# Add a box around one specific circle (centered at x=0.5)
from matplotlib.patches import Rectangle
scale = 0
box_x = -.5  # Slightly left of x=0.5 to center the box
box_y = 0 # Center the box vertically
box_width = 1-scale
box_height = 1-scale
box = Rectangle((box_x, box_y-np.sqrt(3)/12), box_width, box_height, 
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
zoom_field = eabs[int((box_y+ylim/2+np.sqrt(3)/6)*300/(ylim+np.sqrt(3)/6)):int((box_y+box_height+ylim/2+np.sqrt(3)/6)*300/(ylim+np.sqrt(3)/6)),:]  # Extract portion around box with padding
zoom_field = np.concatenate((zoom_field[:,int((box_x)*100):],zoom_field[:,:int((box_x+box_width)*100)]),axis=1)
zoom_extent = [zoom_x, zoom_x+zoom_width, zoom_y, zoom_y+zoom_height]
ax.imshow(zoom_field, extent=zoom_extent, cmap='plasma', vmax=maxF, vmin=0,origin='lower', zorder=2)

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
#--------------------------------crystal structure schematic--------------------------------
import legume
def W1(Ny=10,dslab=170/266,eps_slab=3.4638,ra=.3):

    lattice = legume.Lattice(np.array([1,0]),np.array([0,Ny*np.sqrt(3)]))

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
        y = iy*np.sqrt(3)/2

        #now we can add a circle with the given positions
        phc.add_shape(legume.Circle(x_cent=x,y_cent=y,r=ra))

    return(phc)


def ZIW(Ny=10,dslab=170/266,eps_slab=3.4638,r0=.105,r1=.235):

    lattice = legume.Lattice(np.array([1,0]),np.array([0,Ny*np.sqrt(3)-.25]))

    #now we define a photonic crystal that goes over our lattice and add the one layer of our W1 waveguide
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=dslab,eps_b=eps_slab**2)

    #now we want to add the air holes to our photonic crystal to make it the W1 waveguide we do this in a loop
    flip = False
    for i in range(Ny*2):
        iy = i-Ny+1
        y = iy*np.sqrt(3)/2-np.sqrt(3)/12

        if iy>0 and not flip:
            r0, r1 = r1, r0
            flip = True

        #add hole below and above
        phc.add_shape(legume.Circle(x_cent=.5*((iy+1)%2),y_cent=y-np.sqrt(3)/6,r=r1))
        phc.add_shape(legume.Circle(x_cent=.5*(iy%2),y_cent=y,r=r0))
    
    return(phc)

#%%
phc = ZIW(Ny=10,dslab=170/266,eps_slab=3.4638,r0=.105,r1=.235)
circleZIW = phc.layers[0].shapes
phc = W1(Ny=10,dslab=170/266,eps_slab=3.4638,ra=.3)
circleW1 = phc.layers[0].shapes

# Create figure with two subplots with extra space for text
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,6))
plt.subplots_adjust(hspace=0.1, top=0.5, bottom=.1)  # Add space above, below and between plots

# Add darker background to both plots
ax1.set_facecolor('#0077BB')  # Lighter slate gray
ax2.set_facecolor('#EE7733')  # Lighter slate blue


# Plot circles from W1 waveguide in first subplot
for circle in circleW1:
    # Original circle
    if -0.5 <= circle.x_cent <= 0.5:
        # Swap x and y coordinates to make horizontal
        c = plt.Circle((circle.y_cent, circle.x_cent), circle.r, 
                      facecolor='white', edgecolor='white', linewidth=2)
        ax1.add_patch(c)
    
    # Shifted circle (-1)
    if -0.5 <= circle.x_cent <= 0.5:
        c = plt.Circle((circle.y_cent, circle.x_cent-1), circle.r,
                      facecolor='white', edgecolor='white', linewidth=2)
        ax1.add_patch(c)

# Add black box around W1 waveguide
rect1 = plt.Rectangle((-np.sqrt(3)*2, -0.5), np.sqrt(3)*4, 1, 
                     fill=False, color='black', linewidth=2,zorder=10)
ax1.add_patch(rect1)

# Plot circles from ZIW waveguide in second subplot
for circle in circleZIW:
    # Original circle
    if -0.5 <= circle.x_cent <= 0.5:
        # Swap x and y coordinates to make horizontal
        c = plt.Circle((circle.y_cent-np.sqrt(3)/12, circle.x_cent), circle.r,
                      facecolor='white', edgecolor='white', linewidth=2)
        ax2.add_patch(c)
    
    
    # Shifted circle (-1)
    if -0.5 <= circle.x_cent <= 0.5:
        c = plt.Circle((circle.y_cent-np.sqrt(3)/12, circle.x_cent-1), circle.r,
                      facecolor='white', edgecolor='white', linewidth=2)
        ax2.add_patch(c)

# Add black box around ZIW waveguide
rect2 = plt.Rectangle((-np.sqrt(3)*2, -0.5), np.sqrt(3)*4, 1,
                     fill=False, color='black', linewidth=2,zorder=10)
ax2.add_patch(rect2)

# Set equal aspect ratio and limits for both plots
for ax in [ax1, ax2]:
    ax.set_aspect('equal')
    # Swap x and y limits to match rotated orientation
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(-np.sqrt(3)*2, np.sqrt(3)*2)
    
    # Remove axes for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove border
    for spine in ax.spines.values():
        spine.set_visible(False)

#add a label to the top of each plot 
ax1.text(0,0.6,'W1',fontsize=24,color=ax1.get_facecolor(),ha='center')
ax2.text(0,0.6,'ZIW',fontsize=24,color=ax2.get_facecolor(),ha='center')

# Add radius indicator line for W1
# Find a representative circle in W1
r_w1 = 0.3
ax1.plot([-np.sqrt(3)/2, -np.sqrt(3)/2], [-.5,-.5+r_w1], 'k-', linewidth=2)
ax1.text(-np.sqrt(3)/2, -0.8, '$r$', fontsize=22, ha='center',color='k')

# Add radius indicator lines for ZIW
r0_ziw = 0.105
r1_ziw = 0.235
ax2.plot([-np.sqrt(3)/6, -np.sqrt(3)/6], [0, 0-r0_ziw], 'k-', linewidth=2)
ax2.plot([-np.sqrt(3)/3, -np.sqrt(3)/3], [-0.5, -0.5+r1_ziw], 'k-', linewidth=2)
ax2.text(-np.sqrt(3)/6+.2, -.4, '$r_0$', fontsize=22, ha='center',color='k')
ax2.text(-np.sqrt(3)/3, -0.8, '$r_1$', fontsize=22, ha='center',color='k')

# Add dashed rhombus on ZIW waveguide
rhombus_points = np.array([
    [-np.sqrt(3)/2-np.sqrt(3)+np.sqrt(3)/12, 0],  # Left point
    [-np.sqrt(3)+np.sqrt(3)/12, 0.5],           # Top point
    [np.sqrt(3)/2-np.sqrt(3)+np.sqrt(3)/12, 0],  # Right point
    [-np.sqrt(3)+np.sqrt(3)/12, -0.5],          # Bottom point
    [-np.sqrt(3)/2-np.sqrt(3)+np.sqrt(3)/12, 0]  # Close the shape
])
ax2.plot(rhombus_points[:,0], rhombus_points[:,1], '--', color='white', linewidth=2,zorder=1)

# Add dashed hexagon on W1 waveguide
hex_points = np.array([
    [-np.sqrt(3) + np.sqrt(3)/3*np.cos(0*np.pi/3), np.sqrt(3)/3*np.sin(0*np.pi/3)-.007],      # Point 1
    [-np.sqrt(3) + np.sqrt(3)/3*np.cos(1*np.pi/3), np.sqrt(3)/3*np.sin(1*np.pi/3)-.007],      # Point 2
    [-np.sqrt(3) + np.sqrt(3)/3*np.cos(2*np.pi/3), np.sqrt(3)/3*np.sin(2*np.pi/3)-.007],      # Point 3
    [-np.sqrt(3) + np.sqrt(3)/3*np.cos(3*np.pi/3), np.sqrt(3)/3*np.sin(3*np.pi/3)-.007],      # Point 4
    [-np.sqrt(3) + np.sqrt(3)/3*np.cos(4*np.pi/3), np.sqrt(3)/3*np.sin(4*np.pi/3)-.007],      # Point 5
    [-np.sqrt(3) + np.sqrt(3)/3*np.cos(5*np.pi/3), np.sqrt(3)/3*np.sin(5*np.pi/3)-.007],      # Point 6
    [-np.sqrt(3) + np.sqrt(3)/3*np.cos(0*np.pi/3), np.sqrt(3)/3*np.sin(0*np.pi/3)-.007]       # Close the shape
])
ax1.plot(hex_points[:,0], hex_points[:,1], '--', color='white', linewidth=2,zorder=1)

# Add scale bars
# Vertical line on W1
ax1.plot([np.sqrt(3)/2, np.sqrt(3)/2], [-.5, .5], 'k-', linewidth=2)
# Line between holes on W1
ax1.plot([np.sqrt(3)/2, np.sqrt(3)], [-0.5, 0], 'k-', linewidth=2)
ax1.text(np.sqrt(3)/2, -0.8, '$a$', fontsize=22, ha='center',color='black')

# Vertical line on ZIW
ax2.plot([np.sqrt(3)/3,np.sqrt(3)/3], [-.5, .5], 'k-', linewidth=2)
# Line between holes on ZIW
ax2.plot([np.sqrt(3)/3, np.sqrt(3)/3+np.sqrt(3)/2], [-0.5, 0], 'k-', linewidth=2)
ax2.text(np.sqrt(3)/3, -0.8, '$a$', fontsize=22, ha='center',color='black')



plt.show()



#%%
#---------------------------------Improvments to losses from different sources--------------------------------
# Create a figure for comparing backscattering losses from different sources
plt.figure(figsize=(8, 12))  # Increased height

# Define a variable for font size to easily change it across the whole plot
font_size = 20
plt.rcParams.update({'font.size': font_size})  # Larger font size

# Define data
groups = ["krauss", "Thales", "Steve", "us"]
experimental_losses = np.array([.01067, .0125, .0053, .00055])*1000
theoretical_losses = 100*np.ones(len(groups))

# Set up x positions
x = np.arange(len(groups))

# Create the plot
fig, ax = plt.subplots(figsize=(12, 9))  # Increased height

# Plot points with different shapes instead of bars - larger markers
ax.scatter(x, experimental_losses, s=250, marker='o', color='#3498db', label='Theoretical', zorder=2)
ax.scatter(x, theoretical_losses, s=250, marker='^', color='#e74c3c', label='Experimental', zorder=2)

# Add labels, title and custom x-axis tick labels
ax.set_xlabel('Research Group', fontweight='bold', fontsize=font_size + 4)  # Adjusted font size
ax.set_ylabel(r'Loss ($\alpha_{back}/n_g^2$) [cm$^{-1}$] $\times 10^3$', fontweight='bold', fontsize=font_size + 6)  # Adjusted font size
#ax.set_title('Comparison of Backscattering Losses from Different Sources', fontweight='bold', fontsize=font_size + 8)  # Adjusted font size
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=font_size + 4)  # Adjusted font size
ax.legend(fontsize=font_size + 2)  # Adjusted font size

# Add grid lines for better readability
ax.grid(axis='y', linestyle='--', alpha=0.7, which='both')  # Add grid lines for both major and minor ticks
ax.yaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=10))  # Set major ticks for logarithmic scale
ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs='auto', numticks=10))  # Set minor ticks for logarithmic scale


# Add value labels above each experimental point
for i, value in enumerate(experimental_losses):
    ax.annotate(f'{value:.2f}', 
                xy=(x[i], value),
                xytext=(0, 10),  # 10 points vertical offset
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontweight='bold',
                fontsize=font_size)


# Adjust layout and add a note
plt.tight_layout()

plt.ylim(5E-4*1000,.02*1000)
plt.yscale('log')

plt.show()
#%%
#---------------------------------bar plot of different optomizations
# Define data for the bar plot
authors = ["Notomi et al.", "O'Faolain et al.", "Hauff et al.", "Mann et al.", "This Work"]
experimental_data = [.044, .1945, 0, .00138, 0]  # Loss/ng values (x10^-3 cm^-1)
theoretical_data = [0, .00967, .00538, .001543, 0.0004411]  # Loss/ng values (x10^-3 cm^-1)

# Replace None with 0 for plotting, but keep track of which bars should be visible
exp_visible = [True, True, False, True, False]
theo_visible = [False, True, True, True, True]

# Import required libraries
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Set up figure with specific size for publication quality
fig, ax = plt.subplots(figsize=(10, 7))

# Set width of bars and positions
bar_width = 0.35
x = np.arange(len(authors))

# Set up the log scale
ax.set_yscale('log')

# Calculate the position of each bar and tick based on visibility
tick_positions = []
for i in range(len(authors)):
    # Determine tick position based on which bars are visible
    if exp_visible[i] and theo_visible[i]:
        # Both bars visible - tick in the middle
        tick_pos = x[i]
    elif exp_visible[i]:
        # Only experimental bar visible - tick centered on it
        tick_pos = x[i]
    elif theo_visible[i]:
        # Only theoretical bar visible - tick centered on it
        tick_pos = x[i]
    else:
        # No bars visible - use default position
        tick_pos = x[i]
    
    tick_positions.append(tick_pos)

# Create the bars directly using matplotlib's bar function which works well with log scale
exp_bars = []
theo_bars = []

# Add experimental bars
for i in range(len(authors)):
    if exp_visible[i] and experimental_data[i] > 0:
        if theo_visible[i]:
            # Both visible - position on either side of tick
            bar = ax.bar(tick_positions[i] - bar_width/2, experimental_data[i], 
                         width=bar_width, color='#EE7733', 
                         edgecolor='black', linewidth=1.5, zorder=3)
        else:
            # Only experimental visible - center on tick
            bar = ax.bar(tick_positions[i], experimental_data[i], 
                         width=bar_width, color='#EE7733', 
                         edgecolor='black', linewidth=1.5, zorder=3)
        exp_bars.append(bar)
        # Add hatching
        for patch in bar:
            patch.set_hatch('///')
    else:
        exp_bars.append(None)

# Add theoretical bars
for i in range(len(authors)):
    if theo_visible[i] and theoretical_data[i] > 0:
        if exp_visible[i]:
            # Both visible - position on either side of tick
            bar = ax.bar(tick_positions[i] + bar_width/3, theoretical_data[i], 
                         width=bar_width, color='#0077BB', 
                         edgecolor='black', linewidth=1.5, zorder=3)
        else:
            # Only theoretical visible - center on tick
            bar = ax.bar(tick_positions[i], theoretical_data[i], 
                         width=bar_width, color='#0077BB', 
                         edgecolor='black', linewidth=1.5, zorder=3)
        theo_bars.append(bar)
        # Add hatching
        for patch in bar:
            patch.set_hatch('\\\\\\')
    else:
        theo_bars.append(None)

# Create custom patches for the legend
exp_patch = mpatches.Patch(facecolor='#EE7733', edgecolor='black', linewidth=1.5, hatch='///', label='Experimental')
theo_patch = mpatches.Patch(facecolor='#0077BB', edgecolor='black', linewidth=1.5, hatch='\\\\\\', label='Theoretical')

# Add labels, title and custom x-axis tick labels
ax.set_ylabel(r'Loss $\langle\alpha_{\text{back}}\rangle/n_g^2$ $\text{[cm}^{-1}\text{]}$', fontweight='bold', fontsize=30)
ax.set_xticks(tick_positions)  # Use calculated tick positions
ax.set_xticklabels(authors, rotation=45, ha='right', fontsize=25)


# Add grid lines for better readability (y-axis only)
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# Customize y-axis for log scale
min_value = min([v for v in experimental_data + theoretical_data if v > 0]) * 0.5
max_value = max(experimental_data + theoretical_data) * 2
ax.set_ylim(min_value, max_value)
ax.tick_params(axis='y', labelsize=25)

# Add minor grid lines for log scale
ax.grid(which='minor', axis='y', linestyle=':', alpha=0.4, zorder=0)

# Add legend with custom position and larger font
ax.legend(handles=[exp_patch, theo_patch], fontsize=25, loc='upper right')

# Adjust layout for better spacing
plt.tight_layout()

# Set the axis limits to make sure all bars are visible
ax.set_xlim(-0.5, len(authors) - 0.5)

plt.show()





# %%
