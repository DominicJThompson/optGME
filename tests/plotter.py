#%%
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import json
import numpy as np
import autograd.numpy as npa
import legume
from optomization import Backscatter
import matplotlib as mpl
from optomization import NG
plt.rcParams.update({'font.size':20}) 
print(legume.__version__)

# Disable LaTeX but use Computer Modern fonts
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern math fonts
mpl.rcParams['font.family'] = 'STIXGeneral'  # Use STIX fonts (similar to Computer Modern)
#%%
nk=150


def W1(vars=npa.zeros((0,0)),NyChange=3,Ny=10,dslab=170/266,eps_slab=3.4638,ra=.3):

    vars = vars.reshape((3,NyChange*2))

    lattice = legume.Lattice(npa.array([1,0]),npa.array([0,Ny*npa.sqrt(3)]))

    #now we define a photonic crystal that goes over our lattice and add the one layer of our W1 waveguide
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=dslab,eps_b=eps_slab**2)

    for i in range(vars.shape[1]):
        phc.add_shape(legume.Circle(x_cent=vars[0,i],y_cent=vars[1,i],r=vars[2,i]))

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
        phc.add_shape(legume.Circle(x_cent=x,y_cent=y,r=ra))

    return(phc)

def W1Vars(NyChange=3,ra=.3,key=None):
    vars = npa.zeros((3,NyChange*2))
    vars[2,:] = ra
    for i in range(NyChange*2):
        iy = i-NyChange
        if i>=NyChange:
            iy+=1
        if iy%2==1: 
            vars[0,i] = .5
        vars[1,i] = iy*npa.sqrt(3)/2

    vars = vars.flatten()

    if key is not None:
        np.random.seed(key)
        vars += np.random.normal(loc=0,scale=1/266,size=vars.size)

    return(vars)


def ZIW(vars=npa.zeros((0,0)),NyChange=3,Ny=10,dslab=170/266,eps_slab=3.4638,r0=.105,r1=.235):

    vars = vars.reshape((3,NyChange*2*2))

    lattice = legume.Lattice(npa.array([1,0]),npa.array([0,Ny*npa.sqrt(3)-.25]))

    #now we define a photonic crystal that goes over our lattice and add the one layer of our W1 waveguide
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=dslab,eps_b=eps_slab**2)

    for i in range(vars.shape[1]):
        phc.add_shape(legume.Circle(x_cent=vars[0,i],y_cent=vars[1,i],r=vars[2,i]))

    #now we want to add the air holes to our photonic crystal to make it the W1 waveguide we do this in a loop
    flip = False
    for i in range(Ny*2):
        iy = i-Ny+1
        y = iy*npa.sqrt(3)/2-npa.sqrt(3)/12

        if np.abs(iy-.5)<=NyChange:
            continue

        if iy>0 and not flip:
            r0, r1 = r1, r0
            flip = True
        
        #add hole below and above
        phc.add_shape(legume.Circle(x_cent=.5*((iy+1)%2),y_cent=y-npa.sqrt(3)/6,r=r1))
        phc.add_shape(legume.Circle(x_cent=.5*(iy%2),y_cent=y,r=r0))
    
    return(phc)


def ZIWVars(NyChange=3,r0=.105,r1=.235,key=None):
    vars = npa.zeros((3,NyChange*2*2))
    flip = False
    for i in range(NyChange*2):

        iy = i-NyChange+1
        y = iy*npa.sqrt(3)/2-npa.sqrt(3)/12

        if iy>0 and not flip:
            r0, r1 = r1, r0
            flip = True
        
        vars[0,2*i],vars[0,2*i+1] = .5*((iy+1)%2),.5*(iy%2)
        vars[1,2*i],vars[1,2*i+1] = y-npa.sqrt(3)/6, y
        vars[2,2*i],vars[2,2*i+1] = r1,r0

    vars = vars.flatten()
    if key is not None:
        np.random.seed(key)
        vars += np.random.normal(loc=0,scale=1/266,size=vars.size)

    return(vars)

def runSims(xs,crystal,params):
    phc = crystal(xs)

    kmin,kmax = .5*np.pi,np.pi
    gmeParams = params['gmeParams'].copy()
    gmeParams['kpoints']=np.vstack((np.linspace(kmin,kmax,nk),np.zeros(nk)))
    gmeParams['verbose']=True
    gmeParams['numeig']+=5
    gme = legume.GuidedModeExp(phc,gmax=params['gmax'])
    gme.run(**gmeParams)

    print('running alpha')
    cost = Backscatter(**params['cost'])
    alphas = []
    ngs = []
    gmeParams['numeig']-=1
    for i in range(nk):
        if i%10==0:
            print(i)
        gmeParams['kpoints']=np.vstack(([np.linspace(kmin,kmax,nk)[i],np.linspace(kmin,kmax,nk)[i]+.001],[0,0]))
        gmeParams['verbose']=False
        gmeAlphacalc = legume.GuidedModeExp(phc,gmax=params['gmax'])
        gmeAlphacalc.run(**gmeParams)
        alphas.append(10**cost.cost(gmeAlphacalc,phc,params['mode']))
        ngs.append((1/2/np.pi)*np.abs(.001/(gmeAlphacalc.freqs[1,params['mode']]-gmeAlphacalc.freqs[0,params['mode']])))
    return(phc,gme,alphas,ngs)
#%%
def plotBands(gme,ng,params,color='red',plotback=True,index=0):

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
        freqmin, freqmax = .238, .282
    mode = params['mode']
    kindex = np.abs(gme.kpoints[0,:] - params['gmeParams']['kpoints'][0][0]).argmin()
    print('kindex ',kindex)
    print('ng ',ng[kindex])
    # Generate some sample data
    ks = np.linspace(0.25, 0.5, nk)

    # Create figure and define gridspec layout
    fig = plt.figure(figsize=(6.4, 4.8))

    # Main subplot (dispersion curve)
    ax2 = plt.gca()
    ax2.set_xlabel(r"Wavevector $\tilde k$",fontsize=LABEL_FONT_SIZE)
    ax2.set_ylabel(r"Frequency $\omega a / 2\pi c$",fontsize=LABEL_FONT_SIZE)
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
    if plotback:
        ax2.plot(ks,gme.freqs[:,mode+1],color='darkviolet',linewidth=2,linestyle='--') #other band
    else:
        ax2.plot(ks[50:],gme.freqs[50:,mode+1],color='darkviolet',linewidth=2,linestyle='--') #other band
    #ax2.scatter(ks[kindex],gme.freqs[kindex,mode],s=200,color=color,zorder=3) #optomized point

    se = [24,113,15,149,39,84,13,89]
    ax2.scatter([ks[se[2*index]],ks[se[2*index+1]]],[gme.freqs[se[2*index],mode],gme.freqs[se[2*index+1],mode]],s=150,color=color,zorder=3,edgecolor='black', linewidth=1.5)
    
    # Add text labels with slight offset for better visibility
    if index==0:a,b = r'$\mathbf{a}^\prime$',r'$\mathbf{b}^\prime$'
    elif index==1:a,b = r'$\mathbf{a}$',r'$\mathbf{b}$'
    elif index==2:a,b = r'$\mathbf{c}^\prime$',r'$\mathbf{d}^\prime$'
    elif index==3:a,b = r'$\mathbf{c}$',r'$\mathbf{d}$'
    ax2.annotate(a, 
                 (ks[se[2*index]], gme.freqs[se[2*index], mode]),
                 xytext=(5, 5),
                 textcoords='offset points',
                 color=color,
                 fontsize=ANNOTATION_FONT_SIZE) 
    ax2.annotate(b, 
                 (ks[se[2*index+1]], gme.freqs[se[2*index+1], mode]),
                 xytext=(40, -30),
                 textcoords='offset points',
                 color=color,
                 fontsize=ANNOTATION_FONT_SIZE)


    #find the bandwidth in frequncy
    intersect = np.where(np.sign((gme.freqs[:,mode]-ks)[:-1]) != np.sign((gme.freqs[:,mode]-ks))[1:])[0]
    bandMax = min(np.max(gme.freqs[:,mode]),np.min(gme.freqs[50:,mode+1]),gme.freqs[intersect,mode])
    bandMin = np.max(np.hstack((gme.freqs[:,mode-1],np.min(gme.freqs[:,mode]))))
    #if color=='red':
    #    bandMin+=.0015
    ax2.fill_between(ks,np.ones_like(ks)*bandMax,np.ones_like(ks)*bandMin,color='cyan',alpha=.5,zorder=0)

    # Find k-points corresponding to bandMin and bandMax
    k_bandMin_idx = np.abs(gme.freqs[:,mode] - bandMin).argmin()
    k_bandMax_idx = np.abs(gme.freqs[:,mode] - bandMax).argmin()
    
    k_bandMin = ks[k_bandMin_idx]
    k_bandMax = ks[k_bandMax_idx]
    
    print(f"Band minimum: frequency = {bandMin} at k = {k_bandMin_idx}")
    print(f"Band maximum: frequency = {bandMax} at k = {k_bandMax_idx}")
    print(f"Bandwidth: {(bandMax - bandMin)*conFac}")
    # Show plot
    plt.show()

plotBands(gme,ng,out[-1],color='#EE7733',plotback=False,index=2)
plotBands(gmeOG,ngOG,out[-1],color='#0077BB',plotback=False,index=3)

#plotBands(gmeW1,ngW1,out[-1],color='#EE7733',index=0)
#plotBands(gmeW1OG,ngW1OG,out[-1],color='#0077BB',index=1)
#%%
def alphaImprove(alpha,alphaOG,ng,ngOG,params):
    ks = np.linspace(0.25, 0.5, nk)
    kindex = np.abs(ks-params['gmeParams']['kpoints'][0][0]/np.pi/2).argmin()

    print('lossOpt ',alpha[kindex]*ng[kindex]**2/266/1E-7)
    print('lossOG ',alphaOG[kindex]*ngOG[kindex]**2/266/1E-7)
    print('lossOpt no ng ',alpha[kindex]/266/1E-7)
    print('lossOG no ng ',alphaOG[kindex]/266/1E-7)

    fig, ax1 = plt.subplots()

    # Plot alpha values
    ax1.plot(ks, np.array(alpha)/266/1E-7, '#0077BB', linewidth=2, label='Optimized',zorder=0)
    ax1.plot(ks, np.array(alphaOG)/266/1E-7, '#0077BB', linewidth=2, label='Original',linestyle='--',zorder=0)
    ax1.scatter(ks[kindex], alpha[kindex]/266/1E-7, s=200, color='#005577',zorder=3)  # Changed to a different color of the same relative brightness
    ax1.scatter(ks[kindex], alphaOG[kindex]/266/1E-7, s=200, color='limegreen',zorder=3)
    ax1.set_yscale('log')
    ax1.set_ylim(.8E-3, 1E-1)
    ax1.set_ylabel(r"$\langle\alpha_{back}\rangle/n_g^2$ [cm$^{-1}$]", color='#0077BB')
    ax1.set_xlabel(r"$k_x 2\pi/a$")
    ax1.set_xlim(.25, .5)
    ax1.tick_params(axis='y', labelcolor='#0077BB')

    # Add second y-axis for ng and ngo values
    ax2 = ax1.twinx()
    ax2.plot(ks, ng, '#EE7733', linewidth=2)
    ax2.plot(ks, ngOG, '#EE7733', linewidth=2, linestyle='--')
    ax2.set_ylabel(r'$n_g$', rotation=270, labelpad=25, color='#EE7733')
    ax2.set_yscale('log')
    ax2.set_ylim(.8, 100)
    ax2.tick_params(axis='y', labelcolor='#EE7733')

    # Create custom legend with black lines
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2, label='Optimized'),
        Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Original')
    ]
    ax1.legend(handles=legend_elements, frameon=False)

    plt.show()

def filedPlots(phc,phcOG,gme,gmeOG,params):
    # Font size variables for consistent styling
    TITLE_SIZE = 28
    LABEL_SIZE = 26
    TICK_SIZE = 20
    LEGEND_SIZE = 24
    COLORBAR_LABEL_SIZE = 24
    COLORBAR_TICK_SIZE = 20
    
    # Set up variables
    ylim = 8*np.sqrt(3)/2
    ys = np.linspace(-ylim/2, ylim/2, 300)
    mode = params['mode']
    kindex = np.abs(gme.kpoints[0,:] - params['gmeParams']['kpoints'][0][0]).argmin()
    # kindex = 190
    z = phc.layers[0].d/2

    # Get field of original crystal
    fieldsOG, _, _ = gmeOG.get_field_xy('E', kindex, mode, z, ygrid=ys, component='xyz')
    eabsOG = np.abs(np.conj(fieldsOG['x'])*fieldsOG['x'] + np.conj(fieldsOG['y'])*fieldsOG['y'] + np.conj(fieldsOG['z'])*fieldsOG['z'])
    fields, _, _ = gme.get_field_xy('E', kindex, mode, z, ygrid=ys, component='xyz')
    eabs = np.abs(np.conj(fields['x'])*fields['x'] + np.conj(fields['y'])*fields['y'] + np.conj(fields['z'])*fields['z'])
    maxF = np.max([np.max(eabsOG), np.max(eabs)])

    epsOG = legume.viz.eps_xy(phcOG,Nx = 100,Ny = 100,plot=False)
    eps = legume.viz.eps_xy(phc,Nx = 100,Ny = 100,plot=False)
    fOG, _, _ = gmeOG.get_field_xy('E', kindex, mode, z,Ny = 100,Nx = 100, component='xyz')
    eOG = np.abs(np.conj(fOG['x'])*fOG['x'] + np.conj(fOG['y'])*fOG['y'] + np.conj(fOG['z'])*fOG['z'])
    f, _, _ = gme.get_field_xy('E', kindex, mode, z,Ny = 100,Nx = 100, component='xyz')
    e = np.abs(np.conj(f['x'])*f['x'] + np.conj(f['y'])*f['y'] + np.conj(f['z'])*f['z'])
    vOG = 1/np.max(eOG*epsOG)
    v = 1/np.max(e*eps)
    print(r"Mode volume $\lambda^3$: (OG,Opt) ",vOG*(gmeOG.freqs[kindex,mode]*2*np.pi)**3, v*(gme.freqs[kindex,mode]*2*np.pi)**3)
    print(r"Mode volume $\mu m^3$: (OG,Opt) ",vOG/.266**3, v/.266**3)

    #calculate the maximum percell enhancement
    ng_OG = np.abs(NG(gmeOG,kindex,mode))
    ng_ = np.abs(NG(gme,kindex,mode))
    cPF = 3*np.pi*(3E8)**2*266E-9/(12**(3/2))
    conFac = 2*np.pi*299792458/params['cost']['a']/1e-9
    print(r"Maximum percell enhancement: (OG,Opt) ",cPF*ng_OG/(vOG*(266E-9)**3)/(gmeOG.freqs[kindex,mode]*conFac)**2,cPF*ng_/(v*(266E-9)**3)/(gme.freqs[kindex,mode]*conFac)**2)
    # Create a figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 3))
    
    # Optional parameters to control the field view|
    x_offset = 0  # Adjust to shift the center of the view left or right
    x_crop = 0    # Adjust to crop from the right side (0 for no cropping)
    
    # Calculate the actual x limits based on the parameters
    x_min = -ylim/2 + x_offset
    x_max = ylim/2 + x_offset - x_crop
    
    # Plot original field on top subplot
    cax1 = ax1.imshow(eabsOG.T, extent=[-ylim/2, ylim/2, -.5, .5], cmap='plasma', vmax=maxF, vmin=0)
    ax1.set_title('Original', fontsize=TITLE_SIZE,pad=10)
    circles1 = [Circle((-s.y_cent, s.x_cent), s.r, edgecolor='white', facecolor='none', linewidth=3) for s in phcOG.layers[0].shapes]
    cirlcesArround1 = [Circle((0, 0), s.r, edgecolor='white', facecolor='none', linewidth=3) for s in phcOG.layers[0].shapes]
    for c, ca in zip(circles1, cirlcesArround1):
        ax1.add_patch(c)
        ca.center = (c.center[0], c.center[1]-np.sign(c.center[1]))
        ax1.add_patch(ca)
    ax1.set_ylim(-.5, .5)
    ax1.set_xlim(x_min, x_max)
    ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                   labelbottom=False, labelleft=False)
    
    # Plot optimized field on bottom subplot
    cax2 = ax2.imshow(eabs.T, extent=[-ylim/2, ylim/2, -.5, .5], cmap='plasma', vmax=maxF, vmin=0)
    ax2.set_title('Optimized', fontsize=TITLE_SIZE, pad=10)  # Added pad parameter to move title up
    circles2 = [Circle((-s.y_cent, s.x_cent), s.r, edgecolor='white', facecolor='none', linewidth=3) for s in phc.layers[0].shapes]
    cirlcesArround2 = [Circle((0, 0), s.r, edgecolor='white', facecolor='none', linewidth=3) for s in phc.layers[0].shapes]
    for c, ca in zip(circles2, cirlcesArround2):
        ax2.add_patch(c)
        ca.center = (c.center[0], c.center[1]-np.sign(c.center[1]))
        ax2.add_patch(ca)
    ax2.set_ylim(-.5, .5)
    ax2.set_xlim(x_min, x_max)
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                   labelbottom=False, labelleft=False)
    
    # Add a single colorbar for both plots
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])  # Position for the colorbar
    cbar = fig.colorbar(cax1, cax=cbar_ax)
    cbar.set_label(r"$|\mathbf{e}_{\tilde k=0.33}|^2$ [a$^{-3}$]", rotation=90, labelpad=20, fontsize=COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    # Format the tick labels to show 0 instead of 0.0 while keeping other decimals
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: '0' if x == 0 else f'{x:.1f}'))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make room for colorbar
    plt.show()

filedPlots(phc,phcOG,gme,gmeOG,out[-1])
filedPlots(phcW1,phcW1OG,gmeW1,gmeW1OG,out[-1])
#%%
def lossVng(phc,gme,alphas,ng,phcOG,gmeOG,alphasOG,ngOG,crystal):
    # Plot styling parameters
    STYLE = {
        'font_sizes': {
            'title': 30,
            'label': 36,
            'legend': 32,
            'marker': 30,
            'tick': 30,
            'annotation': 36  # For the a, b, a', b' labels
        },
        'colors': {
            'optimized': '#EE7733',  # orange
            'original': '#0077BB'    # blue
        },
        'line': {
            'width': 3
        },
        'scatter': {
            'size': 150,
            'edge_width': 1.5,
            'zorder': 3
        },
        'axes': {
            'x_lim': (3, 200),
            'y_lim': (1E-3, 1E-1),
            #'x_lim': (2.5,10000),
            #'y_lim': (2.5E-4,1E-2),
            'legend_bbox': (1,1)
        },
        'figure': {
            'dpi': 300  # Increased DPI for higher resolution
        },
        'ticks': {
            'major_length': 10,
            'major_width': 2,
            'minor_length': 5,
            'minor_width': 1.5
        }
    }
    plt.close('all')
    # Create figure with higher DPI
    plt.figure(dpi=STYLE['figure']['dpi'])
    
    # Extract parameters for cleaner code
    fs = STYLE['font_sizes']
    colors = STYLE['colors']
    
    if crystal == 'W1':
        ks = [24,113,15,149]
    elif crystal == 'ZIW':
        ks = [39,84,13,89]

    # Plot lines
    plt.plot(ngOG[ks[2]:], alphasOG[ks[2]:]/266/1E-7, 
             color=colors['original'], label='Original', 
             linewidth=STYLE['line']['width'])
    plt.plot(ng[ks[0]:], alphas[ks[0]:]/266/1E-7, 
             color=colors['optimized'], label='Optimized', 
             linewidth=STYLE['line']['width'])

    # Add annotation labels
    if crystal == 'W1':
        annotations = [
            # Optimized (orange) annotations
            {'x': ng[ks[0]]+4, 'y': alphas[ks[0]]/266/1E-7+.00005, 
             'text': r'$\mathbf{a}^\prime$', 'color': colors['optimized'], 
             'ha': 'right', 'va': 'bottom'},
            {'x': ng[ks[1]-1]+10, 'y': alphas[ks[1]-1]/266/1E-7+.002, 
             'text': r'$\mathbf{b}^\prime$', 'color': colors['optimized'], 
             'ha': 'left', 'va': 'top'},
            # Original (blue) annotations
            {'x': ngOG[ks[2]]+.3, 'y': alphasOG[ks[2]]/266/1E-7+.0002, 
             'text': r'$\mathbf{a}$', 'color': colors['original'], 
             'ha': 'right', 'va': 'bottom'},
            {'x': ngOG[ks[3]-1], 'y': alphasOG[ks[3]-1]/266/1E-7+.0025, 
             'text': r'$\mathbf{b}$', 'color': colors['original'], 
             'ha': 'left', 'va': 'top'}
        ]
    elif crystal == 'ZIW':
        annotations = [
            # Optimized (orange) annotations
            {'x': ng[ks[0]]+3, 'y': alphas[ks[0]]/266/1E-7-.001, 
             'text': r'$\mathbf{c}^\prime$', 'color': colors['optimized'], 
             'ha': 'right', 'va': 'bottom'},
            {'x': ng[ks[1]-1]-1, 'y': alphas[ks[1]-1]/266/1E-7+.0035, 
             'text': r'$\mathbf{d}^\prime$', 'color': colors['optimized'], 
             'ha': 'left', 'va': 'top'},
            # Original (blue) annotations
            {'x': ngOG[ks[2]]+40, 'y': alphasOG[ks[2]]/266/1E-7-.0006, 
             'text': r'$\mathbf{c}$', 'color': colors['original'], 
             'ha': 'right', 'va': 'bottom'},
            {'x': ngOG[ks[3]-1]-1.25, 'y': alphasOG[ks[3]-1]/266/1E-7+.002, 
             'text': r'$\mathbf{d}$', 'color': colors['original'], 
             'ha': 'left', 'va': 'top'}
        ]

    for ann in annotations:
        plt.text(ann['x'], ann['y'], ann['text'], 
                color=ann['color'], fontsize=fs['annotation'], 
                fontweight='bold', ha=ann['ha'], va=ann['va'])

    # Add scatter points
    plt.scatter([ng[ks[0]], ng[ks[1]-1]], 
                [alphas[ks[0]]/266/1E-7, alphas[ks[1]-1]/266/1E-7],
                s=STYLE['scatter']['size'], color=colors['optimized'],
                zorder=STYLE['scatter']['zorder'], edgecolor='black', 
                linewidth=STYLE['scatter']['edge_width'])
    
    plt.scatter([ngOG[ks[2]], ngOG[ks[3]-1]], 
                [alphasOG[ks[2]]/266/1E-7, alphasOG[ks[3]-1]/266/1E-7],
                s=STYLE['scatter']['size'], color=colors['original'],
                zorder=STYLE['scatter']['zorder'], edgecolor='black', 
                linewidth=STYLE['scatter']['edge_width'])

    # Set axes properties
    plt.ylim(STYLE['axes']['y_lim'])
    plt.xlim(STYLE['axes']['x_lim'])
    plt.xscale('log')
    plt.yscale('log')
    
    # Add labels
    plt.xlabel(r"Group Index $n_g$", fontsize=fs['label'])
    
    # Set y-label based on crystal type
    plt.ylabel(rf"$\tilde L^{{\text{{{crystal}}}}}$ [cm$^{{-1}}$]", fontsize=fs['label'])
    
    # Add legend for W1 crystal
    if crystal == 'W1':
        plt.legend(loc='lower right', frameon=False, fontsize=fs['legend'])
        # Set custom x-axis labels for major ticks
        ax.set_xticks([10**1, 10**2, 10**3, 10**4])
        ax.set_xticklabels(['$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    
    # Configure tick parameters for both major and minor ticks
    plt.tick_params(axis='both', which='major', 
                   labelsize=fs['tick'],
                   length=STYLE['ticks']['major_length'],
                   width=STYLE['ticks']['major_width'])
    plt.tick_params(axis='both', which='minor',
                   length=STYLE['ticks']['minor_length'],
                   width=STYLE['ticks']['minor_width'])
    
    # Force minor ticks on x-axis by setting subs and numticks
    ax = plt.gca()
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    
    # Add grid for both crystal types
    plt.grid(True, which='minor', alpha=0.2)
    plt.grid(True, which='major', alpha=0.4)
    
    plt.show()

lossVng(phc,gme,alphas,ng,phcOG,gmeOG,alphasOG,ngOG,'ZIW')
#lossVng(phcW1,gmeW1,alphasW1,ngW1,phcW1OG,gmeW1OG,alphasW1OG,ngW1OG,'W1')

#%%
def lossVfreq(phc,gme,alphas,ng,phcOG,gmeOG,alphasOG,ngOG,crystal):
    # Style parameters
    STYLE = {
        'font_sizes': {
            'title': 30,
            'label': 36,
            'legend': 28,
            'marker': 30,
            'tick': 30,
            'annotation': 36 
        },
        'colors': {
            'optimized': '#EE7733',  # orange
            'original': '#0077BB'    # blue
        },
        'plot': {
            'linewidth': 3,
            'scatter_size': 120,
            'scatter_zorder': 10,
            'scatter_edge_width': 1.5,
            'scatter_edge_color': 'black'
        },
        'axes': {
            #'y_lim': (2.5E-4, 1E-2),
            'y_lim': (1E-3,1E-1),
            'legend_bbox': (.7, .37)
        },
        'ticks': {
            'major_length': 10,
            'major_width': 2,
            'minor_length': 5,
            'minor_width': 1.5
        }
    }

    # Key points to plot
    if crystal == 'W1':
        ks = [24,113,15,149]
    elif crystal == 'ZIW':
        ks = [39, 84, 13, 89]

    # Plot lines
    plt.plot(gmeOG.freqs[ks[2]:,20], alphasOG[ks[2]:]/266/1E-7, 
             color=STYLE['colors']['original'], label='Original', 
             linewidth=STYLE['plot']['linewidth'])
    plt.plot(gme.freqs[ks[0]:,20], alphas[ks[0]:]/266/1E-7, 
             color=STYLE['colors']['optimized'], label='Optimized', 
             linewidth=STYLE['plot']['linewidth'])

    # ZIW labels
    if crystal == 'W1':
        annotations = [
            {'x': gme.freqs[ks[0],20]-.00075, 'y': alphas[ks[0]]/266/1E-7,
             'text': r'$\mathbf{a}^\prime$', 'color': STYLE['colors']['optimized'],
             'ha': 'center', 'va': 'bottom'},
            {'x': gme.freqs[ks[1],20], 'y': alphas[ks[1]-1]/266/1E-7+.0035,
             'text': r'$\mathbf{b}^\prime$', 'color': STYLE['colors']['optimized'],
             'ha': 'left', 'va': 'top'},
            {'x': gmeOG.freqs[ks[2],20]-.001, 'y': alphasOG[ks[2]]/266/1E-7,
             'text': r'$\mathbf{a}$', 'color': STYLE['colors']['original'],
             'ha': 'right', 'va': 'bottom'},
            {'x': gmeOG.freqs[ks[3],20]+.001, 'y': alphasOG[ks[3]]/266/1E-7,
             'text': r'$\mathbf{b}$', 'color': STYLE['colors']['original'],
             'ha': 'left', 'va': 'top'}
        ]
    elif crystal == 'ZIW':
        annotations = [
            {'x': gme.freqs[ks[0],20]-.00075, 'y': alphas[ks[0]]/266/1E-7,
             'text': r'$\mathbf{c}^\prime$', 'color': STYLE['colors']['optimized'],
             'ha': 'center', 'va': 'bottom'},
            {'x': gme.freqs[ks[1],20], 'y': alphas[ks[1]-1]/266/1E-7+.004,
             'text': r'$\mathbf{d}^\prime$', 'color': STYLE['colors']['optimized'],
             'ha': 'left', 'va': 'top'},
            {'x': gmeOG.freqs[ks[2],20]-.001, 'y': alphasOG[ks[2]]/266/1E-7,
             'text': r'$\mathbf{c}$', 'color': STYLE['colors']['original'],
             'ha': 'right', 'va': 'bottom'},
            {'x': gmeOG.freqs[ks[3],20]+.001, 'y': alphasOG[ks[3]]/266/1E-7+.015,
             'text': r'$\mathbf{d}$', 'color': STYLE['colors']['original'],
             'ha': 'left', 'va': 'top'}
        ]

    for ann in annotations:
        plt.text(ann['x'], ann['y'], ann['text'],
                color=ann['color'], fontsize=STYLE['font_sizes']['annotation'],
                fontweight='bold', ha=ann['ha'], va=ann['va'])

    # Add scatter points
    scatter_points = [
        (gme.freqs[ks[0],20], alphas[ks[0]]/266/1E-7, STYLE['colors']['optimized']),
        (gme.freqs[ks[1],20], alphas[ks[1]-1]/266/1E-7, STYLE['colors']['optimized']),
        (gmeOG.freqs[ks[2],20], alphasOG[ks[2]]/266/1E-7, STYLE['colors']['original']),
        (gmeOG.freqs[ks[3],20], alphasOG[ks[3]]/266/1E-7, STYLE['colors']['original'])
    ]

    for x, y, color in scatter_points:
        plt.scatter(x, y, color=color, s=STYLE['plot']['scatter_size'],
                   zorder=STYLE['plot']['scatter_zorder'],
                   edgecolor=STYLE['plot']['scatter_edge_color'],
                   linewidth=STYLE['plot']['scatter_edge_width'])

    # Set axes properties
    plt.ylim(STYLE['axes']['y_lim'])
    plt.yscale('log')
    plt.xlabel(r"Frequency $\omega a / 2\pi c$",
              fontsize=STYLE['font_sizes']['label'])

    # Configure tick parameters for both major and minor ticks
    plt.tick_params(axis='both', which='major', 
                   labelsize=STYLE['font_sizes']['tick'],
                   length=STYLE['ticks']['major_length'],
                   width=STYLE['ticks']['major_width'])
    plt.tick_params(axis='both', which='minor',
                   length=STYLE['ticks']['minor_length'],
                   width=STYLE['ticks']['minor_width'])

    # Add grid
    plt.grid(True, which='minor', alpha=0.2)
    plt.grid(True, which='major', alpha=0.4)

    plt.show()

#lossVfreq(phcW1,gmeW1,alphasW1,ngW1,phcW1OG,gmeW1OG,alphasW1OG,ngW1OG,'W1')
lossVfreq(phc,gme,alphas,ng,phcOG,gmeOG,alphasOG,ngOG,'ZIW')

#%%
print(len(ngW1))
#%%
with open('/home/dominic/Desktop/optGME/optGME/tests/media/ginds3/ziwBest.json','r') as file:
    out = json.load(file)

phc,gme,alphas,ng = runSims(np.array(out[-1]['result']['x']),ZIW,out[-1])
phcOG,gmeOG,alphasOG,ngOG = runSims(ZIWVars(),ZIW,out[-1]) 
#%%
with open('/home/dominic/Desktop/optGME/optGME/tests/media/ginds3/W1Best.json','r') as file:
    out = json.load(file)

phcW1,gmeW1,alphasW1,ngW1 = runSims(np.array(out[-1]['result']['x']),W1,out[-1])
phcW1OG,gmeW1OG,alphasW1OG,ngW1OG = runSims(W1Vars(),W1,out[-1]) 
#%%
alphas = np.array(alphas)
alphasOG = np.array(alphasOG)
ng = np.array(ng)
ngOG = np.array(ngOG)
alphasW1 = np.array(alphasW1)
alphasW1OG = np.array(alphasW1OG)
ngW1 = np.array(ngW1)
ngW1OG = np.array(ngW1OG)


#%%
plotBands(gme,ng,out[-1],color='#EE7733')
plotBands(gmeOG,ngOG,out[-1],color='#0077BB')

plotBands(gmeW1,ngW1,out[-1],color='#EE7733')
plotBands(gmeW1OG,ngW1OG,out[-1],color='#0077BB')

#%%
alphaImprove(alphas,alphasOG,ng,ngOG,out[-1])
alphaImprove(alphasW1,alphasW1OG,ngW1,ngW1OG,out[-1])

#%%
filedPlots(phc,phcOG,gme,gmeOG,out[-1])

# %%
print(gmeOG.freqs[80,20])

#%%
# %%
plt.plot(gme.freqs[150:160,19:21],'o')

# %%
import copy 

gmeCopy = copy.deepcopy(gme)
alphasCopy = copy.deepcopy(alphas)
ngCopy = copy.deepcopy(ng)
#%%
with open('/Users/dominic/Desktop/optGME/tests/media/nonlinopt0_v2.json','r') as file:
    out = json.load(file)

phcW1,gmeW1,alphasW1,ngW1 = runSims(W1Vars(),W1,out[-1]) 

#%%
plt.plot(gme.kpoints[0],np.array(alphasOG))
plt.plot(gme.kpoints[0],np.array(alphasW1))
plt.yscale('log')
#plt.ylim(.1,1e6)
plt.show()
# %%
with open('/Users/dominic/Desktop/optGME/tests/media/nonlinopt0_v2.json','r') as file:
    out = json.load(file)

phcW1,gmeW1,alphasW1,ngW1 = runSims(W1Vars(),W1,out[-1]) 
#%%
with open('/Users/dominic/Desktop/optGME/tests/media/ziwOptBasic39.json','r') as file:
    outZIW = json.load(file)

phcZIW,gmeZIW,alphasZIW,ngZIW = runSims(ZIWVars(),ZIW,outZIW[-1]) 

#%%
plt.plot(gmeW1.freqs[20:,20],((np.array(alphasW1[20:])*np.array(ngW1[20:])**2)/266/1E-7),'r',label='W1',linewidth=2)
plt.plot(gmeZIW.freqs[20:118,20],((np.array(alphasZIW[20:118])*np.array(ngZIW[20:118])**2)/266/1E-7),'g',label='ZIW',linewidth=2)
#plt.plot(gmeW1.freqs[20:,20],1/((np.array(alphasW1[20:]))/266/1E-7),'r',label='W1',linewidth=2)
#plt.plot(gmeZIW.freqs[20:118,20],1/((np.array(alphasZIW[20:118]))/266/1E-7),'g',label='ZIW',linewidth=2)
plt.yscale('log')
plt.yticks([1E-1,10,1000,100000])
plt.xlabel(r"Frequency $\omega a / 2\pi c$")
plt.ylabel(r"$\langle\alpha_{back}\rangle$ [cm$^{-1}$]")
# Add twin axis for ng
ax2 = plt.twinx()
ax2.plot(gmeW1.freqs[20:,20], np.array(ngW1[20:])**2, 'r--', alpha=0.5, linewidth=2)
ax2.plot(gmeZIW.freqs[20:118,20], np.array(ngZIW[20:118])**2, 'g--', alpha=0.5, linewidth=2)
ax2.set_ylabel(r"$n_g^2$")
ax2.set_yscale('log')
ax2.set_yticks([1E1,1E3,1E5,1E7])

plt.show()
# %%
# %%
plt.rcParams.update({'font.size':22})
# Create figure with appropriate size for publication
plt.figure(figsize=(8, 6))

# Plot backscattering on left y-axis
ax1 = plt.gca()
ax1.plot(gmeW1.freqs[20:,20], 
         ((np.array(alphasW1[20:])*np.array(ngW1[20:])**2)/266/1E-7),
         color='#0077BB', linewidth=3, linestyle='-')  # Strong blue
ax1.plot(gmeZIW.freqs[20:118,20],
         ((np.array(alphasZIW[20:118])*np.array(ngZIW[20:118])**2)/266/1E-7),
         color='#0077BB', linewidth=3, linestyle='--')  # Strong blue

# Configure left y-axis
ax1.set_yscale('log')
plt.ylim(.5E-1,1E3)
ax1.set_yticks([1E-1, 1E1, 1E3])
ax1.set_ylabel(r"Loss $\langle\alpha_{back}\rangle$ [cm$^{-1}$]", color='#0077BB')
ax1.set_xlabel(r"Frequency $\omega a / 2\pi c$")
ax1.tick_params(axis='y', labelcolor='#0077BB')

# Add right y-axis for group index
ax2 = ax1.twinx()
ax2.plot(gmeW1.freqs[20:,20], 
         np.array(ngW1[20:])**2,
         color='#EE7733', label='W1', linewidth=3, linestyle='-')  # Orange
ax2.plot(gmeZIW.freqs[20:118,20],
         np.array(ngZIW[20:118])**2, 
         color='#EE7733', label='ZIW', linewidth=3, linestyle='--')  # Orange

# Configure right y-axis  
ax2.set_yscale('log')
ax2.set_ylim(1E1,1E3)
ax2.set_yticks([1E1, 1E2, 1E3])
ax2.set_ylabel(r"$n_g^2$", color='#EE7733')
ax2.tick_params(axis='y', labelcolor='#EE7733')

# Add simplified legend
ax1.plot([], [], 'k-', label='W1', linewidth=2)
ax1.plot([], [], 'k--', label='ZIW', linewidth=2)
ax1.legend(bbox_to_anchor=(.4, .6), frameon=False)

# Adjust layout and display
plt.tight_layout()
plt.show()

# %%
orange = '#EE7733'
blue = '#0077BB'

plt.plot(np.array(ngW1[20:])**2,((np.array(alphasW1[20:]))/266/1E-7),color=orange,label='W1',linewidth=3)
plt.plot(np.array(ngZIW[20:118])**2,((np.array(alphasZIW[20:118]))/266/1E-7),color=blue,label='ZIW',linewidth=3)
plt.yscale('log')
plt.xscale('log')
plt.xlim(1E1,1E4)
plt.ylim(2E-3,1E-1)
plt.legend(bbox_to_anchor=(.6, .45), frameon=False,fontsize=24)
plt.xlabel(r"Group Index $n_g^2$",fontsize=28)
plt.ylabel(r"Loss $\langle\alpha_{back}\rangle/n_g^2$ [cm$^{-1}$]",fontsize=28)
plt.show()


# %%
legume.viz.eps_xy(phc)
# %%
ks = np.linspace(0.25, 0.5, 25)
gmeParamss = {'verbose':True,'numeig':21,'compute_im':False,'kpoints':np.array([[float(ks[8])],[0]]),'gmode_inds':[0,2,4]}
out = legume.GuidedModeExp(phc,gmax=4.01)
out.run(**gmeParamss)
# %%

# %%
from scipy.constants import c
boundNG = []
for i in range(len(gmeW1OG.freqs)):
    Efield,_,_ = gmeW1OG.get_field_xy('E',i,20,phcW1OG.layers[0].d/2,Nx=60,Ny=125)
    Hfield,_,_ = gmeW1OG.get_field_xy('H',i,20,phcW1OG.layers[0].d/2,Nx=60,Ny=125)
    Efield = np.array([[Efield['x']],[Efield['y']],[Efield['z']]])
    Hfield = np.array([[Hfield['x']],[Hfield['y']],[Hfield['z']]])
    ng = 1/(np.sum(np.real(np.cross(np.conj(Efield),Hfield,axis=0)))*phcW1OG.lattice.a2[1]/60/125*phcW1OG.layers[0].d)
    boundNG.append(ng)
#%%
boundNG = np.array(boundNG)
# Find indices where ng is approximately 5, 10, 20, 50, and 100
target_ngs = [5, 10, 30, 100]
indices = []

for target in target_ngs:
    # Find the index where ng is closest to the target value
    idx = np.argmin(np.abs(boundNG - target))
    indices.append(idx)

# Plot the full ng data
plt.figure(figsize=(10, 6))
plt.plot(boundNG)

# Add dots at the specific indices
for i, idx in enumerate(indices):
    plt.plot(idx, boundNG[idx], 'ro', markersize=8)
    plt.annotate(f'ng={target_ngs[i]}, idx={idx}', 
                 xy=(idx, boundNG[idx]), 
                 xytext=(10, 0), 
                 textcoords='offset points',
                 fontsize=12)

plt.yscale('log')
plt.xlabel('Index', fontsize=14)
plt.ylabel('Group Index (ng)', fontsize=14)
plt.title('Group Index with Markers at Specific Values', fontsize=16)
plt.grid(True, which='both', linestyle='--', alpha=0.7)
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
ZIWopt = 10**(np.array(ZIWopt))/266/1E-7
# %%
# Create separate figures for W1 and ZIW optimization plots

# First figure: W1 Optimization
fig1, ax1 = plt.subplots(figsize=(5, 4))

# Set log scales for W1 plot
ax1.set_yscale('log')
ax1.set_xscale('log')

# Highlight regions where constraint violation is non-zero for W1
violation_regions_w1 = []
start = None

# Find continuous regions of constraint violations for W1
for i in range(len(W1opt)):
    if W1CV[i] > 0 and start is None:
        start = i
    elif W1CV[i] <= 0 and start is not None:
        violation_regions_w1.append((start, i))
        start = None

# Add the last region if it extends to the end
if start is not None:
    violation_regions_w1.append((start, len(W1opt)))

# Add dashed horizontal lines for initial and final values for W1
initial_value_w1 = W1opt[0]
final_value_w1 = W1opt[-1]
ax1.axhline(y=initial_value_w1, color='k', linestyle='--', alpha=0.7, zorder=0)
ax1.axhline(y=final_value_w1, color='k', linestyle='--', alpha=0.7, zorder=0)

# Plot W1 optimization data
x_values = np.arange(1, len(W1opt) + 1)  # Start from 1 for log scale

# Plot base line first
ax1.plot(x_values, W1opt, 'b-', linewidth=2.5, zorder=2)

# Add markers with different colors based on constraint violation
for i, val in enumerate(W1opt):
    idx = i + 1  # Adjust for 1-based indexing in plot
    in_violation = any(start <= i < end for start, end in violation_regions_w1)
    if in_violation:
        ax1.plot(idx, val, 'ro', markersize=4, zorder=3)
    else:
        ax1.plot(idx, val, 'o', markersize=4, markerfacecolor='blue', markeredgecolor='darkblue', zorder=3)

# Add violation regions
for start, end in violation_regions_w1:
    ax1.axvspan(start + 1, end + 1, alpha=0.2, color='#FF000080', edgecolor=None)

# Add labels for initial and final values for W1, formatted for small numbers
ax1.text(len(W1opt)*0.1, initial_value_w1*1.2, f'Initial: {initial_value_w1:.5f}', 
         verticalalignment='bottom', horizontalalignment='right', fontsize=14, 
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
ax1.text(len(W1opt)*0.1, final_value_w1*0.8, f'Final: {final_value_w1:.5f}', 
         verticalalignment='top', horizontalalignment='right', fontsize=14,
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

ax1.set_title('W1 Optimization', fontsize=18, fontweight='bold')
ax1.set_xlabel('Iteration', fontsize=16)
ax1.set_ylabel(r'Loss $\langle\alpha_{back}\rangle/n_g^2$ [cm$^{-1}$]', fontsize=16)
ax1.set_xlim(1, len(W1opt))  # Start from 1 for log scale
ax1.grid(True, linestyle='--', alpha=0.3, which='both')
ax1.tick_params(axis='both', which='major', labelsize=13)
ax1.set_ylim(.9E-4,6.5E-3)

# Make the plot square by adjusting the aspect ratio
ax1.set_box_aspect(0.8)

plt.tight_layout()
plt.show()

# Second figure: ZIW Optimization
fig2, ax2 = plt.subplots(figsize=(5, 4))

# Set log scales for ZIW plot
ax2.set_yscale('log')
ax2.set_xscale('log')

# Highlight regions where constraint violation is non-zero for ZIW
violation_regions_ziw = []
start = None

# Find continuous regions of constraint violations for ZIW
for i in range(len(ZIWopt)):
    if ZIWCV[i] > 0 and start is None:
        start = i
    elif ZIWCV[i] <= 0 and start is not None:
        violation_regions_ziw.append((start, i))
        start = None

# Add the last region if it extends to the end
if start is not None:
    violation_regions_ziw.append((start, len(ZIWopt)))

# Add dashed horizontal lines for initial and final values for ZIW
initial_value_ziw = ZIWopt[0]
final_value_ziw = ZIWopt[-1]
ax2.axhline(y=initial_value_ziw, color='k', linestyle='--', alpha=0.7, zorder=0)
ax2.axhline(y=final_value_ziw, color='k', linestyle='--', alpha=0.7, zorder=0)

# Plot ZIW optimization data
x_values = np.arange(1, len(ZIWopt) + 1)  # Start from 1 for log scale

# Plot base line first
ax2.plot(x_values, ZIWopt, 'g-', linewidth=2.5, zorder=2)

# Add markers with different colors based on constraint violation
for i, val in enumerate(ZIWopt):
    idx = i + 1  # Adjust for 1-based indexing in plot
    in_violation = any(start <= i < end for start, end in violation_regions_ziw)
    if in_violation:
        ax2.plot(idx, val, 'ro', markersize=4, zorder=3)
    else:
        ax2.plot(idx, val, 'o', markersize=4, markerfacecolor='green', markeredgecolor='darkgreen', zorder=3)

# Add violation regions
for start, end in violation_regions_ziw:
    ax2.axvspan(start + 1, end + 1, alpha=0.2, color='#FF000080', edgecolor=None)

# Add labels for initial and final values for ZIW, formatted for small numbers
ax2.text(len(ZIWopt)*0.9, initial_value_ziw*1.2, f'Initial: {initial_value_ziw:.5f}', 
         verticalalignment='bottom', horizontalalignment='right', fontsize=14,
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
ax2.text(len(ZIWopt)*0.9, final_value_ziw*0.8, f'Final: {final_value_ziw:.5f}', 
         verticalalignment='top', horizontalalignment='right', fontsize=14,
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

ax2.set_title('ZIW Optimization', fontsize=18, fontweight='bold')
ax2.set_xlabel('Iteration', fontsize=16)
ax2.set_ylabel(r'Loss $\langle\alpha_{back}\rangle/n_g^2$ [cm$^{-1}$]', fontsize=16)
ax2.set_xlim(1, len(ZIWopt))  # Start from 1 for log scale
ax2.grid(True, linestyle='--', alpha=0.3, which='both')
ax2.tick_params(axis='both', which='major', labelsize=13)

# Make the plot square by adjusting the aspect ratio
ax2.set_box_aspect(0.8)

plt.tight_layout()
plt.show()
# %%
