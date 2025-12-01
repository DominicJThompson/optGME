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
    print(gmeParams,params['gmax'])
    gme.run(**gmeParams)


    print('running alpha')
    cost = Backscatter(**params['cost'])
    alphas = []
    ngs = []
    gmeParams['numeig']-=1
    for i in range(nk):
        if i%10==0:
            print(i)
        gmeParams['kpoints']=np.vstack(([np.linspace(kmin,kmax,nk)[i]],[0]))
        gmeParams['verbose']=False
        gmeAlphacalc = legume.GuidedModeExp(phc,gmax=params['gmax'])
        gmeAlphacalc.run(**gmeParams)
        alphas.append(10**cost.cost(gmeAlphacalc,phc,params['mode']))
        ngs.append(np.abs(NG(gmeAlphacalc,0,params['mode'])))
    return(phc,gme,alphas,ngs)

def runNoiseSweep(xs,crystal,params):
    phc = crystal(xs)
    gmeParams = params['gmeParams'].copy()
    gmeParams['kpoints']=np.array(gmeParams['kpoints'])
    gmeParams['verbose']=False
    cost = Backscatter(**params['cost'])
    gme = legume.GuidedModeExp(phc,gmax=params['gmax'])
    gme.run(**gmeParams)
    
    lp_alphas = []
    for lp in np.logspace(np.log10(10),np.log10(300),500):
        cost.lp = lp
        lp_alphas.append(10**cost.cost(gme,phc,params['mode']))

    return(lp_alphas)
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

#plotBands(gme,ng,out[-1],color='#EE7733',plotback=False,index=2)
#plotBands(gmeOG,ngOG,out[-1],color='#0077BB',plotback=False,index=3)

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
    conversionFactor = (266E-9)*(266E-9*z)*(266E-9*phc.lattice.a2[1])
    vOG = (1/np.max(eOG*epsOG))*conversionFactor
    v = (1/np.max(e*eps))*conversionFactor
    print(r"Mode volume $(\lambda/n)^3$: (OG,Opt) ",
          vOG/(266E-9/gmeOG.freqs[kindex,mode]/np.sqrt(phcOG.layers[0].eps_b))**3, 
          v/(266E-9/gme.freqs[kindex,mode]/np.sqrt(phc.layers[0].eps_b))**3)
    print(r"Mode volume $\mu m^3$: (OG,Opt) ",vOG/(1E-6)**3, v/(1E-6)**3)

    #calculate the maximum percell enhancement
    ng_OG = np.abs(NG(gmeOG,kindex,mode))
    ng_ = np.abs(NG(gme,kindex,mode))
    cPF = 3*np.pi*(299792458)**2*266E-9/(phc.layers[0].eps_b**(3/2))
    conFac = 2*np.pi*299792458/params['cost']['a']/1e-9
    print(r"Maximum percell enhancement: (OG,Opt) ",
          cPF*ng_OG/(vOG)/(gmeOG.freqs[kindex,mode]*conFac)**2,
          cPF*ng_/(v)/(gme.freqs[kindex,mode]*conFac)**2)
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

#filedPlots(phc,phcOG,gme,gmeOG,out[-1])
filedPlots(phcW1,phcW1OG,gmeW1,gmeW1OG,out[-1])
#%%
#%%
def lossVng(phc,gme,alphas,ng,phcOG,gmeOG,alphasOG,ngOG,crystal):
    # Plot styling parameters
    STYLE = {
        'font_sizes': {
            'title': 30,
            'label': 32,
            'legend': 24,
            'marker': 30,
            'tick': 27,
            'annotation': 36
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
            'x_lim': (2.5, 200),
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
    if crystal == 'W1':
        STYLE['axes']['x_lim'] = (2, 15000)
        STYLE['axes']['y_lim'] = (2.5E-4,1E-2)
    plt.close('all')
    # Create figure with higher DPI
    plt.figure(dpi=STYLE['figure']['dpi'])

    alphasOG = alphasOG*np.sqrt(np.pi)
    alphas = alphas*np.sqrt(np.pi)
    
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
            {'x': ngOG[ks[3]-1]-3000, 'y': alphasOG[ks[3]-1]/266/1E-7+.0025, 
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
        plt.gca().set_xticks([10**1, 10**2, 10**3, 10**4])
        plt.gca().set_xticklabels(['$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    
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

lossVng(phcZIW,gmeZIW,np.array(alphasZIW),np.array(ngZIW),phcZIWOG,gmeZIWOG,np.array(alphasZIWOG),np.array(ngZIWOG),'ZIW')
lossVng(phcW1,gmeW1,np.array(alphasW1),np.array(ngW1),phcW1OG,gmeW1OG,np.array(alphasW1OG),np.array(ngW1OG),'W1')

#%%
def lossVfreq(phc,gme,alphas,ng,phcOG,gmeOG,alphasOG,ngOG,crystal):
    # Style parameters
    STYLE = {
        'font_sizes': {
            'title': 30,
            'label': 32,
            'legend': 24,
            'marker': 30,
            'tick': 27,
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
with open('/Users/dominic/Desktop/optGME/tests_old/media/ginds3/ziwBest.json','r') as file:
    out = json.load(file)

#phcZIWmid,gmeZIWmid,alphasZIWmid,ngZIWmid = runSims(np.array([out[6]['x_values']]),ZIW,out[-1])
phcZIW,gmeZIW,alphasZIW,ngZIW = runSims(np.array(out[-1]['result']['x']),ZIW,out[-1])
phcZIWOG,gmeZIWOG,alphasZIWOG,ngZIWOG = runSims(ZIWVars(),ZIW,out[-1]) 
#%%
with open('/Users/dominic/Desktop/optGME/tests_old/media/ginds3/W1Best.json','r') as file:
    out = json.load(file)

#phcW1mid,gmeW1mid,alphasW1mid,ngW1mid = runSims(np.array([out[5]['x_values']]),W1,out[-1])
phcW1,gmeW1,alphasW1,ngW1 = runSims(np.array(out[-1]['result']['x']),W1,out[-1])
phcW1OG,gmeW1OG,alphasW1OG,ngW1OG = runSims(W1Vars(),W1,out[-1]) 
#%%
alphas = np.array(alphasZIW)
alphasOG = np.array(alphasZIWOG)
ng = np.array(ngZIW)
ngOG = np.array(ngZIWOG)
alphasW1 = np.array(alphasW1)
alphasW1OG = np.array(alphasW1OG)
ngW1 = np.array(ngW1)
ngW1OG = np.array(ngW1OG)
#%%
with open('/home/dominic/Desktop/optGME/optGME/tests/media/ginds3/ziwBest.json','r') as file:
    out = json.load(file)
lp_alphasZIW = np.array(runNoiseSweep(np.array(out[-1]['result']['x']),ZIW,out[-1]))
#%%
with open('/home/dominic/Desktop/optGME/optGME/tests/media/ginds3/W1Best.json','r') as file:
    out = json.load(file)
lp_alphasW1 = np.array(runNoiseSweep(np.array(out[-1]['result']['x']),W1,out[-1]))

#%%
# Convert lp_alphasZIW and lp_alphasW1 to 2D arrays
noise_alphas_ziw = np.tile(lp_alphasZIW, (len(lp_alphasZIW), 1))
noise_alphas_w1 = np.tile(lp_alphasW1, (len(lp_alphasW1), 1))
sigmas = np.linspace(.5,7,noise_alphas_ziw.shape[0])

# Loop through and perform function on each row
for i,s in enumerate(sigmas):
    # Replace this with your desired function that operates on entire rows
    noise_alphas_ziw[i, :] = s**2*noise_alphas_ziw[i, :]/9

for i,s in enumerate(sigmas):
    # Replace this with your desired function that operates on entire rows
    noise_alphas_w1[i, :] = s**2*noise_alphas_w1[i, :]/9



# %%
# Font size parameters
import matplotlib
TITLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 15
CBAR_FONT_SIZE = 17
MAJOR_TICK_LENGTH = 6
MINOR_TICK_LENGTH = 3
TICK_WIDTH = 1.5

# Create figure with specific size for side-by-side square plots and high DPI
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.5), dpi=400, 
                               gridspec_kw={'wspace': 0.1, 'hspace': 0.0})
normed_noise_alphas_ziw = noise_alphas_ziw/266/1E-7
normed_noise_alphas_w1 = noise_alphas_w1/266/1E-7

# Find global min/max for consistent colorbar (using raw values, not log10)
vmin = min(normed_noise_alphas_ziw.min(), normed_noise_alphas_w1.min())
vmax = max(normed_noise_alphas_ziw.max(), normed_noise_alphas_w1.max())

# Define x and y axis values
x_vals = np.logspace(np.log10(10), np.log10(300), 500)
y_vals = sigmas

# Create log-scaled image for W1 (left) - using LogNorm for proper log scaling
im1 = ax1.imshow(normed_noise_alphas_w1, origin='lower', aspect='auto', 
                 cmap='plasma', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                 extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]])
ax1.set_ylabel(r'Roughness $\sigma$ [nm]', fontsize=LABEL_FONT_SIZE)
ax1.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE, 
                length=MAJOR_TICK_LENGTH, width=TICK_WIDTH)
ax1.tick_params(axis='both', which='minor', length=MINOR_TICK_LENGTH, width=TICK_WIDTH)
ax1.set_xscale('log')

# Add inset title for W1 with white backing
ax1.text(0.92, 0.92, 'W1', transform=ax1.transAxes, fontsize=TITLE_FONT_SIZE, 
         bbox=dict(boxstyle="round,pad=0.25", facecolor='white', alpha=0, edgecolor='none'),
         verticalalignment='top',horizontalalignment='right')

# Add red X with black outline at sigma=3, lp=40 for W1
ax1.plot(40, 3, 'rx', markersize=9, markeredgewidth=3, markeredgecolor='black', 
         transform=ax1.transData, zorder=10)

# Create log-scaled image for ZIW (right) - using LogNorm for proper log scaling
im2 = ax2.imshow(normed_noise_alphas_ziw, origin='lower', aspect='auto', 
                 cmap='plasma', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                 extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]])
ax2.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE,
                length=MAJOR_TICK_LENGTH, width=TICK_WIDTH)
ax2.tick_params(axis='both', which='minor', length=MINOR_TICK_LENGTH, width=TICK_WIDTH)
ax2.set_xscale('log')
# Remove y-axis ticks and labels for right plot since they're shared
ax2.tick_params(axis='y', which='both', left=False, labelleft=False)

# Add inset title for ZIW with white backing
ax2.text(0.92, 0.92, 'ZIW', transform=ax2.transAxes, fontsize=TITLE_FONT_SIZE, 
         bbox=dict(boxstyle="round,pad=0.25", facecolor='white', alpha=0, edgecolor='none'),
         verticalalignment='top',horizontalalignment='right')

# Add red X with black outline at sigma=3, lp=40 for ZIW
ax2.plot(40, 3, 'rx', markersize=9, markeredgewidth=3, markeredgecolor='black', 
         transform=ax2.transData, zorder=10)

# Add shared x-axis label below both plots in the middle
fig.text(0.55, -.1, r'Correlation Length $l_p$ [nm]', fontsize=LABEL_FONT_SIZE, 
         ha='center', va='center')

# Create chunkier colorbar above the plots with text and ticks above the colorbar
cbar_ax = fig.add_axes([0.135, .95, 0.75, 0.07])  # [left, bottom, width, height] - positioned above plots, made chunkier
cbar = plt.colorbar(im1, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r'$\tilde L$ [cm$^{-1}$]', fontsize=CBAR_FONT_SIZE, labelpad=10)
cbar.ax.tick_params(labelsize=TICK_FONT_SIZE, length=MAJOR_TICK_LENGTH, width=TICK_WIDTH, top=True, bottom=False)
cbar.ax.tick_params(axis='both', which='minor', length=MINOR_TICK_LENGTH, width=1.5)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.tick_top()
# Set specific tick locations and labels
cbar.set_ticks([.01,.001,.0001,.00001])
cbar.set_ticklabels([r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$'])
# Reverse the colorbar direction to go from large to small values left to right

# Adjust layout to prevent label cutoff
plt.tight_layout()
# %%
x_vals
# %%
ibm_colors={
    "blue": "#5596E6",
    "purple": "#9855D4",
    "magenta": "#E71D73",
    "orange": "#FF6F00",
    "yellow": "#FDD600",
}

def plotBands(gme,gmemid,gmefinal,ng,params,color='red',plotback=True,index=0,index2=1):
    blue = '#0077BB'
    orange = '#EE7733'
    green = '#66CC99'
    dark_blue = '#004488'
    dark_orange = '#CC6622'

    # Font size parameters
    TITLE_FONT_SIZE = 34
    LABEL_FONT_SIZE = 34
    ANNOTATION_FONT_SIZE = 40
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
    fig = plt.figure(figsize=(6.4, 5.8),dpi=400)

    # Main subplot (dispersion curve)
    ax2 = plt.gca()
    ax2.set_xlabel(r"Wavevector $\tilde k$",fontsize=LABEL_FONT_SIZE)
    ax2.set_ylabel(r"Frequency $\omega a / 2\pi c$",fontsize=LABEL_FONT_SIZE)
    ax2.set_xlim(0.25, 0.5)
    ax2.set_ylim(freqmin,freqmax)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)

    #plot frequency plot 
    ax2.fill_between(ks,ks,np.max(ks),color='darkGray',alpha=1) #light line
    ax2.fill_between(ks,gmefinal.freqs[:,mode-1],np.zeros_like(ks),color=ibm_colors['orange'],alpha=.7) #continums
    ax2.fill_between(ks,gme.freqs[:,mode-1],np.zeros_like(ks),color=ibm_colors['purple'],alpha=.7) #continums\
    ax2.plot(ks,gme.freqs[:,mode],color=ibm_colors['purple'],linewidth=3,zorder=2) #band of interest
    if plotback:
        ax2.plot(ks,gme.freqs[:,mode+1],color=ibm_colors['purple'],linewidth=3,linestyle='--') #other band
    else:
        ax2.plot(ks[50:],gme.freqs[50:,mode+1],color=ibm_colors['purple'],linewidth=3,linestyle='--') #other band
    ax2.plot(ks,gmemid.freqs[:,mode],color=ibm_colors['blue'],linewidth=3,zorder=2) #band of interest
    ax2.plot(ks,gmefinal.freqs[:,mode],color=ibm_colors['orange'],linewidth=3,zorder=2) #band of interest
    if plotback:
        ax2.plot(ks,gmefinal.freqs[:,mode+1],color=ibm_colors['orange'],linewidth=3,linestyle='--') #other band
    else:
        ax2.plot(ks[50:],gmefinal.freqs[50:,mode+1],color=ibm_colors['orange'],linewidth=3,linestyle='--') #other band

    se = [24,113,15,149,39,84,13,89]
    ax2.scatter([ks[se[2*index]],ks[se[2*index+1]]],[gme.freqs[se[2*index],mode],gme.freqs[se[2*index+1],mode]],s=150,color=ibm_colors['purple'],zorder=3,edgecolor='black', linewidth=1.5)
    ax2.scatter([ks[se[2*index2]],ks[se[2*index2+1]]],[gmefinal.freqs[se[2*index2],mode],gmefinal.freqs[se[2*index2+1],mode]],s=150,color=ibm_colors['orange'],zorder=3,edgecolor='black', linewidth=1.5)

    # Add text labels with slight offset for better visibility
    if index==0:a,b = r'$\mathbf{a}^\prime$',r'$\mathbf{b}^\prime$'
    elif index==1:a,b = r'$\mathbf{a}$',r'$\mathbf{b}$'
    elif index==2:a,b = r'$\mathbf{c}^\prime$',r'$\mathbf{d}^\prime$'
    elif index==3:a,b = r'$\mathbf{c}$',r'$\mathbf{d}$'
    if index>=2:
        xytext1 = (5, -25)
        xytext2 = (15, -5)
    else:
        xytext1 = (5, 5)
        xytext2 = (-25, 10)
    ax2.annotate(a, 
                 (ks[se[2*index]], gme.freqs[se[2*index], mode]),
                 xytext=xytext1,
                 textcoords='offset points',
                 color=ibm_colors['purple'],
                 fontsize=ANNOTATION_FONT_SIZE) 
    ax2.annotate(b, 
                 (ks[se[2*index+1]], gme.freqs[se[2*index+1], mode]),
                 xytext=xytext2,
                 textcoords='offset points',
                 color=ibm_colors['purple'],
                 fontsize=ANNOTATION_FONT_SIZE)
    
    # Add vertical red dashed line at specified value
    ax2.axvline(x=0.33333, color=ibm_colors['magenta'], linestyle='--', alpha=0.8, linewidth=3)

    # Add text labels with slight offset for better visibility
    if index2==0:a,b = r'$\mathbf{a}^\prime$',r'$\mathbf{b}^\prime$'
    elif index2==1:a,b = r'$\mathbf{a}$',r'$\mathbf{b}$'
    elif index2==2:a,b = r'$\mathbf{c}^\prime$',r'$\mathbf{d}^\prime$'
    elif index2==3:a,b = r'$\mathbf{c}$',r'$\mathbf{d}$'
    if index2>=2:
        xytext1 = (-8, 12)
        xytext2 = (5, 15)
    else:
        xytext1 = (-25, -40)
        xytext2 = (-15, -40)
    ax2.annotate(a, 
                 (ks[se[2*index2]], gmefinal.freqs[se[2*index2], mode]),
                 xytext=xytext1,
                 textcoords='offset points',
                 color=ibm_colors['orange'],
                 fontsize=ANNOTATION_FONT_SIZE) 
    ax2.annotate(b, 
                 (ks[se[2*index2+1]], gmefinal.freqs[se[2*index2+1], mode]),
                 xytext=xytext2,
                 textcoords='offset points',
                 color=ibm_colors['orange'],
                 fontsize=ANNOTATION_FONT_SIZE)

    # Add inset title for ZIW with white backing
    ax2.text(0.95, 0.95, 'W1', transform=ax2.transAxes, fontsize=TITLE_FONT_SIZE, 
             bbox=dict(boxstyle="round,pad=0.25", facecolor='white', alpha=0.7, edgecolor='none'),
             verticalalignment='top',horizontalalignment='right')

    
    plt.show()

plotBands(gmeZIWOG,gmeZIWmid,gmeZIW,ngZIW,out[-1],color='#EE7733',plotback=False,index=3,index2=2)
plotBands(gmeW1OG,gmeW1mid,gmeW1,ngW1,out[-1],color='#0077BB',plotback=True,index=1,index2=0)
# %%

def filedPlots(phc,phcMid,phcOG,gme,gmeMid,gmeOG,params):
    # Font size variables for consistent styling
    TITLE_SIZE = 28
    LABEL_SIZE = 26
    TICK_SIZE = 20
    LEGEND_SIZE = 24
    COLORBAR_LABEL_SIZE = 32
    COLORBAR_TICK_SIZE = 28
    
    # Set up variables
    ylim = 8*np.sqrt(3)/2
    ys = np.linspace(-ylim/2, ylim/2, 300)
    mode = params['mode']
    kindex = np.abs(gme.kpoints[0,:] - params['gmeParams']['kpoints'][0][0]).argmin()
    # kindex = 190
    z = phc.layers[0].d/2

    # Get field of original crystal
    fieldsOG, _, _ = gmeOG.get_field_xy('E', kindex, mode, z, ygrid=ys, component='xyz')
    eabsOG = np.sqrt(np.abs(np.conj(fieldsOG['x'])*fieldsOG['x'] + np.conj(fieldsOG['y'])*fieldsOG['y'] + np.conj(fieldsOG['z'])*fieldsOG['z']))
    fields, _, _ = gme.get_field_xy('E', kindex, mode, z, ygrid=ys, component='xyz')
    eabs = np.sqrt(np.abs(np.conj(fields['x'])*fields['x'] + np.conj(fields['y'])*fields['y'] + np.conj(fields['z'])*fields['z']))
    fieldsMid, _, _ = gmeMid.get_field_xy('E', kindex, mode, z, ygrid=ys, component='xyz')
    eabsMid = np.sqrt(np.abs(np.conj(fieldsMid['x'])*fieldsMid['x'] + np.conj(fieldsMid['y'])*fieldsMid['y'] + np.conj(fieldsMid['z'])*fieldsMid['z']))
    maxF = np.max([np.max(eabsOG), np.max(eabs), np.max(eabsMid)])
    
    # Optional parameters to control the field view|
    x_offset = 0  # Adjust to shift the center of the view left or right
    x_crop = 0    # Adjust to crop from the right side (0 for no cropping)
    
    # Calculate the actual x limits based on the parameters
    x_min = -ylim/2 + x_offset
    x_max = ylim/2 + x_offset - x_crop
    
    # Create a figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3, 4.8),dpi=400)
    
    # Plot optimized field on left subplot (vertical orientation)
    cax1 = ax1.imshow(eabs, extent=[-.5, .5, -ylim/2, ylim/2], cmap='plasma', vmax=maxF, vmin=0, zorder=1)
    circles1 = [Circle((s.x_cent, s.y_cent), s.r, edgecolor='white', facecolor='none', linewidth=3, zorder=2) for s in phc.layers[0].shapes]
    cirlcesArround1 = [Circle((0, 0), s.r, edgecolor='white', facecolor='none', linewidth=3, zorder=2) for s in phc.layers[0].shapes]
    for c, ca in zip(circles1, cirlcesArround1):
        ax1.add_patch(c)
        ca.center = (c.center[0]-np.sign(c.center[0]), c.center[1])
        ax1.add_patch(ca)
    ax1.set_xlim(-.5, .5)
    ax1.set_ylim(x_min, x_max)
    ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                   labelbottom=False, labelleft=False)
    
    # Plot mid field on center subplot (vertical orientation)
    cax2 = ax2.imshow(eabsMid, extent=[-.5, .5, -ylim/2, ylim/2], cmap='plasma', vmax=maxF, vmin=0, zorder=1)
    circles2 = [Circle((s.x_cent, s.y_cent), s.r, edgecolor='white', facecolor='none', linewidth=3, zorder=2) for s in phcMid.layers[0].shapes]
    cirlcesArround2 = [Circle((0, 0), s.r, edgecolor='white', facecolor='none', linewidth=3, zorder=2) for s in phcMid.layers[0].shapes]
    for c, ca in zip(circles2, cirlcesArround2):
        ax2.add_patch(c)
        ca.center = (c.center[0]-np.sign(c.center[0]), c.center[1])
        ax2.add_patch(ca)
    ax2.set_xlim(-.5, .5)
    ax2.set_ylim(x_min, x_max)
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                   labelbottom=False, labelleft=False)
    
    # Plot original field on right subplot (vertical orientation)
    cax3 = ax3.imshow(eabsOG, extent=[-.5, .5, -ylim/2, ylim/2], cmap='plasma', vmax=maxF, vmin=0, zorder=1)
    circles3 = [Circle((s.x_cent, s.y_cent), s.r, edgecolor='white', facecolor='none', linewidth=3, zorder=2) for s in phcOG.layers[0].shapes]
    cirlcesArround3 = [Circle((0, 0), s.r, edgecolor='white', facecolor='none', linewidth=3, zorder=2) for s in phcOG.layers[0].shapes]
    for c, ca in zip(circles3, cirlcesArround3):
        ax3.add_patch(c)
        ca.center = (c.center[0]-np.sign(c.center[0]), c.center[1])
        ax3.add_patch(ca)
    ax3.set_xlim(-.5, .5)
    ax3.set_ylim(x_min, x_max)
    ax3.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                   labelbottom=False, labelleft=False)
    
    # Hyperparameters to control box positioning and appearance
    box_margin = 0.02      # Margin between subplot and box edge
    box_linewidth = 3      # Line width of the boxes
    box_colors = [ibm_colors['purple'], ibm_colors['blue'], ibm_colors['orange']]  # Colors for each subplot box
    
    # Calculate box positions based on subplot layout
    subplot_width = 0.2   # Width of each subplot (adjust if needed)
    subplot_spacing = 0.0725 # Spacing between subplots
    box_height = 0.77       # Height of boxes
    box_y_offset = 0.11    # Vertical offset from bottom
    
    # Calculate x positions for each box
    box_x_positions = [
        0.14,                                    # Left subplot
        0.14 + subplot_width + subplot_spacing,  # Center subplot  
        0.14 + 2*(subplot_width + subplot_spacing)  # Right subplot
    ]
    
    # Add boxes around each subplot using calculated coordinates
    for i, (x_pos, color) in enumerate(zip(box_x_positions, box_colors)):
        fig.add_artist(plt.Rectangle((x_pos - box_margin, box_y_offset - box_margin), 
                                   subplot_width + 2*box_margin, box_height + 2*box_margin,
                                   facecolor='none', edgecolor=color, 
                                   linewidth=box_linewidth, zorder=10))
    
    # Add a horizontal colorbar at the bottom
    cbar_ax = fig.add_axes([0.115, 0.05, 0.8, 0.03])  # Position for horizontal colorbar at bottom
    cbar = fig.colorbar(cax1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r"$|\mathbf{e}_{\tilde k=0.33}|$ [a$^{-\frac{3}{2}}$]", fontsize=COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    # Set specific tick locations - you can modify these values
    cbar.set_ticks([0, .25, .5])
    # Format the tick labels to show 0 instead of 0.0 while keeping other decimals
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: '0' if x == 0 else f'{x:.2f}'))

    plt.show()

#filedPlots(phcZIWOG,phcZIWmid,phcZIW,gmeZIWOG,gmeZIWmid,gmeZIW,out[-1])
filedPlots(phcW1OG,phcW1mid,phcW1,gmeW1OG,gmeW1mid,gmeW1,out[-1])
# %%
## Mode volume calculation

def modeVolume(gme,phc,ngs,k):
    eps = np.real(legume.viz.eps_xy(phc,Nx=20,Ny=600,plot=False))
    f,_,_ = gme.get_field_xy('E',k,20,phc.layers[0].d/2,Nx=20,Ny=600,component='xyz')
    eabs = np.abs(np.conj(f['x'])*f['x'] + np.conj(f['y'])*f['y'] + np.conj(f['z'])*f['z'])
    v = (1/np.max(eps*eabs))*(266E-9)**3
    print("mode volume micro meters cubed: ",v/(1E-6)**3)

    wavelength = 266E-9/gme.freqs[k,20]
    print("mode volume wavelength of index: ",v/(wavelength/np.sqrt(12))**3)

    #purcell enhancement calculation
    purcell = eabs*3*np.pi*(299792458)**2*266E-9*np.abs(ngs[k])/(gme.freqs[k,20]*2*np.pi*299792458/266E-9)**2/np.sqrt(12)/(266E-9)**3
    print("purcell enhancement: ",np.max(purcell))

print("W1 Og")
modeVolume(gmeW1OG,phcW1OG,ngW1OG,50)
print("W1 Final")
modeVolume(gmeW1,phcW1,ngW1,50)
print("ZIW Og")
modeVolume(gmeZIWOG,phcZIWOG,ngZIWOG,50)
print("ZIW Final")
modeVolume(gmeZIW,phcZIW,ngZIW,50)
# %%
print(ngW1OG[50])
print(ngW1[50])
print(ngZIWOG[50])
print(ngZIW[50])

# %%
print((np.log10(e^alphasW1OG[50]/10)/np.log10(alphasW1[50]/10)))
print(alphasW1[50])
print(alphasZIWOG[50]/alphasZIW[50])
print(alphasZIW[50])
# %%
print(np.log10(np.exp(-alphasW1OG[50]*ngW1OG[50]**2/266E-7))/np.log10(np.exp(-alphasW1[50]*ngW1[50]**2/266E-7)))
# %%
print(10**np.log10(alphasW1OG[50])/10**np.log10(alphasW1[50]))
# %%
print(alphasW1OG[50]/alphasW1[50])
# %%
