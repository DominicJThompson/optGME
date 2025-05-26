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
plt.rcParams.update({'font.size':20})

# Disable LaTeX but use Computer Modern fonts
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern math fonts
mpl.rcParams['font.family'] = 'STIXGeneral'  # Use STIX fonts (similar to Computer Modern)

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

def plotBands(gme,ng,params,color='red',plotback=True,index=0):

    # Font size parameters
    TITLE_FONT_SIZE = 34
    LABEL_FONT_SIZE = 34
    ANNOTATION_FONT_SIZE = 34
    TICK_FONT_SIZE = 28
    
    #missilanius variables needed
    conFac = 1e-12*299792458/params['cost']['a']/1e-9
    if index==0 or index==1:
        freqmin, freqmax = .248, .282
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
    ax2.set_xlabel(r"Wavevector $k_x 2\pi/a$",fontsize=LABEL_FONT_SIZE)
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
    if index==0:a,b = r'$a^\prime$',r'$b^\prime$'
    elif index==1:a,b = r'$a$',r'$b$'
    elif index==2:a,b = r'$c^\prime$',r'$d^\prime$'
    elif index==3:a,b = r'$c$',r'$d$'
    ax2.annotate(a, 
                 (ks[se[2*index]], gme.freqs[se[2*index], mode]),
                 xytext=(5, 5),
                 textcoords='offset points',
                 color=color,
                 fontsize=ANNOTATION_FONT_SIZE) 
    ax2.annotate(b, 
                 (ks[se[2*index+1]], gme.freqs[se[2*index+1], mode]),
                 xytext=(+35, -30),
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
    COLORBAR_LABEL_SIZE = 28
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
    print(1/np.max(eabsOG), 1/np.max(eabs))

    # Create a figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 3))
    
    # Optional parameters to control the field view
    x_offset = 0  # Adjust to shift the center of the view left or right
    x_crop = 0    # Adjust to crop from the right side (0 for no cropping)
    
    # Calculate the actual x limits based on the parameters
    x_min = -ylim/2 + x_offset
    x_max = ylim/2 + x_offset - x_crop
    
    # Plot original field on top subplot
    cax1 = ax1.imshow(eabsOG.T, extent=[-ylim/2, ylim/2, -.5, .5], cmap='plasma', vmax=maxF, vmin=0)
    ax1.set_title('Original', fontsize=TITLE_SIZE)
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
    ax2.set_title('Optimized', fontsize=TITLE_SIZE)
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
    cbar.set_label(r"$|\mathbf{E}|$", rotation=0, labelpad=20, fontsize=COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make room for colorbar
    plt.show()

filedPlots(phc,phcOG,gme,gmeOG,out[-1])
filedPlots(phcW1,phcW1OG,gmeW1,gmeW1OG,out[-1])
#%%
def lossVng(phc,gme,alphas,ng,phcOG,gmeOG,alphasOG,ngOG):
    # Font size parameters
    TITLE_SIZE = 30
    LABEL_SIZE = 30
    LEGEND_SIZE = 28
    MARKER_SIZE = 30
    TICK_SIZE = 26
    
    orange = '#EE7733'
    blue = '#0077BB'

    ks = [24,113,15,149]
    #ks = [39,84,13,89]

    plt.plot(ngOG[ks[2]:],alphasOG[ks[2]:]/266/1E-7,color=blue,label='Original',linewidth=3)
    plt.plot(ng[ks[0]:],alphas[ks[0]:]/266/1E-7,color=orange,label='Optimized',linewidth=3)
    kindex = 50
    # Add letters to mark the beginning and ending of each line
    plt.text(ng[ks[0]]+4, alphas[ks[0]]/266/1E-7, r'$a^\prime$', color=orange, fontsize=24, fontweight='bold', ha='right', va='bottom')
    plt.text(ng[ks[1]-1]+4, alphas[ks[1]-1]/266/1E-7+.0015, r'$b^\prime$', color=orange, fontsize=24, fontweight='bold', ha='left', va='top')
    
    plt.text(ngOG[ks[2]]+2, alphasOG[ks[2]]/266/1E-7-.0006, r'$a$', color=blue, fontsize=24, fontweight='bold', ha='right', va='bottom')
    plt.text(ngOG[ks[3]-1]+100, alphasOG[ks[3]-1]/266/1E-7+.0015, r'$b$', color=blue, fontsize=24, fontweight='bold', ha='left', va='top')

    #plt.text(ng[ks[0]]+2.5, alphas[ks[0]]/266/1E-7-.001, r'$c^\prime$', color=orange, fontsize=MARKER_SIZE, fontweight='bold', ha='right', va='bottom')
    #plt.text(ng[ks[1]-1]-1, alphas[ks[1]-1]/266/1E-7+.003, r'$d^\prime$', color=orange, fontsize=MARKER_SIZE, fontweight='bold', ha='left', va='top')
    
    #plt.text(ngOG[ks[2]]+25, alphasOG[ks[2]]/266/1E-7+.0005, r'$c$', color=blue, fontsize=MARKER_SIZE, fontweight='bold', ha='right', va='bottom')
    #plt.text(ngOG[ks[3]-1]-1.25, alphasOG[ks[3]-1]/266/1E-7+.008, r'$d$', color=blue, fontsize=MARKER_SIZE, fontweight='bold', ha='left', va='top')

    plt.scatter([ng[ks[0]],ng[ks[1]-1]],[alphas[ks[0]]/266/1E-7,alphas[ks[1]-1]/266/1E-7],s=150,color=orange,zorder=3,edgecolor='black', linewidth=1.5)
    plt.scatter([ngOG[ks[2]],ngOG[ks[3]-1]],[alphasOG[ks[2]]/266/1E-7,alphasOG[ks[3]-1]/266/1E-7],s=150,color=blue,zorder=3,edgecolor='black', linewidth=1.5)
    #plt.ylim(1E-3,1E-1)
    #plt.xlim(3.1,200)

    plt.ylim(2.5E-4,1E-2)
    plt.xlim(3.1,10000)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(.35, .4), frameon=False, fontsize=LEGEND_SIZE)
    plt.xlabel(r"Group Index $n_g$", fontsize=LABEL_SIZE)
    plt.ylabel(r"Loss $\langle\alpha_{\text{back}}\rangle/n_g^2$ [cm$^{-1}$]", fontsize=LABEL_SIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    plt.show()

#lossVng(phc,gme,alphas,ng,phcOG,gmeOG,alphasOG,ngOG)
lossVng(phcW1,gmeW1,alphasW1,ngW1,phcW1OG,gmeW1OG,alphasW1OG,ngW1OG)

#%%
def lossVfreq(phc,gme,alphas,ng,phcOG,gmeOG,alphasOG,ngOG):
    # Font size parameters
    TITLE_SIZE = 30
    LABEL_SIZE = 30
    LEGEND_SIZE = 28
    MARKER_SIZE = 30
    TICK_SIZE = 26
    TEXT_SIZE = 30
    
    orange = '#EE7733'
    blue = '#0077BB'

    #ks = [24,113,15,149]
    ks = [39,84,13,89]

    plt.plot(gmeOG.freqs[ks[2]:,20],alphasOG[ks[2]:]/266/1E-7,color=blue,label='Original',linewidth=3)
    plt.plot(gme.freqs[ks[0]:,20],alphas[ks[0]:]/266/1E-7,color=orange,label='Optimized',linewidth=3)
    kindex = 50


    # Add text labels (W1 labewls)
    #plt.text(gme.freqs[ks[0],20]+.001, alphas[ks[0]]/266/1E-7, r'$a^\prime$', color=orange, fontsize=TEXT_SIZE, fontweight='bold', ha='center', va='bottom')
    #plt.text(gme.freqs[ks[1],20]+.0004, alphas[ks[1]-1]/266/1E-7+.003, r'$b^\prime$', color=orange, fontsize=TEXT_SIZE, fontweight='bold', ha='left', va='top')
    
    #plt.text(gmeOG.freqs[ks[2],20], alphasOG[ks[2]]/266/1E-7+.0001, r'$a$', color=blue, fontsize=TEXT_SIZE, fontweight='bold', ha='right', va='bottom')
    #plt.text(gmeOG.freqs[ks[3],20]+.0005, alphasOG[ks[3]]/266/1E-7+.0003, r'$b$', color=blue, fontsize=TEXT_SIZE, fontweight='bold', ha='left', va='top')
    
    #ZIW labels
    plt.text(gme.freqs[ks[0],20]-.00075, alphas[ks[0]]/266/1E-7, r'$c^\prime$', color=orange, fontsize=TEXT_SIZE, fontweight='bold', ha='center', va='bottom')
    plt.text(gme.freqs[ks[1],20], alphas[ks[1]-1]/266/1E-7+.003, r'$d^\prime$', color=orange, fontsize=TEXT_SIZE, fontweight='bold', ha='left', va='top')
    
    plt.text(gmeOG.freqs[ks[2],20]-.001, alphasOG[ks[2]]/266/1E-7, r'$c$', color=blue, fontsize=TEXT_SIZE, fontweight='bold', ha='right', va='bottom')
    plt.text(gmeOG.freqs[ks[3],20], alphasOG[ks[3]]/266/1E-7+.015, r'$d$', color=blue, fontsize=TEXT_SIZE, fontweight='bold', ha='left', va='top')


    # Add points at each location
    plt.scatter(gme.freqs[ks[0],20], alphas[ks[0]]/266/1E-7, color=orange, s=120, zorder=10, edgecolor='black', linewidth=1.5)
    plt.scatter(gme.freqs[ks[1],20], alphas[ks[1]-1]/266/1E-7, color=orange, s=120, zorder=10, edgecolor='black', linewidth=1.5)
    
    plt.scatter(gmeOG.freqs[ks[2],20], alphasOG[ks[2]]/266/1E-7, color=blue, s=120, zorder=10, edgecolor='black', linewidth=1.5)
    plt.scatter(gmeOG.freqs[ks[3],20], alphasOG[ks[3]]/266/1E-7, color=blue, s=120, zorder=10, edgecolor='black', linewidth=1.5)

    #W1 limits
    #plt.ylim(1E-4,1E-2)  # Increased upper limit to provide more space at the top

    #ZIW limits
    plt.ylim(.8E-3,1E-1)

    plt.yscale('log')
    # Remove minor ticks for log scale
    #plt.gca().yaxis.set_minor_locator(plt.NullLocator())
    plt.legend(bbox_to_anchor=(.7, .37), frameon=False, fontsize=LEGEND_SIZE)
    plt.xlabel(r"Frequency $\omega a / 2\pi c$", fontsize=LABEL_SIZE)
    plt.ylabel(r"Loss $\langle\alpha_{\text{back}}\rangle/n_g^2$ [cm$^{-1}$]", fontsize=LABEL_SIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    plt.show()

#lossVfreq(phcW1,gmeW1,alphasW1,ngW1,phcW1OG,gmeW1OG,alphasW1OG,ngW1OG)
lossVfreq(phc,gme,alphas,ng,phcOG,gmeOG,alphasOG,ngOG)

#%%
#%%
with open('/Users/dominic/Desktop/optGME/tests/media/ginds3/ziwBest.json','r') as file:
    out = json.load(file)

phc,gme,alphas,ng = runSims(np.array(out[-1]['result']['x']),ZIW,out[-1])
phcOG,gmeOG,alphasOG,ngOG = runSims(ZIWVars(),ZIW,out[-1]) 
#%%
with open('/Users/dominic/Desktop/optGME/tests/media/ginds3/W1Best.json','r') as file:
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
