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
plt.rcParams.update({'font.size':20})
nk=200


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

def runSims(xs,params):
    phc = W1(xs)

    kmin,kmax = .5*np.pi,np.pi
    gmeParams = params['gmeParams'].copy()
    gmeParams['kpoints']=np.vstack((np.linspace(kmin,kmax,nk),np.zeros(nk)))
    gmeParams['verbose']=True
    gmeParams['numeig']+=1
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

def plotBands(gme,ng,params,color='red'):

    #missilanius variables needed
    conFac = 1e-12*299792458/params['cost']['a']/1e-9
    freqmin, freqmax = .243, .295
    mode = params['mode']
    kindex = np.abs(gme.kpoints[0,:] - params['gmeParams']['kpoints'][0][0]).argmin()
    print('kindex ',kindex)
    print('ng ',ng[kindex])
    # Generate some sample data
    ks = np.linspace(0.25, 0.5, nk)

    # Create figure and define gridspec layout
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(3, 1, height_ratios=[1, 0.1, 2], hspace=0.05)

    # Upper subplot (group index vs. kx)
    ax1 = fig.add_subplot(gs[0])
    ax1.set_ylabel(r"$n_g$")
    ax1.set_xticks([])  # Hide x-axis ticks
    ax1.set_xlim(0.25, 0.5)
    ax1.set_ylim(0, 20)

    # Lower subplot (dispersion curve)
    ax2 = fig.add_subplot(gs[2])
    ax2.set_xlabel(r"$k_x 2\pi/a$")
    ax2.set_ylabel(r"$\omega a / 2\pi c$")
    ax2.set_xlim(0.25, 0.5)
    ax2.set_ylim(freqmin,freqmax)

    #add twin axis
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel("Frequency [THz]")
    ax2_twin.set_ylim(freqmin*conFac,freqmax*conFac)

    #plot the ng
    ax1.plot(ks,ng,linewidth=2,color='darkviolet')
    ax1.scatter(ks[kindex],ng[kindex],s=200,color=color,zorder=3) #optomized point

    #plot frequency plot 
    ax2.fill_between(ks,ks,np.max(ks),color='darkGray',alpha=1) #light line
    ax2.fill_between(ks,gme.freqs[:,mode-1],np.zeros_like(ks),color='navy',alpha=.7) #continums
    ax2.plot(ks,gme.freqs[:,mode],color='darkviolet',linewidth=2,zorder=2) #band of interest
    ax2.plot(ks,gme.freqs[:,mode+1],color='darkviolet',linewidth=2,linestyle='--') #other band
    ax2.scatter(ks[kindex],gme.freqs[kindex,mode],s=200,color=color,zorder=3) #optomized point

    #find the bandwidth in frequncy
    intersect = np.where(np.sign((gme.freqs[:,mode]-ks)[:-1]) != np.sign((gme.freqs[:,mode]-ks))[1:])[0]
    bandMax = min(np.max(gme.freqs[:,mode]),np.min(gme.freqs[:,mode+1]),gme.freqs[intersect,mode])
    bandMin = np.max(np.hstack((gme.freqs[:,mode-1],np.min(gme.freqs[:,mode]))))
    #if color=='red':
    #    bandMin+=.0015
    ax2.fill_between(ks,np.ones_like(ks)*bandMax,np.ones_like(ks)*bandMin,color='cyan',alpha=.5,zorder=0)

    # Show plot
    plt.show()

def alphaImprove(alpha,alphaOG,params):
    ks = np.linspace(0.25, 0.5, nk)
    kindex = np.abs(ks-params['gmeParams']['kpoints'][0][0]/np.pi/2).argmin()

    print('lossOpt ',alpha[kindex]/266/1E-7)
    print('lossOG ',alphaOG[kindex]/266/1E-7)

    plt.plot(ks,np.array(alpha)/266/1E-7,'r',linewidth=2,label='Optimized')
    plt.plot(ks,np.array(alphaOG)/266/1E-7,'g',linewidth=2,label='Original')
    plt.scatter(ks[kindex],alpha[kindex]/266/1E-7,s=200,color='r')
    plt.scatter(ks[kindex],alphaOG[kindex]/266/1E-7,s=200,color='g')
    plt.yscale('log')
    plt.ylim(3E-4,3E-2)
    plt.legend()
    plt.ylabel(r"$\langle\alpha_{back}\rangle/n_g^2$ [cm$^{-1}$]")
    plt.xlabel(r"$k_x 2\pi/a$")
    plt.xlim(.25,.5)
    plt.show()

def filedPlots(phc,phcOG,gme,gmeOG,params):
    #set up variables
    ylim= 8*np.sqrt(3)/2
    ys = np.linspace(-ylim/2,ylim/2,300)
    mode = params['mode']
    kindex = np.abs(gme.kpoints[0,:] - params['gmeParams']['kpoints'][0][0]).argmin()
    z = phc.layers[0].d/2

    #get field of original crystal
    fieldsOG,_,_ = gmeOG.get_field_xy('E',kindex,mode,z,ygrid=ys,component='xyz')
    eabsOG = np.abs(np.conj(fieldsOG['x'])*fieldsOG['x']+np.conj(fieldsOG['y'])*fieldsOG['y']+np.conj(fieldsOG['z'])*fieldsOG['z'])
    fields,_,_ = gme.get_field_xy('E',kindex,mode,z,ygrid=ys,component='xyz')
    eabs = np.abs(np.conj(fields['x'])*fields['x']+np.conj(fields['y'])*fields['y']+np.conj(fields['z'])*fields['z'])
    maxF = np.max([np.max(eabsOG),np.max(eabs)])

    #plot the fields
    fix,ax = plt.subplots()
    cax = ax.imshow(eabs.T,extent=[-ylim/2,ylim/2,.5,-.5],cmap='plasma',vmax=maxF,vmin=0)
    circles = [Circle((s.y_cent,s.x_cent),s.r,edgecolor='white',facecolor='none',linewidth=3) for s in phc.layers[0].shapes]
    cirlcesArround = [Circle((0,0),s.r,edgecolor='white',facecolor='none',linewidth=3) for s in phc.layers[0].shapes]
    for c,ca in zip(circles,cirlcesArround):
        plt.gca().add_patch(c)
        ca.center = (c.center[0],c.center[1]-np.sign(c.center[1]))
        plt.gca().add_patch(ca)
    plt.ylim(-.5,.5)
    plt.xlim(-ylim/2,ylim/2)
    ax.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
    plt.show()

    #plot the fields for the original
    fix,ax = plt.subplots()
    cax = ax.imshow(eabsOG.T,extent=[-ylim/2,ylim/2,.5,-.5],cmap='plasma',vmax=maxF,vmin=0)
    circles = [Circle((s.y_cent,s.x_cent),s.r,edgecolor='white',facecolor='none',linewidth=3) for s in phcOG.layers[0].shapes]
    cirlcesArround = [Circle((0,0),s.r,edgecolor='white',facecolor='none',linewidth=3) for s in phcOG.layers[0].shapes]
    for c,ca in zip(circles,cirlcesArround):
        plt.gca().add_patch(c)
        ca.center = (c.center[0],c.center[1]-np.sign(c.center[1]))
        plt.gca().add_patch(ca)
    plt.ylim(-.5,.5)
    plt.xlim(-ylim/2,ylim/2)
    ax.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
    plt.show()




#%%
with open('/Users/dominic/Desktop/optGME/tests/media/nonlinopt32.json','r') as file:
    out = json.load(file)

phc,gme,alphas,ng = runSims(np.array(out[-1]['result']['x']),out[-1])
phcOG,gmeOG,alphasOG,ngOG = runSims(W1Vars(),out[-1]) 

#%%
plotBands(gme,ng,out[-1])
plotBands(gmeOG,ngOG,out[-1],color='green')

#%%
    
alphaImprove(alphas,alphasOG,out[-1])
#%%
filedPlots(phc,phcOG,gme,gmeOG,out[-1])

#%%
filedPlots(phc,phcOG,gme,gmeOG,out[-1])
# %%
print(np.log10(alphas[66]))
print(np.log10(alphasOG[66]))
print(alphas[66]/266/1E-7/(alphasOG[66]/266/1E-7))
# %% 
#%%
0.0004673460766618901*7.716625423762982**2
# %%
0.005045916208333005*6.822117123068426**2
# %%
