#%%
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import legume 
legume.set_backend('autograd')
import numpy as np
import autograd.numpy as npa
from autograd import grad
import time
import matplotlib.pyplot as plt
import optomization
import itertools


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


def worker_function(input):

    #make directory to save files
    with open(input['path'], "w") as f:
        pass

    cost = optomization.Backscatter()
    ks = npa.linspace(npa.pi*.5,npa.pi,150)
    vars = W1Vars(key=input['key'])

    gmeParams = {'verbose':False,'numeig':21,'compute_im':False,'kpoints':npa.array([[int(ks[input['index']])],[0]]),'gmode_inds':[0,2,4]}
    
    #define constraints
    manager = optomization.ConstraintManager(x0=vars,
                                            numberHoles=3,
                                            crystal=W1,
                                            phcParams={},
                                            gmeParams=gmeParams,
                                            gmax=4.01,
                                            mode=20,
                                            gmode_inds=[0])
    
    manager.add_inside_unit_cell('Inside',.5)
    manager.add_rad_bound('minimumRadius',.15,.42)
    manager.add_min_dist('minDist',40/266,3)
    manager.add_gme_constrs_complex('gme_constrs_ng',minFreq=.26,maxFreq=.28,minNG=input['ng']*.9,maxNG=input['ng']*1.1,ksBefore=[float(ks[input['indexB'][0]]),float(ks[input['indexB'][1]])],ksAfter=[float(ks[input['indexA'][0]]),float(ks[input['indexA'][1]])],bandwidth=.002,slope='down')
    #run minimization
    minim = optomization.TrustConstr(vars,W1,cost,mode=20,maxiter=400,gmeParams=gmeParams,constraints=manager,path=input['path'],initial_tr_radius=.1,xtol=1e-4)
    minim.minimize()
    minim.save(input['path'])
#%%
if __name__=='__main__':
    #print working directory
    ngs = [30,10,5]
    indexs = [62,45,30]    
    indexsA = [[85,120],[82,120],[75,120]]
    indexsB = [[10,45],[10,30],[10,20]]
    for j in np.arange(50):
        for i,ng in enumerate(ngs):
            input = {'path':f"./media/w1/sweepNG/ng{ng}_{j}.json",'ng':ng,'key':j,'index':indexs[i],'indexB':indexsB[i],'indexA':indexsA[i]}
            minim = worker_function(input)  # Compute the result
# %%
