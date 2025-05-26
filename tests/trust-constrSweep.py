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
    ks = npa.linspace(npa.pi*.5,npa.pi,25)

    gmeParams = {'verbose':False,'numeig':21,'compute_im':False,'kpoints':npa.array([[float(ks[8])],[0]])}
    vars = W1Vars(key=input['key'])
    
    #define constraints
    manager = optomization.ConstraintManager(x0=vars,
                                            numberHoles=3,
                                            crystal=W1,
                                            phcParams={},
                                            gmeParams=gmeParams,
                                            gmax=2.01,
                                            mode=20)
    
    manager.add_inside_unit_cell('Inside',.15)
    manager.add_rad_bound('minimumRadius',27.5/266,.5-40/266)
    manager.add_min_dist('minDist',40/266,3)
    manager.add_gme_constrs_complex('gme_constrs',minFreq=.26,maxFreq=.3,ksBefore=[float(ks[3]),float(ks[6])],ksAfter=[float(ks[13]),float(ks[24])],bandwidth=.002,slope='down')   
    
    #run minimization
    trustConstrParams = {'initial_tr_radius':input['tr'],'initial_constr_penalty':input['cp'],'initial_barrier_parameter':input['bp']}
    minim = optomization.TrustConstr(vars,W1,cost,mode=20,maxiter=400,gmeParams=gmeParams,constraints=manager,path=input['path'],xtol=1e-4,**trustConstrParams)
    minim.minimize()
    minim.save(input['path'])
#%%
if __name__=="__main__":

    
    for tr in [1,.5,.1,.05,.01,.005,.001,.0005]:
        for i in range(10):
            input = {'path':f"tests/media/w1/trustRad/w1OptBasic{tr}_{i}.json",'key':i,'tr':tr, 'cp':1, 'bp':.1}
            minim = worker_function(input)  # Compute the result

    for cp in [10, 5, 1, .5, .1, .05, .01, .005]:
        for i in range(10):
            input = {'path':f"tests/media/w1/constrPenalty/w1OptBasic{tr}_{i}.json",'key':i,'tr':1, 'cp':cp, 'bp':.1}
            minim = worker_function(input)  # Compute the result

    for bp in [1, .5, .1, .05, .01, .005, .001, .0005]:
        for i in range(10):
            input = {'path':f"tests/media/w1/barrierParam/w1OptBasic{tr}_{i}.json",'key':i,'tr':1, 'cp':1, 'bp':bp}
            minim = worker_function(input)  # Compute the result
# %%
