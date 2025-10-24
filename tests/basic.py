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

    gmeParams = {'verbose':False,'numeig':21,'compute_im':False,'gmode_inds':[0,2,4],'kpoints':np.vstack([input['ks_interest'],[0]])}
    vars = W1Vars(key=input['key'])
    
    #define constraints
    manager = optomization.ConstraintManager(x0=vars,
                                            numberHoles=3,
                                            crystal=W1,
                                            phcParams={},
                                            gmeParams=gmeParams,
                                            gmax=2.5,
                                            mode=20,
                                            gmode_inds=[0,2,4],
                                            keep_feasible=input['feasible'])
    
    manager.add_inside_unit_cell('Inside',.15)
    manager.add_rad_bound('minimumRadius',.22,.4)
    manager.add_min_dist('minDist',40/266,3)
    manager.add_gme_constrs_complex('gme_constrs',minFreqHard=input['minfreqHard'],minFreqSoft=input['minfreqSoft'],maxFreq=.28,ksBefore=input['ks_before'],ksAfter=input['ks_after'],
                                    bandwidth=.001,slope='down',minNG=input['ngs_target']*.9,maxNG=input['ngs_target']*1.1,path=input['pathc'])


    #run minimization
    tcParams = input['tcParams']
    minim = optomization.TrustConstr(vars,W1,cost,mode=20,maxiter=800,gmeParams=gmeParams,constraints=manager,path=input['path'],**tcParams)
    minim.minimize()
    minim.save(input['path'])
#%%
if __name__=='__main__':
    ks = list(np.linspace(npa.pi*.5,npa.pi,100))
    for i in range(len(ks)):
        ks[i] = float(ks[i])

    ks_interest = [ks[31],ks[45],ks[51],ks[55]]
    ngs_target = [5,10,15,20]
    ks_before = [[ks[15],ks[25]],[ks[15],ks[35]],[ks[15],ks[41]],[ks[15],ks[45]]]
    ks_after = [[ks[45],ks[95]],[ks[60],ks[90]],[ks[65],ks[90]],[ks[65],ks[90]]]
    minfreqHard = [.253,.253,.253,.253]
    minfreqSoft = [.26,.26,.26,.26]

    path = f"media/constr/data/W1Opt_ng10.json"
    pathc = f"media/constr/constr/W1Opt_ng10.json"
    input = {'path':path,'pathc':pathc,'tcParams':{'xtol':1e-3,'initial_tr_radius':.1,'initial_barrier_parameter':.1,'initial_constr_penalty':.1},
            'key':2,'ks_interest':ks_interest[1],'ngs_target':ngs_target[1],'ks_before':ks_before[1],'ks_after':ks_after[1],'feasible':False,'minfreqHard':minfreqHard[1],'minfreqSoft':minfreqSoft[1]}
    minim = worker_function(input)  # Compute the result

    path = f"media/constr/data/W1Opt_ng15.json"
    pathc = f"media/constr/constr/W1Opt_ng15.json"
    input = {'path':path,'pathc':pathc,'tcParams':{'xtol':1e-3,'initial_tr_radius':.1,'initial_barrier_parameter':.1,'initial_constr_penalty':.1},
            'key':1,'ks_interest':ks_interest[2],'ngs_target':ngs_target[2],'ks_before':ks_before[2],'ks_after':ks_after[2],'feasible':False,'minfreqHard':minfreqHard[2],'minfreqSoft':minfreqSoft[2]}
    minim = worker_function(input)  # Compute the result

# %%
