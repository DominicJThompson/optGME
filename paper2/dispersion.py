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
#%%
def worker_function(input):

    #make directory to save files
    with open(input['path'], "w") as f:
        pass

    cost = optomization.dispersion(ng_target=input['ngs_target'])

    gmeParams = {'verbose':False,'numeig':15,'compute_im':False,'gmode_inds':[0],'kpoints':np.vstack([input['ks_interest'],[0,0,0,0]])}
    phcParams = {"Ny":7,"dslab":270/input['a'],"eps_slab":3.13}
    vars = optomization.W1Vars(key=input['key'])
    
    #define constraints
    manager = optomization.ConstraintManager(x0=vars,
                                            numberHoles=3,
                                            crystal=optomization.W1,
                                            phcParams=phcParams,
                                            gmeParams=gmeParams,
                                            gmax=2.5,
                                            mode=14,
                                            gmode_inds=[0])
    
    manager.add_inside_unit_cell('Inside',.15)
    manager.add_rad_bound('minimumRadius',.22,.4)
    manager.add_min_dist('minDist',40/455,3)
    manager.add_gme_constrs_dispersion('gme_constrs',minFreq=input['minfreq'],maxFreq=.305,ksBefore=input['ks_before'],ksAfter=input['ks_after'],
                                    bandwidth=.001,slope='down',path=input['pathc'])


    #run minimization
    tcParams = input['tcParams']
    minim = optomization.TrustConstr(vars,optomization.W1,cost,mode=14,maxiter=200,gmeParams=gmeParams,phcParams=phcParams,constraints=manager,path=input['path'],**tcParams)
    minim.minimize()
    minim.save(input['path'])
#%%
if __name__=='__main__':
    ks = list(np.linspace(npa.pi*.5,npa.pi,100))

    ks_interest = [ks[29],ks[36],ks[44],ks[51]]
    ngs_target = [10]
    ks_before = ks[15]
    ks_after = ks[90]
    minfreq = .285

    for i in range(10):
        path = f"media/test{i}.json"
        pathc = f"media/test{i}_constr.json"
        input = {'path':path,'pathc':pathc,'tcParams':{'xtol':1e-3,'initial_tr_radius':.1,'initial_barrier_parameter':.1,'initial_constr_penalty':.1},
                'key':i,'ks_interest':ks_interest,'ngs_target':ngs_target,'ks_before':ks_before,'ks_after':ks_after,'minfreq':minfreq,'a':455}
        minim = worker_function(input)  # Compute the result

# %%
