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

    cost = optomization.Backscatter()

    gmeParams = {'verbose':False,'numeig':21,'compute_im':False,'gmode_inds':[0,2,4],'kpoints':np.vstack([input['ks_interest'],[0]])}
    vars = optomization.W1Vars(key=input['key'])
    
    #define constraints
    manager = optomization.ConstraintManager(x0=vars,
                                            numberHoles=3,
                                            crystal=optomization.W1,
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
    minim = optomization.TrustConstr(vars,optomization.W1,cost,mode=20,maxiter=1,gmeParams=gmeParams,constraints=manager,path=input['path'],**tcParams)
    minim.minimize()
    minim.save(input['path'])
#%%
if __name__=='__main__':
    ks = list(np.linspace(npa.pi*.5,npa.pi,100))

    ks_interest = [ks[31]]
    ngs_target = [5]
    ks_before = [[ks[15],ks[25]]]
    ks_after = [[ks[45],ks[95]]]
    minfreqHard = [.253]
    minfreqSoft = [.26]

    path = f"media/test.json"
    pathc = f"media/test_constr.json"
    input = {'path':path,'pathc':pathc,'tcParams':{'xtol':1e-3,'initial_tr_radius':.1,'initial_barrier_parameter':.1,'initial_constr_penalty':.1},
            'key':2,'ks_interest':ks_interest[0],'ngs_target':ngs_target[0],'ks_before':ks_before[0],'ks_after':ks_after[0],'feasible':False,'minfreqHard':minfreqHard[0],'minfreqSoft':minfreqSoft[0]}
    minim = worker_function(input)  # Compute the result

# %%
