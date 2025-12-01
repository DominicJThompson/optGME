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

    os.makedirs(input['path'], exist_ok=True)

    cost = optomization.dispersion(ng_target=input['ngs_target'])

    gmeParams = {'verbose':False,'numeig':15,'compute_im':False,'gmode_inds':[0],'kpoints':np.vstack([input['ks_interest'],[0]*len(input['ks_interest'])])}
    phcParams = {"Ny":7,"dslab":270/input['a'],"eps_slab":3.13}
    backscatterParams = {'a':input['a'],'sig':3,'lp':40,'phidiv':45,'zdiv':10}
    vars = optomization.W1Vars(key=input['key'])
    
    #define constraints
    manager = optomization.ConstraintManager(x0=vars,
                                            numberHoles=3,
                                            crystal=optomization.W1,
                                            phcParams=phcParams,
                                            gmeParams=gmeParams,
                                            gmax=3.5,
                                            mode=14,
                                            gmode_inds=[0])
    
    manager.add_inside_unit_cell('Inside',.2)
    manager.add_rad_bound('minimumRadius',.15,.4)
    manager.add_min_dist('minDist',40/455,3)
    manager.add_gme_constrs_dispersion_backscatter('gme_constrs',minFreq=input['minfreq'],maxFreq=.33,ksBefore=input['ks_before'],ksAfter=input['ks_after'],
                                    bandwidth=.001,slope='down',maxBackscatter=input['maxBackscatter'],backscatterParams=backscatterParams)


    #run minimization
    tcParams = input['tcParams']
    minim = optomization.TrustConstr(vars,
                                    optomization.W1,cost,
                                    mode=14,
                                    maxiter=500,
                                    gmeParams=gmeParams,
                                    phcParams=phcParams,
                                    constraints=manager,
                                    path=input['path']+'/raw_data.json',
                                    gmax=3.01,
                                    **tcParams)
    minim.minimize()
    minim.save(input['path']+'/raw_data.json')

    optomization.dispLossPlot(np.array(minim.result["x"]),
                            optomization.W1,
                            input['ks_interest'],
                            input['path']+'/meta_data.png',
                            gmax=4.01,
                            phcParams=phcParams,
                            mode=14,
                            a=input['a'],
                            final_cost=float(minim.result['fun']),
                            execution_time=minim.result['execution_time'],
                            niter=minim.result['niter'])
#%%
if __name__=='__main__':
    np.random.seed(42)
    npa.random.seed(42)
    ks = list(np.linspace(npa.pi*.5,npa.pi,100))
    loss_index = int(os.environ["LOSS_INDEX"])

    #first run the ng=20 test
    ks_interest = [ks[32],ks[40],ks[48],ks[57],ks[65],ks[74],ks[82],ks[91]]
    ngs_target = [28]
    ks_before = ks[20]
    ks_after = ks[95]
    minfreq = .26
    maxBackscatter = [1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,1e-2,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,1e-1]

    for i in range(2):
        path = f"media/loss_tests{loss_index}/test{i}"
        input = {'path':path,'tcParams':{'xtol':1e-3,'initial_tr_radius':.1,'initial_barrier_parameter':.1,'initial_constr_penalty':.1},
                'key':i,'ks_interest':ks_interest,'ngs_target':ngs_target,'ks_before':ks_before,'ks_after':ks_after,'minfreq':minfreq,'a':455,'maxBackscatter':maxBackscatter[loss_index]}
        minim = worker_function(input)  # Compute the result

    optomization.runBatchReport(28,maxBackscatter[loss_index],8,f'media/loss_tests{loss_index}',f'media/loss_tests{loss_index}/report.html')

# %%
