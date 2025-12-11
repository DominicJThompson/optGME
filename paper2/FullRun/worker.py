#%%
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
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
    manager.add_gme_constrs_dispersion_backscatter('gme_constrs',minFreq=input['minfreq'],maxFreq=.327,ksBefore=input['ks_before'],ksAfter=input['ks_after'],
                                    bandwidth=.0025,slope='down',maxBackscatter=input['maxBackscatter'],backscatterParams=backscatterParams)


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
    
    #get the input values 
    parser = argparse.ArgumentParser()
    parser.add_argument("--LOSS_INDEX", type=int)
    parser.add_argument("--NDBP_INDEX", type=int)
    parser.add_argument("--NG_INDEX", type=int)
    parser.add_argument("--SEED", type=int)
    args = parser.parse_args()

    loss_index = args.LOSS_INDEX
    ndbp_index = args.NDBP_INDEX
    ngs_index = args.NG_INDEX
    seed = args.SEED

    np.random.seed(seed)
    npa.random.seed(seed)
    ks = list(np.linspace(npa.pi*.5,npa.pi,200))

    ks_interest = ks_array = [
        [ks[97], ks[103], ks[110], ks[117], ks[123], ks[130], ks[137], ks[144]],
        [ks[88], ks[97], ks[106], ks[115], ks[125], ks[134], ks[143], ks[153]],
        [ks[79], ks[90], ks[102], ks[114], ks[126], ks[138], ks[150], ks[162]],
        [ks[70], ks[84], ks[98], ks[113], ks[127], ks[142], ks[156], ks[171]],
        [ks[62], ks[78], ks[95], ks[112], ks[128], ks[145], ks[162], ks[179]],
    ]
    ks_before = [
        [ks[32], ks[64]],
        [ks[29], ks[58]],
        [ks[26], ks[52]],
        [ks[23], ks[46]],
        [ks[20], ks[41]],
    ]
    ks_after = [
        [ks[162], ks[181]],
        [ks[168], ks[184]],
        [ks[174], ks[187]],
        [ks[180], ks[190]],
        [ks[186], ks[193]],
    ]

    ngs_target = [10,12,14,16,18,20,22,24,26,28,30]
    minfreq = .26
    maxBackscatter = [1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,1e-2,2e-2,3e-2,4e-2,5e-2]

    if seed == 42:
        add = 10
    else:
        add = 0
    for i in range(1):
        path = f"media/ng{ngs_target[ngs_index]}//ndbp{ndbp_index}/loss_tests{loss_index}/test{i+add}"
        input = {'path':path,'tcParams':{'xtol':1e-3,'initial_tr_radius':.1,'initial_barrier_parameter':.1,'initial_constr_penalty':.1},
                'key':i,'ks_interest':ks_interest[ndbp_index],'ngs_target':ngs_target[ngs_index],'ks_before':ks_before[ndbp_index],'ks_after':ks_after[ndbp_index],'minfreq':minfreq,'a':455,'maxBackscatter':maxBackscatter[loss_index]}
        minim = worker_function(input)  # Compute the result

# %%

