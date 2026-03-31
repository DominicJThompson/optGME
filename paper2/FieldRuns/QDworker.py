#%%
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import legume 
legume.set_backend('autograd')
import numpy as np
import autograd.numpy as npa
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
    manager.add_gme_constrs_QDs('gme_constrs',minFreq=input['minfreq'],maxFreq=.327,ksBefore=input['ks_before'],ksAfter=input['ks_after'],
                                    bandwidth=.002,slope='down',maxBackscatter=input['maxBackscatter'],backscatterParams=backscatterParams,minPurcell=input['minPurcell'])


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
    
    # #get the input values 
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--LOSS_INDEX", type=int)
    # parser.add_argument("--FIELD_INDEX", type=int)
    # parser.add_argument("--NDBP_INDEX", type=int)
    # parser.add_argument("--NG_INDEX", type=int)
    # parser.add_argument("--SEED", type=int)
    # args = parser.parse_args()

    # loss_index = args.LOSS_INDEX
    # field_index = args.FIELD_INDEX
    # ndbp_index = args.NDBP_INDEX
    # ngs_index = args.NG_INDEX
    # seed = args.SEED

    loss_index = 0
    field_index = 0
    ndbp_index = 0
    ngs_index = 0
    seed = 0

    np.random.seed(seed)
    npa.random.seed(seed)
    ks = list(np.linspace(npa.pi*.5,npa.pi,200))

    
    ks_interest = {
        0:{   # bandwidth index
            0:[96,99,103,107,111,115,119,123], # ng index
            1:[89,94,100,105,112,118,124,130],
            2:[82,89,97,105,113,121,129,137],
        },
        1:{
            0:[102,104,106,108,110,112,114,117],
            1:[99,102,105,108,111,114,117,120],
            2:[96,99,103,107,111,115,119,123],
        },
        2:{
            0:[105,106,107,108,110,111,112,114],
            1:[104,105,107,108,110,111,113,115],
            2:[102,104,106,108,110,112,114,117],
        },
    }

    ks_before = {
        0:{
            0:[32,64],
            1:[29,59],
            2:[27,54],
        },
        1:{
            0:[34,68],
            1:[33,66],
            2:[32,64],
        },
        2:{
            0:[35,70],
            1:[34,69],
            2:[34,68],
        },
    }

    ks_after = {
        0:{
            0:[148,174],
            1:[153,176],
            2:[158,179],
        },
        1:{
            0:[144,172],
            1:[146,173],
            2:[148,174],
        },
        2:{
            0:[142,171],
            1:[143,171],
            2:[144,172],
        },
    }

    ngs_target = [20,30,40]
    minfreq = .26
    maxBackscatter = [0.01,0.02,0.03,0.04,0.05]
    minPurcell = [0.1,0.125,0.15,0.175,0.2]

    if seed == 420:
        add = 10
    else:
        add = 0
    for i in range(10):
        path = f"media/QDs_yDipole/ng{ngs_target[ngs_index]}/ndbp{ndbp_index}/loss_tests{loss_index}/field_tests{field_index}/test{i+add}"
        input = {'path':path,'tcParams':{'xtol':1e-3,'initial_tr_radius':.1,'initial_barrier_parameter':.1,'initial_constr_penalty':.1},
                'key':i,'ks_interest':ks_interest[ndbp_index][ngs_index],'ngs_target':ngs_target[ngs_index],'ks_before':ks_before[ndbp_index][ngs_index],'ks_after':ks_after[ndbp_index][ngs_index],
                'minfreq':minfreq,'a':405,'maxBackscatter':maxBackscatter[loss_index],'minPurcell':minPurcell[field_index]}
        minim = worker_function(input)  # Compute the result

# %%

