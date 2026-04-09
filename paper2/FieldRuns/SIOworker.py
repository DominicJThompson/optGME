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
    phcParams = {"Ny":7,"dslab":220/input['a'],"eps_slab":3.4784,'eps_clad':1.44427}
    backscatterParams = {'a':input['a'],'sig':3,'lp':40,'phidiv':45,'zdiv':15}
    dopingParams = {'wi':32/input['a'],'wf':113/input['a'],'Ne':1E18,'Nh':1e18,'dl':14/input['a'],'zdiv':15,'max_y':4*np.sqrt(3)/2}
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
    manager.add_min_dist('minDist',40/input['a'],3)
    manager.add_gme_constrs_MZMs('gme_constrs',minFreq=input['minfreq'],maxFreq=.28,ksBefore=input['ks_before'],ksAfter=input['ks_after'],
                                    bandwidth=.002,slope='down',maxLoss=input['maxLoss'],backscatterParams=backscatterParams,minTheta=input['minTheta'],
                                    dopingParams=dopingParams,ng_target=input['ng_target'])


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
                            niter=minim.result['niter'],
                            field='MZMs',
                            backscatterParams=backscatterParams,
                            dopingParams=dopingParams)
#%%
if __name__=='__main__':
    
    #get the input values 
    parser = argparse.ArgumentParser()
    parser.add_argument("--LOSS_INDEX", type=int)
    parser.add_argument("--FIELD_INDEX", type=int)
    parser.add_argument("--NDBP_INDEX", type=int)
    parser.add_argument("--NG_INDEX", type=int)
    parser.add_argument("--SEED", type=int)
    args = parser.parse_args()

    loss_index = args.LOSS_INDEX
    field_index = args.FIELD_INDEX
    ndbp_index = args.NDBP_INDEX
    ngs_index = args.NG_INDEX
    seed = args.SEED


    np.random.seed(seed)
    npa.random.seed(seed)
    ks = list(np.linspace(npa.pi*.5,npa.pi,200))

    ks_interest = ks_array = [
        [ks[120], ks[127], ks[134], ks[141], ks[149], ks[156], ks[163], ks[171]], #NDBP = 0.25
        [ks[115], ks[123], ks[132], ks[141], ks[159], ks[158], ks[167], ks[176]], #NDBP = 0.3
        [ks[110], ks[120], ks[130], ks[140], ks[150], ks[160], ks[170], ks[181]], #NDBP = 0.35
    ]
    ks_before = [
        [ks[32], ks[65]],
        [ks[32], ks[65]],
        [ks[32], ks[65]],
    ]
    ks_after = [
        [ks[186], ks[199]],
        [ks[188], ks[199]],
        [ks[190], ks[199]],
    ]

    ngs_target = [10,20,30]
    minfreq = .248
    maxLosses = {0:[25,20,15,10,5],
                1:[12,10,8,6,4],
                2:[3.5,3,2.5,2,1.5]}
    minThetas = [0.7,0.6,0.5,0.4,0.3]

    if seed == 420:
        add = 5
    if seed == 69:
        add = 10
    if seed == 67:
        add = 15
    else:
        add = 0
    for i in range(5):
        path = f"media/MZM_totLoss/ng{ngs_target[ngs_index]}/ndbp{ndbp_index}/loss_tests{loss_index}/field_tests{field_index}/test{i+add}"
        input = {'path':path,'tcParams':{'xtol':1e-3,'initial_tr_radius':.1,'initial_barrier_parameter':.1,'initial_constr_penalty':.1},
                'key':add+i,'ks_interest':ks_interest[ndbp_index],'ngs_target':ngs_target[ngs_index],'ks_before':ks_before[ndbp_index],'ks_after':ks_after[ndbp_index],
                'minfreq':minfreq,'a':390,'maxLoss':maxLosses[ndbp_index][loss_index],'minTheta':minThetas[field_index],'ng_target':ngs_target[ngs_index]}
        minim = worker_function(input)  # Compute the result

# %%

