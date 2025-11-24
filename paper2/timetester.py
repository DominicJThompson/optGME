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
import psutil
import time
import multiprocessing as mp
#%%
def worker_function(input):

    os.makedirs(input['path'], exist_ok=True)

    cost = optomization.dispersion(ng_target=input['ngs_target'])

    gmeParams = {'verbose':False,'numeig':15,'compute_im':False,'gmode_inds':[0],'kpoints':np.vstack([input['ks_interest'],[0]*len(input['ks_interest'])])}
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
    manager.add_gme_constrs_dispersion('gme_constrs',minFreq=input['minfreq'],maxFreq=.33,ksBefore=input['ks_before'],ksAfter=input['ks_after'],
                                    bandwidth=.001,slope='down')


    #run minimization
    tcParams = input['tcParams']
    minim = optomization.TrustConstr(vars,
                                    optomization.W1,cost,
                                    mode=14,
                                    maxiter=1,
                                    gmeParams=gmeParams,
                                    phcParams=phcParams,
                                    constraints=manager,
                                    path=input['path']+'/raw_data.json',
                                    gmax=3.01,
                                    **tcParams)
    minim.minimize()
    minim.save(input['path']+'/raw_data.json')
#%%

# -------------------------------------------------
# Helper: set CPU affinity per process
# -------------------------------------------------
def set_cpu_affinity(cpu_ids):
    p = psutil.Process(os.getpid())
    p.cpu_affinity(cpu_ids)

# -------------------------------------------------
# Wrapped worker so we can time + pin CPUs
# -------------------------------------------------
def timed_worker(cpu_ids, input_data, return_dict, run_name):
    set_cpu_affinity(cpu_ids)

    t0 = time.perf_counter()
    worker_function(input_data)
    t1 = time.perf_counter()

    return_dict[run_name] = {
        "cpus": cpu_ids,
        "time_sec": t1 - t0
    }

# -------------------------------------------------
# Benchmark driver
# -------------------------------------------------
def run_cpu_scaling_tests():

    np.random.seed(42)
    npa.random.seed(42)

    ks = list(np.linspace(npa.pi * .5, npa.pi, 100))

    ks_interest = [ks[32], ks[40], ks[48], ks[57], ks[65], ks[74], ks[82], ks[91]]
    ngs_target = [28]
    ks_before = ks[20]
    ks_after = ks[95]
    minfreq = .26

    base_input = {
        'tcParams': {
            'xtol': 1e-3,
            'initial_tr_radius': .1,
            'initial_barrier_parameter': .1,
            'initial_constr_penalty': .1
        },
        'ks_interest': ks_interest,
        'ngs_target': ngs_target,
        'ks_before': ks_before,
        'ks_after': ks_after,
        'minfreq': minfreq,
        'a': 455
    }

    total_cpus = psutil.cpu_count(logical=True)
    print(f"Detected {total_cpus} CPUs")

    # Try these core group sizes
    cpu_group_sizes = [18]
    cpu_group_sizes = [n for n in cpu_group_sizes if n <= total_cpus]

    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []

    core_cursor = 0

    for i, cores_per_job in enumerate(cpu_group_sizes):
        # assign CPU slice
        cpu_ids = list(range(core_cursor, core_cursor + cores_per_job))
        core_cursor += cores_per_job

        path = f"media/cpu_scaling/test_scale_{cores_per_job}"
        os.makedirs(path, exist_ok=True)

        input_data = dict(base_input)
        input_data.update({
            'path': path,
            'key': i
        })

        run_name = f"{cores_per_job}_cores"

        p = mp.Process(
            target=timed_worker,
            args=(cpu_ids, input_data, return_dict, run_name)
        )
        processes.append(p)

    # start all
    for p in processes:
        p.start()

    # wait for all
    for p in processes:
        p.join()

    # -------------------------------------------------
    # Print timing results
    # -------------------------------------------------
    print("\nCPU Scaling Results:")
    for k, v in return_dict.items():
        print(f"{k}: {v['time_sec']:.3f}s  | CPUs -> {v['cpus']}")

    return dict(return_dict)


#%%
if __name__=='__main__':
    results = run_cpu_scaling_tests()
# %%
