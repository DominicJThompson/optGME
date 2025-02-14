#%%
import numpy as np
import matplotlib.pyplot as plt
import json
from pandas.plotting import parallel_coordinates
import pandas as pd
plt.rcParams.update({'font.size':14})

def W1Vars(NyChange=3,ra=.3,key=None):
    vars = np.zeros((3,NyChange*2))
    vars[2,:] = ra
    for i in range(NyChange*2):
        iy = i-NyChange
        if i>=NyChange:
            iy+=1
        if iy%2==1: 
            vars[0,i] = .5
        vars[1,i] = iy*np.sqrt(3)/2

    vars = vars.flatten()

    if key is not None:
        np.random.seed(key)
        vars += np.random.normal(loc=0,scale=1/266,size=vars.size)

    return(vars)


# %%
final = []
xs = []
runs = []
for i in range(67):
    with open(f"/Users/dominic/Desktop/optGME/tests/media/W1/nonlinopt{i}.json", "r") as file:
        data = json.load(file)
    final.append(data[-1]['result']['fun'])
    run = []
    for d in data[:-1]:
        run.append(d['objective_value'])
    runs.append(run)
    xs.append(np.array(data[-1]['result']['x'])-W1Vars())
final = np.array(final)
xs = np.array(xs)
#%%
for i,x in enumerate(xs):
    plt.scatter(np.arange(18),x,s=2/((10**final[i]/10**data[-1]['inital_cost'])**2))
plt.ylabel('ofset from original [a]')
labels = [f"x{i}" for i in range(6)] + [f"y{i}" for i in range(6)] + [f"r{i}" for i in range(6)]
plt.xticks(np.arange(18), labels, rotation=45)
plt.xlabel('Parameters')

plt.show()
#%%
for r in runs:
    plt.plot(10**np.array(r)/10**data[-1]['inital_cost'])
plt.ylim(0,1.2)
plt.ylabel(r'$\alpha/\alpha_0$')
plt.xlabel('Iteration')
plt.show()

# %%
plt.plot(10**final/10**data[-1]['inital_cost'],'o')
plt.ylabel(r'$\alpha/\alpha_0$')
plt.xlabel('Different Optomizations')
# %%
