#%%
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import legume 
legume.set_backend('autograd')
import numpy as np
import autograd.numpy as npa
from autograd import grad, jacobian, hessian
import time
import matplotlib.pyplot as plt
import optomization
import json


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

# %%
with open("/Users/dominic/Desktop/optGME/tests/media/nonlinopt0.json", "r") as file:
    data = json.load(file)

vars = np.array(data[52]['x_values'])
#%%
ks = npa.linspace(npa.pi*.5,npa.pi,25)

gmeParams = {'verbose':False,'numeig':21,'compute_im':False,'kpoints':npa.array([[float(ks[8])],[0]])}
    
#define constraints
manager = optomization.ConstraintManager(x0=vars,
                                        numberHoles=3,
                                        crystal=W1,
                                        phcParams={},
                                        gmeParams=gmeParams,
                                        gmax=2.01,
                                        mode=20)
    
manager.add_inside_unit_cell('Inside',.5)
manager.add_rad_bound('minimumRadius',.15,.42)
#manager.add_min_dist('minDist',40/266,3,W1Vars(NyChange=3+3))
manager.add_gme_constrs_complex('gme_constrs',minFreq=.26,maxFreq=.28,ksBefore=[float(ks[4]),float(ks[6])],ksAfter=[float(ks[14]),float(ks[20])],bandwidth=.005,slope='down')

for value in list(manager.constraints.values()):
    print(value.fun(vars))

#%%
print(list(manager.constraints.values()))
# %%
print(data[0]['x_values'])
bounds = manager.build_bounds()
x = data[0]['x_values']
violations = [(i, lb, x, ub) for i, (lb, x, ub) in enumerate(zip(bounds.lb, x, bounds.ub)) if not (lb <= x <= ub)]
# %%
print("Bounds Respected:", all(lb <= x <= ub for x, lb, ub in zip(x, bounds.lb, bounds.ub)))
# %%
nk = 25
kmin, kmax = .5*np.pi,np.pi
path = np.vstack((np.linspace(kmin,kmax,nk),np.zeros(nk)))
cost = optomization.Backscatter()

def objective(vars):
    phc = W1(vars=vars)
    gme = legume.GuidedModeExp(phc,2.001)
    gme.run(kpoints=np.array([[path[0,8]],[path[1,8]]]),numeig=21,compute_im=False)
    out = cost.cost(gme,phc,20)
    return(out)

gradfunc = grad(objective)

plt.plot(gradfunc(W1Vars()))

# %%
data[-1]['result']['v']
# %%
ks = npa.linspace(npa.pi*.5,npa.pi,25)
cost = optomization.Backscatter()
gmeParams = {'verbose':False,'numeig':21,'compute_im':False,'kpoints':npa.array([[int(ks[8])],[0]])}
manager = optomization.ConstraintManager(x0=W1Vars(),
                                        numberHoles=3,
                                        crystal=W1,
                                        phcParams={},
                                        gmeParams=gmeParams,
                                        gmax=2.01,
                                        mode=20)
    
manager.add_inside_unit_cell('Inside',.5)
manager.add_rad_bound('minimumRadius',.15,.42)
#manager.add_min_dist('minDist',40/266,3,W1Vars(NyChange=3+3))
manager.add_gme_constrs('gme_constrs',minFreq=.26,maxFreq=.28,ksBefore=[float(ks[6])],ksAfter=[float(ks[14]),float(ks[20])],bandwidth=.005,slope='down')


list = []
for key, val in manager.constraints.items():
    #try:
    #    list.append(grad(lambda x: -npa.log(-val['fun'](x)))(W1Vars()))
    #except TypeError:
    list.append(jacobian(lambda x: -npa.log(-val['fun'](x)))(W1Vars())) 

# %%
plt.imshow(list[0],'bwr')
plt.colorbar()
# %%
plt.plot(np.sum(list[0],axis=0))
# %%
# Load JSON file
with open("/Users/dominic/Desktop/optGME/tests/media/opt1.json", "r") as file:
    data = json.load(file)

phc = W1(vars = np.array(data[14]['x_values']))

nk = 25
kmin, kmax = .5*np.pi,np.pi
path = np.vstack((np.linspace(kmin,kmax,nk),np.zeros(nk)))

gme = legume.GuidedModeExp(phc,gmax=2.01)
gme.run(kpoints=path,numeig=30,compute_im=False,verbose=True)
#%%
plt.plot(gme.freqs[:,15:25])



# %%
ks = npa.linspace(npa.pi*.5,npa.pi,25)
print(ks[4],ks[6],ks[8],ks[12],ks[20])
# %%
final = []
for i in range(15):
    with open(f"/Users/dominic/Desktop/optGME/tests/media/nonlinopt{i}.json", "r") as file:
        data = json.load(file)
    final.append(data[-1]['result']['fun'])

plt.plot(np.array(final).T,'o')
# %%
