#%%
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import legume 
legume.set_backend('autograd')
import numpy as np
import autograd.numpy as npa
import optomization


def BIW(vars=npa.zeros((0,0)),NyChange=3,Ny=10,dslab=170/266,eps_slab=3.4638,r0=.105,r1=.235):

    vars = vars.reshape((3,NyChange*2*2))

    lattice = legume.Lattice(npa.array([1,0]),npa.array([0,Ny*npa.sqrt(3)+.45]))

    #now we define a photonic crystal that goes over our lattice and add the one layer of our W1 waveguide
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=dslab,eps_b=eps_slab**2)

    for i in range(vars.shape[1]):
        phc.add_shape(legume.Circle(x_cent=vars[0,i],y_cent=vars[1,i],r=vars[2,i]))

    #now we want to add the air holes to our photonic crystal to make it the W1 waveguide we do this in a loop
    flip = False
    for i in range(Ny*2):
        iy = i-Ny+1
        y = iy*npa.sqrt(3)/2-npa.sqrt(3)/12

        if np.abs(iy-.5)<=NyChange:
            continue
        
        #move the hole over by half a unit cell if they are on odd rows
        if iy%2==1:
            x = .5
        else:
            x = 0

        if iy>0 and not flip:
            r0, r1 = r1, r0
            flip = True
        
        #add hole below and above
        phc.add_shape(legume.Circle(x_cent=x,y_cent=y-npa.sqrt(3)/3,r=r1))
        phc.add_shape(legume.Circle(x_cent=x,y_cent=y,r=r0))
    
    return(phc)

def ZIW(vars=npa.zeros((0,0)),NyChange=3,Ny=10,dslab=170/266,eps_slab=3.4638,r0=.105,r1=.235):

    vars = vars.reshape((3,NyChange*2*2))

    lattice = legume.Lattice(npa.array([1,0]),npa.array([0,Ny*npa.sqrt(3)-.25]))

    #now we define a photonic crystal that goes over our lattice and add the one layer of our W1 waveguide
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=dslab,eps_b=eps_slab**2)

    for i in range(vars.shape[1]):
        phc.add_shape(legume.Circle(x_cent=vars[0,i],y_cent=vars[1,i],r=vars[2,i]))

    #now we want to add the air holes to our photonic crystal to make it the W1 waveguide we do this in a loop
    flip = False
    for i in range(Ny*2):
        iy = i-Ny+1
        y = iy*npa.sqrt(3)/2-npa.sqrt(3)/12

        if np.abs(iy-.5)<=NyChange:
            continue

        if iy>0 and not flip:
            r0, r1 = r1, r0
            flip = True
        
        #add hole below and above
        phc.add_shape(legume.Circle(x_cent=.5*((iy+1)%2),y_cent=y-npa.sqrt(3)/6,r=r1))
        phc.add_shape(legume.Circle(x_cent=.5*(iy%2),y_cent=y,r=r0))
    
    return(phc)

def BIWVars(NyChange=3,r0=.105,r1=.235,key=None):
    vars = npa.zeros((3,NyChange*2*2))
    flip = False
    for i in range(NyChange*2):

        iy = i-NyChange+1
        y = iy*npa.sqrt(3)/2-npa.sqrt(3)/12

        if iy>0 and not flip:
            r0, r1 = r1, r0
            flip = True

        if iy%2==1:
            x = .5
        else:
            x = 0
        
        vars[0,2*i],vars[0,2*i+1] = x,x
        vars[1,2*i],vars[1,2*i+1] = y-npa.sqrt(3)/3, y
        vars[2,2*i],vars[2,2*i+1] = r1,r0

    vars = vars.flatten()
    if key is not None:
        np.random.seed(key)
        vars += np.random.normal(loc=0,scale=1/266,size=vars.size)

    return(vars)

def ZIWVars(NyChange=3,r0=.105,r1=.235,key=None):
    vars = npa.zeros((3,NyChange*2*2))
    flip = False
    for i in range(NyChange*2):

        iy = i-NyChange+1
        y = iy*npa.sqrt(3)/2-npa.sqrt(3)/12

        if iy>0 and not flip:
            r0, r1 = r1, r0
            flip = True
        
        vars[0,2*i],vars[0,2*i+1] = .5*((iy+1)%2),.5*(iy%2)
        vars[1,2*i],vars[1,2*i+1] = y-npa.sqrt(3)/6, y
        vars[2,2*i],vars[2,2*i+1] = r1,r0

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
    ks = npa.linspace(npa.pi*.5,npa.pi,25)

    gmeParams = {'verbose':False,'numeig':21,'compute_im':False,'kpoints':npa.array([[float(ks[12])],[0]])}
    vars = ZIWVars(NyChange=3,key=input['key'])
    
    #define constraints
    manager = optomization.ConstraintManager(x0=vars,
                                            numberHoles=12,
                                            crystal=ZIW,
                                            phcParams={},
                                            gmeParams=gmeParams,
                                            gmax=2.01,
                                            mode=19)
    
    manager.add_inside_unit_cell('Inside',.5)
    manager.add_rad_bound('minimumRadius',27.5/266,.35)
    manager.add_min_dist('minDist',40/266,3)
    manager.add_gme_constrs_complex('gme_constrs',minFreq=.25,maxFreq=.27,ksBefore=[float(ks[3]),float(ks[6])],ksAfter=[float(ks[16]),float(ks[24])],bandwidth=.005,slope='down')

    #the eigenvalue of interest changes depending on gmax: gmax=2.01, eig=19 but gmax=4.01, eig=20
    #run minimization
    tcParams = {'xtol':1e-4,'initial_tr_radius':.1}
    minim = optomization.TrustConstr(vars,ZIW,cost,mode=20,maxiter=400,gmeParams=gmeParams,constraints=manager,path=input['path'],**tcParams)
    minim.minimize()
    minim.save(input['path'])
# %%
if __name__=='__main__':
    for i in range(300):
        input = {'path':f"/Users/dominic/Desktop/optGME/tests/media/ZIW{i}.json",'key':i}
        worker_function(input)  # Compute the result

# %%
