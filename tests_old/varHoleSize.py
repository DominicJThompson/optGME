#%%
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import legume 
legume.set_backend('autograd')
import numpy as np
import autograd.numpy as npa
import optomization
import json

#-------------ZIW stuff------------------

def ZIWVarH(vars=np.array([.105,.2]),Ny=10,dslab=170/266,eps_slab=3.4638,r0=.105,r1=.235):

    lattice = legume.Lattice(npa.array([1,0]),npa.array([0,Ny*npa.sqrt(3)-.25]))

    #now we define a photonic crystal that goes over our lattice and add the one layer of our W1 waveguide
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=dslab,eps_b=eps_slab**2)

    #now we want to add the air holes to our photonic crystal to make it the W1 waveguide we do this in a loop
    flip = False
    for i in range(Ny*2):
        iy = i-Ny+1
        y = iy*npa.sqrt(3)/2-npa.sqrt(3)/12

        if iy>0 and not flip:
            r0, r1 = r1, r0
            flip = True

        if iy==0:
            phc.add_shape(legume.Circle(x_cent=.5*((iy+1)%2),y_cent=y-npa.sqrt(3)/6,r=r1))
            phc.add_shape(legume.Circle(x_cent=.5*(iy%2),y_cent=vars[1]+npa.sqrt(3)/6,r=vars[0]))
            continue

        if iy==1:
            phc.add_shape(legume.Circle(x_cent=.5*((iy+1)%2),y_cent=-vars[1],r=vars[0]))
            phc.add_shape(legume.Circle(x_cent=.5*(iy%2),y_cent=y,r=r0))
            continue
        
        #add hole below and above
        phc.add_shape(legume.Circle(x_cent=.5*((iy+1)%2),y_cent=y-npa.sqrt(3)/6,r=r1))
        phc.add_shape(legume.Circle(x_cent=.5*(iy%2),y_cent=y,r=r0))
    
    return(phc)

phc = ZIWVarH(vars=np.array([.15,.3]))
_ = legume.viz.eps_xy(phc,Ny=300)
# %%
rs = np.linspace(27.5/266,.18,50)
ys = np.linspace(27.5/2/266+40/2/266,.3,50)
file_path = '/Users/dominic/Desktop/optGME/tests/media/holeSize/ZIWSweep2D_big.json'

if not os.path.exists(file_path):
    with open(file_path, "w") as f:
        json.dump([], f)

for i,r in enumerate(rs):
    print(f"r: {i}")
    for j,y in enumerate(ys):
        cost = optomization.Backscatter()
        ks = npa.linspace(npa.pi*.5,npa.pi,25)

        gmeParams = {'verbose':False,'numeig':21,'compute_im':False,'kpoints':npa.array([[float(ks[8])],[0]])}
        vars = npa.array([r,y])
        
        #define constraints
        manager = optomization.ConstraintManager(x0=vars,
                                                numberHoles=12,
                                                crystal=ZIWVarH,
                                                phcParams={},
                                                gmeParams=gmeParams,
                                                gmax=2.01,
                                                mode=19)

        #set custom upper and lower bounds 
        manager.upperBounds = np.array([np.sqrt(3)/3-.235-40/266, .3])
        manager.lowerBounds = np.array([27.5/266, .1])
        manager.add_gme_constrs_complex('gme_constrs',minFreq=.261,maxFreq=.28,ksBefore=[float(ks[3]),float(ks[6])],ksAfter=[float(ks[13]),float(ks[24])],bandwidth=.005,slope='down')

        phc = ZIWVarH(vars=vars)
        gme = legume.GuidedModeExp(phc,4.01)
        gme.run(**gmeParams)
        out = cost.cost(gme,phc,20)

        #check constraints
        cVals = manager.constraints['gme_constrs'].fun(vars)
        lowB = manager.constraints['gme_constrs'].lb
        upB = manager.constraints['gme_constrs'].ub

        within_bounds = [bool(low <= val <= high) for low, val, high in zip(lowB, cVals, upB)]
        
        #save to json 
        data = {'val':[float(r),float(y)],'cost':float(out),'cs':within_bounds}
        with open(file_path, "r") as f:
            results = json.load(f)

        # Append new data
        results.append(data)

        # Write the updated list back to the file
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)

# %%
path = "/Users/dominic/Desktop/optGME/tests/media/holeSize/ZIWOpt2.json"
cost = optomization.Backscatter()
ks = npa.linspace(npa.pi*.5,npa.pi,25)

gmeParams = {'verbose':False,'numeig':21,'compute_im':False,'kpoints':npa.array([[float(ks[8])],[0]])}
vars = npa.ones(1)*.125

#define constraints
manager = optomization.ConstraintManager(x0=vars,
                                        numberHoles=1,
                                        crystal=ZIWVarH,
                                        phcParams={},
                                        gmeParams=gmeParams,
                                        gmax=2.01,
                                        mode=19)

#set custom upper and lower bounds 
manager.upperBounds = np.array([np.sqrt(3)/3-.235-40/266])
manager.lowerBounds = np.array([27.5/266])
manager.add_gme_constrs_complex('gme_constrs',minFreq=.261,maxFreq=.27,ksBefore=[float(ks[3]),float(ks[6])],ksAfter=[float(ks[13]),float(ks[24])],bandwidth=.005,slope='down')

tcParams = {'xtol':1e-4,'initial_tr_radius':.1}
minim = optomization.TrustConstr(vars,ZIWVarH,cost,mode=20,maxiter=400,gmeParams=gmeParams,constraints=manager,path=path,**tcParams)
minim.minimize()
minim.save(path)
# %%

#-----------------W1 stuff-------------------

def W1VarH(vars=np.array([]),Ny=10,dslab=170/266,eps_slab=3.4638,ra=.3):

    lattice = legume.Lattice(npa.array([1,0]),npa.array([0,Ny*npa.sqrt(3)]))

    #now we define a photonic crystal that goes over our lattice and add the one layer of our W1 waveguide
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=dslab,eps_b=eps_slab**2)

    
    #now we want to add the air holes to our photonic crystal to make it the W1 waveguide we do this in a loop
    for i in range(Ny*2-1):
        iy = i-Ny
        if i>=Ny:
            iy+=1
        
        #move the hole over by half a unit cell if they are on odd rows
        if iy%2==1:
            x = .5
        else:
            x = 0

        #the y component should be scaled by the factor of np.sqrt(3)/2
        y = iy*npa.sqrt(3)/2

        if npa.abs(iy)==1:
            phc.add_shape(legume.Circle(x_cent=x,y_cent=npa.sign(iy)*vars[1],r=vars[0]))
            continue

        #now we can add a circle with the given positions
        phc.add_shape(legume.Circle(x_cent=x,y_cent=y,r=ra))

    return(phc)

phc = W1VarH(vars=np.array([.2,np.sqrt(3)/2]))
_ = legume.viz.eps_xy(phc,Ny=300)
#%%

rs = np.linspace(.22,.44,50)
ys = np.linspace(.6,1.05,50)
file_path = '/Users/dominic/Desktop/optGME/tests/media/holeSize/W1Sweep2D_big.json'


if not os.path.exists(file_path):
    with open(file_path, "w") as f:
        json.dump([], f)

for i,r in enumerate(rs):
    for j,y in enumerate(ys):

        cost = optomization.Backscatter()
        ks = npa.linspace(npa.pi*.5,npa.pi,25)

        gmeParams = {'verbose':False,'numeig':21,'compute_im':False,'kpoints':npa.array([[float(ks[8])],[0]])}
        vars = npa.array([r,y])
        
        #define constraints
        manager = optomization.ConstraintManager(x0=vars,
                                                numberHoles=1,
                                                crystal=W1VarH,
                                                phcParams={},
                                                gmeParams=gmeParams,
                                                gmax=2.01,
                                                mode=20)

        #set custom upper and lower bounds 
        manager.upperBounds = np.array([1-.3-40/266])
        manager.lowerBounds = np.array([27.5/266])
        manager.add_gme_constrs_complex('gme_constrs',minFreq=.26,maxFreq=.32,ksBefore=[float(ks[3]),float(ks[6])],ksAfter=[float(ks[13]),float(ks[24])],bandwidth=.002,slope='down')

        phc = W1VarH(vars=vars)
        gme = legume.GuidedModeExp(phc,4.01)
        gme.run(**gmeParams)
        out = cost.cost(gme,phc,20)

        #check if constraints fit 
        cVals = manager.constraints['gme_constrs'].fun(vars)
        lowB = manager.constraints['gme_constrs'].lb
        upB = manager.constraints['gme_constrs'].ub

        within_bounds = [bool(low <= val <= high) for low, val, high in zip(lowB, cVals, upB)]
        
        #save to json 
        data = {'val':[r,y],'cost':out,'cs':within_bounds}

        with open(file_path, "r") as f:
            results = json.load(f)

        # Append new data
        results.append(data)

        # Write the updated list back to the file
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)
    print(i)
# %%
path = "/Users/dominic/Desktop/optGME/tests/media/holeSize/W1Opt4.json"
cost = optomization.Backscatter()
ks = npa.linspace(npa.pi*.5,npa.pi,25)

gmeParams = {'verbose':False,'numeig':21,'compute_im':False,'kpoints':npa.array([[float(ks[8])],[0]])}
vars = npa.ones(1)*.3

#define constraints
manager = optomization.ConstraintManager(x0=vars,
                                        numberHoles=1,
                                        crystal=W1VarH,
                                        phcParams={},
                                        gmeParams=gmeParams,
                                        gmax=2.01,
                                        mode=20)

#set custom upper and lower bounds 
manager.upperBounds = np.array([1-.3-40/266])
manager.lowerBounds = np.array([27.5/266])
manager.add_gme_constrs_complex('gme_constrs',minFreq=.26,maxFreq=.28,ksBefore=[float(ks[3]),float(ks[6])],ksAfter=[float(ks[13]),float(ks[24])],bandwidth=.005,slope='down')

tcParams = {'xtol':1e-4,'initial_tr_radius':.01}
minim = optomization.TrustConstr(vars,W1VarH,cost,mode=20,maxiter=400,gmeParams=gmeParams,constraints=manager,path=path,**tcParams)
minim.minimize()
minim.save(path)
# %%

#--------------------playground--------------------

cost = optomization.Backscatter()
ks = npa.linspace(npa.pi*.5,npa.pi,25)

gmeParams = {'verbose':False,'numeig':25,'compute_im':False,'kpoints':npa.array([[float(ks[8])],[0]])}
vars = npa.ones(1)*.2

#define constraints
manager = optomization.ConstraintManager(x0=vars,
                                        numberHoles=1,
                                        crystal=W1VarH,
                                        phcParams={},
                                        gmeParams=gmeParams,
                                        gmax=2.01,
                                        mode=20)

#set custom upper and lower bounds 
manager.upperBounds = np.array([1-.3-40/266])
manager.lowerBounds = np.array([27.5/266])
manager.add_gme_constrs_complex('gme_constrs',minFreq=.26,maxFreq=.28,ksBefore=[float(ks[3]),float(ks[6])],ksAfter=[float(ks[13]),float(ks[24])],bandwidth=.002,slope='down')

phc = W1VarH(vars=vars)
gme = legume.GuidedModeExp(phc,4.01)
gme.run(**gmeParams)
out = cost.cost(gme,phc,20)
gmefull = legume.GuidedModeExp(phc,4.01)
gmeParams['kpoints']= np.vstack((ks,np.zeros_like(ks)))
gmefull.run(**gmeParams)

#check if constraints fit 
cVals = manager.constraints['gme_constrs'].fun(vars)
lowB = manager.constraints['gme_constrs'].lb
upB = manager.constraints['gme_constrs'].ub

within_bounds = [bool(low <= val <= high) for low, val, high in zip(lowB, cVals, upB)]

|# %%
import matplotlib.pyplot as plt
plt.plot(gmefull.freqs)
#plt.ylim(.24,.32)
plt.show()
# %%
print(gme.freqs.shape)
# %%
