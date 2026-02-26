import autograd.numpy as npa
import legume
import numpy as np


#--------------------------------W1--------------------------------
def W1(vars=npa.zeros((0,0)),NyChange=3,Ny=10,dslab=170/266,eps_slab=3.4638,ra=.3,eps_clad=1):

    vars = vars.reshape((3,NyChange*2))

    lattice = legume.Lattice(npa.array([1,0]),npa.array([0,Ny*npa.sqrt(3)]))

    #now we define a photonic crystal that goes over our lattice and add the one layer of our W1 waveguide
    phc = legume.PhotCryst(lattice,eps_l=eps_clad**2,eps_u=eps_clad**2)
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
        phc.add_shape(legume.Circle(x_cent=x,y_cent=y,r=ra,eps=eps_clad**2))

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

#--------------------------------ZIW--------------------------------
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