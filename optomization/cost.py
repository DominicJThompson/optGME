import legume.backend as bd
import legume
import autograd.numpy as npa

class Cost(object):
    """
    Class defining the cost function for the optomization
    """

    def __init__(self, a = 266):
        """
            Holds all the perameters for the cost fucntion

            Args:
                phc_def : this is a function that defines your photonic crystal
                phc_def_inputs : the requred inputs to your phc definition fucntion except
                                 for the variables we alter during optoization. Stored in dictionary
        """
        self.a = a


    def __str__(self):
        """Print the class parameters."""
        out = ""
        for key, value in self.__dict__.items():
            out += f"{key}: {value}\n"
        return(out)
    
    def return_params(self):
        return self.__dict__.copy()

    def cost(self, vars):
        """
        Defines the cost that the optomizer will use
        """
        raise NotImplementedError("cost() needs to be implemented by"
                                  "cost subclasses")
    

class Backscatter(Cost):
    """
    Defines the cost function associate with backscattering
    """

    def __init__(self, phidiv = 45, lp = 40, sig = 3, **kwargs):
        # Call the master class constructor
        super().__init__(**kwargs)
        self.phidiv = phidiv
        self.lp = lp
        self.sig = sig


    def hole_borders(self, phc):
        """
        returns the points at the hole borders that we will use to compute alpha
        """

        #get relevent information for computation
        shapes = phc.layers[0].shapes
        phis = npa.linspace(0,2*npa.pi,self.phidiv,endpoint=False)
        cphis = bd.cos(phis); sphis = bd.sin(phis)

        #get the array of hole atributes [xpos,ypos,r]
        holeCords = bd.array([[s.x_cent, s.y_cent, s.r] for s in shapes])

        #get the coordinates of the hole borders
        # Compute the initial borders using broadcasting
        x_coords = holeCords[:, 0:1] + holeCords[:, 2:3] * cphis  # Add dimensions for broadcasting
        y_coords = holeCords[:, 1:2] + holeCords[:, 2:3] * sphis

        # Combine x and y coordinates
        initialBorders = bd.stack([x_coords, y_coords], axis=-1)

        # Apply lattice corrections
        xCorrected = bd.where(initialBorders[..., 0] > phc.lattice.a1[0] / 2, 
                             initialBorders[..., 0] - phc.lattice.a1[0], 
                             initialBorders[..., 0])
        yCorrected = bd.where(initialBorders[..., 1] > phc.lattice.a2[0] / 2, 
                             initialBorders[..., 1] - phc.lattice.a2[0], 
                             initialBorders[..., 1])

        # Combine corrected coordinates
        borders = bd.stack([xCorrected, yCorrected], axis=-1)

        return(borders, phis, holeCords[:,2])
    
    def get_xyfield(self,gme,n,xys,z,field='E',components='xyz'):
        """
        returns the field at arbitrary points
        """

        #setup 
        ft = {}
        ft['x'],ft['y'],ft['z'] = gme.ft_field_xy(field,0,n,z)
        fis = {}
        _, ind_unique = npa.unique(gme.gvec,return_index=True,axis=1)

        #loop through adding the field
        for comp in components:
            if not (comp in fis.keys()):
                fis[comp] = bd.zeros(xys[:,:,0].shape,dtype=bd.complex)
                for indg in ind_unique:
                    fis[comp] += bd.sqrt(npa.pi)*ft[comp][indg]*bd.exp(1j*gme.gvec[0,indg]*xys[:,:,0]+1j*gme.gvec[1,indg]*xys[:,:,1])
            else:
                raise ValueError("component can be any combiation of xyz")

        return(fis)
    
    
    def comp_pdote(self,gme,phc,n,z,borders,phis):
        """
        Computes the E and D dot products for the alpha calculation
        """


        E = self.get_xyfield(gme,n,borders,z,components='xyz')
        D = self.get_xyfield(gme,n,borders,z,field='D',components='xy')

        Epara = bd.array([-bd.sin(phis)*E['x'],bd.cos(phis)*E['y'],E['z']])
        Dperp = bd.array([bd.cos(phis)*D['x'],bd.sin(phis)*D['y'],npa.zeros_like(E['z'])])

        p = Epara+(phc.layers[0].eps_b+1)*Dperp/(2*phc.layers[0].eps_b*1)

        pdeR = bd.conj(E['x'])*bd.conj(p[0])+bd.conj(E['y'])*bd.conj(p[1])+bd.conj(E['z'])*bd.conj(p[2])
        pdeRP = E['x']*p[0]+E['y']*p[1]+E['z']*p[2]

        return(pdeR,pdeRP)
    
    def comp_backscatter(self, gme, phc, n):
        """
        This runs the calculation of the backscattering divided by the group index
        Given the simulation results
        """
        #get the points around the hole
        borders, phis, holeRad = self.hole_borders(phc)
        
        #proccess phis so that they work with the formula
        phisLooped = npa.arctan(npa.tan(phis))
   
        #get the necicary field information around the holes
        pdeR, pdeRP = self.comp_pdote(gme,phc,n,phc.layers[0].d,borders,phis)

        #do the multiplication for the p dot e part, we will add the jacobian determinate after
        pdeMeshs = npa.array([[[pdeR[j] for _ in range(pdeR.shape[1])],npa.array([pdeRP[j] for _ in range(pdeR.shape[1])]).T ] for j in range(pdeR.shape[0])])
        preSumPde = pdeMeshs[:,0]*pdeMeshs[:,1]

        #the real exponential term
        phiMesh, phiPMesh = npa.meshgrid(phisLooped,phisLooped)
        realExp = (bd.abs(phiMesh-phiPMesh)*(-holeRad[:,npa.newaxis,npa.newaxis]))/(self.lp/self.a) #the unites cancle

        #the imaginary exponential term
        xMeshs = npa.array([[[borders[j,:,0] for _ in range(borders.shape[1])],npa.array([borders[j,:,0] for _ in range(borders.shape[1])]).T ] for j in range(borders.shape[0])])
        imagExp = 2*(bd.norm(gme.kpoints[:,0]))*(xMeshs[:,0]-xMeshs[:,1]) #units cancle

        #run the intigral, including the jacobian determinite
        intigrand = preSumPde*bd.exp(realExp+1j*imagExp)
        intigral = bd.sum(intigrand,axis=(1,2))*(holeRad*npa.pi*2/self.phidiv)**2

        #calculate the leading coeficnets for each of the holes
        cirleCoeffs = ((299792458*2*npa.pi*gme.freqs[0,n])*(self.sig/self.a)*(phc.layers[0].eps_b-1)/2)**2

        #compute the final result
        alpha = bd.real(cirleCoeffs*bd.sum(intigral)*(phc.layers[0].d*self.a*10**-9)**2)
        
        return(alpha*266*1E-9) #this puts it in units of a^-1
    
    
    def cost(self,gme,phc,n):
        """
        returns the cost associated with the backscattering
        """
        alpha = npa.log10(self.comp_backscatter(gme,phc,n))

        return(alpha)
    
