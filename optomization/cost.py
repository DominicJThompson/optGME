import legume.backend as bd
import legume
import autograd.numpy as npa
from optomization.utils import NG
from optomization.utils import backscatterLog

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

    def __init__(self, phidiv = 45, zdiv = 1, lp = 40, sig = 3, **kwargs):
        # Call the master class constructor
        super().__init__(**kwargs)
        self.phidiv = phidiv
        self.lp = lp
        self.sig = sig
        self.zdiv = zdiv

    def cost(self,gme,phc,n,k=0):
        """
        returns the cost associated with the backscattering
        """
        alpha = backscatterLog(gme,phc,n,k=k,a=self.a,sig=self.sig,lp=self.lp,phidiv=self.phidiv,zdiv=self.zdiv)

        return(alpha)

class dispersion(Cost):
    """
    Defines the cost function associated with the dispersion
    """

    def __init__(self, ng_target=10, Nx=100, Ny=125, **kwargs):
        super().__init__(**kwargs)
        self.ng_target = ng_target
        self.Nx = Nx
        self.Ny = Ny

    def cost(self,gme,phc,n):
        """
        returns the cost associated with the dispersion
        """
        cost = 0
        for i in range(len(gme.kpoints[0])):
            ng = npa.abs(NG(gme,i,n,Nx=self.Nx,Ny=self.Ny))
            cost += (ng - self.ng_target)**2
        return(cost)