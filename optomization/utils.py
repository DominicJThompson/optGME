import numpy as np
import legume
from legume import backend as bd

def NG(gme,kIndex,nIndex,height=None,Nx=60,Ny=125):
    """
    Calculate the group index (ng) for a specific mode and k-point in a photonic crystal waveguide.
    
    This function computes the group index by calculating the Poynting vector from the electric 
    and magnetic fields at a specific k-point and mode index. The group index is proportional to 
    the inverse of the energy velocity in the structure.
    
    Args:
        gme: Guided mode expansion object containing the photonic crystal information
        kIndex: Index of the k-point to use for calculation
        nIndex: Index of the mode to use for calculation
        height: Height within the slab to evaluate the fields (defaults to middle of slab)
        Nx: Number of points in x-direction for field calculation
        Ny: Number of points in y-direction for field calculation
        
    Returns:
        ng: Group index value
    """

    if height is None:
        height = gme.phc.layers[0].d/2

    Efield,_,_ = gme.get_field_xy('E',kIndex,nIndex,height,Nx=Nx,Ny=Ny)
    Hfield,_,_ = gme.get_field_xy('H',kIndex,nIndex,height,Nx=Nx,Ny=Ny)
    Efield = bd.array([[Efield['x']],[Efield['y']],[Efield['z']]])
    Hfield = bd.array([[Hfield['x']],[Hfield['y']],[Hfield['z']]])
    ng = -1/(bd.sum(bd.real(bd.cross(bd.conj(Efield),Hfield,axis=0)))*gme.phc.lattice.a2[1]/Nx/Ny*gme.phc.layers[0].d)
    return(ng)