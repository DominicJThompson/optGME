import legume
from scipy.optimize import Bounds, NonlinearConstraint
import autograd.numpy as np
from autograd import grad,jacobian
import legume.backend as bd
from optomization.utils import NG, backscatterLog

#temporary library
import os
import json


class ConstraintManager(object):
    """
    The class that defines constraints to be placed on the optomization.
    Each function of the class is a wrapper that defines another function
    """
    def __init__(self,x0=[],numberHoles=0,keep_feasible=False,**kwargs):
        """
        Initalize the relevent fields

        Args:
            x0: inital variables
            numberHoles: number of holes in total being changed
        """

        #this contains the constraints to be easily converted to scipy constraints:
        #{'name': {'type': 'ineq'/'eq', 'fun': constraint function, 'args': (tuple of args)},..}
        #the idea is to use list(self.constraints.values()) later to get scipy constraints
        self.constraints = {}

        #this contains the lower and uppoer bounds for all the input variables, 
        #these are used for rad limits and unit cell confinment
        self.lowerBounds = np.ones(len(x0))*-100
        self.upperBounds = np.ones(len(x0))*100

        #this contains the name and discription of the constraint:
        #{'name': {'discription': short discription, 'args': any relevent arguments}},..}
        #this is for saving purposes
        self.constraintsDisc = {}

        #defualt arguments from optimizer and initialized that need to be set
        self.defaultArgs = {'x0':x0,
                            'numberHoles':numberHoles,
                            **kwargs
                            }
        
        #set keep_feasible
        self.keep_feasible = keep_feasible

    def remove_constraint(self, name):
        """
        Remove a constraint by name.
        """
        if name in self.constraints:
            del self.constraints[name]
    
    def update_args(self, name, args):
        """
        Update the arguments of a specific constraint.
        """
        if name not in self.constraints:
            raise ValueError(f"Constraint {name} does not exist.")
        func = self.defaultArgs.get(name, (None,))[0]
        self.constraints[name]['fun'] = self._wrap_function(func, args)
        self.defaultArgs[name] = args

    def get_active_constraints(self):
        """
        Get a list of active constraints for optimization, formatted for SciPy.
        """
        return list(self.constraints.values())
    
    def _wrap_function(self, func, args):
        """
        Wrap the constraint function to pass updated arguments dynamically.
        """
        def wrapped(x):
            return func(x, *args)
        return wrapped
    
    def _wrap_function_vector_out(self, func, args):
        """
        Wrap the constraint function to pass updated arguments dynamically.
        This implementation for functions that output vectors
        """
        def wrapped(x):
            return func(x, *args)
        return wrapped
    
    def _wrap_grad(self, func, args):
        """
        Wraps the gradient of the constraints
        """
        def wrapped(x):
            return func(x,*args).tolist()
        return wrapped
    
    def build_bounds(self):
        return(Bounds(self.lowerBounds,self.upperBounds,keep_feasible=self.keep_feasible))

    #------------default constraints to add-------------
        
    def add_inside_unit_cell(self, name, bound):
        """
        Keep x and y values bound in box of [-bound,bound], assume
        Assume xs of shape [*xs,*ys,*rs]
        """
        xyIndices = len(self.defaultArgs['x0'])//3*2
        self.lowerBounds[:xyIndices] = self.defaultArgs['x0'][:xyIndices]-bound
        self.upperBounds[:xyIndices] = self.defaultArgs['x0'][:xyIndices]+bound

        self.constraintsDisc[name] = {
            'discription': """Keeps the x and y values bound in [-.5,.5] so they stay in the unit cell""",
            'args': bound,
            'type': 'bound'
        }
    
    def add_rad_bound(self,name,minRad,maxRad):
        """
        Enforces a min and max radius that the holes may not go below or above
        Assume xs of shape [*xs,*ys,*rs]
        """
        xyIndices = len(self.defaultArgs['x0'])//3*2
        self.lowerBounds[xyIndices:] = minRad
        self.upperBounds[xyIndices:] = maxRad
        
        self.constraintsDisc[name] = {
            'discription':  """Enforces the radius bounds""",
            'args': {'minRad': minRad, 'maxRad': maxRad},
            'type': 'bound'
        }
    
    def add_min_dist(self,name,minDist,buffer):
        """
        Enforces the minimum distance between holes. 
        Changes bounds so that they will keep the holes from interacting 
        with the holes we are not moving.

        Args: 
            buffer: the numbers of holes on either side that we enforce the distance 
        """
        nH = self.defaultArgs['numberHoles']
        self.constraints[name] = NonlinearConstraint(self._wrap_function_vector_out(self._min_dist,(buffer,)),
                                                     minDist,
                                                     np.inf,
                                                     jac=self._wrap_grad(jacobian(self._min_dist),(buffer,)))
        
        #change bounds on edges so the y value doesn't become too large or small 
        self.lowerBounds[nH:nH*2] = np.maximum(self.lowerBounds[nH:nH*2],np.ones(nH)*np.min(self.defaultArgs['x0'][nH:nH*2]))
        self.upperBounds[nH:nH*2] = np.minimum(self.upperBounds[nH:nH*2],np.ones(nH)*np.max(self.defaultArgs['x0'][nH:nH*2]))

        self.constraintsDisc[name] = {
            'discription': """Enforces a minimum radius between the holes within a buffer number of holes""",
            'args': {'minDist': minDist, 'buffer': buffer},
            'type': 'constraint'
        }

    def add_gme_constrs_complex(self,name,minFreqHard=0,minFreqSoft=0,maxFreq=100,minNG=0,maxNG=100,ksBefore=[0],ksAfter=[np.pi],bandwidth=.01,slope='down',path="temp.json"):
        """
        implements the folowing constraints:
        freq_bound,
        monotonic_band,
        bandwidth
        In addition to adding ng sign constraints on the k points on either side

        Args:
            minFreq: minimum alowable frequency
            maxFreq: maximum alowable frequency
            minNg: minimum alowable group index
            maxNg: maximum alowable group index
            ksBefore: a list of k values before the optomized kpoint
            ksAfter: a list of k values after the optomized kpoint
            bandwidth: The total bandwidth from the optomized mode, half on each side
            slope: either 'up' or 'down' depending on the inital band
        """


        self.constraints[name] = NonlinearConstraint(self._wrap_function_vector_out(self._gme_constrs_complex,(ksBefore,ksAfter,bandwidth,slope,path)),
                                                     np.array([minFreqHard,minFreqSoft,minNG,-np.inf,-np.inf,-np.inf]),
                                                     np.array([maxFreq,maxFreq,maxNG,0,0,0]),
                                                     jac=self._wrap_grad(jacobian(self._gme_constrs_complex),(ksBefore,ksAfter,bandwidth,slope,path)),
                                                     keep_feasible=[True,False,False,False,False,False])
        self.constraintsDisc[name] = {
            'discription': """implements the folowing constraints: freq_bound, ng_bound, monotonic_band, bandwidth, and ng sign constraints on either side """,
            'args': {'minFreqHard': minFreqHard, 'minFreqSoft': minFreqSoft, 'maxFreq': maxFreq,'minNG': minNG, 'maxNG': maxNG,'ksBefore': ksBefore, 'ksAfter': ksAfter, 'bandwidth': bandwidth, 'slope': slope},
            'type': 'constraint'
        }

    def add_gme_constrs_dispersion(self,name,minFreq=0,maxFreq=100,ksBefore=[0],ksAfter=[np.pi],bandwidth=.01,slope='down',path="temp.json"):
        """
        implements the folowing constraints:
        freq_bound,
        monotonic_band,
        bandwidth
        In addition to adding ng sign constraints on the k points on either side

        Args:
            minFreq: minimum alowable frequency
            maxFreq: maximum alowable frequency
            ksBefore: a list of k values before the optomized kpoint
            ksAfter: a list of k values after the optomized kpoint
            bandwidth: The total bandwidth from the optomized mode, half on each side
            slope: either 'up' or 'down' depending on the inital band
        """


        self.constraints[name] = NonlinearConstraint(self._wrap_function_vector_out(self._gme_constrs_dispersion,(ksBefore,ksAfter,bandwidth,slope,path)),
                                                     np.array([minFreq,-np.inf,-np.inf]),
                                                     np.array([maxFreq,0,0]),
                                                     jac=self._wrap_grad(jacobian(self._gme_constrs_dispersion),(ksBefore,ksAfter,bandwidth,slope,path)),
                                                     keep_feasible=[False,False,False])
        self.constraintsDisc[name] = {
            'discription': """implements the folowing constraints: freq_bound, monotonic_band, bandwidth""",
            'args': {'minFreq': minFreq, 'maxFreq': maxFreq,'ksBefore': ksBefore, 'ksAfter': ksAfter, 'bandwidth': bandwidth, 'slope': slope},
            'type': 'constraint'
        }

    def add_gme_constrs_dispersion_backscatter(self,name,minFreq=0,maxFreq=100,ksBefore=[0],ksAfter=[np.pi],bandwidth=.01,maxBackscatter=1,slope='down',backscatterParams={}):
        """
        implements the folowing constraints:
        freq_bound,
        monotonic_band,
        no overlapping bands
        maximum allowable backscatter
        In addition to adding ng sign constraints on the k points on either side

        Args:
            minFreq: minimum alowable frequency
            maxFreq: maximum alowable frequency
            ksBefore: a list of k values before the optomized kpoint
            ksAfter: a list of k values after the optomized kpoint
            bandwidth: The band should not be touching neighboring bands
            maxBackscatter: maximum allowable backscatter units of [loss/ng^2 [dB/cm]]
            slope: either 'up' or 'down' depending on the inital band
        """


        self.constraints[name] = NonlinearConstraint(self._wrap_function_vector_out(self._gme_constrs_dispersion_backscatter,(ksBefore,ksAfter,bandwidth,slope,backscatterParams)),
                                                     np.array([minFreq,-np.inf,-np.inf,0]),
                                                     np.array([maxFreq,0,0,maxBackscatter]),
                                                     jac=self._wrap_grad(jacobian(self._gme_constrs_dispersion_backscatter),(ksBefore,ksAfter,bandwidth,slope,backscatterParams)),
                                                     keep_feasible=[False,False,False,False])
        self.constraintsDisc[name] = {
            'discription': """implements the folowing constraints: freq_bound, monotonic_band, bandwidth""",
            'args': {'minFreq': minFreq, 'maxFreq': maxFreq,'ksBefore': ksBefore, 'ksAfter': ksAfter, 'bandwidth': bandwidth, 'slope': slope},
            'type': 'constraint'
        }
    
    #----------functions that define default constraints----------
    
    def _min_dist(self,x,buffer):
        varsR = x.reshape(3,-1)
        cs = bd.array([])
        for i in range(buffer):
            xOff = varsR[0,i+1:]-varsR[0,:-i-1]
            xOffW = bd.abs(xOff-np.floor(xOff+.5))
            c = bd.sqrt(xOffW**2+(varsR[1,i+1:]-varsR[1,:-i-1])**2)-varsR[2,i+1:]-varsR[2,:-i-1]
            cs = bd.hstack((cs,c))
        return(cs)

    
    def _gme_constrs_complex(self,x,ksBefore,ksAfter,bandwidth,slope,path):

        #start by setting up GME and running it for all points 
        phc = self.defaultArgs['crystal'](vars=x,**self.defaultArgs['phcParams'])
        gme = legume.GuidedModeExp(phc,self.defaultArgs['gmax'])

        #set up kpoints and run gme
        gmeParams = self.defaultArgs['gmeParams'].copy()
        kpointsBefore = bd.vstack((bd.array(ksBefore),bd.zeros(len(ksBefore))))
        kpointsAfter = bd.vstack((bd.array(ksAfter),bd.zeros(len(ksAfter))))
        kpoints = bd.hstack((kpointsBefore,gmeParams['kpoints'],kpointsAfter))
        gmeParams['kpoints'] = kpoints
        gmeParams['gmode_inds'] = self.defaultArgs['gmode_inds']
        gmeParams['numeig'] = self.defaultArgs['gmeParams']['numeig']+1
        gme.run(**gmeParams)

        #get adjustment for slope that will be used repeatedly
        if slope=='up':
            c = 1
        elif slope=='down':
            c = -1
        else:
            raise ValueError("slope within ng bound must be either 'up' or 'down'")

        #get frequency bound constraint
        freq = gme.freqs[len(ksBefore),self.defaultArgs['mode']]

        #get ng bound constraint
        ng = c*NG(gme,len(ksBefore),self.defaultArgs['mode'],Nx=100,Ny=125)

        #monotonic constraint
        monotonic = c*(gme.freqs[:-1,self.defaultArgs['mode']]-gme.freqs[1:,self.defaultArgs['mode']])
        monotonicOut = bd.max(monotonic)

        #bandwidth constraint
        above = bandwidth/2+freq-gme.freqs[-1,self.defaultArgs['mode']+1]
        below = bandwidth/2-freq+gme.freqs[1,self.defaultArgs['mode']-1]

        #combine constraints and return
        return(bd.hstack((freq,freq,ng,monotonicOut,above,below)))

    def _gme_constrs_dispersion(self,x,ksBefore,ksAfter,bandwidth,slope,path):

        #start by setting up GME and running it for all points 
        phc = self.defaultArgs['crystal'](vars=x,**self.defaultArgs['phcParams'])
        gme = legume.GuidedModeExp(phc,self.defaultArgs['gmax'])

        #set up kpoints and run gme
        gmeParams = self.defaultArgs['gmeParams'].copy()
        kpointsBefore = bd.vstack((bd.array(ksBefore),[0]))
        kpointsAfter = bd.vstack((bd.array(ksAfter),[0]))
        kpoints = bd.hstack((kpointsBefore,gmeParams['kpoints'][:,0].reshape(2,1),gmeParams['kpoints'][:,-1].reshape(2,1),kpointsAfter))
        gmeParams['kpoints'] = kpoints
        gmeParams['gmode_inds'] = self.defaultArgs['gmode_inds']
        gmeParams['numeig'] = self.defaultArgs['gmeParams']['numeig']+1
        gme.run(**gmeParams)

        #get adjustment for slope that will be used repeatedly
        if slope=='up':
            c = 1
        elif slope=='down':
            c = -1
        else:
            raise ValueError("slope within ng bound must be either 'up' or 'down'")

        #get frequency bound constraint
        freq_start = gme.freqs[1,self.defaultArgs['mode']]
        freq_end = gme.freqs[2,self.defaultArgs['mode']]

        #monotonic constraint
        monotonic = c*(gme.freqs[:-1,self.defaultArgs['mode']]-gme.freqs[1:,self.defaultArgs['mode']])
        monotonicOut = bd.max(monotonic)

        #bandwidth constraint
        above = bandwidth/2+freq_start-gme.freqs[-1,self.defaultArgs['mode']+1]
        below = bandwidth/2-freq_end+gme.freqs[0,self.defaultArgs['mode']-1]
        bandwidthOut = bd.max(bd.hstack((above,below)))

        #combine constraints and return
        return(bd.hstack((freq_start,monotonicOut,bandwidthOut)))

    def _gme_constrs_dispersion_backscatter(self,x,ksBefore,ksAfter,bandwidth,slope,backscatterParams):

        #start by setting up GME and running it for all points 
        phc = self.defaultArgs['crystal'](vars=x,**self.defaultArgs['phcParams'])
        gme = legume.GuidedModeExp(phc,self.defaultArgs['gmax'])

        #set up kpoints and run gme
        gmeParams = self.defaultArgs['gmeParams'].copy()
        kpointsBefore = bd.vstack((bd.array(ksBefore),len(ksBefore)*[0]))
        kpointsAfter = bd.vstack((bd.array(ksAfter),len(ksAfter)*[0]))
        kpoints = bd.hstack((kpointsBefore,gmeParams['kpoints'],kpointsAfter))
        gmeParams['kpoints'] = kpoints
        gmeParams['gmode_inds'] = self.defaultArgs['gmode_inds']
        gmeParams['numeig'] = self.defaultArgs['gmeParams']['numeig']+1
        gme.run(**gmeParams)

        #get adjustment for slope that will be used repeatedly
        if slope=='up':
            c = 1
        elif slope=='down':
            c = -1
        else:
            raise ValueError("slope within ng bound must be either 'up' or 'down'")

        #get frequency bound constraint
        freq_start = gme.freqs[len(ksBefore),self.defaultArgs['mode']]
        freq_end = gme.freqs[len(kpoints[0])-len(ksAfter)-1,self.defaultArgs['mode']]

        #monotonic constraint
        monotonic = c*(gme.freqs[:-1,self.defaultArgs['mode']]-gme.freqs[1:,self.defaultArgs['mode']])
        monotonicOut = bd.max(monotonic)

        #bandwidth constraint
        above = bandwidth/2+freq_start-gme.freqs[-1,self.defaultArgs['mode']+1]
        below = bandwidth/2-freq_end+gme.freqs[0,self.defaultArgs['mode']-1]
        bandwidthOut = bd.max(bd.hstack((above,below)))

        #backscatter constraint
        backscatters = []
        for i in range(len(gmeParams['kpoints'][0])-2):
            backscatters.append(10**backscatterLog(gme,phc,self.defaultArgs['mode'],k=1+i,**backscatterParams)/backscatterParams['a']/1E-7*10*np.log10(np.e))
        backscatterOut = bd.max(backscatters)

        #combine constraints and return
        return(bd.hstack((freq_start,monotonicOut,bandwidthOut,backscatterOut)))