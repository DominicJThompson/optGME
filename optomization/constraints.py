import legume
from scipy.optimize import Bounds, NonlinearConstraint
import autograd.numpy as np
from autograd import grad,jacobian
import legume.backend as bd


class ConstraintManager(object):
    """
    The class that defines constraints to be placed on the optomization.
    Each function of the class is a wrapper that defines another function
    """
    def __init__(self,x0=[],numberHoles=0,**kwargs):
        """
        Initalize the relevent fields

        Args:
            x0: inital variables
            numberHoles: number of holes on each side being changed
        """

        #this contains the constraints to be easily converted to scipy constraints:
        #{'name': {'type': 'ineq'/'eq', 'fun': constraint function, 'args': (tuple of args)},..}
        #the idea is to use list(self.constraints.values()) later to get scipy constraints
        self.constraints = {}

        #this contains the lower and uppoer bounds for all the input variables, 
        #these are used for rad limits and unit cell confinment
        self.lowerBounds = np.ones(len(x0))*100
        self.upperBounds = np.ones(len(x0))*-100

        #this contains the name and discription of the constraint:
        #{'name': {'discription': short discription, 'args': any relevent arguments}},..}
        #this is for saving purposes
        self.constraintsDisc = {}

        #defualt arguments from optimizer and initialized that need to be set
        self.defaultArgs = {'x0':x0,
                            'numberHoles':numberHoles,
                            **kwargs
                            }

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
        return(Bounds(self.lowerBounds,self.upperBounds))

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
    
    def add_min_dist(self,name,minDist,buffer,varsPadded):
        """
        Enforces a minimum distance between holes including the radius. 
        Enforce this distance for a buffer number of holes on each side
        Need the vars padded to get aditional positional information from holes we are not optomizing over
        Assume xs of shape [*xs,*ys,*rs], also assumes holes are ordered in the order they appear

        Args: 
            buffer: the numbers of holes on either side that we enforce the distance 
            varsPadded: inital variables with padding added on either side for additinoal holes, 
                        should be 2*buffer larger then vars for each variable
        """
        for i in range(self.defaultArgs['numberHoles']*2+buffer):
            for j in range(buffer):
                if i+j+1<buffer: #if hole is looking at another hole that doesn't move
                    continue
                self.constraints[name+str(i)+str(j)] = NonlinearConstraint(self._wrap_function(self._min_dist,(minDist,i,j+1,buffer,varsPadded)),
                                                             minDist,
                                                             np.inf,
                                                             jac = self._wrap_grad(grad(self._min_dist),(minDist,i,j+1,buffer,varsPadded)))
        self.constraintsDisc[name] = {
            'discription': """Enforces a minimum radius between the holes within a buffer number of holes""",
            'args': {'minDist': minDist, 'buffer': buffer},
            'type': 'constraint'
        }

    def add_freq_bound(self,name,minFreq,maxFreq):
        """
        Enforces the frequency beging bound within the region [minFreq,maxFreq]

        Args:
            minFreq: minimum alowable frequency
            maxFreq: maximum alowable frequency
        """
        self.constraints[name] = {
            'type': 'ineq',
            'fun': self._wrap_function(self._freq_bound,(minFreq,maxFreq,)),
            'jac': self._wrap_grad(grad(self._freq_bound),(minFreq,maxFreq,))
        }
        self.constraintsDisc[name] = {
            'discription': """Enforces the frequency to be within the given frequency range""",
            'args': {'minFreq': minFreq, 'maxFreq': maxFreq},
            'type': 'constraint'
        }
    
    def add_ng_bound(self,name,minNg,maxNg,slope='down'):
        """
        Enforces the ng beging bound within the region [minNg,maxNg], with the correct slope

        Args:
            minNg: minimum alowable group index
            maxNg: maximum alowable group index
            slope: either 'up' or 'down' depending on the desired slope
        """
        self.constraints[name] = {
            'type': 'ineq',
            'fun': self._wrap_function(self._ng_bound,(minNg,maxNg,slope,)),
            'jac': self._wrap_grad(grad(self._ng_bound),(minNg,maxNg,slope,))
        }
        self.constraintsDisc[name] = {
            'discription': """Enforces the group index to be within the given range with the correct slope""",
            'args': {'minNg': minNg, 'maxNg': maxNg, 'slope': slope},
            'type': 'constraint'
        }

    def add_monotonic_band(self,name,ksBefore,ksAfter,slope='down'):
        """
        forces the band to be monotonic in the selected ks

        Args:
            ksBefore: a list of k values before the optomized kpoint
            ksAfter: a list of k values after the optomized kpoint
            slope: either 'up' or 'down' depending on the inital band
        """
        self.constraints[name] = {
            'type': 'ineq',
            'fun': self._wrap_function_vector_out(self._monotonic_band,(ksBefore,ksAfter,slope,)),
            'jac': self._wrap_grad(jacobian(self._monotonic_band),(ksBefore,ksAfter,slope,))
        }
        self.constraintsDisc[name] = {
            'discription': """Enforces the band to be monatonic""",
            'args': {'ksBefore': ksBefore, 'ksAfter': ksAfter, 'slope': slope},
            'type': 'constraint'
        }

    def add_bandwidth(self,name,ksBefore,ksAfter,bandwidth):
        """
        Forces modes above and below to fall outside of the given bandwidth

        Args: 
            ksBefore: a list of k values before the optomized kpoint
            ksAfter: a list of k values after the optomized kpoint
            bandwidth: The total bandwidth from the optomized mode, half on each side
        """
        self.constraints[name] = {
            'type': 'ineq',
            'fun': self._wrap_function_vector_out(self._bandwidth,(ksBefore,ksAfter,bandwidth)),
            'jac': self._wrap_grad(jacobian(self._bandwidth),(ksBefore,ksAfter,bandwidth))
        }
        self.constraintsDisc[name] = {
            'discription': """Enforces a specific bandwidth that the bands above and below must obey, half the given bandwidth on each side """,
            'args': {'ksBefore': ksBefore, 'ksAfter': ksAfter, 'bandwidth': bandwidth},
            'type': 'constraint'
        }
        
    #def add_ng_others() #ensures that the ngs of the other k points are the correct sign near the optomized point
    def add_ng_others(self,name,ksBefore,ksAfter,slope='down'):
        """
        Force the group index of the modes coming before and after to be the same as the slope, 
        helps ensure monotonic

        Args: 
            ksBefore: a list of k values before the optomized kpoint
            ksAfter: a list of k values after the optomized kpoint
            slope: either 'up' or 'down' depending on the inital band
        """
        ks = bd.hstack((ksBefore,ksAfter))
        for i,k in enumerate(ks):
            self.constraints[name+str(i)] = {
                'type': 'ineq',
                'fun': self._wrap_function(self._ng_others,(k,slope)),
                'jac': self._wrap_grad(jacobian(self._ng_others),(k,slope))
            }
        self.constraintsDisc[name] = {
            'discription': """Enforces the group index of the neighboring modes to be the same slope, helps enforce monotonic""",
            'args': {'ksBefore': ksBefore, 'ksAfter': ksAfter, 'slope': slope},
            'type': 'constraint'
        }
    
    #This is a cheep version of the cache that adds all the constraints that require GME calculation
    #Currently missing ng constraints to prevent loop issues, need full jax backend implementaion
    def add_gme_constrs(self,name,minFreq=0,maxFreq=100,ksBefore=[0],ksAfter=[np.pi],bandwidth=.01,slope='down'):
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

        self.constraints[name] = {
            'type': 'ineq',
            'fun': self._wrap_function_vector_out(self._gme_constrs,(minFreq,maxFreq,ksBefore,ksAfter,bandwidth,slope)),
            'jac': self._wrap_grad(jacobian(self._gme_constrs),(minFreq,maxFreq,ksBefore,ksAfter,bandwidth,slope))
        }
        self.constraintsDisc[name] = {
            'discription': """implements the folowing constraints: freq_bound, ng_bound, monotonic_band, bandwidth, and ng sign constraints on either side """,
            'args': {'minFreq': minFreq, 'maxFreq': maxFreq,'ksBefore': ksBefore, 'ksAfter': ksAfter, 'bandwidth': bandwidth, 'slope': slope},
            'type': 'constraint'
        }

    def add_gme_constrs_complex(self,name,minFreq=0,maxFreq=100,ksBefore=[0],ksAfter=[np.pi],bandwidth=.01,slope='down'):
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


        self.constraints[name] = NonlinearConstraint(self._wrap_function_vector_out(self._gme_constrs_complex,(ksBefore,ksAfter,bandwidth,slope)),
                                                     np.array([minFreq,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]),
                                                     np.array([maxFreq,0,0,0,0,0]),
                                                     jac=self._wrap_grad(jacobian(self._gme_constrs_complex),(ksBefore,ksAfter,bandwidth,slope)))
        self.constraintsDisc[name] = {
            'discription': """implements the folowing constraints: freq_bound, ng_bound, monotonic_band, bandwidth, and ng sign constraints on either side """,
            'args': {'minFreq': minFreq, 'maxFreq': maxFreq,'ksBefore': ksBefore, 'ksAfter': ksAfter, 'bandwidth': bandwidth, 'slope': slope},
            'type': 'constraint'
        }
    
    #----------functions that define default constraints----------
    
    def _min_dist(self,x,minDist,i,j,buffer,varsPadded):
        #i indicates the hole to look at
        #j indicates the number of holes about i we should look

        numH = self.defaultArgs['numberHoles']
        if i<buffer:
            #we are looking at hole that is in the padding and comparing to variable hole
            xi, yi, ri = varsPadded[i], varsPadded[i+(numH+buffer)*2], varsPadded[i+4*(numH+buffer)]
            xj, yj, rj = x[i+j-buffer], x[i+j-buffer+2*numH], x[i+j-buffer+numH*4]
        
        elif i>=numH*2 and i+j>=numH*2+buffer:
            #we are looking at the last few changing holes
            xi,yi,ri = x[i-buffer], x[i-buffer+2*numH], x[i-buffer+numH*4]
            xj,yj,rj = varsPadded[i+j], varsPadded[i+j+(numH+buffer)*2], varsPadded[i+j+4*(numH+buffer)]

        else:
            #we ar fully inside the region where we are changing holes
            xi,yi,ri = x[i-buffer], x[i-buffer+2*numH], x[i-buffer+numH*4]
            xj,yj,rj = x[i-buffer+j], x[i-buffer+2*numH+j], x[i-buffer+numH*4+j]

        #distance between the holes
        dist = bd.sqrt((xi-xj)**2+(yi-yj)**2)-rj-ri

        return(dist)

    def _freq_bound(self,x,minFreq,maxFreq):

        #calculates the gme and then ensure the frequency is between the two value
        phc = self.defaultArgs['crystal'](vars=x,**self.defaultArgs['phcParams'])
        gme = legume.GuidedModeExp(phc,self.defaultArgs['gmax'])
        gme.run(**self.defaultArgs['gmeParams'])
        
        freq = gme.freqs[0,self.defaultArgs['mode']]

        return(bd.abs(freq-(minFreq+maxFreq)/2)-(maxFreq-minFreq)/2)

    def _ng_bound(self,x,minNg,maxNg,slope):

        #caclculate the gme and then ensure the gropu index is within the range
        phc = self.defaultArgs['crystal'](vars=x,**self.defaultArgs['phcParams'])
        gme = legume.GuidedModeExp(phc,self.defaultArgs['gmax'])
        gme.run(**self.defaultArgs['gmeParams'])

        Efield,_,_ = gme.get_field_xy('E',0,self.defaultArgs['mode'],phc.layers[0].d/2,Nx=60,Ny=125)
        Hfield,_,_ = gme.get_field_xy('H',0,self.defaultArgs['mode'],phc.layers[0].d/2,Nx=60,Ny=125)
        Efield = bd.array([[Efield['x']],[Efield['y']],[Efield['z']]])
        Hfield = bd.array([[Hfield['x']],[Hfield['y']],[Hfield['z']]])

        #adjust for the slope we want
        if slope=='up':
            c = 1
        elif slope=='down':
            c = -1
        else:
            raise ValueError("slope within ng bound must be either 'up' or 'down'")
        
        #calculate ng, accounting for slope
        ng = -c*1/(bd.sum(bd.real(bd.cross(bd.conj(Efield),Hfield,axis=0)))*phc.lattice.a2[1]/60/125*phc.layers[0].d)
        
        return(bd.abs(ng-(minNg+maxNg)/2)-(maxNg-minNg)/2)
    
    def _monotonic_band(self,x,ksBefore,ksAfter,slope):

        #caclculate the gme and then ensure the gropu index is within the range
        phc = self.defaultArgs['crystal'](vars=x,**self.defaultArgs['phcParams'])
        gme = legume.GuidedModeExp(phc,self.defaultArgs['gmax'])

        #set up kpoints and run gme
        gmeParams = self.defaultArgs['gmeParams'].copy()
        kpointsBefore = bd.vstack((bd.array(ksBefore),bd.zeros(len(ksBefore))))
        kpointsAfter = bd.vstack((bd.array(ksAfter),bd.zeros(len(ksAfter))))
        kpoints = bd.hstack((kpointsBefore,gmeParams['kpoints'],kpointsAfter))
        gmeParams['kpoints'] = kpoints
        gme.run(**gmeParams)

        #adjust for the slope we want
        if slope=='up':
            c = 1
        elif slope=='down':
            c = -1
        else:
            raise ValueError("slope within monotonic must be either 'up' or 'down'")

        return(c*(gme.freqs[:-1,self.defaultArgs['mode']]-gme.freqs[1:,self.defaultArgs['mode']]))
        
    def _bandwidth(self,x,ksBefore,ksAfter,bandwidth):

        phc = self.defaultArgs['crystal'](vars=x,**self.defaultArgs['phcParams'])
        gme = legume.GuidedModeExp(phc,self.defaultArgs['gmax'])

        #set up kpoints and run gme
        gmeParams = self.defaultArgs['gmeParams'].copy()
        kpointsBefore = bd.vstack((bd.array(ksBefore),bd.zeros(len(ksBefore))))
        kpointsAfter = bd.vstack((bd.array(ksAfter),bd.zeros(len(ksAfter))))
        kpoints = bd.hstack((kpointsBefore,gmeParams['kpoints'],kpointsAfter))
        gmeParams['kpoints'] = kpoints
        gmeParams['numeig'] = self.defaultArgs['gmeParams']['numeig']+1
        gme.run(**gmeParams)

        #freqs above
        optFreq = gme.freqs[len(ksBefore),self.defaultArgs['mode']]
        above = bandwidth/2+optFreq-gme.freqs[:,self.defaultArgs['mode']+1]

        #freqs below
        below = bandwidth/2-optFreq+gme.freqs[:,self.defaultArgs['mode']-1]

        return(bd.hstack((above,below)))
    
    def _ng_others(self,x,k,slope):

        phc = self.defaultArgs['crystal'](vars=x,**self.defaultArgs['phcParams'])
        gme = legume.GuidedModeExp(phc,self.defaultArgs['gmax'])

        #set up kpoints and run gme
        gmeParams = self.defaultArgs['gmeParams'].copy()
        kpoints = bd.vstack((bd.array(k),bd.zeros(1)))
        gmeParams['kpoints'] = kpoints
        gme.run(**gmeParams)

        #adjust for the slope we want
        if slope=='up':
            c = 1
        elif slope=='down':
            c = -1
        else:
            raise ValueError("slope within monotonic must be either 'up' or 'down'")

        #calculate the ng values
        Efield,_,_ = gme.get_field_xy('E',0,self.defaultArgs['mode'],phc.layers[0].d/2,Nx=60,Ny=125)
        Hfield,_,_ = gme.get_field_xy('H',0,self.defaultArgs['mode'],phc.layers[0].d/2,Nx=60,Ny=125)
        Efield = bd.array([[Efield['x']],[Efield['y']],[Efield['z']]])
        Hfield = bd.array([[Hfield['x']],[Hfield['y']],[Hfield['z']]])
        ng = c*1/(bd.sum(bd.real(bd.cross(bd.conj(Efield),Hfield,axis=0)))*phc.lattice.a2[1]/60/125*phc.layers[0].d)

        return(ng)
    
    def _gme_constrs(self,x,minFreq,maxFreq,ksBefore,ksAfter,bandwidth,slope):

        #start by setting up GME and running it for all points 
        phc = self.defaultArgs['crystal'](vars=x,**self.defaultArgs['phcParams'])
        gme = legume.GuidedModeExp(phc,self.defaultArgs['gmax'])

        #set up kpoints and run gme
        gmeParams = self.defaultArgs['gmeParams'].copy()
        kpointsBefore = bd.vstack((bd.array(ksBefore),bd.zeros(len(ksBefore))))
        kpointsAfter = bd.vstack((bd.array(ksAfter),bd.zeros(len(ksAfter))))
        kpoints = bd.hstack((kpointsBefore,gmeParams['kpoints'],kpointsAfter))
        gmeParams['kpoints'] = kpoints
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
        freq = gme.freqs[0,self.defaultArgs['mode']]
        boundF = (freq-(minFreq+maxFreq)/2)**2-((maxFreq-minFreq)/2)**2

        #get the ng bound constraint
        #Efield,_,_ = gme.get_field_xy('E',len(ksBefore),self.defaultArgs['mode'],phc.layers[0].d/2,Nx=60,Ny=125)
        #Hfield,_,_ = gme.get_field_xy('H',len(ksBefore),self.defaultArgs['mode'],phc.layers[0].d/2,Nx=60,Ny=125)
        #Efield = bd.array([[Efield['x']],[Efield['y']],[Efield['z']]])
        #Hfield = bd.array([[Hfield['x']],[Hfield['y']],[Hfield['z']]])
        #ng = -c*1/(bd.sum(bd.real(bd.cross(bd.conj(Efield),Hfield,axis=0)))*phc.lattice.a2[1]/60/125*phc.layers[0].d)
        #boundNG = -ng

        #monotonic constraint
        monotonic = c*(gme.freqs[:-1,self.defaultArgs['mode']]-gme.freqs[1:,self.defaultArgs['mode']])

        #bandwidth constraint
        above = bandwidth/2+freq-gme.freqs[-1,self.defaultArgs['mode']+1]
        below = bandwidth/2-freq+gme.freqs[1,self.defaultArgs['mode']-1]

        #ng direction correct before
        #Efield,_,_ = gme.get_field_xy('E',len(ksBefore)-1,self.defaultArgs['mode'],phc.layers[0].d/2,Nx=60,Ny=125)
        #Hfield,_,_ = gme.get_field_xy('H',len(ksBefore)-1,self.defaultArgs['mode'],phc.layers[0].d/2,Nx=60,Ny=125)
        #Efield = bd.array([[Efield['x']],[Efield['y']],[Efield['z']]])
        #Hfield = bd.array([[Hfield['x']],[Hfield['y']],[Hfield['z']]])
        #ng = -c*1/(bd.sum(bd.real(bd.cross(bd.conj(Efield),Hfield,axis=0)))*phc.lattice.a2[1]/60/125*phc.layers[0].d)
        #boundNGBefore = -ng

        #ng direction correct
        #Efield,_,_ = gme.get_field_xy('E',len(ksBefore)+1,self.defaultArgs['mode'],phc.layers[0].d/2,Nx=60,Ny=125)
        #Hfield,_,_ = gme.get_field_xy('H',len(ksBefore)+1,self.defaultArgs['mode'],phc.layers[0].d/2,Nx=60,Ny=125)
        #Efield = bd.array([[Efield['x']],[Efield['y']],[Efield['z']]])
        #Hfield = bd.array([[Hfield['x']],[Hfield['y']],[Hfield['z']]])
        #ng = -c*1/(bd.sum(bd.real(bd.cross(bd.conj(Efield),Hfield,axis=0)))*phc.lattice.a2[1]/60/125*phc.layers[0].d)
        #boundNGAfter = -ng

        #combine constraints and return
        return(bd.hstack((boundF,monotonic,above,below)))
    
    def _gme_constrs_complex(self,x,ksBefore,ksAfter,bandwidth,slope):

        #start by setting up GME and running it for all points 
        phc = self.defaultArgs['crystal'](vars=x,**self.defaultArgs['phcParams'])
        gme = legume.GuidedModeExp(phc,self.defaultArgs['gmax'])

        #set up kpoints and run gme
        gmeParams = self.defaultArgs['gmeParams'].copy()
        kpointsBefore = bd.vstack((bd.array(ksBefore),bd.zeros(len(ksBefore))))
        kpointsAfter = bd.vstack((bd.array(ksAfter),bd.zeros(len(ksAfter))))
        kpoints = bd.hstack((kpointsBefore,gmeParams['kpoints'],kpointsAfter))
        gmeParams['kpoints'] = kpoints
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
        freq = gme.freqs[0,self.defaultArgs['mode']]

        #get the ng bound constraint
        #Efield,_,_ = gme.get_field_xy('E',len(ksBefore),self.defaultArgs['mode'],phc.layers[0].d/2,Nx=60,Ny=125)
        #Hfield,_,_ = gme.get_field_xy('H',len(ksBefore),self.defaultArgs['mode'],phc.layers[0].d/2,Nx=60,Ny=125)
        #Efield = bd.array([[Efield['x']],[Efield['y']],[Efield['z']]])
        #Hfield = bd.array([[Hfield['x']],[Hfield['y']],[Hfield['z']]])
        #ng = -c*1/(bd.sum(bd.real(bd.cross(bd.conj(Efield),Hfield,axis=0)))*phc.lattice.a2[1]/60/125*phc.layers[0].d)
        #boundNG = -ng

        #monotonic constraint
        monotonic = c*(gme.freqs[:-1,self.defaultArgs['mode']]-gme.freqs[1:,self.defaultArgs['mode']])

        #bandwidth constraint
        above = bandwidth/2+freq-gme.freqs[-1,self.defaultArgs['mode']+1]
        below = bandwidth/2-freq+gme.freqs[1,self.defaultArgs['mode']-1]

        #ng direction correct before
        #Efield,_,_ = gme.get_field_xy('E',len(ksBefore)-1,self.defaultArgs['mode'],phc.layers[0].d/2,Nx=60,Ny=125)
        #Hfield,_,_ = gme.get_field_xy('H',len(ksBefore)-1,self.defaultArgs['mode'],phc.layers[0].d/2,Nx=60,Ny=125)
        #Efield = bd.array([[Efield['x']],[Efield['y']],[Efield['z']]])
        #Hfield = bd.array([[Hfield['x']],[Hfield['y']],[Hfield['z']]])
        #ng = -c*1/(bd.sum(bd.real(bd.cross(bd.conj(Efield),Hfield,axis=0)))*phc.lattice.a2[1]/60/125*phc.layers[0].d)
        #boundNGBefore = -ng

        #ng direction correct
        #Efield,_,_ = gme.get_field_xy('E',len(ksBefore)+1,self.defaultArgs['mode'],phc.layers[0].d/2,Nx=60,Ny=125)
        #Hfield,_,_ = gme.get_field_xy('H',len(ksBefore)+1,self.defaultArgs['mode'],phc.layers[0].d/2,Nx=60,Ny=125)
        #Efield = bd.array([[Efield['x']],[Efield['y']],[Efield['z']]])
        #Hfield = bd.array([[Hfield['x']],[Hfield['y']],[Hfield['z']]])
        #ng = -c*1/(bd.sum(bd.real(bd.cross(bd.conj(Efield),Hfield,axis=0)))*phc.lattice.a2[1]/60/125*phc.layers[0].d)
        #boundNGAfter = -ng

        #combine constraints and return
        return(bd.hstack((freq,monotonic,above,below)))