from legume.backend import backend as bd
import numpy as np 
from scipy.optimize import minimize
import time
import json
import types
from collections.abc import Iterable
import legume
import optomization
from autograd import grad


class Minimize(object):
    """
    Class that defines the optomization method that we will be using.
    """
    def __init__(self, x0, crystal, cost, mode=0, constraints={}, gmax=4.01, gmeParams={}, phcParams={}, tol=None):
        """
        Initalizes the class with all relevent general perameters

        Args:
            x0 : The inital perameters for hole potisition and radius
            crystal : The function defining the crystal we are optomizing on, should have input vars
            cost : the class object that defines cost function we are using
            mode : the mode we are optomizing on
            constraints : the class object that defines the types of constraints
            gmax : the gmax used in the gme computation
            gmeParams : the parameters to our GME simulation defined in dictionary
            phcParams : the parameters to our Photonic crystal definition, except for vars defined in dictionary
            tol : tolerance perameter needed for scipy.optomize.minimize. Default None
        """

        self.x0 = x0
        self.crystal = crystal
        self.cost = cost
        self.mode = mode
        self.constraints = constraints
        self.gmax = gmax
        self.gmeParams = gmeParams
        self.phcParams = phcParams
        self.tol = tol
        self.result = None
        

    def __str__(self):
        """Print the class parameters."""
        out = ""
        for key, value in self.__dict__.items():
            if key == 'crystal':
                out += f"{key}: {value.__name__}\n"
            elif key == 'cost':
                out += '-----cost-----\n'
                out += str(self.cost)
                out += '-----cost-----\n'
            else:
                out += f"{key}: {value}\n"
        return(out)

    def objective(self,vars):
        """
        defines the function that will be optomizaed over
        """
        phc = self.crystal(vars=vars,**self.phcParams)
        gme = legume.GuidedModeExp(phc,self.gmax)
        gme.run(**self.gmeParams)
        out = self.cost.cost(gme,phc,self.mode)
        return(out)
    
    def scipy_objective(self,vars):
        """
        defines the scipy wrapper for the objective functions written in jax
        """
        return(np.array(self.objective(vars)))

    def minimize(self):
        """
        defines the optomization method that will be used
        """
        raise NotImplementedError("minimize() needs to be implemented by"
                                  "minimize subclasses")
    
    def save(self,file_path):
        """
        defines the function that saves the information about the class and 
        results of the optomization to a json file. It clears any file there
        and then writes 

        Args:
            file_path : the path the file should be stored at
        """

        try:
            with open(self.path, "r") as f:
                savedData = json.load(f)  # Load existing data
        except (FileNotFoundError, json.JSONDecodeError):
            savedData = []  # Initialize if file doesn't exist or is empty\


        #builds the dict recersivly
        def build_dict(data):
            for key, value in data.items():
                if isinstance(value,(np.ndarray)):
                    data[key] = np.array(value).tolist()
                elif isinstance(value,set):
                    data[key] = list(value)
                elif isinstance(value,types.FunctionType):
                    data[key] = value.__name__
                elif isinstance(value, (optomization.Cost)):
                    data[key] = build_dict(value.__dict__)
                elif isinstance(value,optomization.ConstraintManager):
                    data[key] = value.constraintsDisc
                elif isinstance(value, dict):
                    data[key] = build_dict(value)
                elif not isinstance(value, Iterable):
                    data[key] = value
                elif all(isinstance(v, np.ndarray) for v in value):  
                    data[key] = [v.tolist() for v in value]
                else:
                    data[key] = value
            return(data)
            
        dataToSave = build_dict(self.__dict__)

        savedData.append(dataToSave)

        try:
            with open(file_path, 'w') as json_file:
                json.dump(savedData, json_file, indent=4)
        except IOError as e:
            raise IOError(f"An error occurred while writing to {file_path}: {e}")


class BFGS(Minimize):
    """
    runs the BFGS minimization method from scipy.optomize.minimize
    """

    def __init__(self, x0, crystal, cost, disp=False, maxiter=None, gtol=1e-5, return_all=False, **kwargs):
        """
        Defines all of the inputs needed to run optomizations with BFGS optomization class.
        Default values are the defaults from scipy

        Args: 
            disp : Bool, set to True to print convergance messages
            maxiter : Int or None, The maximimum number of iterations that will be run
            gtol : Float, the tolerance on the gradent to stop optoimization
            return_all : Bool, returns all intermediate steps at the end of optimization
        """
        super().__init__(x0,crystal,cost,**kwargs)
        self.disp = disp
        self.maxiter = maxiter
        self.gtol = gtol
        self.return_all = return_all

    def minimize(self):
        """
        function to call to preform minimization
        """

        #define grad function
        gradFunc = grad(self.objective)

        #scipy needs the gradient as a numpy array
        def scipy_grad(var):
            return(np.array(gradFunc(var)))
        
        #get the inital value to save
        self.inital_cost = self.objective(self.x0)
        
        #run optomization
        t1 = time.time()
        result = minimize(fun=self.scipy_objective,    
                          x0=self.x0,            
                          jac=scipy_grad, 
                          method='BFGS',
                          options={'disp': self.disp,
                                   'maxiter': self.maxiter,
                                   'gtol': self.gtol,
                                   'return_all': self.return_all}
                        )
        
        #save result
        self.result = dict(result.items())
        self.time = time.time()-t1


class TrustConstr(Minimize):
    """
    Class that defines the trust trust-constr scipy method
    """
    def __init__(self, x0, crystal, cost, 
                 gtol=1e-8, xtol=1e-8, barrier_tol=1e-8, 
                 sparse_jacobian=None, initial_tr_radius=1, 
                 initial_constr_penalty=1, initial_barrier_parameter=.1, 
                 initial_barrier_tolerance=.1, factorization_method=None, 
                 finite_diff_rel_step=None, maxiter=1000, verbose=0, 
                 disp=False, path=None, **kwargs):
        """
        Defines the optomization routine for the trust-constr scipy class,
        for a definition of arguments please look to scipy docs at:
        https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html
        """
        super().__init__(x0,crystal,cost,**kwargs)
        self.gtol = gtol
        self.xtol = xtol
        self.barrier_tol = barrier_tol
        self.sparse_jacobian = sparse_jacobian
        self.initial_tr_radius = initial_tr_radius
        self.initial_constr_penalty = initial_constr_penalty
        self.initial_barrier_parameter = initial_barrier_parameter
        self.initial_barrier_tolerance = initial_barrier_tolerance
        self.factorization_method = factorization_method
        self.finite_diff_rel_step = finite_diff_rel_step
        self.maxiter = maxiter
        self.verbose = verbose
        self.disp = disp

        #for saving the mid run information to a file
        self.path = path

    def minimize(self):
        
        #define grad function
        gradFunc = grad(self.objective)

        #scipy needs the gradient as a numpy array
        def scipy_grad(var):
            return(np.array(gradFunc(var)))
        
        #get the inital value to save
        self.inital_cost = self.objective(self.x0)

        #if we want to save data, set it up
        if self.path is not None:
            # Ensure the JSON file starts as an empty list (if it doesn't exist)
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)  # Load existing data
            except (FileNotFoundError, json.JSONDecodeError):
                data = []  # Initialize if file doesn't exist or is empty\

        #callback function that saves data if we want to 
        def callback(xk, result=None):
            if result is not None and self.path is not None:
                iteration_data = {
                    "iteration": getattr(result, "nit", None),
                    "objective_value": getattr(result, "fun", None),
                    "x_values": xk.tolist(),
                    "trust_region_radius": getattr(result, "tr_radius", None),
                    "optimality": getattr(result, "optimality", None),
                    "constraint_violation": getattr(result, "constr_violation", None),
                    "penalty": getattr(result, "penalty", None),
                    "barrier_parameter": getattr(result, "barrier_parameter", None),
                    "cg_iterations": getattr(result, "cg_niter", None),
                    "cg_stop_condition": getattr(result, "cg_stop_cond", None),
                }

                # Append the new iteration to the JSON file
                data.append(iteration_data)
                with open(self.path, "w") as f:
                    json.dump(data, f, indent=4)

        #run optomization
        t1 = time.time()
        result = minimize(fun=self.scipy_objective,    
                          x0=self.x0,            
                          jac=scipy_grad, 
                          method='trust-constr',
                          constraints=list(self.constraints.constraints.values()),
                          bounds=self.constraints.build_bounds(),
                          callback=callback,
                          options={'gtol': self.gtol,
                                   'xtol': self.xtol,
                                   'barrier_tol': self.barrier_tol,
                                   'sparse_jacobian': self.sparse_jacobian,
                                   'initial_tr_radius': self.initial_tr_radius,
                                   'initial_constr_penalty': self.initial_constr_penalty,
                                   'initial_barrier_parameter': self.initial_barrier_parameter,
                                   'initial_barrier_tolerance': self.initial_barrier_tolerance,
                                   'factorization_method': self.factorization_method,
                                   'finite_diff_rel_step':self.finite_diff_rel_step,
                                   'maxiter': self.maxiter,
                                   'verbose': self.verbose,
                                   'disp': self.disp}
                        )
        
        #save result
        self.result = dict(result.items())
        self.time = time.time()-t1


class Adam(Minimize):
    """
    Runs the adam minimization routine
    """

    def __init__(self, x0, crystal, cost, beta1 = .9,
                beta2 = .999, numiter = 100, alpha = .1,**kwargs):
        """
        Initiallizes the adam optomization method

        Args: 
            beta1: controls first momentum (default .9)
            beta2: controls second momentum (default .999)
            numIter: the maximum number of iterations it will run (default 100)
        """

        super().__init__(x0,crystal,cost,**kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.numIter = numiter
        self.alpha = alpha

    def minimize(self):
        "runs the minimization routine for adam"

        #define grad function
        gradFunc = jax.grad(self.objective)

        #get the inital value to save
        self.inital_cost = self.objective(self.x0)

        #initialize variables for adam
        m = np.zeros(len(self.x0))
        v = np.zeros(len(self.x0))
        vars = self.x0

        for i in range(self.numIter):
            
            #get the derivative
            g = gradFunc(vars)

            #run the adam routine
            m = self.beta1*m+(1-self.beta1)*g
            v = self.beta2*v+(1-self.beta2)*(g**2)

            #get the intermediate varibales 
            mhat = m/(1-self.beta1**(i+1))
            vhat = v/(1-self.beta2**(i+1))

            #update variable
            vars = vars-self.alpha*(mhat/(bd.sqrt(vhat)+1E-8))

        

