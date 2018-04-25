
# coding: utf-8

import subprocess
import os

def get_procs():
    pid = os.getpid()
    procs = subprocess.check_output([ "lsof", '-w', '-Ff', "-p", str( pid ) ] ).splitlines()
    return [p for p in procs if p != b'ftxt']

# # 04 - Full waveform inversion with Devito and Dask

# ## Introduction
# 
# In this tutorial we show how [Devito](http://www.opesci.org/devito-public) and [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) are used with [Dask](https://dask.pydata.org/en/latest/#dask) to perform [full waveform inversion](https://www.slim.eos.ubc.ca/research/inversion) (FWI) on distributed memory parallel computers.

# ## scipy.optimize.minimize 
# 
# In this tutorial we use [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) to solve the FWI gradient based minimization problem rather than the simple grdient decent algorithm in the previous tutorial.
# 
# ```python
# scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
# ```
# 
# > Minimization of scalar function of one or more variables.
# >
# > In general, the optimization problems are of the form:
# >
# > minimize f(x) subject to
# >
# > g_i(x) >= 0,  i = 1,...,m
# > h_j(x)  = 0,  j = 1,...,p
# > where x is a vector of one or more variables. g_i(x) are the inequality constraints. h_j(x) are the equality constrains.
# 
# [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) provides a wide variety of methods for solving minimization problems depending on the context. Here we are going to focus on using L-BFGS via [scipy.optimize.minimize(method=’L-BFGS-B’)](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb)
# 
# ```python
# scipy.optimize.minimize(fun, x0, args=(), method='L-BFGS-B', jac=None, bounds=None, tol=None, callback=None, options={'disp': None, 'maxls': 20, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})```
# 
# The argument `fun` is a callable function that returns the misfit between the simulated and the observed data. If `jac` is a Boolean and is `True`, `fun` is assumed to return the gradient along with the objective function - as is our case when applying the adjoint-state method.

# ## What is Dask?
# 
# > [Dask](https://dask.pydata.org/en/latest/#dask) is a flexible parallel computing library for analytic computing.
# >
# > Dask is composed of two components:
# >
# > * Dynamic task scheduling optimized for computation...
# > * “Big Data” collections like parallel arrays, dataframes, and lists that extend common interfaces like NumPy, Pandas, or Python iterators to larger-than-memory or distributed environments. These parallel collections run on top of the dynamic task schedulers.
# >
# > Dask emphasizes the following virtues:
# > 
# > * Familiar: Provides parallelized NumPy array and Pandas DataFrame objects
# > * Flexible: Provides a task scheduling interface for more custom workloads and integration with other projects.
# > * Native: Enables distributed computing in Pure Python with access to the PyData stack.
# > * Fast: Operates with low overhead, low latency, and minimal serialization necessary for fast numerical algorithms
# > * Scales up: Runs resiliently on clusters with 1000s of cores
# > * Scales down: Trivial to set up and run on a laptop in a single process
# > * Responsive: Designed with interactive computing in mind it provides rapid feedback and diagnostics to aid humans
# 
# **We are going to use it here to parallelise the computation of the functional and gradient as this is the vast bulk of the computational expense of FWI and it is trivially parallel over data shots.**

# ## Setting up (synthetic) data
# In a real world scenario we work with collected seismic data; for the tutorial we know what the actual solution is and we are using the workers to also generate the synthetic data.

# In[1]:


#NBVAL_IGNORE_OUTPUT

import numpy as np

import scipy
from scipy import signal, optimize

from devito import configuration, Grid

from distributed import Client, LocalCluster, wait

import cloudpickle as pickle

# Import acoustic solver, source and receiver modules.
from examples.seismic import Model, demo_model
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import TimeAxis, PointSource, RickerSource, Receiver

# Import convenience function for plotting results
from examples.seismic import plot_image

def get_true_model():
    ''' Define the test phantom; in this case we are using
    a simple circle so we can easily see what is going on.
    '''
    return demo_model('circle-isotropic', vp=3.0, vp_background=2.5, 
                      origin=param['origin'], shape=param['shape'],
                      spacing=param['spacing'], nbpml=param['nbpml'])

def get_initial_model():
    '''The initial guess for the subsurface model.
    '''
    return demo_model('circle-isotropic', vp=2.5, vp_background=2.5, 
                      origin=param['origin'], shape=param['shape'],
                      spacing=param['spacing'], nbpml=param['nbpml'])

def load_shot_data(shot_id, dt):
    ''' Load shot data from disk, resampling to the model time step.
    '''
    src, rec = pickle.load(open("shot_%d.p"%shot_id, "rb"))
    
    return src.resample(dt), rec.resample(dt)

def dump_shot_data(shot_id, src, rec):
    ''' Dump shot data to disk.
    '''
    pickle.dump((src, rec), open('shot_%d.p'%shot_id, "wb"))
    
def generate_shotdata_i(param):
    """ Inversion crime alert! Here the worker is creating the
        'observed' data using the real model. For a real case
        the worker would be reading seismic data from disk.
    """
    shot_id = param['shot_id']

    import os.path
    if os.path.exists("shot_%d.p"%shot_id):
        return

    true_model = get_true_model()
    
    # Time step from model grid spacing
    dt = true_model.critical_dt

    # Set up source data and geometry.

    time_range = TimeAxis(start=param['t0'], stop=param['tn'], step=dt)
    src = RickerSource(name='src', grid=true_model.grid, f0=param['f0'],
                       time_range=time_range)

    src.coordinates.data[0, :] = [30, param['shot_id']*1000./(param['nshots']-1)]
    
    # Number of receiver locations per shot.
    nreceivers = 101

    # Set up receiver data and geometry.
    rec = Receiver(name='rec', grid=true_model.grid, time_range=time_range,
                   npoint=nreceivers)
    rec.coordinates.data[:, 1] = np.linspace(0, true_model.domain_size[0], num=nreceivers)
    rec.coordinates.data[:, 0] = 980. # 20m from the right end

    # Set up solver.
    solver = AcousticWaveSolver(true_model, src, rec, space_order=4)

    # Generate synthetic receiver data from true model.
    true_d, _, _ = solver.forward(src=src, m=true_model.m)

    dump_shot_data(shot_id, src, true_d)

    return None

def generate_shotdata(param, cluster):
    # print("def generate_shotdata")
    # Define work list
    work = [dict(param) for i in range(param['nshots'])]
    for i in  range(param['nshots']):
        work[i]['shot_id'] = i
        generate_shotdata_i(work[i])
        
    # Map worklist to cluster
    client = Client(cluster, nanny=False, nthreads=1)

    # print("mapping work")
    futures = client.map(generate_shotdata_i, work)

    # print("gathering")
    # Wait for all futures
    results = client.gather(futures)

    # print('closing')
    client.close()

    # print("generate_shotdata results ", results)


# ## Dask specifics
# 
# Previously we defined a function to calculate the individual contribution to the functional and gradient for each shot, which was then used in a loop over all shots. However, when using distributed frameworks such as Dask we instead think in terms of creating a worklist which gets *mapped* onto the worker pool. The sum reduction is also performed in parallel. For now however we assume that the scipy.optimize.minimize itself is running on the *master* process; this is a reasonable simplification because the computational cost of calculating (f, g) far exceeds the other compute costs.

# Because we want to be able to use standard reduction operators such as sum on (f, g) we first define it as a type so that we can define the `__add__` (and `__rand__` method).

# In[3]:


# Define a type to store the functional and gradient.
class fg_pair:
    def __init__(self, f, g):
        self.f = f
        self.g = g
    
    def __add__(self, other):
        f = self.f + other.f
        g = self.g + other.g
        
        return fg_pair(f, g)
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)


# ## Create operators for gradient based inversion
# To perform the inversion we are going to use [scipy.optimize.minimize(method=’L-BFGS-B’)](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb).
# 
# First we define the functional, ```f```, and gradient, ```g```, operator (i.e. the function ```fun```) for a single shot of data. This is the work that is going to be performed by the worker on a unit of data.

# In[4]:
import os
import signal
import sys
import time  

def handle_ipdb(sig, frame):
    import ipdb
    ipdb.Pdb().set_trace(frame)

from devito import Function

# Create FWI gradient kernel for a single shot
def gradient_i(param):
    # signal.signal(signal.SIGUSR1, handle_ipdb)

    from devito import clear_cache

    print("clear_cache")
    # Need to clear the workers cache.
    clear_cache()

    # Load the current model and the shot data for this worker.
    # Note, unlike the serial example the model is not passed in
    # as an argument. Broadcasting large datasets is considered
    # a programming anti-pattern and at the time of writing it
    # it only worked relaiably with Dask master. Therefore, the
    # the model is communicated via a file.
    print("pickle.load")
    model0 = pickle.load(open(param['model_filename'], "rb"))
    # print("load min/max m :: ", np.min(model0.m.data.flatten()), np.max(model0.m.data.flatten()))

    
    print("critical_dt")
    dt = model0.critical_dt
    # print("model0.critical_dt = ", model0.critical_dt)
    assert np.isfinite(dt)
     
    print("load_shot_data")
    src, rec = load_shot_data(param['shot_id'], dt)

    # Set up solver.
    print("AcousticWaveSolver")
    solver = AcousticWaveSolver(model0, src, rec, space_order=4)

    # Compute simulated data and full forward wavefield u0
    d, u0, _ = solver.forward(src=src, m=model0.m, save=True)
        
    # Compute the data misfit (residual) and objective function
    residual = Receiver(name='rec', grid=model0.grid,
                        time_range=rec.time_range,
                        coordinates=rec.coordinates.data)

    residual.data[:] = d.data[:] - rec.data[:]
    f = .5*np.linalg.norm(residual.data.flatten())**2
    
    # Compute gradient using the adjoint-state method. Note, this
    # backpropagates the data misfit through the model.
    grad = Function(name="grad", grid=model0.grid)
    solver.gradient(rec=residual, u=u0, m=model0.m, grad=grad)
    
    # Copying here to avoid a (probably overzealous) destructor deleting
    # the gradient before Dask has had a chance to communicate it.
    g = np.array(grad.data[:])
    
    # return the objective functional and gradient.
    return fg_pair(f, g)


# Define the global functional-gradient operator. This does the following:
# * Maps the worklist (shots) to the workers so that the invidual contributions to (f, g) are computed.
# * Sum individual contributions to (f, g) and returns the result.

def gradient(param, cluster):
    # Define work list
    worklist = []
    
    # print("def gradient::create work list")
    for i in  range(param['nshots']):
        work = {}
        for k in (key for key in param if key is not 'model'):
            work[k] = param[k]
        work['shot_id'] = i
        worklist.append(work)

    # Distribute worklist to workers.
    # print("def gradient::client")
    client = Client(cluster, nanny=False, nthreads=1)

    # print("map work")
    fgi = client.map(gradient_i, worklist, retries=1)
    
    # Perform reduction.
    # print("reduction")
    fg = client.submit(sum, fgi).result()

    # print("shutdown")
    client.close()

    # print("return")
    return fg.f, fg.g

# ## FWI with L-BFGS-B
# Equipped with a function to calculate the functional and gradient, we are finally ready to define the optimization function.

# In[6]:


from scipy import optimize

# Define bounding box constraints on the solution.
def apply_box_constraint(m):
    # Maximum possible 'realistic' velocity is 3.5 km/sec
    # Minimum possible 'realistic' velocity is 2 km/sec
    return np.clip(m, 1/3.5**2, 1/2**2)

def gradient_decent(model, param, cluster, ftol=0.1, maxiter=5):
    # Dump a copy of the current model for the workers
    # to pick up when they are ready.
    param['model'] = model

    # Run FWI with gradient descent
    history = np.zeros((param['fwi_iterations'], 1))
    
    for i in range(0, param['fwi_iterations']):
        print("iteration :: ", len(get_procs()), get_procs())

        param['model_filename'] = "model_%d.pkl"%i

        dtype = param['model'].dtype
        shape = param['model'].m.shape
        print(i, "dump min/max m :: ", np.min(param['model'].m.data.flatten()),
              np.max(param['model'].m.data.flatten()))
    
        pickle.dump(param['model'], open(param['model_filename'], "wb"))

        print("dumped")

        # Compute the functional value and gradient for the current
        # model estimate
        phi, direction = gradient(param, cluster)
        print("phi = ", phi)
        print("direction = ", np.max(direction))
        
        # Store the history of the functional values
        history[i] = phi
    
        # Artificial Step length for gradient descent
        # In practice this would be replaced by a Linesearch (Wolfe, ...)
        # that would guarantee functional decrease Phi(m-alpha g) <= epsilon Phi(m)
        # where epsilon is a minimum decrease constant
        alpha = .001 / np.max(direction)
        print("alpha = ", alpha)
        # Update the model estimate and inforce minimum/maximum values
        # param['model'].m.data[:] = apply_box_constraint(param['model'].m - alpha * direction)
        param['model'].m.data[:] = param['model'].m.data - alpha * direction
    
        # Log the progress made
        print('Objective value is %f at iteration %d' % (phi, i+1))


if __name__ == '__main__':
    configuration['log_level'] = 'ERROR'

    # Set up inversion parameters.
    param = {'t0': 0.,
             'tn': 1000.,              # Simulation last 1 second (1000 ms)
             'f0': 0.010,              # Source peak frequency is 10Hz (0.010 kHz)
             'nshots': 4,              # Number of shots to create gradient from
             'm_bounds': (0.08, 0.25), # Set the min and max slowness
             'shape': (101, 101),      # Number of grid points (nx, nz).
             'spacing': (10., 10.),    # Grid spacing in m. The domain size is now 1km by 1km.
             'origin': (0, 0),         # Need origin to define relative source and receiver locations.
             'nbpml': 40,              # nbpml thickness.
             'fwi_iterations': 30}

    print(LocalCluster.__doc__)
    print(Client.__doc__)
    cluster = LocalCluster(threads_per_worker=1)

    # Generate shot data.
    generate_shotdata(param, cluster)
    model0 = get_initial_model()
    result = gradient_decent(model0, param, cluster)

    cluster.close()
