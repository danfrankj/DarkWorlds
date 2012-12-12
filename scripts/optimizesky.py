import scipy
from skytools import *
from haloness import *
import numpy as np
import viz

'''
from now on, all dark matter coordinates are stored [x1,x2,x3,y1,y2,y3]
they are only converted from/to [x1,y1,x2,y2,x3,y3] when doing I/O
'''

def model_elipticity(dm_x, dm_y, gal_x, gal_y, gal_e1, gal_e2, kernel):
    nhalo = dm_x.size
    ngal = gal_x.size
    
    # compute kernel weights and angles
    weights = np.zeros([ngal, nhalo])
    phis = np.zeros([ngal, nhalo])
    for ihalo in range(nhalo):
        weights[:, ihalo] = kernel(np.sqrt(np.power(gal_x - dm_x[ihalo], 2) +
                                           np.power(gal_y - dm_y[ihalo], 2)))
        phis[:, ihalo] = np.arctan((gal_y - dm_y[ihalo]) / (gal_x - dm_x[ihalo]))
        
    # solve analytically for strengths
    b = np.zeros(nhalo)
    A = np.zeros([nhalo, nhalo])
    for ihalo in range(nhalo):
        e_tang = -(gal_e1 * np.cos(2. * phis[:, ihalo]) +
                   gal_e2 * np.sin(2. * phis[:, ihalo]))
        b[ihalo] = np.sum(weights[:, ihalo] * e_tang)
        for jhalo in range(nhalo):
            A[ihalo, jhalo] = np.sum(weights[:, ihalo] * weights[:, jhalo] *
                                     np.cos(2 * (phis[:, ihalo] - phis[:, jhalo])))

    if np.linalg.cond(A) > 1e9:
        return np.zeros(gal_e1.shape), np.zeros(gal_e2.shape)
    alpha_star = np.linalg.solve(A, b)
    
    # negative strengths are not allowed
    #if np.min(alpha_star) <= 0.0:
    #    raise OptimizationException()
    # eventually we'd like a model where strengths greater than one are not allowed...
    alpha_star = np.minimum(alpha_star, 1.0)
    alpha_star = np.maximum(alpha_star, 0.0)
    
    return -np.sum(alpha_star * weights * np.cos(2 * phis), axis=1), -np.sum(alpha_star * weights * np.sin(2 * phis), axis=1)


def elipticity_error(gal_e1, gal_e2, model_e1, model_e2):
    return np.sum(np.power(gal_e1 - model_e1, 2) +
                  np.power(gal_e2 - model_e2, 2))

def fwrapper(gal_x, gal_y, gal_e1, gal_e2, nhalo, kernel):
    
    def f(halo_coords):
        dm_x = halo_coords[0:nhalo]
        dm_y = halo_coords[nhalo: 2 * nhalo]
        
        model_e1, model_e2 = model_elipticity(dm_x=dm_x, dm_y=dm_y,
                                              gal_x=gal_x, gal_y=gal_y,
                                              gal_e1=gal_e1, gal_e2=gal_e2,
                                              kernel=kernel)
        
        # add penalty for leaving domain (prevents simplex algo from wandering...)
        penalty = 1.*(-np.sum(np.minimum(halo_coords,0)) + np.sum(np.maximum(halo_coords-4200,0)))
        denom = elipticity_error(gal_e1, gal_e2, np.zeros(gal_e1.shape), np.zeros(gal_e2.shape)) 
        return elipticity_error(gal_e1, gal_e2, model_e1, model_e2)/denom + penalty

    return f

def fmin_random(f, nhalo, Ns):

    val_min = 1e30
    sol_min = None
    for ii in xrange(Ns):
        # produce a random halo configuration in the domain
        x0 = 4200*np.random.rand(2*nhalo)
        # hand over to native simplex algorithm
        sol = scipy.optimize.fmin(func=f, x0=x0, disp=0)
        val = f(sol)
        if (val < val_min):
            val_min = val
            sol_min = sol
        
    assert(sol_min != None)
    
    return sol_min

GRID_SCHEDULE = [20, 50, 100]

def predict(skynum, kernel=gaussian(1000.), Ngrid=None, plot=False, test=False, verbose=True):

    nhalo, halo_coords = read_halos(skynum, test=test)
    
    if (Ngrid == None):
        Ngrid = GRID_SCHEDULE[nhalo-1]

    if (verbose):
        print "Ngrid: " + str(Ngrid)
    
    sky = read_sky(skynum, test=test)
    gal_x, gal_y, gal_e1, gal_e2 = sky.T
        
    f = fwrapper(gal_x=gal_x, gal_y=gal_y, 
                 gal_e1=gal_e1, gal_e2=gal_e2,
                 nhalo=nhalo, kernel=kernel)
    
    # brute is deprecated in favor of simplex w/random starts...
    #grid_range = [(0, 4200)] * nhalo * 2
    #sol = scipy.optimize.brute(f, grid_range, Ns=Ngrid) #, finish=None)
    
    sol = fmin_random(f=f, nhalo=nhalo, Ns=Ngrid) 
    val = f(sol)
    
    #print sol, val 
    dm_x = sol[0: nhalo]
    dm_y = sol[nhalo: 2 * nhalo]
    
    if plot:
        viz.plot_sky(skynum, dm_x, dm_y, test=test)
        
    return dm_x, dm_y, val

def optimizeparam(Ns=100):
    
    Nparam = 2
    val_min = 1e30
    param_min = None
    for ii in xrange(Ns):
        x0 = np.random.rand(Nparam)
        # exponents are (0.0,2.0)
        x0[0] = 2.0*np.random.rand()
        #x0[1] = 2.0*np.random.rand()
        # coefficients are anyone's guess...
        x0[1] = 100.0*np.random.rand()
        #x0[3] = 100.0*np.random.rand()
        
        param = scipy.optimize.fmin(func=kernel_fun2, x0=x0, disp=0)
        val = kernel_fun(param)
        print param, val
        if (val < val_min):
            val_min = val
            param_min = param
    
    print "HEREHERE: ", param_min, val_min

def kernel_fun(param):
    param = np.abs(param)
    def kernel(dist):
        return np.exp(-np.power(dist/param[1], param[0]))
    error = 0.0
    for skynum in range(1, 101):
        n_halos, halo_coords = read_halos(skynum)
        gal_x,gal_y,gal_e1,gal_e2 = read_sky(skynum).T
        
        assert(n_halos==1)
        
        phi = np.arctan((gal_y - halo_coords[1])/(gal_x - halo_coords[0]))
        e_tang = -(gal_e1 * np.cos(2. * phi) +
                   gal_e2 * np.sin(2. * phi))
        dist = np.sqrt(np.power(gal_x - halo_coords[0], 2) + 
                       np.power(gal_y - halo_coords[1], 2))
        weight = kernel(dist)
        if (np.sum(weight*weight) > 1e-6):
            alpha = np.sum(weight*e_tang)/np.sum(weight*weight)
            alpha = np.minimum(alpha, 1.0)
            
            error += np.sum(np.power(e_tang - alpha*weight, 2)) #/np.sum(np.power(e_tang,2))
        else:
            error += 1000.0

    return error

def kernel_fun2(param):
    param = np.abs(param)
    def kernel(dist):
        return np.exp(-np.power(dist/param[1], param[0]))
    error = 0.0
    for skynum in range(1, 301):
        nhalo, halo_coords = read_halos(skynum)
        gal_x,gal_y,gal_e1,gal_e2 = read_sky(skynum).T
        
        f = fwrapper(gal_x=gal_x, gal_y=gal_y, 
                 gal_e1=gal_e1, gal_e2=gal_e2,
                 nhalo=nhalo, kernel=kernel)
        
        error += f(halo_coords)
        
    print param, error
    return error









def finitediff(f, eps):
    """
    Returns a function that computes the finite difference gradient for a given
    function f.  This is useful when you don't know the analytical gradient
    but will rarely be faster.

    Parameters
    ----------
    f - An function that must accept arguments f(x, *args, **kwds) and return
        a float.
    eps - each element of x is perturbed by eps to compute the gradient
        typical value range from 1e-3 to 1e-6 but can depend on the sensitivity
        of the function.

    Returns
    -------
    g - A function that takes arguments g(x, *args, **kwds) and returns the
        gradient of f in the form of an ndarray
    """
    def g(x, *args, **kwds):
        f0 = f(x, *args, **kwds)
        grad = np.empty(x.shape) * np.nan
        perturbation = np.empty(x.shape)
        for i in range(len(x)):
            perturbation.fill(0.)
            perturbation[i] = eps
            fi = f(x + perturbation, *args, **kwds)
            grad[i] = (fi - f0) / eps
        return grad
    return g


def minimize_bfgs(f, x0, fprime=None, **kwargs):
    """
    a wrapper around bfgs
    """
    return scipy.optimize.fmin_bfgs(f, x0, fprime=fprime, retall=True, **kwargs)


def minimize_gdescent(f, x0, fprime, gtol=1e-5, maxiter=10000, alpha=1000.,
                      retall=True):
    path = []
    x = x0
    for i in xrange(maxiter):
        print i
        if retall:
            path.append(x)
        g = fprime(x)
        print np.linalg.norm(g), x
        x = x - alpha * g
        if np.linalg.norm(g) < gtol:
            break
        return x, f(x), fprime(x), i == (maxiter - 1), path
    
    
def optimize_sky(sky, **kwargs):

    def opt_func(dm_xy):
        return -1. * haloness(dm_x=dm_xy[0], dm_y=dm_xy[1], sky=sky)
    grad = finitediff(opt_func, eps=1.)
    x0 = np.mean(sky, axis=0)[:2]
    x0 = np.array([1000., 2500.])
    #return minimize_bfgs(opt_func, x0)#, epsilon=1.0)
    return minimize_gdescent(f=opt_func, x0=x0, fprime=grad, **kwargs)


