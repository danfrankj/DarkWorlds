import numpy as np
from scipy import optimize
from haloness import haloness

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
    return scipy.optimize.fmin_bfgs(f, x0, fprime=fprime, **kwargs)
    
def minimize_gdescent(f, x0, fprime, gtol=1e-5, maxiter=100, alpha=1., 
                        retall=True):
    path = []
    x = x0
    for i in xrange(maxiter):
        print i
        if retall:
            path.append(x)
        g = fprime(x)
        x = x - alpha * g
        if np.linalg.norm(g) < gtol:
            break
    return x, f(x), fprime(x), i == (maxiter - 1), path

def optimize_sky(sky, **kwargs):
    
    def opt_func(dm_xy):
        return haloness(dm_x=dm_xy[0], dm_y=dm_xy[1], sky=sky)
    grad = finitediff(opt_func, eps=10.)
    x0 = np.mean(sky, axis=0)[:2]
    return minimize_gdescent(f = opt_func, x0=x0, fprime=grad, **kwargs)


