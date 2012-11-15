
import numpy as np
from scipy import optimize

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
        grad = numpylib.nans(x.shape)
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
    
def minimize_gdescent(f, x0, fprime, gtol=1e-5, maxiter=10000, alpha=.001, retall=True):
    x = x0
    for i in xrange(maxiter):
        g = fprime(x)
        x = x - alpha * g
        if np.linalg.norm(g) < gtol:
            break
    return x, f(x), fprime(x), i == (maxiter - 1)
