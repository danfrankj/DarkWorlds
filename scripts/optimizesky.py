import numpy as np
from scipy import optimize
import numpy.linalg
# probably verboten, but namespace is tiny for this project...
from haloness import *
from viz import plot_sky


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
    return minimize_gdescent(f = opt_func, x0=x0, fprime=grad, **kwargs)


'''
brute force optimization of ellipticity error function...
'''

def error(skynum, kernel):
    
    def f(halo_coords):

        # halo_coords is stacked coordinates - dm_x first, then dm_y
        nhalo = halo_coords.size/2
        dm_x = halo_coords[0:nhalo]
        dm_y = halo_coords[nhalo:2*nhalo]
        
        sky = read_sky(skynum)
        gal_x,gal_y,gal_e1,gal_e2 = sky.T
        ngal = gal_x.size
        
        # compute weights and angles
        weights = np.zeros([ngal, nhalo]) 
        phis    = np.zeros([ngal, nhalo]) 
        for ihalo in range(nhalo):
            weights[:,ihalo] = kernel(np.sqrt(np.power(gal_x - dm_x[ihalo], 2) + 
                                              np.power(gal_y - dm_y[ihalo], 2)))
            phis[:,ihalo] = np.arctan((gal_y - dm_y[ihalo]) / (gal_x - dm_x[ihalo]))
            
        # solve analytically for strengths
        b = np.zeros(nhalo)
        A = np.zeros([nhalo, nhalo])
        for ihalo in range(nhalo):
            e_tang = -(gal_e1 * np.cos(2.*phis[:,ihalo]) + 
                       gal_e2 * np.sin(2.*phis[:,ihalo]))
            b[ihalo] = np.sum(weights[:,ihalo]*e_tang)
            for jhalo in range(nhalo):
                A[ihalo, jhalo] = np.sum(weights[:,ihalo]*weights[:,jhalo]*
                                         np.cos(2*(phis[:,ihalo]-phis[:,jhalo])))
        
        if (np.linalg.cond(A) > 1e9):
            return 1e20
        alpha_star = np.linalg.solve(A,b)
        #print alpha_star
        # negative strengths are not allowed
        if (np.min(alpha_star) <= 0.0):
            return 1e20
        
        # compute error 
        for ihalo in range(nhalo):
            gal_e1 += alpha_star[ihalo]*weights[:,ihalo]*np.cos(2*phis[:,ihalo])
            gal_e2 += alpha_star[ihalo]*weights[:,ihalo]*np.sin(2*phis[:,ihalo])
                        
        return np.sum(gal_e1**2 + gal_e2**2)

    return f
        


def predict(skynum, kernel=gaussian(1000.), Ngrid=20, plot=False):
    
    nhalo, halo_coords = read_halos(skynum)
    sky = read_sky(skynum)
    
    grid_range = (0, 4200)*2*nhalo
    grid_range = [grid_range[i:i+2] for i in range(0, len(grid_range), 2)]
    
    sol = scipy.optimize.brute(error(skynum, kernel), grid_range, Ns=Ngrid, finish=None)
        
    print sol
    dm_x = sol[0:nhalo]
    dm_y = sol[nhalo:2*nhalo]
    
    if (plot==True):
        plot_sky(skynum, dm_x, dm_y)
    
    return 
