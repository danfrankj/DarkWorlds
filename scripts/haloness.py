import numpy as np
import scipy.stats

def exppow(d=0.4, bw=6.3):
    def mykernel(dist):
        return np.exp(-np.power(dist,d)/bw)
    return mykernel

def general_exp(param=[0.0, 0.0, 0.0, 0.0]):
    def mykernel(dist):
        return np.exp(-np.power(dist/param[2], param[0]) - 
                       np.power(dist/param[3], param[1]))
    return mykernel

def expsqrt(bw = 15.414):
    def mykernel(dist):
        return np.exp(-np.sqrt(dist)/bw)
    return mykernel

def invexp(bw = 976.884):
    def mykernel(dist):
        return np.exp(-dist/bw)
    return mykernel

def distpow(exp = -1.):
    def mykernel(dist):
        return np.power(dist, exp)
    return mykernel

def gaussian(bw=1000.):
    def mykernel(dist):
        return scipy.stats.norm.pdf(dist, loc=0, scale=bw)
    return mykernel

def haloness(dm_x, dm_y, sky, kernel):
    gal_x,gal_y,gal_e1,gal_e2 = sky.T
    dist = np.sqrt(np.power(gal_x - dm_x, 2) + np.power(gal_y - dm_y, 2))
    weight = kernel(dist)
    phi = np.arctan((gal_y - dm_y) / (gal_x - dm_x))
    e_tangental = -1. * (gal_e1 * np.cos( 2 * phi) + gal_e2 * np.sin(2 * phi))
    score = np.dot(e_tangental, weight)/np.sum(weight)

    return score



