import numpy as np
import scipy
from viz import read_sky



def haloness(dm_x, dm_y, sky, bw=1000.):
    score = 0
    # SHOULD WE NORMALIZE THE WEIGHTS?
    for gal_x, gal_y, gal_e1, gal_e2 in sky:
        dist = np.sqrt(np.power(gal_x - dm_x, 2) + np.power(gal_y - dm_y, 2))
        weight = scipy.stats.norm.pdf(dist, loc=0, scale=bw)
        phi = np.arctan(gal_y - dm_y / (gal_x - dm_x))
        e_tangental = -1. * (gal_e1 * np.cos( 2 * phi) + gal_e2 * np.sin(2 * phi))
        score += e_tangental * weight
    return score
        
    
    