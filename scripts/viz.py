import numpy as np
import matplotlib.pyplot as plt
from skytools import read_sky, read_halos
from haloness import *

def plot_haloness(skynum, kernel=gaussian(1000.), N=200.):
    mysky = read_sky(skynum)
    # I hope these just point to mysky...
    # this is where a class might be useful
    gal_x,gal_y,gal_e1,gal_e2 = mysky.T
    
    margin = 100
    xplot = np.linspace(min(gal_x)-margin, max(gal_x)+margin, N)
    yplot = np.linspace(min(gal_y)-margin, max(gal_y)+margin, N)
    
    X,Y = np.meshgrid(xplot,yplot)
    Z = np.zeros(X.shape)
    
    # looping is NOT the right thing to do...
    for idx, x in enumerate(xplot):
        for idy, y in enumerate(yplot):
            Z[idy, idx] = haloness(x, y, mysky, kernel)
    
    #plt.contourf(X, Y, Z)
    plt.pcolor(X, Y, Z)
    plt.colorbar()
    plt.hold(True);
    
    n_halos, halo_coords = read_halos(skynum)
    for ihalo in range(n_halos):
        plt.scatter(halo_coords[ihalo*2], halo_coords[ihalo*2 + 1],\
                        color='white', s=100)
        
    plt.axis((min(gal_x)-margin, max(gal_x)+margin, min(gal_y)-margin, max(gal_y)+margin))
    
    plt.show()
    

def plot_sky(skynum, dm_x=None, dm_y=None):    
    
    n_halos, halo_coords = read_halos(skynum)
    
    gal_x,gal_y,gal_e1,gal_e2 = read_sky(skynum).T
    n_gal = gal_x.size
    ax = plt.figure().add_subplot(1,1,1)
    ax.patch.set_facecolor('black')
    ax.set_title('Training Sky ' + str(skynum) + ': ' +\
                  str(n_gal) + ' galaxies, ' + str(n_halos) +\
                   ' halos')
    
    margin = 100
    ax.axis((min(gal_x)-margin, max(gal_x)+margin, min(gal_y)-margin, max(gal_y)+margin))
        
    # plot galaxy centers
    ax.scatter(gal_x, gal_y, s=5, color='white')
    plt.hold(True);
        
    # plot ellipticity
    '''
    ellipticity definition is still slightly mysterious
    the definition (taken from the forums) of,
    \theta = \frac{1}{2} atan(e_2,e_1)
    has the right behavoir i.e,
    e_2 = 0, e_1 > 0 \implies \theta = 0
    e_1 = 0, e_2 > 0 \implies \theta = pi/4
    e_2 = 0, e_1 < 0 \implies \theta = pi/2
    etc.
    '''
    scale = 200.0; # scaling factor for ellipticity
    for igal in xrange(n_gal):
        assert(gal_e1[igal] > -1.0 and gal_e1[igal] < 1.0)
        assert(gal_e2[igal] > -1.0 and gal_e2[igal] < 1.0)
        e_mag = np.sqrt(gal_e1[igal]*gal_e1[igal] +
                           gal_e2[igal]*gal_e2[igal]);
        theta = 0.5*np.arctan2(gal_e2[igal],gal_e1[igal]);
        dx = scale*e_mag*np.cos(theta);
        dy = scale*e_mag*np.sin(theta);
        tmp_coords = np.empty((2,2))
        tmp_coords[0][0] = gal_x[igal] - dx
        tmp_coords[0][1] = gal_x[igal] + dx
        tmp_coords[1][0] = gal_y[igal] - dy
        tmp_coords[1][1] = gal_y[igal] + dy
        ax.plot(tmp_coords[0], tmp_coords[1], color='white')
                
    # plot halos
    for ihalo in range(n_halos):
        # try to get the blurred effect...
        for ii in range(1,10):
            ax.scatter(halo_coords[ihalo*2], halo_coords[ihalo*2 + 1],\
                                 color='white', alpha=0.5-ii/20.0, s=ii*50)
        

    plt.show();


'''
keep for paul reference... 
mysky = Sky(1)

class Sky(object):
    val = 1
    array = None

    def __init__(skynum):
        self.array = read_sky(skynum)

    def dostuff():
        return val

    def get_elipticiies():
'''
    

