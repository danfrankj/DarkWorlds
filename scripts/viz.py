import numpy as np
import matplotlib.pyplot as plt
from skytools import read_sky, read_halos
from haloness import *

def plot_haloness(skynum, kernel=gaussian(1000.), N=200.):
    mysky = read_sky(skynum)
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
    plt.hold(True)
    
    n_halos, halo_coords = read_halos(skynum)
    for ihalo in range(n_halos):
        plt.scatter(halo_coords[ihalo*2], halo_coords[ihalo*2 + 1],\
                        color='white', s=50)
        
    plt.axis((min(gal_x)-margin, max(gal_x)+margin, min(gal_y)-margin, max(gal_y)+margin))
    
    plt.show()

def plot_etang(skynum, kernel=gaussian(1000.)):
    n_halos, halo_coords = read_halos(skynum)
    gal_x,gal_y,gal_e1,gal_e2 = read_sky(skynum).T

    assert(n_halos==1)
    print halo_coords
    
    phi = np.arctan((gal_y - halo_coords[1])/(gal_x - halo_coords[0]))
    e_tang = -(gal_e1 * np.cos(2. * phi) +
               gal_e2 * np.sin(2. * phi))
    dist = np.sqrt(np.power(gal_x - halo_coords[0], 2) + 
                   np.power(gal_y - halo_coords[1], 2))
    weight = kernel(dist)
    alpha = np.sum(weight*e_tang)/np.sum(weight*weight)
    print alpha
    
    dist_model = np.linspace(0.0, np.max(dist), 100.0)
    weight_model = kernel(dist_model)
    e_tang_model = alpha*weight_model
    
    plt.plot(dist, e_tang, '.')
    plt.hold(True)
    plt.plot(dist_model, e_tang_model, 'r-', linewidth=1.5)
    
    plt.xlabel(r'$r$', fontsize=20)
    plt.ylabel(r'$e_{\mathrm{tangential}}$', fontsize=20)
    plt.axis([0.0, np.max(dist)+100.0, -1.0, 1.0])
    plt.show()

def plot_etang_train(kernel=gaussian(1000.)):
    
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
        alpha = np.sum(weight*e_tang)/np.sum(weight*weight)
        print alpha
    
        dist_model = np.linspace(0.0, np.max(dist), 100.0)
        weight_model = kernel(dist_model)
        e_tang_model = alpha*weight_model
    
        plt.plot(dist, e_tang*(alpha*kernel(0.)), 'k.')
        plt.hold(True)
        #plt.plot(dist_model, e_tang_model, 'r-', linewidth=1.5)
    
    plt.xlabel(r'$r$', fontsize=20)
    plt.ylabel(r'$e_{\mathrm{tangential}}$', fontsize=20)
    plt.axis([0.0, np.max(dist)+100.0, -1.0, 1.0])
    plt.show()
    

def plot_sky(skynum, dm_x=None, dm_y=None, test=False):    
    
    n_halos, halo_coords = read_halos(skynum, test=test)
    
    gal_x,gal_y,gal_e1,gal_e2 = read_sky(skynum, test=test).T
    n_gal = gal_x.size
    ax = plt.figure().add_subplot(1,1,1)
    ax.patch.set_facecolor('black')
    if (test):
        ax.set_title('Test Sky ' + str(skynum) + ': ' +\
                         str(n_gal) + ' galaxies, ' + str(n_halos) +\
                         ' halos')
    else:
        ax.set_title('Training Sky ' + str(skynum) + ': ' +\
                         str(n_gal) + ' galaxies, ' + str(n_halos) +\
                         ' halos')
    
    margin = 100
    ax.axis((min(gal_x)-margin, max(gal_x)+margin, min(gal_y)-margin, max(gal_y)+margin))
        
    # plot galaxy centers
    ax.scatter(gal_x, gal_y, s=5, color='white')
    plt.hold(True);
    
    # plot ellipticity
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
                
    if (test==False):
        # plot halos        
        for ihalo in range(n_halos):
            # try to get the blurred effect...
            for ii in range(1,10):
                ax.scatter(halo_coords[ihalo*2], halo_coords[ihalo*2 + 1],\
                               color='white', alpha=0.5-ii/20.0, s=ii*50)
            
    # plot predicted halo locations, if given
    if ((dm_x != None) | (dm_y != None)):
        assert ((dm_x != None) & (dm_y != None))
        assert ((n_halos == dm_x.size) & (n_halos == dm_y.size)) 
        
        plt.scatter(dm_x, dm_y, color="red", s=50)    

    plt.show();


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
    

