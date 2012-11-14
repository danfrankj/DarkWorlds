import numpy as np
import pylab
import sys
import os
import inspect
import matplotlib.pyplot as plt

def read_sky(skynum):
    data_dir = os.path.join(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))), '../data/Train_Skies/')
    
    return np.loadtxt(
        os.path.join(data_dir, skyfile_name(skynum)), delimiter=',', 
        unpack=True, usecols=(1,2,3,4), skiprows=1).T;
        
def skyfile_name(skynum):
    return 'Training_Sky' + str(skynum) + '.csv'

def plot_sky(skynum):    
    # load in all halos for now...
    # how to not use 2 file reads?
    n_halos = np.loadtxt('../data/Training_halos.csv',\
                                delimiter=',', unpack=True,\
                                usecols=(1,), dtype=int, skiprows=1)
    # don't have any use for x_ref, y_ref (only used for metric), so skip..
    halo_coords = np.loadtxt('../data/Training_halos.csv',\
                             delimiter=',', unpack=True,\
                             usecols=range(4,10),skiprows=1)
            
    gal_x,gal_y,gal_e1,gal_e2 = read_sky(skynum).T
    n_gal = gal_x.size
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.patch.set_facecolor('black')
    ax.set_title('Training Sky ' + str(skynum) + ': ' +\
                   str(n_gal) + ' galaxies, ' + str(n_halos[skynum-1]) +\
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
    \theta = \frac{1}{2} atan(e_2/e_1)
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
    for ihalo in range(n_halos[skynum - 1]):
        # try to get the blurred effect...
        for ii in range(1,10):
            ax.scatter(halo_coords[ihalo*2][skynum - 1], halo_coords[ihalo*2 + 1][skynum - 1],\
                                 color='white', alpha=0.5-ii/20.0, s=ii*50)
        

    plt.show();

