import numpy as np
import matplotlib.pyplot as plt
from skytools import read_sky, read_halos
from haloness import *
import optimizesky as optsky

def plot_sky(skynum, dm_x=None, dm_y=None, test=False):    
    
    nhalo, halo_coords = read_halos(skynum, test=test)
    
    gal_x,gal_y,gal_e1,gal_e2 = read_sky(skynum, test=test).T
    n_gal = gal_x.size
    ax = plt.figure().add_subplot(1,1,1)
    ax.patch.set_facecolor('black')
    if (test):
        ax.set_title('Test Sky ' + str(skynum) + ': ' +\
                         str(n_gal) + ' galaxies, ' + str(nhalo) +\
                         ' halos')
    else:
        ax.set_title('Training Sky ' + str(skynum) + ': ' +\
                         str(n_gal) + ' galaxies, ' + str(nhalo) +\
                         ' halos')
    
    margin = 100
    ax.axis((min(gal_x)-margin, max(gal_x)+margin, 
             min(gal_y)-margin, max(gal_y)+margin))
        
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
        for ihalo in range(nhalo):
            # try to get the blurred effect...
            for ii in range(1,10):
                ax.scatter(halo_coords[ihalo], halo_coords[ihalo + nhalo],\
                               color='white', alpha=0.5-ii/20.0, s=ii*50)
            
    # plot predicted halo locations, if given
    if ((dm_x != None) | (dm_y != None)):
        assert ((dm_x != None) & (dm_y != None))
        assert ((nhalo == dm_x.size) & (nhalo == dm_y.size)) 
        
        plt.scatter(dm_x, dm_y, color="red", s=50)    

    plt.show();

def plot_etang(skynum, kernel=gaussian(1000.)):
    nhalo, halo_coords = read_halos(skynum)
    gal_x,gal_y,gal_e1,gal_e2 = read_sky(skynum).T

    assert(nhalo==1)
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
    weight = gaussian(1159.)(dist)
    alpha = np.sum(weight*e_tang)/np.sum(weight*weight)
    weight_model = gaussian(1159.)(dist_model)
    e_tang_model = alpha*weight_model
    plt.plot(dist_model, e_tang_model, 'g-', linewidth=1.5)

    plt.xlabel(r'$r$', fontsize=20)
    plt.ylabel(r'$e_{\mathrm{tangential}}$', fontsize=20)
    plt.axis([0.0, np.max(dist)+100.0, -1.0, 1.0])
    plt.show()

def diagnostic(skynum, Nrange, kernel=exppow()):
    nhalo, halo_coords = read_halos(skynum)
    
    sky = read_sky(skynum)
    gal_x, gal_y, gal_e1, gal_e2 = sky.T
    
    f = optsky.fwrapper(gal_x=gal_x, gal_y=gal_y, 
                 gal_e1=gal_e1, gal_e2=gal_e2,
                 nhalo=nhalo, kernel=kernel)
    
    sol_data = np.zeros(2*nhalo)
    for ihalo in range(nhalo):
        sol_data[ihalo] = halo_coords[2*ihalo]
        sol_data[nhalo+ihalo] = halo_coords[2*ihalo+1]
        
    val_data = f(sol_data)
    print halo_coords, sol_data, val_data
    
    val_array = np.zeros(len(Nrange))
    for ii in range(len(Nrange)):
        dm_x, dm_y, val = optsky.predict(skynum, Ngrid=Nrange[ii])
        print val
        if (val > 1e5):
            val = -1.0
        val_array[ii] = val
        
    val_max = np.max(val_array)
    for ii in range(len(Nrange)):
        if (val_array[ii] < 0.0):
            val_array[ii] = val_max
    
            
    plt.plot(Nrange, val_array, linewidth=2.5, color='blue', linestyle='-')
    plt.hold(True)
    tmprange = np.linspace(min(Nrange), max(Nrange),100)
    tmpval = val_data*np.ones(100)
    plt.plot(tmprange, tmpval, linewidth=2.5, color='red', linestyle='--')
    
    plt.title('Training Sky ' + str(skynum) + ': '\
                  + str(nhalo) + ' halos')
    plt.xlabel('Ngrid')
    plt.ylabel('f(xstar)')

    plt.show()

def plot_polar(kernel=exppow()):
    
    all_data = []
        
    for skynum in range(1, 101):
        nhalo, halo_coords = read_halos(skynum)
        gal_x,gal_y,gal_e1,gal_e2 = read_sky(skynum).T
        print skynum, halo_coords
        dm_x = halo_coords[0:nhalo]
        dm_y = halo_coords[nhalo: 2 * nhalo]
        
        model_e1, model_e2 = optsky.model_elipticity(dm_x=dm_x, dm_y=dm_y,
                                                    gal_x=gal_x, gal_y=gal_y,
                                                    gal_e1=gal_e1, gal_e2=gal_e2,
                                                    kernel=kernel)
        model_e1 = model_e1.T
        model_e2 = model_e2.T
        assert(nhalo==1)
        '''
        phi = np.arctan((gal_y - halo_coords[1])/(gal_x - halo_coords[0]))
        e_tang = -(gal_e1 * np.cos(2. * phi) +
                   gal_e2 * np.sin(2. * phi))
        dist = np.sqrt(np.power(gal_x - halo_coords[0], 2) + 
                   np.power(gal_y - halo_coords[1], 2))
        weight = kernel(dist)
        alpha = np.sum(weight*e_tang)/np.sum(weight*weight)
        
        model_e1 = -alpha*weight*np.cos(2.0*phi)
        model_e2 = -alpha*weight*np.sin(2.0*phi)
        '''
        model_emag = np.sqrt(model_e1*model_e1 + model_e2*model_e2)
        # limit predicted ellipticity to emag = 1
        coeff_emag = np.maximum(model_emag, 1.0)
        model_e1 /= coeff_emag
        model_e2 /= coeff_emag
        model_emag = np.sqrt(model_e1*model_e1 + model_e2*model_e2)
        assert(max(model_emag) <= (1.0 + 1e-10))
        
        gal_emag = np.sqrt(gal_e1*gal_e1 + gal_e2*gal_e2)
        theta = np.arccos((gal_e1*model_e1 + gal_e2*model_e2)/gal_emag/model_emag)
        
        all_data.append(np.array([model_emag,theta,gal_emag]).T)
    
    # convert list into monster array
    all_data = np.vstack(all_data)
    # sort based on pred_emag
    all_data = all_data[all_data[:,0].argsort()]
    
    # place data into pred_emag bins
    Nbins = 10
    emag_bins = np.power(np.linspace(0.0,1.0,Nbins+1),2.0);
    ind = np.zeros(Nbins+1)
    print emag_bins
    for ii in range(Nbins-1):
        ind[ii+1] = np.min(np.where(all_data[:,0] > emag_bins[ii+1]))
    
    ind[Nbins] = all_data.shape[0]-1     # last index should always be end of array
    ind = ind.astype(int)
    print ind
    print all_data[ind,0]
    
    ax = plt.subplot(111, polar=True)
    
    i0 = 4
    i1 = 5
    
    plt.scatter(-all_data[ind[i0]:ind[i1], 1], all_data[ind[i0]:ind[i1], 2], s=5)
    plt.scatter([0.0,0.0], [all_data[ind[i0],0], all_data[ind[i1],0]], s=10, color='red')
    plt.show()

    return all_data
    

def plot_etang_train(kernel=gaussian(1000.)):
    
    for skynum in range(1, 101):
        nhalo, halo_coords = read_halos(skynum)
        gal_x,gal_y,gal_e1,gal_e2 = read_sky(skynum).T

        assert(nhalo==1)
        
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
    
    nhalo, halo_coords = read_halos(skynum)
    for ihalo in range(nhalo):
        plt.scatter(halo_coords[ihalo*2], halo_coords[ihalo*2 + 1],\
                        color='white', s=50)
        
    plt.axis((min(gal_x)-margin, max(gal_x)+margin, min(gal_y)-margin, max(gal_y)+margin))
    
    plt.show()


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
    

