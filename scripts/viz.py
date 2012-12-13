import numpy as np
import matplotlib.pyplot as plt
from skytools import read_sky, read_halos
from haloness import *
import optimizesky as optsky

def plot_ellipticity(Ngal=20):
    
    e1, e2 = np.meshgrid(np.linspace(-1.0,1.0,Ngal), np.linspace(-1.0,1.0,Ngal))
    e1 = e1.ravel()
    e2 = e2.ravel()
    emag = np.sqrt(np.power(e1,2.0) + np.power(e2,2.0))
    ind = np.where(emag <= 1.0)
    e1 = e1[ind]
    e2 = e2[ind]
    Ngal = e1.size
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='equal')

    ax.scatter(e1, e2, s=2.5, color='black')
    ax.hold(True);
    
    # plot ellipticity
    scale = 0.08; # scaling factor for ellipticity
    for igal in xrange(Ngal):
        e_mag = np.sqrt(e1[igal]*e1[igal] +
                        e2[igal]*e2[igal]);
        theta = 0.5*np.arctan2(e2[igal], e1[igal]);
        dx = scale*e_mag*np.cos(theta);
        dy = scale*e_mag*np.sin(theta);
        tmp_coords = np.empty((2,2))
        tmp_coords[0][0] = e1[igal] - dx
        tmp_coords[0][1] = e1[igal] + dx
        tmp_coords[1][0] = e2[igal] - dy
        tmp_coords[1][1] = e2[igal] + dy
        ax.plot(tmp_coords[0], tmp_coords[1], color='black')
    
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
        
    theta = np.linspace(0.0, 2.0*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), '--', color='blue', linewidth=1.5)
    
    plt.annotate(r'$e_1$',
         xy=(1.0, 0.0), xycoords='data',
         xytext=(+10, +10), textcoords='offset points', fontsize=20)
    plt.annotate(r'$e_2$',
         xy=(0.0, 1.0), xycoords='data',
         xytext=(+5, +10), textcoords='offset points', fontsize=20)

    #plt.axis('equal')
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    plt.xticks([-1.0, 1.0])
    plt.yticks([-1.0, 1.0])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.85 ))
    #[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

    #plt.figure()
    #plt.xlabel(r'$e_1$', fontsize='20')
    #plt.ylabel(r'$e_2$', fontsize='20')

    plt.show()

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
    e_rad = -(gal_e1 * np.sin(2. * phi) -
              gal_e2 * np.cos(2. * phi))
    dist = np.sqrt(np.power(gal_x - halo_coords[0], 2) + 
                   np.power(gal_y - halo_coords[1], 2))
    weight = kernel(dist)
    alpha = np.sum(weight*e_tang)/np.sum(weight*weight)
    print alpha
    
    dist_model = np.linspace(0.0, np.max(dist), 100.0)
    weight_model = kernel(dist_model)
    e_tang_model = alpha*weight_model
    
    plt.figure()
    plt.plot(dist_model, e_tang_model, 'r-', linewidth=2.0)
    plt.hold(True)
    
    weight = gaussian()(dist)
    alpha = np.sum(weight*e_tang)/np.sum(weight*weight)
    
    weight_model = gaussian(1159.)(dist_model)
    e_tang_model = alpha*weight_model
    plt.plot(dist_model, e_tang_model, 'b-', linewidth=2.0)
    plt.plot(dist, e_tang, '.', color='black')
    plt.legend(['learned exponential model', 'gaussian model'])
    plt.xlabel(r'$\mathrm{radius} \ \ r$', fontsize=18)
    plt.ylabel(r'$\mathrm{tangential \ ellipticity} \ \ e_{\mathrm{tan}}$', fontsize=18)
    plt.axis([0.0, np.max(dist)+100.0, -1.0, 1.0])

    plt.figure()
    plt.plot(dist, e_rad, '.', color='black')
    plt.xlabel(r'$\mathrm{radius} \ \ r$', fontsize=18)
    plt.ylabel(r'$\mathrm{radial \ ellipticity} \ \ e_{\mathrm{rad}}$', fontsize=18)
    plt.axis([0.0, np.max(dist)+100.0, -1.0, 1.0])
    
    plt.show()

def diagnostic(Nrange, kernel=exppow()):
    
    plt.figure()
    plt.hold(True)
    
    for skynum in range(201, 231):
        nhalo, halo_coords = read_halos(skynum)
        gal_x,gal_y,gal_e1,gal_e2 = read_sky(skynum).T
        print skynum
        
        f = optsky.fwrapper(gal_x=gal_x, gal_y=gal_y, 
                            gal_e1=gal_e1, gal_e2=gal_e2,
                            nhalo=nhalo, kernel=kernel)
    
        val_true = f(halo_coords)
        
        
        val_array = np.zeros(len(Nrange))
        for ii in range(len(Nrange)):
            dm_x, dm_y, val = optsky.predict(skynum, Ngrid=Nrange[ii])
            val_array[ii] = val
            
        val_array = (val_array - val_true)/val_true
        
        plt.plot(Nrange, val_array, linewidth=0.5, color='black', linestyle='-')
        
        #plt.title('Training Sky ' + str(skynum) + ': '\
        #          + str(nhalo) + ' halos')
        plt.xlabel(r'$\mathrm{number \ of \ random \ starts}$')
        plt.ylabel(r'$\mathrm{normalized \ objective \ value}$')
      
    plt.plot(Nrange, np.zeros(len(Nrange)), linewidth=2.0, color='blue', linestyle='--')
    plt.show()

def plot_likelihood(skynum, pdf=None, kernel=exppow_lim(), N=20):
    
    if (pdf == None):
        pdf = optsky.build_pdf(kernel)

    nhalo, halo_coords = read_halos(skynum)
    sky = read_sky(skynum)
    gal_x, gal_y, gal_e1, gal_e2 = sky.T

    f = optsky.fwrapper(gal_x=gal_x, gal_y=gal_y, 
                        gal_e1=gal_e1, gal_e2=gal_e2,
                        nhalo=nhalo, kernel=kernel, pdf=pdf)

    margin = 0
    xplot = np.linspace(0.0, 4200.0, N)
    yplot = np.linspace(0.0, 4200.0, N)
    
    X,Y = np.meshgrid(xplot,yplot)
    Z = np.zeros(X.shape)
    
    for idx, x in enumerate(xplot):
        for idy, y in enumerate(yplot):
            Z[idy, idx] = f(np.array([x,y]))
    
    #plt.contourf(X, Y, Z)
    plt.pcolor(X, Y, Z)
    plt.colorbar()
    plt.hold(True)
    
    nhalo, halo_coords = read_halos(skynum)
    for ihalo in range(nhalo):
        plt.scatter(halo_coords[ihalo], halo_coords[ihalo + nhalo],\
                        color='white', s=50)
        
    plt.axis((min(gal_x)-margin, max(gal_x)+margin, min(gal_y)-margin, max(gal_y)+margin))
    
    plt.show()

def plot_pdf(kernel=exppow_lim(), N=10):
    my_pdf = optsky.build_pdf(kernel)
        
    X, Y = np.meshgrid(np.linspace(-1.0,1.0,N), np.linspace(-1.0,1.0,N))
    
    for model_emag in np.linspace(0.0,0.5,11):
        Z = my_pdf(np.array([model_emag*np.ones(X.size), X.ravel(), Y.ravel()]))
        Z = Z.reshape(X.shape)
        plt.figure()
        plt.pcolor(X, Y, Z)
        plt.colorbar()
        plt.hold(True)
        plt.scatter(model_emag, 0.0, s=10, color='white')
        plt.axis([-1.0,1.0,-1.0,1.0])
        plt.title("model_emag " + str(model_emag))

    plt.show()
    
    
    

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
    

