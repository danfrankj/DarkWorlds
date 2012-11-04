import numpy
import pylab
import sys

def main(skies):
    # load in all halos for now...
    # how to not use 2 file reads?
    n_halos = numpy.loadtxt('../data/Training_halos.csv',\
                                delimiter=',', unpack=True,\
                                usecols=(1,), dtype=int, skiprows=1)
    # don't have any use for x_ref, y_ref (only used for metric), so skip..
    halo_coords = numpy.loadtxt('../data/Training_halos.csv',\
                                    delimiter=',', unpack=True,\
                                    usecols=range(4,10),skiprows=1)
    
    for skystr in skies:
        # skies are 1-indexed...
        isky = int(skystr) - 1;
        assert(isky >= 0)                
        
        gal_x,gal_y,gal_e1,gal_e2 = numpy.loadtxt('../data/Train_Skies/Training_Sky' + skystr + '.csv',\
                                                      delimiter=',', unpack=True,\
                                                      usecols=(1,2,3,4), skiprows=1);
        
        n_gal = gal_x.size;

        this_fig = pylab.figure(isky).add_subplot(1,1,1);
	this_fig.patch.set_facecolor('black')
        this_fig.set_title('Training Sky ' + skystr + ': ' +\
                           str(n_gal) + ' galaxies, ' + str(n_halos[isky]) +\
                           ' halos');
        
	margin = 100
	this_fig.axis((min(gal_x)-margin, max(gal_x)+margin, 
                       min(gal_y)-margin, max(gal_y)+margin))
        
        # plot galaxy centers
	this_fig.scatter(gal_x, gal_y, s=5, color='white')
	pylab.hold(True);
        
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
            e_mag = numpy.sqrt(gal_e1[igal]*gal_e1[igal] +
                               gal_e2[igal]*gal_e2[igal]);
            theta = 0.5*numpy.arctan2(gal_e2[igal],gal_e1[igal]);
            dx = scale*e_mag*numpy.cos(theta);
            dy = scale*e_mag*numpy.sin(theta);
            tmp_coords = numpy.empty((2,2))
            tmp_coords[0][0] = gal_x[igal] - dx
            tmp_coords[0][1] = gal_x[igal] + dx
            tmp_coords[1][0] = gal_y[igal] - dy
            tmp_coords[1][1] = gal_y[igal] + dy
            this_fig.plot(tmp_coords[0], tmp_coords[1], color='white')
                    
        # plot halos
        for ihalo in range(n_halos[isky]):
            # try to get the blurred effect...
            for ii in range(1,10):
                this_fig.scatter(halo_coords[ihalo*2][isky], halo_coords[ihalo*2 + 1][isky],\
                                     color='white', alpha=0.5-ii/20.0, s=ii*50)
        

    pylab.show();
    
if __name__ == "__main__":
    main(sys.argv[1:])
