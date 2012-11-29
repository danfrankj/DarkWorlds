import numpy as np
import sys
import os
import inspect

def read_halos(skynum):
    halo_path = os.path.join(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))), '../data/Training_halos.csv')
    
    n_halos = np.loadtxt(halo_path,\
                                delimiter=',', unpack=True,\
                                usecols=(1,), dtype=int, skiprows=1)
    # don't have any use for x_ref, y_ref for now, so skip..
    halo_coords = np.loadtxt(halo_path,\
                             delimiter=',', unpack=True,\
                             usecols=range(4,10),skiprows=1).T

    # skies are one-indexed!
    return(n_halos[skynum-1], halo_coords[skynum-1, :])

def read_sky(skynum):
    data_dir = os.path.join(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))), '../data/Train_Skies/')
    
    return np.loadtxt(
        os.path.join(data_dir, skyfile_name(skynum)), delimiter=',', 
        unpack=True, usecols=(1,2,3,4), skiprows=1).T;
        
def skyfile_name(skynum):
    return 'Training_Sky' + str(skynum) + '.csv'



