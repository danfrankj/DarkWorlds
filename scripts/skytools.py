import numpy as np
import os
import inspect

def read_halos(skynum, test=False):
    prefix = '../data/Test_haloCounts.csv' if test else '../data/Training_halos.csv'
    halo_path = os.path.join(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))),
        prefix)

    nhalo = np.loadtxt(halo_path, delimiter=',', unpack=True,
                         usecols=(1,), dtype=int, skiprows=1)
    nhalo = nhalo[skynum-1]
    
    if test:
        # don't have coordinates for test set, of course!
        return nhalo, None
    
    # don't have any use for x_ref, y_ref for now, so skip..
    halo_coords = np.loadtxt(halo_path,
                             delimiter=',', unpack=True,
                             usecols=range(4, 10), skiprows=1).T
    
    tmp = halo_coords[skynum-1,:]
    # from file coordinates are in the format [x1,y1,x2,y2...] 
    # let's convert them to our format [x1,x2,x3,y1,y2...]
    halo_coords = np.zeros(2*nhalo)
    for ihalo in range(nhalo):
        halo_coords[ihalo] = tmp[2*ihalo]
        halo_coords[nhalo+ihalo] = tmp[2*ihalo+1]
    
    # skies are one-indexed!
    return nhalo, halo_coords

def read_sky(skynum, test=False):
    prefix = '../data/Test_Skies/' if test else '../data/Train_Skies'
    data_dir = os.path.join(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))), prefix)

    return np.loadtxt(
        os.path.join(data_dir, skyfile_name(skynum, test=test)), delimiter=',',
        unpack=True, usecols=(1, 2, 3, 4), skiprows=1).T


def skyfile_name(skynum, test=False):
    prefix = 'Test_Sky' if test else 'Training_Sky'
    return prefix + str(skynum) + '.csv'
