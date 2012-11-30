import numpy as np
import os
import inspect


def read_halos(skynum, test=False):
    prefix = '../data/Test_haloCounts.csv' if test else '../data/Training_halos.csv'
    halo_path = os.path.join(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))),
        prefix)

    n_halos = np.loadtxt(halo_path, delimiter=',', unpack=True,
                         usecols=(1,), dtype=int, skiprows=1)
    # don't have any use for x_ref, y_ref for now, so skip..
    if test:
        return n_halos[skynum -1], None
    halo_coords = np.loadtxt(halo_path,
                             delimiter=',', unpack=True,
                             usecols=range(4, 10), skiprows=1).T

    # skies are one-indexed!
    return n_halos[skynum - 1], halo_coords[skynum - 1, :]


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
