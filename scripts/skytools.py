import numpy as np
import sys
import os
import inspect


def read_sky(skynum):
    data_dir = os.path.join(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))), '../data/Train_Skies/')
    
    return np.loadtxt(
        os.path.join(data_dir, skyfile_name(skynum)), delimiter=',', 
        unpack=True, usecols=(1,2,3,4), skiprows=1).T;
        
def skyfile_name(skynum):
    return 'Training_Sky' + str(skynum) + '.csv'

