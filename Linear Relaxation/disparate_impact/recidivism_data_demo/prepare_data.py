import os,sys
import urllib.request, urllib.error, urllib.parse
sys.path.insert(0, '../fair_classification/') # the code for fair classification is in this directory
import utils as ut
import numpy as np
import pandas as pd
from random import seed, shuffle
SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

"""
    The adult dataset can be obtained from: http://archive.ics.uci.edu/ml/datasets/Adult
    The code will look for the data files (adult.data, adult.test) in the present directory, if they are not found, it will download them from UCI archive.
"""

def load_recidivism_data(load_data_size=None):

    """
        if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
        if it is a number, say 10000, then we will return randomly selected 10K examples
    """

    # Load in the data
    data = pd.read_csv('new_Recidivism.csv')

    # split it out into the proper groups
    X = data.drop(['Race', 'Recidivism'], axis=1)
    y = data['Recidivism']
    x_control = {'Race': data['Race']}

    x_control['Race'] = x_control['Race'].replace(-1,0)

    # convert to numpy arrays for easy handling
    print('X Shape: ', X.shape)
    print('y shape: ', y.shape)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype = float)
    for k, v in list(x_control.items()): x_control[k] = np.array(v, dtype=float)

    # shuffle the data
    perm = list(range(0,len(y))) # shuffle the data before creating each fold
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in list(x_control.keys()):
        x_control[k] = x_control[k][perm]

    # see if we need to subsample the data
    if load_data_size is not None:
        print("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in list(x_control.keys()):
            x_control[k] = x_control[k][:load_data_size]

    print(X)
    print(y)
    print(x_control)
    return X, y, x_control

def main():
  load_recidivism_data()

if __name__ == '__main__':
	main()