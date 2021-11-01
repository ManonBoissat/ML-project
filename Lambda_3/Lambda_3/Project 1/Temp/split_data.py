# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    data_pts = len(y)
    rand_index = np.random.permutation(data_pts)
    split_distri_index = int(np.floor(ratio * data_pts))
    tr_idx = rand_index[ : split_distri_index]
    ts_idx = rand_index[split_distri_index:]
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    x_tr = x[tr_idx]
    y_tr = y[tr_idx]
    x_ts = x[ts_idx]
    y_ts = y[ts_idx]
    
    return x_tr, y_tr, x_ts, y_ts