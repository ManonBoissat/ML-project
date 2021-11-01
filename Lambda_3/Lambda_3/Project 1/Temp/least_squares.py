# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # least squares: TODO
    # returns mse, and optimal weights
    #return np.dot(np.linalg.inv(np.dot(a.T, a)), np.dot(a.T, y))
    # ***************************************************
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.dot(np.linalg.inv(a),b)
    
    #w = np.linalg.solve(a,b) 
    return w
