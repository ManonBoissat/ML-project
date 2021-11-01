# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
import matplotlib.pyplot as plt

def build_poly(x, degree):
    #print("Inside build poly function...\n")
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    poly_basis = np.ones(len(x))
    for j in range (1, degree+1):
        poly_basis = np.c_[poly_basis, np.power(x, j)]
    return poly_basis

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.xlim(1e-7, 1)
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    #plt.savefig("cross_validation")
    
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    error = y - tx.dot(w)
    mse = 1/2 * np.mean(error**2)
    #mae = np.mean(np.abs(error))
    return mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    lambda_dash = 2 * lambda_* tx.shape[0]
    lambda_I = lambda_dash * np.identity(tx.shape[1]) # feature dimension space
    a = tx.T.dot(tx) + lambda_I
    b = tx.T.dot(y)
    wts = np.linalg.solve(a,b)
    return wts

def Regressor_prep(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    ts_idx = k_indices[k]
    tr_idx = k_indices[np.arange(k_indices.shape[0]) != k]  # k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_idx = tr_idx.reshape(-1)
    #print(ts_idx.shape, tr_idx.shape)
    
    y_ts = y[ts_idx]
    x_ts = x[ts_idx]
    
    y_tr = y[tr_idx]
    x_tr = x[tr_idx]
    
    #print(y_ts.shape, y_tr.shape)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # form data with polynomial degree: TODO
    # ***************************************************
    tr_poly = build_poly (x_tr, degree)
    ts_poly = build_poly (x_ts, degree)
    
    #print(tr_poly.shape, ts_poly.shape)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    
    wts = ridge_regression(y_tr, tr_poly, lambda_)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate the loss for train and test data: TODO
    # ***************************************************
    #print(y_tr.shape, tr_poly.shape, wts_ridge.shape)
    mse_loss_tr = compute_mse(y_tr, tr_poly, wts)
    loss_tr = np.sqrt(2*mse_loss_tr)

    mse_loss_ts = compute_mse(y_ts, ts_poly, wts)
    loss_te = np.sqrt(2*mse_loss_ts)
    return wts, loss_tr, loss_te

def cross_validation_ridge(seed, k_fold,lambdas, y, tX):
    degree=6
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = [] 
    rmse_te = []
    all_wts = []
    # ***************************************************
    # INSERT YOUR CODE HERE
    # cross validation: TODO
    # ***************************************************
    for l in lambdas:
        fold_rmse_tr = []
        fold_rmse_ts = []
        fold_wts =[]
        for k in range (k_fold):
            #print("x shape = {}, yshape = {}, k = {}, lambda ={}, degree={}".format(x.shape, y.shape, k,l,degree))
            wts, loss_tr, loss_ts = Regressor_prep(y, tX, k_indices, k, l, degree)
            fold_rmse_tr.append(loss_tr)
            fold_rmse_ts.append(loss_ts)
            fold_wts.append(wts)
        rmse_tr.append(np.mean(fold_rmse_tr))
        rmse_te.append(np.mean(fold_rmse_ts))
        all_wts.append(fold_wts)
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    return rmse_tr, rmse_te, all_wts