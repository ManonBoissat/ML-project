import numpy as np
from Implement_helpers import compute_mse, compute_gradient, compute_stoch_gradient, sigmoid, calculate_sigmoid_loss
from Implement_helpers import calculate_logistic_gradient,penalized_logistic_regression

##################################   Gradient Descent ######################
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_mse(y, tx, w)
        grad = compute_gradient(y, tx, w)
        w =  w - gamma*grad 
        ws.append(w)
        losses.append(loss)
    return losses[-1], ws[-1]

################################# SGD #####################################3

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    batch_size =1 
    ws = [initial_w]
    losses = []
    w = initial_w
    for itr in range (max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size= batch_size, num_batches=1):
            grad = compute_stoch_gradient(y_batch, tx_batch, w)
            #print("grad shape = ", grad.shape)
            w = w - gamma * grad
            loss = compute_mse(y_batch, tx_batch, w)
            #print("wt_appended", w.shape)
            ws.append(w)
            losses.append(loss)
            #print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(bi=itr, ti=max_iters - 1, l=loss))
    return losses[-1], ws[-1]

############################### Least squsres ##########################

def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # returns mse, and optimal weights
    # ***************************************************
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    wts = np.dot(np.linalg.inv(a),b) 
    loss_tx = compute_mse(y, tx, wts)
    return loss_tx, wts

################################## Ridge Regression ####################################

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lambda_dash = 2 * lambda_* tx.shape[0]
    lambda_I = lambda_dash * np.identity(tx.shape[1]) # feature dimension space
    a = tx.T.dot(tx) + lambda_I
    b = tx.T.dot(y)
    wts = np.linalg.solve(a,b)
    
    loss_tx = compute_mse(y, tx, wts)
    rmse_loss_tr = np.sqrt(2*mse_loss_tr)
    
    return loss_tx, wts

##################################### Logistic Regression ####################################################

def logistic_regression(y, tx, w, max_iter, gamma ):
    threshold = 1e-8
    losses = []
    all_wts = [w]
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss = calculate_sigmoid_loss(y, tx, w)
        grad = calculate_logistic_gradient(y, tx, w)
        w = w - gamma * grad
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        all_wts.append(w)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return losses[-1], all_wts[-1]

#####################################  Regularized Logistic Regression ####################################################

def reg_logistic_regression(y, x, lambda_, w, max_iter, gamma ):
    threshold = 1e-8
    losses = []
    all_wts = [w]
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, grad = penalized_logistic_regression(y, x, w, lambda_)
        w = w - gamma * grad
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        all_wts.append(w)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return losses[-1], all_wts[-1]

##########################################################################################################################
