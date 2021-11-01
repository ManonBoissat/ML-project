import numpy as np

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    error = y - tx.dot(w)
    mse = 1/2 * np.mean(error**2)
    #mae = np.mean(np.abs(error))
    return mse

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient and error vector
    # ***************************************************
    error = y - tx.dot(w)
    grad = -1* ((tx.T).dot(error))/len(error)
    return grad

def gradient_descent(y, tx, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    initial_w = np.ones((tx.shape[1],1))*0.0007
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # TODO: compute gradient and loss
        # ***************************************************
        loss = compute_mse(y, tx, w)
        grad = compute_gradient(y, tx, w)
        # ***************************************************
        # TODO: update w by gradient
        # ***************************************************
        # store w and loss
        w=  w - gamma*grad 
        ws.append(w)
        losses.append(loss)
        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #       bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws