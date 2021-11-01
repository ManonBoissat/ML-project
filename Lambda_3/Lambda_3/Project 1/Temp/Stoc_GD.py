import numpy as np

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    error = y - tx.dot(w)
    mse = 1/2 * np.mean(error**2)
    #mae = np.mean(np.abs(error))
    return mse



def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the origin  al data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y  
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]



def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
    #raise NotImplementedError
    #print("y shape",y.shape)
    #print("tx shape", tx.shape)
    #print("w shape", w.shape)
    error = y - tx.dot(w)
    #print("error shape", error.shape)
    grad = -tx.T.dot(error)/len(error)
    #print("grad shape", grad.shape)
    return grad

def stochastic_gradient_descent(y, tx, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # TODO: implement stochastic gradient descent.
    # ***************************************************
    #raise NotImplementedError
    initial_w = np.ones((tx.shape[1],1))*0.0007
    w = initial_w
    ws = []
    losses = []
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
    return losses, ws