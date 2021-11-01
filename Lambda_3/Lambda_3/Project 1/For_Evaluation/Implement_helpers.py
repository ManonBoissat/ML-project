import numpy as np
import matplotlib.pyplot as plt

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    error = y - tx.dot(w)
    mse = 1/2 * np.mean(error**2)
    #mae = np.mean(np.abs(error))
    return mse

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # compute gradient and error vector
    # ***************************************************
    error = y - tx.dot(w)
    grad = -1* ((tx.T).dot(error))/len(error)
    return grad


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    error = y - tx.dot(w)
    #print("error shape", error.shape)
    grad = -tx.T.dot(error)/len(error)
    #print("grad shape", grad.shape)
    return grad


def sigmoid(z):
    """apply the sigmoid function on t."""
    return 1.0 / (1 + np.exp(-z))


def calculate_sigmoid_loss(y, tx, w):
    ep = 1e-5
    """compute the loss: negative log likelihood."""
    a = sigmoid(tx.dot(w))
    loss = - np.average(y * np.log(a+ep) + (1 - y) * np.log(1 - a+ep))
    return loss


def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    delta = pred - y
    grad = tx.T.dot(delta)
    return grad


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    loss = calculate_sigmoid_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    grad = calculate_logistic_gradien(y, tx, w) + 2 * lambda_ * w
    return loss, grad



#########################################################################################################################
                         ######## Additional helper Function ##########
#########################################################################################################################


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


