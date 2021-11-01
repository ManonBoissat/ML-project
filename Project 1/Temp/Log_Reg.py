import numpy as np


def sigmoid(z):
    """apply the sigmoid function on t."""
    return 1.0 / (1 + np.exp(-z))


def calculate_loss(y, tx, w):
    ep = 1e-5
    """compute the loss: negative log likelihood."""
    a = sigmoid(tx.dot(w))
    loss = - np.average(y * np.log(a+ep) + (1 - y) * np.log(1 - a+ep))
    return loss


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    delta = pred - y
    grad = tx.T.dot(delta)
    return grad


def logistic_regression_gradient_descent_demo(y, tx):
    # init parameters
    max_iter = 100
    threshold = 1e-8
    gamma = 10**(-6)
    losses = []
    all_wts = []
    # build tx
    w = np.zeros((tx.shape[1], 1))#*0.0007
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss = calculate_loss(y, tx, w)
        grad = calculate_gradient(y, tx, w)
        w = w - gamma * grad
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        all_wts.append(w)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    #visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent", True)
    #print("loss={l}".format(l=calculate_loss(y, tx, w)))
    return losses, all_wts




