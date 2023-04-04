from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # Add regularization to the gradient
    dW += 2 * reg * W

    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    # Compute the scores for all samples in a vectorized form
    scores = X.dot(W)

    # Compute the correct class scores for each sample in a vectorized form
    correct_class_scores = scores[np.arange(len(y)), y]

    # Compute the margins for all classes for each sample in a vectorized form
    margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1)

    # Set the margin of the correct class to zero
    margins[np.arange(len(y)), y] = 0

    # Compute the loss
    loss = np.mean(np.sum(margins, axis=1)) + reg * np.sum(W * W)

    # Compute the gradient
    binary = margins > 0
    binary[np.arange(len(y)), y] = -np.sum(binary, axis=1)

    print(binary.shape)
    print(binary)
    dW = (X.T).dot(binary)
    dW /= len(y)
    dW += 2 * reg * W

    return loss, dW
