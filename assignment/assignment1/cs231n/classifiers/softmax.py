import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
 
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]

  scores = X.dot(W)
  score_max = np.max(scores, axis=1)
  scores = scores - score_max.reshape(num_train,1)
  prob = np.exp(scores) / np.exp(scores).sum(axis=1).reshape(num_train, 1)
  dW_each = np.zeros_like(dW)
  y_trueClass = np.zeros_like(prob)
  y_trueClass[np.arange(num_train), y] = 1.0
  for i in xrange(num_train):
    for j in xrange(num_class):
        loss += -(y_trueClass[i, j] * np.log(prob[i,j]))
        dW_each[:, j] = -(y_trueClass[i, j] - prob[i, j]) * X[i]        
    dW += dW_each
    
  loss /= num_train
  dW /= num_train
  
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)
  score_max = np.max(scores, axis=1).reshape(num_train, 1)
  scores -= score_max
  prob = np.exp(scores) / np.exp(scores).sum(axis=1).reshape(num_train, 1)
  y_trueClass = np.zeros_like(prob)
  y_trueClass[np.arange(num_train), y] = 1.0
  loss = -np.sum(y_trueClass * np.log(prob)) / num_train + reg * np.sum(W * W)
  dW = - np.dot(X.T, y_trueClass - prob) / num_train + 2 * reg * W
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

