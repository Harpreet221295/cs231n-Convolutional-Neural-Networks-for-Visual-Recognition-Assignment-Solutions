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
  num_train = X.shape[0]
  num_classes = W[1]
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  

  scores = X.dot(W)
  exp_scores = np.exp(scores)
  scores_sum = np.sum(exp_scores, axis = 1)
  exp_label = exp_scores[np.arange(num_train),y]

  softmax_scores = exp_label/scores_sum
  loss = -1 * np.sum(np.log(softmax_scores))
  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)
  
  fscores = exp_scores/scores_sum[:,np.newaxis]

  idx = np.zeros_like(fscores)
  idx[np.arange(num_train),y] = 1
  dW = X.T.dot(fscores-idx)

  dW /= num_train
  dW += reg*W


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
  num_train = X.shape[0]
  num_classes = W[1]
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  exp_scores = np.exp(scores)
  scores_sum = np.sum(exp_scores, axis = 1)
  exp_label = exp_scores[np.arange(num_train),y]

  softmax_scores = exp_label/scores_sum
  loss = -1 * np.sum(np.log(softmax_scores))
  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)
  
  fscores = exp_scores/scores_sum[:,np.newaxis]

  idx = np.zeros_like(fscores)
  idx[np.arange(num_train),y] = 1
  dW = X.T.dot(fscores-idx)

  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

