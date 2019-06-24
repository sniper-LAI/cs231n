from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    for i in range(num_train):
        f = scores[i]
        logC = -np.max(f) # logC is used for avoiding the numetic instability
        f += logC
        p = np.exp(f[y[i]]) / np.sum(np.exp(f))
        # Calculate the loss value
        loss += -np.log(p)
        # Calculate the gradient
        for j in range(num_classes):
            if j == y[i]:
                dW[:,j] += X[i].T * (np.exp(f[j]) / np.sum(np.exp(f)) - 1)
            else:
                dW[:,j] += X[i].T * (np.exp(f[j]) / np.sum(np.exp(f)))
                
    # Final loss value
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    logC = np.max(scores,axis = 1)[:,np.newaxis]
    #print(logC)
    scores -= logC
    softmax = np.exp(scores) / np.sum(np.exp(scores),axis = 1,keepdims = True)
    loss = np.sum(-np.log(softmax[np.arange(num_train),y]))
    #print(loss)
    
    #############################################################################
    # dW's shape is D x C. softmax's shape is N x C. X's shape is N x D
    # softmax =[s11 s12 s13 ..... s1C
    #           s21 s22 s23 ..... s2C
    #            .   .   .         .
    #            .   .   .         .
    #           sN1 sN2 sN3 ..... sNC]
    #
    # x =[x11 x12 x13 ..... x1D
    #     x21 x22 x23 ..... x2D
    #      .   .   .         .
    #      .   .   .         .
    #     xN1 xN2 xN3 ..... xND]
    #
    # x.T =[x11 x21 ..... xN1
    #       x12 x22 ..... xN2
    #        .   .         .
    #        .   .         .
    #       x1D x2D ..... xND]
    # dW = [ (s11*x11+s21*x21+...+sN1*xN1)-.. (s12*x11+s22*x21+...+sN2*xN1)-.. .....
    #        (s11*x12+s21*x22+...+sN1*xN2)-.. (s12*x12+s22*x22+...+sN2*xN2)-.. ..... 
    #          .   
    #          .   
    #        (s11*x1D+s21*x2D+...+sN1*xND)-.. (s12*x1D+s22*x2D+...+sN2*xND)-.. ..... ]
    # 
    #############################################################################
    softmax[np.arange(num_train),y] -= 1
    dW = np.dot(X.T,softmax) # D x C shape, the same shape as W
    #print(dW[:,y].shape)
    #dW[:,y] -= X[np.arange(num_train),:].T
    
    loss /= num_train
    dW /= num_train
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
