# -*- coding: utf-8 -*-
__authors__ = "Nicholas Leonard"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Nicholas Leonard"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicholas Leonard"

from collections import OrderedDict
import numpy as np
import warnings
import functools

from theano.gof.op import get_debug_values
from theano import tensor as T
import theano

from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.mlp import Layer, Linear, MLP
from pylearn2.models.model import Model
from pylearn2.space import Conv2DSpace, VectorSpace, CompositeSpace, Space
from pylearn2.utils import sharedX
from pylearn2.costs.cost import Cost

def batched_tensordot(x, y, axes=2):
    """
    :param x: A Tensor with sizes e.g.: for  3D (dim1, dim3, dim2)
    :param y: A Tensor with sizes e.g.: for 3D (dim1, dim2, dim4)
    :param axes: an integer or array. If an integer, the number of axes
                 to sum over. If an array, it must have two array
                 elements containing the axes to sum over in each tensor.

                 Note that the default value of 2 is not guaranteed to work
                 for all values of a and b, and an error will be raised if
                 that is the case. The reason for keeping the default is to
                 maintain the same signature as numpy's tensordot function
                 (and np.tensordot raises analogous errors for non-compatible
                 inputs).

                 If an integer i, it is converted to an array containing
                 the last i dimensions of the first tensor and the first
                 i dimensions of the second tensor:
                     axes = [range(a.ndim - i, b.ndim), range(i)]

                 If an array, its two elements must contain compatible axes
                 of the two tensors. For example, [[1, 2], [2, 0]] means sum
                 over the 2nd and 3rd axes of a and the 3rd and 1st axes of b.
                 (Remember axes are zero-indexed!) The 2nd axis of a and the
                 3rd axis of b must have the same shape; the same is true for
                 the 3rd axis of a and the 1st axis of b.
    :type axes: int or array-like of length 2
    This function computes the tensordot product between the two tensors, by
    iterating over the first dimension using scan.
    Returns a tensor of size e.g. if it is 3D: (dim1, dim3, dim4)
    Example:
    >>> first = T.tensor3('first')
    >>> second = T.tensor3('second')
    >>> result = batched_dot(first, second)
    :note:  This is a subset of numpy.einsum, but we do not provide it for now.
    But numpy einsum is slower than dot or tensordot:
    http://mail.scipy.org/pipermail/numpy-discussion/2012-October/064259.html
    """
    if isinstance(axes, list):
        axes = np.asarray(axes)-1
    elif isinstance(axes, np.ndarray):
        axes -= 1
                
    result, updates = theano.scan(fn=lambda x_mat, y_mat:
            theano.tensor.tensordot(x_mat, y_mat, axes),
            outputs_info=None,
            sequences=[x, y],
            non_sequences=None)
    return result

def kmeans(X, K, numepochs, Winit=None, learningrate=0.01, batchsize=100, verbose=True): 
    if Winit is None:
        W = numpy.random.randn(K, X.shape[1])*0.1
    else:
        W = Winit

    X2 = (X**2).sum(1)[:, None]
    for epoch in range(numepochs):
        for i in range(0, X.shape[0], batchsize):
            W2 = (W**2).sum(1)[:, None]
            D = -2*numpy.dot(W, X[i:i+batchsize,:].T) + W2 + X2[i:i+batchsize].T
            S = (D==D.min(0)[None,:]).astype("float")
            clustersums = numpy.dot(S, X[i:i+batchsize,:])
            pointspercluster = S.sum(1)[:, None]
            
            W += learningrate * (clustersums - pointspercluster * W) 
        if verbose:
            cost = D.min(0).sum()
            print "epoch", epoch, "of", numepochs, " cost: ", cost

    return W

def assign(X, W):
    X2 = (X**2).sum(1)[:, None]
    W2 = (W**2).sum(1)[:, None]
    D = -2*np.dot(W, X.T) + W2 + X2.T
    print X.shape, W.shape, X2.shape, W2.shape, D.shape
    # (8, 7) (9, 7) (8, 1) (9, 1) (9, 8)
    return (D==D.min(0)[None,:]).astype(int)
    
def assign_triangle(X, W):
    X2 = (X**2).sum(1)[:, None]
    W2 = (W**2).sum(1)[:, None]
    D = np.sqrt(-2*np.dot(W, X.T) + W2 + X2.T)
    #print D.shape, D.mean(axis=0).shape, numpy.maximum(0, D.mean(axis=0)-D).shape
    return np.maximum(0, D.mean(axis=0)-D)

def test_tensor_kmeans():
    # 133 sec on gpu, 129 on cpu
    batch_size = 32
    num_input_gates = 100
    num_active_input_gates = 30
    num_output_gates = 200
    input_gate_dims = 100
    
    gate_ind = range(num_input_gates)
    ind = np.random.randint(0,num_input_gates,(num_active_input_gates, batch_size)).astype("int32")
    x = np.random.randn(num_active_input_gates*batch_size*input_gate_dims).reshape( \
        (num_active_input_gates, batch_size, input_gate_dims)).astype("float32")
    w = np.random.uniform(-0.05, 0.05, (num_input_gates, num_output_gates, input_gate_dims)).astype("float32")
    print ind.shape, x.shape, w.shape
    # (3, 8) (3, 8, 7) (5, 9, 7)
    
    assign(x[0],w[0])
    
    X = T.ftensor3('X')
    W = sharedX(w)
    #W = T.ftensor3('W')
    I = T.imatrix('I')
    
    #print x[0].shape, (x[0]**2).shape, (x[0]**2).sum(1).shape, (x[0]**2).sum(1)[:, None].shape
    #(8, 7) (8, 7) (8,) (8, 1)
   
    
    D_list = []
    for gate_idx in xrange(num_active_input_gates):
        Ig = I[gate_idx]
        Xg = X[gate_idx]
        Wg = W[Ig]
        X2g = T.shape_padright(T.sqr(Xg).sum(1))
        W2g = T.sqr(Wg).sum(2)
        Dg = T.batched_dot(Wg, Xg)
        #f = theano.function([X,I],[Xg,Wg,X2g,W2g,Dg])
        #r = f(x,ind)
        #print r[0].shape, r[1].shape, r[2].shape, r[3].shape, r[4].shape
        Dg = -2*Dg + W2g + X2g
        Dg = Dg.T
        #f = theano.function([X,W,I],[Wg,Xg,Dg])
        #r = f(x,w,ind)
        #print r[0].shape, r[1].shape, r[2].shape,
        
        D_list.append(Dg)
    A = T.stack(*D_list)
    D = A.mean(0)
    print 'compiling'
    f = theano.function([X,I],[A,D])
    print 'compiled'
    import time
    start = time.time()
    for i in range(1000):
        [a,d] = f(x,ind)
    print time.time()-start
    print a.shape, d.shape
    
def test_tensor_kmeans2():
    # 17 sec on gpu (problem solved)
    #batch_size = 32
    #num_input_gates = 100
    #num_output_gates = 200
    #input_gate_dims = 100
    batch_size = 8
    num_input_gates = 3
    num_output_gates = 9
    input_gate_dims = 7
    
    x = np.random.randn(num_input_gates*batch_size*input_gate_dims).reshape( \
        (batch_size, num_input_gates, input_gate_dims)).astype("float32")
    w = np.random.uniform(-0.05, 0.05, (num_input_gates, num_output_gates, input_gate_dims)).astype("float32")
    print x.shape, w.shape
    # (8, 3, 7) (3, 9, 7)
    
    assign(x[0],w[0])
    
    # (8, 3, 7)
    X1 = T.ftensor3('X')
    # (3, 9, 7)
    W = sharedX(w, name='W')
    
    # (3, 7, 8) instead of (8,3,7):
    X = X1.dimshuffle(1,2,0)
   
    # (3,1,8) (1,8)
    X2 = (X**2).sum(1).dimshuffle(0,'x',1)
    # (3,9,1) (9,1)
    W2 = (W**2).sum(2).dimshuffle(0,1,'x')
    # (3,9,8) (9,8)
    D = -2*T.batched_dot(W, X) + W2 + X2
    
    A = D.mean(0)
    # The D tensor holds the distance of each example to all clusters 
    # (WITHIN the context of each input gates)
    # The A tensor holds the distance of each example to all clusters
    
    print 'compiling'
    f = theano.function([X1],[X1,X,X2,W,W2,D,A])
    print 'compiled'
    import time
    start = time.time()
    for i in range(1000):
        e = f(x)
    print time.time()-start
    print len(e)
    
def test_tensor_kmeans3():
    numepochs = 10 
    learningrate=0.01
    batch_size = 8
    num_input_gates = 3
    num_output_gates = 9
    input_gate_dims = 7
    dataset_size = batch_size * 4

    x = np.random.randn(num_input_gates*dataset_size*input_gate_dims).reshape( \
        (dataset_size, num_input_gates, input_gate_dims)).astype("float32")
    w = np.random.uniform(-0.05, 0.05, (num_input_gates, num_output_gates, input_gate_dims)).astype("float32")
    print x.shape, w.shape
    # (8, 3, 7) (3, 9, 7)
    
    # (8, 3, 7)
    X1 = T.ftensor3('X')
    # (3, 9, 7)
    W = sharedX(w, name='W')
    
    LR = sharedX(learningrate) #T.fscalar('LR')
    
    # (3, 7, 8) instead of (8,3,7):
    X = X1.dimshuffle(1,2,0)
   
    X2 = (X**2).sum(1).dimshuffle(0,'x',1)
    # (3,9,1) (9,1)
    W2 = (W**2).sum(2).dimshuffle(0,1,'x')
    
    # The D tensor holds the distance of each example to all clusters 
    # (WITHIN the context of each input gates)
    # This, as well as A, can be use to get a cost.
    # (3,9,8) (9,8)
    D = T.sqr(-2*T.batched_dot(W, X) + W2 + X2)
    # The A matrix holds the (mean) distance of each example-gate 
    # to all clusters
    # (9,8)
    A = D.mean(0)
    
    # (
    cost = A.min(0).sum() 
    # + GateIndividuality*D.min(0).sum()
    # )*RouterUnsupervision
    [grad_W] = T.grad(cost, [W], disconnected_inputs = 'ignore')
    f = theano.function([X1],cost,updates=[(W,W-(LR*grad_W))])
    """
    # (9,8) set each example's best cluster to 1, zero otherwise:
    S =  T.cast(T.eq(A,A.min(0).dimshuffle(0,'x')), 'float32')
    # (9,8) <- (9,8) * (8,7)    
    # (3,7,8)
    clustersums = T.dot(S, X)
    # (9,1)
    pointspercluster = S.sum(1)[:, None]
    W += LR * (clustersums - pointspercluster * W) """
    print "learning"
    for epoch in range(numepochs):
        for i in range(0, x.shape[0], batch_size):
            print f(x[i:i+batch_size,:,:])
            # (3,1,8) (1,8)
    print 'compiling'
    #f = theano.function([X1, LR],[X1,X,X2,W,W2,D,A])
    print 'compiled'
    import time
    start = time.time()
    for i in range(1000):
        e = f(x)
    print time.time()-start
    print len(e)
    
def soft_mixture(C, M, X1, x):
    # C : (8, 3, 9, 11)
    # M : (8, 3, 9)
    f = theano.function([X1],[C,M,M.sum(1),M.sum(1).dimshuffle(0,'x',1)])
    print [e.shape for e in f(x[:8,:,:])]
    
    # Make incomming mixture weights sum to one: 
    M /= M.sum(1).dimshuffle(0,'x',1)
    
    # Put scannable dims in front:
    # C2 : (8, 9, 3, 11)
    C2 = C.dimshuffle(0, 2, 1, 3)
    # M2 : (8, 9, 3)
    M2 = M.dimshuffle(0, 2, 1)
    
    # C3 : (72, 3, 11)
    C3 = C2.reshape((C2.shape[0]*C2.shape[1], C2.shape[2], C2.shape[3]))
    # M3 : (72, 3)
    M3 = M2.reshape((M2.shape[0]*M2.shape[1], M2.shape[2]))
    
    f = theano.function([X1],[C2,M2,C3,M3])
    print [e.shape for e in f(x[:8,:,:])]
    
    # R : (72, 11)
    R = batched_tensordot(C3, M3, axes=[[1],[1]])
    
    f = theano.function([X1],[R])
    print [e.shape for e in f(x[:8,:,:])]
    # (8, 9, 11)
    return R.reshape((M2.shape[0], M2.shape[1], R.shape[1]))
    
def test_tensor_layer():
    numepochs = 10 
    learningrate_k=0.01
    learningrate_w=0.01
    batch_size = 8
    num_input_gates = 3
    num_output_gates = 9
    input_gate_dims = 7
    output_gate_dims = 11
    dataset_size = batch_size * 4
    branching_factor = 2
    num_active_output_gates = 2
    
    x = np.random.randn(num_input_gates*dataset_size*input_gate_dims).reshape( \
        (dataset_size, num_input_gates, input_gate_dims)).astype("float32")
    k = np.random.uniform(-0.05, 0.05, (num_input_gates, num_output_gates, input_gate_dims)).astype("float32")
    print x.shape, k.shape
    # (8, 3, 7) (3, 9, 7)
    
    # (8, 3, 7)
    X1 = T.ftensor3('X')
    # (3, 9, 7)
    K = sharedX(k, name='K')
    
    LRk = sharedX(learningrate_k) #T.fscalar('LR')
    
    # (3, 7, 8) instead of (8,3,7):
    X = X1.dimshuffle(1,2,0)
   
    X2 = (X**2).sum(1).dimshuffle(0,'x',1)
    # (3,9,1) (9,1)
    K2 = (K**2).sum(2).dimshuffle(0,1,'x')
    
    # The D tensor holds the distance of each example to all clusters 
    # (WITHIN the context of each input gates)
    # This, as well as A, can be use to get a cost.
    # (3,9,8) (9,8)
    D = T.sqr(-2*T.batched_dot(K, X) + K2 + X2)
    # The A matrix holds the (mean) distance of each example-gate 
    # to all clusters
    # (9,8)
    A = D.mean(0)
    
    
    ############### LAYER PRE FPROP #################
    # (3, 9, 7, 11)
    w = np.random.uniform(-0.05, 0.05, \
        (num_input_gates, num_output_gates, 
         input_gate_dims, output_gate_dims) \
    ).astype("float32")
    print w.shape
    
    
    W = sharedX(w, name='W')
    
    LRw = sharedX(learningrate_w)
    
    # (3, 7, 8) instead of (8,3,7):
    X = X1.dimshuffle(1,2,0)
    
    # The dot product of each connection weight
    # (3, 7, 8) (3, 9, 7, 11) -> (3, 9, 11, 8)
    f = theano.function([X1],[X,W])
    print [e.shape for e in f(x[:8,:,:])]
    C = batched_tensordot(W, X, axes=[[2], [1]])
    
    f = theano.function([X1],[X,W,C])
    print [e.shape for e in f(x[:8,:,:])]
    
    ################# MIXTURE ##################
    """
    # sort each point's clusters by ascending distance, and get the value
    # of the distance at position: branching_factor of each input gate
    # (3,8)
    S = T.sort(D, axis=1)[:,branching_factor,:]
    
    M = T.le(D,S)"""
    
    # Masks: Corridor Mixture, Gate Mixture
    # We will experiment with Hard (max and stochastic),
    # Soft (weighted average, sum, etc), 
    # and Hybrid Mixtures
     
    R = soft_mixture(C.dimshuffle(3,0,1,2), D.dimshuffle(2,0,1), X1, x)
    
    ############## LAYER POST FPROP #############
    
    
    
    
if __name__ == "__main__":
    test_tensor_layer()
