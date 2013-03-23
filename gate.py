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

class GateBatch:
    """
    A batch of gate vectors weighted by gate coefficients where 
    each gate is identified by a gate index
    """
    def __init__(self, gate_activations):
        """
        Initialize a GateBatch.

        Parameters
        ----------
        gate_activations : ftensor3
            Batch of activations of shape :
            (batch_size, num_gates, gate_dims)  
        """
        self.gate_activations = gate_activations
    def set_activations(self, gate_activations):
        self.gate_activations = gate_activations
    def set_coefficients(self, gate_coefficients):
        self.gate_coefficients = gate_coefficients
    def get_coefficients(self):
        return self.gate_coefficients
    def get_activations(self):
        return self.gate_activations

class GateSpace(Space):
    #A Space whose points are weighted tuples of points in other spaces 
    def __init__(self, num_gates, gate_dims):
        """
        Initialize a VectorSpace.

        Parameters
        ----------
        num_gates : int
            number of gates in this space.
        gate_dims : int
            Dimensionality of a vector in this space.
        """
        self.num_gates = num_gates
        self.gate_dims = gate_dims

    def get_total_dimension(self):
        raise NotImplementedError()
    def get_num_gates(self):
        return self.num_gates
    def get_gate_dimension(self):
        return self.gate_dims
    def format_as(self, batch, space):
        raise NotImplementedError()
    def format_from(self, batch, space):
        if isinstance(space, VectorSpace):
            return GateSpace(T.shape_padleft(batch))
        raise NotImplementedError("GateSpace does not know how to format as "+str(space))
        
    def __eq__(self, other):
        return all([type(self) == type(other), 
                    self.num_gates == other.num_gates,
                    self.gate_dims == other.gate_dims])       

    def validate(self, batch):
        if not isinstance(batch, GateBatch):
            raise TypeError("GateSpace batch should be a GateBatch, got "+str(type(batch)))

class GateRouter(Layer):
    """
    Takes as input a GateBatch and returns batch of gate coefficients.
    
    """
    def __init__(self,
                 layer_name,
                 num_gates,   
                 irange = 0.05,
                 routing_protocol = 'nearest'
        ):
            
        self.__dict__.update(locals())
        del self.self
        
        self.output_space = VectorSpace(self.num_gates)
        
    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.pre_input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = True
            self.input_space \
                = GateSpace(num_gates = 1, 
                            gate_dims=space.get_total_dimension())
        elif isinstance(space, GateSpace):
            self.requires_reformat = False
            self.input_space = space
        
        rng = self.mlp.rng
        # a tensor3 of weights. Each dim0 represents the weight matrix 
        # from an input gate. In such weight matrices, 
        # each row (dim1) is a cluster centroid:
        W = rng.uniform(-self.irange, self.irange,
                        (self.input_space.get_num_gates(),
                         self.output_space.get_total_dimension(),
                         self.input_space.get_gate_dimension()))

        self.W = sharedX(W)
        self.W.name = self.layer_name + '_W'

    def fprop(self, G):
        self.pre_input_space.validate(G)

        if self.requires_reformat:
            G = self.input_space.format_from(G, self.pre_input_space)
        # Get the batch of gate activations and indices from G:
        # (batch_size, num_input_gates, input_gate_dims)
        X1 = G.get_activations() 
        # (num_input_gates, input_gate_dims, batch_size)
        X = X1.dimshuffle(1,2,0) 
       
        # (num_input_gates, 1, batch_size) 
        X2 = (X**2).sum(1).dimshuffle(0,'x',1)
        # (num_input_gates, num_output_gates, 1)
        W2 = (self.W**2).sum(2).dimshuffle(0,1,'x')
        
        # The D tensor holds the distance of each example to all clusters 
        # (WITHIN the context of each input gate)
        # This, as well as A, can be use to get a cost.
        # (num_input_gates, num_output_gates, batch_size) 
        D = T.sqr(-2*T.batched_dot(self.W, X) + W2 + X2)
        # The A matrix holds the (mean) distance of each example-gate 
        # to all clusters
        # (num_output_gates, batch_size)
        A = D.mean(0)

        A.name = self.layer_name + '_A'
        D.name = self.layer_name + '_D'

        return A, D
        
    def cost(self, A, D):
        """
        Y must be one-hot binary. Y_hat is a softmax estimate.
        of Y. Returns negative log probability of Y under the Y_hat
        distribution.
        """
        
        # (
        cost = A.min(0).sum() 
        # + GateIndividuality*D.min(0).sum()
        # )*RouterUnsupervision
        [grad_W] = T.grad(cost, [W], disconnected_inputs = 'ignore')
        f = theano.function([X1],cost,updates=[(W,W-(LR*grad_W))]) 
        

class GateLayer(Layer):
    """
    A layer of a mixture of experts takes as input Vector or Conv
    Space.
    
    It holds num_clusters Linear Layers.
    
    TODO:
        try cosinus similarity
        b1 could be the centroid.
    
    """
    def __init__(self,
                 layer_name,
                 gates,
                 gate_dims,                
                 active_gates,
                 gate_activation = None,
                 output_pooling = 'weighted_average',
                 output_activation = 'softmax',
                 routing_protocol = 'nearest',
                 irange = None,
                 sparse_init = None,
                 sparse_stdev = 1.,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 max_col_norm = None
        ):
        """
            layer_name: A name for this layer that will be prepended to
                        monitoring channels related to this layer.
            num_clusters: The number of clusters (output gates) per input gate.
            num_units: The number of units per output gate.
            irange: if specified, initializes each weight randomly in
                U(-irange, irange)
            sparse_init: if specified, irange must not be specified.
                        This is an integer specifying how many weights to make
                        non-zero. All non-zero weights will be initialized
                        randomly in N(0, sparse_stdev^2)
            include_prob: probability of including a weight element in the set
               of weights initialized to U(-irange, irange). If not included
               a weight is initialized to 0. This defaults to 1.
            init_bias: All biases are initialized to this number
            W_lr_scale: The learning rate on the weights for this layer is
                multiplied by this scaling factor
            b_lr_scale: The learning rate on the biases for this layer is
                multiplied by this scaling factor
            max_col_norm: The norm of each column of the weight matrix is
                constrained to have at most this norm. If unspecified, no
                constraint. Constraint is enforced by re-projection (if
                necessary) at the end of each update.
            max_row_norm: Like max_col_norm, but applied to the rows.
        """


        self.__dict__.update(locals())
        del self.self

        self.b = sharedX( np.zeros((self.gates, self.gate_dims)) \
            + self.init_bias, name = self.layer_name + '_b')
        
        self.output_space = VectorSpace(output_dims)

    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            rval[self.W1] = self.W_lr_scale
            rval[self.W2] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b1] = self.b_lr_scale
            rval[self.b2] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dims = space.dim
        else:
            self.requires_reformat = True
            self.input_dims = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dims)
        
        rng = self.mlp.rng
        if self.irange is not None:
            assert self.istdev is None
            assert self.sparse_init is None
            # a tensor3 of weight. Each dim1 represents the weight matrix 
            # from input to gate dim1
            W1 = rng.uniform(-self.irange, self.irange,
                            (self.gates, self.input_dims, self.gate_dims))
            # a tensor3 of weight. Each dim1 represents the weight matrix 
            # from gate dim1 to output
            W2 = rng.uniform(-self.irange, self.irange,
                            (self.gates, self.gate_dims, self.output_dims))
            # each row is a cluster centroid:
            K = rng.uniform(-self.irange, self.irange,
                            (self.gates, self.input_dims))
        else:
            raise NotImplementedError()

        self.W1 = sharedX(W1)
        self.W1.name = self.layer_name + '_W1'
        
        self.W2 = sharedX(W2)
        self.W2.name = self.layer_name + '_W2'
        
        self.K = sharedX(K)
        self.K.name = self.layer_name + '_K'

    def censor_updates(self, updates):

        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            if self.W in updates:
                updated_W = updates[self.W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[self.W] = updated_W * (desired_norms / (1e-7 + col_norms))

    def get_params(self):
        assert self.b1.name is not None
        assert self.b2.name is not None
        assert self.W1.name is not None
        assert self.W2.name is not None
        return [self.W1, self.W2, self.b1, self.b2]

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        return coeff * (T.sqr(self.W1).sum() + T.sqr(self.W2).sum())

    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        return coeff * (T.abs(self.W1).sum() + T.abs(self.W2).sum())

    def get_weights(self):
        return [self.W1.get_value(), self.W2.get_value()]

    def set_weights(self, weights1, weights2):
        self.W1.set_value(weights1)
        self.W2.set_value(weights2)

    def set_biases(self, biases1, biases2):
        self.b1.set_value(biases1)
        self.b2.set_value(biases2)

    def get_biases(self):
        return [self.b1.get_value(), self.b2.get_value()]

    def get_weights_format(self):
        raise NotImplementedError()

    def get_weights_view_shape(self):
        raise NotImplementedError()

    def get_weights_topo(self):
        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        # There was an implementation of this, but it was broken
        raise NotImplementedError()

    def get_monitoring_channels(self):
        return OrderedDict()


    def get_monitoring_channels_from_state(self, state):
        return OrderedDict()

    def fprop(self, X, G):
        """
        X is a tensor3 of gates. G
        
        """

        self.input_space.validate(X)

        if self.requires_reformat:
            X = self.input_space.format_as(X, self.desired_space)

        # Find nearest cluster centroids for each point:
        K = self.K
        # derived from K-means hack I borrowed from Roland Memisevic's code:
        # distances from points to centroids:
        D = -2.0*T.dot(K, X.T) \
            + (K**2).sum(1)[:, None] \
            + (X**2).sum(1)[:, None].T
        # argsort each point's clusters by ascending distance, and truncate 
        # far ones (the remainder will be used for propagation):
        S = T.argsort(D, axis=0)[:self.active_gates,:]

        # feedforward each point through best gates, 
        #  where each gate is represented by a k-means cluster:        
        # holds of list of output activations to be summed, averaged, etc.
        O_list = []
        for idx in range(self.active_gates):
            gate_idx = S[idx,:]
            G = T.batched_dot(X, self.W1[gate_idx]) + self.b1[gate_idx]
            if self.gate_activation is None:
                pass
            elif self.gate_activation == 'tanh':
                G = T.tanh(G)
            elif self.gate_activation == 'sigmoid':
                G = T.nnet.sigmoid(G)
            else:
                raise NotImplementedError()
            # fprop from gate to output:
            O = T.batched_dot(G, self.W2[gate_idx])
            O_list.append(O)
        
        # mixtures:        
        P = T.stack(*O_list)
        if self.output_pooling in ['weighted_average', 'weighted_sum']:
            M = T.sort(D,axis=0)[:self.active_gates,:]
            # now they sum to one for each point:
            M = M/M.sum(axis=0)
            # makes em broadcastable
            M = T.shape_padright(M)
            # this is where the broadcast comes in handy. Each gate
            # is weighted by the normalized cluster centroid distances:
            P = M*P
            if self.output_pooling == 'weighted_sum':
               P = P.sum(axis=0)
            elif self.output_pooling == 'weighted_average':
               P = P.mean(axis=0)
        
        # activate output
        A = P + self.b2
        if self.output_activation is None:
            pass
        elif self.output_activation == 'tanh':
            A = T.tanh(A)
        elif self.output_activation == 'sigmoid':
            A = T.nnet.sigmoid(A)
        elif self.output_activation == 'softmax':
            A = T.nnet.softmax(A)
            

        A.name = self.layer_name + '_A'

        return A
        
    def cost(self, Y, Y_hat):
        """
        Y must be one-hot binary. Y_hat is a softmax estimate.
        of Y. Returns negative log probability of Y under the Y_hat
        distribution.
        """

        assert hasattr(Y_hat, 'owner')
        owner = Y_hat.owner
        assert owner is not None
        op = owner.op
        if isinstance(op, Print):
            assert len(owner.inputs) == 1
            Y_hat, = owner.inputs
            owner = Y_hat.owner
            op = owner.op
        assert isinstance(op, T.nnet.Softmax)
        z ,= owner.inputs
        assert z.ndim == 2

        z = z - z.max(axis=1).dimshuffle(0, 'x')
        log_prob = z - T.log(T.exp(z).sum(axis=1).dimshuffle(0, 'x'))
        # we use sum and not mean because this is really one variable per row
        log_prob_of = (Y * log_prob).sum(axis=1)
        assert log_prob_of.ndim == 1

        rval = log_prob_of.mean()

        return - rval
        
class GateNetwork(MLP):
    pass

