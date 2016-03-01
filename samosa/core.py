#!/usr/bin/python

import numpy
from random import randint
from math import floor, ceil
from collections import OrderedDict

# Theano Packages
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv
from theano.ifelse import ifelse
from theano.tensor.signal.pool import Pool as DownsampleFactorMax
from theano.tensor.signal.pool import pool_2d 
from theano.tensor.signal.pool import max_pool_2d_same_size 

from theano.tensor.nnet.neighbours import images2neibs

#### rectified linear unit
def ReLU(x, alpha = 0):
    y = T.nnet.relu(x,alpha = alpha)
    return(y)
    
#### sigmoid
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
#### tanh
def Tanh(x):
    y = T.tanh(x)
    return(y)
    
#### softmax
def Softmax(x): 
    return T.nnet.softmax(x)
    
### Abs    
def Abs(x):
    return(abs(x))
 
### Squared
def Squared(x):
    return(x ** 2)    

### maxouts   
def max1d (x,i,stride):
    return x[:,i::stride]
    
def max2d (x,i,stride):
    return x[:,i::stride,:,:]
    
def conv2D_border_mode_same( 
                input,
                filters,
                subsample,
                image_shape,):
    filter_shp = T.shape(filters)[2] - 1 
    out = conv.conv2d(input = input ,
                      filters = filters,
                      border_mode='full',
                      subsample = subsample,
                      image_shape = image_shape )                      
    return out[:,:,filter_shp:1+filter_shp,filter_shp:1+filter_shp]    
        
def Maxout(x, maxout_size, max_out_flag = 1, dimension = 1):
    """ 
        max_out =1 Ian Goodfellow et al. " Maxout Networks " on arXiv. (jmlr)        
        max_out =2, max_out = 3, Yu, Dingjun, et al. "Mixed Pooling for Convolutional Neural Networks." Rough Sets and Knowledge 
        Technology. Springer International Publishing, 2014. 364-375.
        Same is also implemeted in the MLP layers also.        
    """   
    if dimension == 1:
        maxing = max1d
    elif dimension == 2:
        maxing = max2d
        
    if max_out_flag == 0:   # Do nothing
        output = x        
    elif max_out_flag == 1:  # Do maxout network.
        maxout_out = None        
        for i in xrange(maxout_size):
            temp = maxing(x,i,maxout_size)                                   
            if maxout_out is None:                                              
                maxout_out = temp                                                  
            else:                                                               
                maxout_out = T.maximum(maxout_out, temp)  
        output = maxout_out               
    elif max_out_flag == 2:  # Do meanout network.
        maxout_out = None                                                       
        for i in xrange(maxout_size):                                            
            temp = maxing(x,i,maxout_size)                                   
            if maxout_out is None:                                              
                maxout_out = temp                                                  
            else:                                                               
                maxout_out = (maxout_out*(i+1)+temp)/(i+2)   
        output = maxout_out            
    elif max_out == 3: # Do mixout network.
        maxout_out = None
        maxout_mean = None
        maxout_max  = None 
        for i in xrange(maxout_size):                                            
            temp = maxing(x,i,maxout_size)                                   
            if maxout_mean is None:                                              
                maxout_mean = temp
                maxout_max = temp
                maxout_out = temp
            else:                                                               
                maxout_mean = (maxout_out*(i+1)+temp)/(i+2) 
                maxout_max = T.maximum(maxout_out, temp)                 
        lambd      = srng.uniform( maxout_mean.shape, low=0.0, high=1.0)
        maxout_out = lambd * maxout_max + (1 - lambd) * maxout_mean
        output = maxout_out    
    return output                
    
def meanpool ( input, ds, ignore_border = False ):
    out_shp = (input.shape[0], input.shape[1], input.shape[2]/ds[0], input.shape[3]/ds[1])    
    neib = images2neibs(input, neib_shape = ds ,mode = 'valid' if ignore_border is False else 'ignore_borders')
    pooled_vectors = neib.mean( axis = - 1 )
    return T.reshape(pooled_vectors, out_shp, ndim = 4 )    
    
def maxrandpool ( input, ds, p, ignore_border = False ):
    """ provide random pooling among the top 'p' sorted outputs p = 0 is maxpool """
    rng = numpy.random.RandomState(24546)
    out_shp = (input.shape[0], input.shape[1], input.shape[2]/ds[0], input.shape[3]/ds[1])        
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    pos = srng.random_integers(size=(1,1), low = ds[0]*ds[1]-1-p, high = ds[0]*ds[1]-1)
    neib = images2neibs(input, neib_shape = ds ,mode = 'valid' if ignore_border is False else 'ignore_borders') 
    neib = neib.sort(axis = -1) 
    pooled_vectors = neib[:,pos]   
    return T.reshape(pooled_vectors, out_shp, ndim = 4 )           
    
#From the Theano Tutorials
def shared_dataset(data_xy, borrow=True, svm_flag = True):

	data_x, data_y = data_xy
	shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
	shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX),  borrow=borrow)
	                                 
	if svm_flag is True:
		# one-hot encoded labels as {-1, 1}
		n_classes = len(numpy.unique(data_y))  # dangerous?
		y1 = -1 * numpy.ones((data_y.shape[0], n_classes))
		y1[numpy.arange(data_y.shape[0]), data_y] = 1
		shared_y1 = theano.shared(numpy.asarray(y1,dtype=theano.config.floatX), borrow=borrow)

		return shared_x, theano.tensor.cast(shared_y, 'int32'), shared_y1
	else:
		return shared_x, theano.tensor.cast(shared_y, 'int32') 
             
# SVM layer from the discussions in this group
# https://groups.google.com/forum/#!msg/theano-users/on4D16jqRX8/IWGa-Gl07g0J
class SVMLayer(object):

    def __init__(self, input, n_in, n_out, rng, W=None, b=None):


        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(value=numpy.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)),
                                             dtype=theano.config.floatX),
                                name='W', borrow=True)
        else:
            self.W = W
        # initialize the baises b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)
        else:
            self.b =b

        # parameters of the model
        self.params = [self.W, self.b]
        self.output = T.dot(input, self.W) + self.b
        self.y_pred = T.argmax(self.output, axis=1)

        self.L1 = abs(self.W).sum()
        self.L2 = (self.W ** 2 ).sum()
        
    def hinge(self, u):
            return T.maximum(0, 1 - u)

    def svm_cost(self, y1):
        y1_printed = theano.printing.Print('this is important')(T.max(y1))
        margin = y1 * self.output
        cost = self.hinge(margin).mean(axis=0).sum()
        return cost


    def errors(self, y):

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.sum(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


# Modified From https://github.com/mdenil/dropout/blob/master/mlp.py
class LogisticRegression(object):

    def __init__(self, input, n_in, n_out, rng, W=None, b=None, use_bias = False ):

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                    value=numpy.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)),
                                     dtype=theano.config.floatX),
                    name='W')
                    
        else:
            self.W = W

        # initialize the baises b as a vector of n_out 0s        
        if b is None:
            self.b = theano.shared(
                    value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                    name='b')
        else:
            self.b = b

        # compute vector of class-membership probabilities in symbolic form
        if use_bias is True:
            self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b + 1e-7) 
        else:
            self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + 1e-7)         
        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.L1 = abs(self.W).sum()
        self.L2 = (self.W ** 2).sum()
        self.params = [self.W, self.b] if use_bias is True else [self.W]
        self.probabilities = T.log(self.p_y_given_x)
        # po = theano.function ( inputs = [index], outputs = MLPlayers.layers[-1].input, givens = {x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size]})
        
    def negative_log_likelihood(self, y ):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) 
        
    def categorical_cross_entropy( self, y ):
        return T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x,y))

    def binary_cross_entropy ( self, y ):
        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x,y))
        
    def errors(self, y):
      
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return T.sum(T.neq(T.cast(self.y_pred,'int32'), T.cast(y,'int32')))      # L1 norm of the error.
            #return T.sum(T.neq(self.y_pred, y))      # L1 norm of the error.  
        else:
            raise NotImplementedError()

    
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation , max_out, maxout_size, W=None, b=None, batch_norm = False, alpha=None,
                 use_bias=False):

        self.input = input
        self.activation = activation
        srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
        if W is None:
            W_values = numpy.asarray(0.01 * rng.standard_normal(
                size=(n_in, n_out)), dtype=theano.config.floatX)

            W = theano.shared(value=W_values, name='W')
        
            if activation == Sigmoid or activation == T.nnet.sigmoid:
                W_values*=4 
            W = theano.shared(value=W_values, name='W')
        
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        if alpha is None:
            alpha_values = numpy.ones((n_out,), dtype = theano.config.floatX)
            alpha = theano.shared(value = alpha_values, name = 'alpha')
            
        self.W = W
        self.b = b
        self.alpha = alpha 
        
        lin_output = T.dot(input, self.W)
                
        if batch_norm is True:
            std = lin_output.std( 0 )
            mean = lin_output.mean( 0 )
            std += 0.001 # To avoid divide by zero like fudge_factor
        
            self.lin_output = lin_output - mean 
            lin_output = lin_output * ( self.alpha / std ) + self.b 
            
        if batch_norm is False and use_bias is True:
            lin_output = lin_output  + self.b
                           
        maxout_output = Maxout(x = lin_output, maxout_size = maxout_size, max_out_flag = max_out, dimension = 1 )         
        self.output = (lin_output if activation is None else activation(lin_output))      
            
        # parameters of the model
        if batch_norm is True:
            self.params = [self.W, self.b, self.alpha]
        else:
            if use_bias:
                self.params = [self.W, self.b]
            else:
                self.params = [self.W]
        self.L1 = abs(self.W).sum()  + abs(self.alpha).sum()
        self.L2 = (self.W**2).sum()  + (self.alpha**2).sum()
        self.n_out = n_out / maxout_size

# dropout thanks to misha denil 
# https://github.com/mdenil/dropout
def _dropout_from_layer(rng, layer, p):

    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, use_bias, max_out, maxout_size, W=None, b=None, batch_norm = False, alpha=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b, batch_norm = batch_norm, alpha = alpha, max_out = max_out, maxout_size = maxout_size,
                activation=activation, use_bias=use_bias)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)      
    
class MLP(object):

    def __init__(self,
            rng,
            input,
            layer_sizes,
            dropout_rates,
            maxout_rates,
            activations,
            copy_from_old,
            freeze,
            use_bias=True,
            max_out=False,
            svm_flag = True,
            batch_norm = False, 
            params = [],
            verbose = True):

        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        next_layer_input, dropout_next_layer_input = input
   
        next_dropout_layer_input = _dropout_from_layer(rng, dropout_next_layer_input, p=dropout_rates[0])
        
        layer_counter = 0    
        prev_maxout_size = 1   

        self.L1 = theano.shared(0)
        self.L2 = theano.shared(0)
        
        count = 0
        if len(layer_sizes) > 2:
            for n_in, n_out in weight_matrix_sizes[:-1]:
                curr_maxout_size = maxout_rates[layer_counter]
                if verbose is True and max_out > 0:
                    print "           -->        initializing mlp Layer with " + str(n_out) + " hidden units taking in input size " + str(n_in / prev_maxout_size) + " after maxout this output becomes " + str(n_out / curr_maxout_size)
                elif verbose is True and max_out == 0:
                    print "           -->        initializing mlp Layer with " + str(n_out) + " hidden units taking in input size " + str(n_in) 
                if len(params) < count + 1:
                    next_dropout_layer = DropoutHiddenLayer(rng=rng,
                                                    input=next_dropout_layer_input,
                                                    activation=activations[layer_counter],
                                                    n_in=n_in / prev_maxout_size, n_out = n_out,
                                                    use_bias=use_bias, 
                                                    max_out = max_out,
                                                    batch_norm = batch_norm,
                                                    dropout_rate=dropout_rates[layer_counter + 1],
                                                    maxout_size = maxout_rates[layer_counter]
                                                    )
                else:
                    next_dropout_layer = DropoutHiddenLayer(rng=rng,
                                            input=next_dropout_layer_input,
                                            activation=activations[layer_counter],
                                            n_in=n_in / prev_maxout_size , n_out = n_out,
                                            use_bias=use_bias,
                                            max_out = max_out,
                                            batch_norm = batch_norm,
                                            dropout_rate=dropout_rates[layer_counter + 1],
                                            maxout_size = maxout_rates[layer_counter],
                                            W = params[count] if copy_from_old[layer_counter] is True else None,
                                            b = params[count+1] if copy_from_old[layer_counter] is True else None,
                                            alpha = params[count+2] if batch_norm is True else None)
    
            
                                                        
                self.dropout_layers.append(next_dropout_layer)                
                next_dropout_layer_input = self.dropout_layers[-1].output
    
                # Reuse the paramters from the dropout layer here, in a different
                # path through the graph.                        
                next_layer = HiddenLayer(rng=rng,
                        input=next_layer_input,
                        activation=activations[layer_counter],
                        # scale the weight matrix W with (1-p)
                        W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
                        b=next_dropout_layer.b,
                        alpha =next_dropout_layer.alpha,
                        max_out = max_out,
                        batch_norm = batch_norm,
                        maxout_size = maxout_rates[ layer_counter ],
                        n_in = n_in / prev_maxout_size, n_out=n_out,
                        use_bias=use_bias)
                self.layers.append(next_layer)
                next_layer_input = self.layers[-1].output
                #first_layer = False
                self.L1 = self.L1 + self.layers[-1].L1
                self.L2 = self.L2 + self.layers[-1].L2 
                                
                layer_counter += 1
                
                count = count + 2 
                if batch_norm is True:
                    count = count + 1 
            # Set up the output layer
            n_in, n_out = weight_matrix_sizes[-1] 
                                    
        else: 
            n_in, n_out = weight_matrix_sizes[-1]
            
        # Again, reuse paramters in the dropout output.
    
        if svm_flag is False:
            if verbose is True:
                print "           -->        initializing regression layer with " + str(n_out) + " output units and " + str(n_in) + " input units"
            if not len(params) < count + 1:      
      
                dropout_output_layer = LogisticRegression(
                    input=next_dropout_layer_input,
                    n_in=n_in, n_out=n_out, rng=rng,
                    W = params[count] if copy_from_old[-1] is True else None,
                    b = params[count+1] if copy_from_old[-1] is True and use_bias is True else None, 
                    use_bias = use_bias)
        
                output_layer = LogisticRegression(
                    input=next_layer_input,
                    # scale the weight matrix W with (1-p)
                    W=dropout_output_layer.W * (1 - dropout_rates[layer_counter]),
                    b=dropout_output_layer.b,
                    n_in=n_in, n_out=n_out, rng=rng,  use_bias = use_bias)
            else:
                dropout_output_layer = LogisticRegression(
                    input=next_dropout_layer_input ,
                    n_in=n_in, n_out=n_out, rng=rng,  use_bias = use_bias
                    )
        
                output_layer = LogisticRegression(
                    input=next_layer_input,
                    # scale the weight matrix W with (1-p)
                    n_in=n_in, n_out=n_out, rng=rng,
                    W=dropout_output_layer.W * (1 - dropout_rates[layer_counter]),
                    b=dropout_output_layer.b,  use_bias = use_bias
                    )

            self.layers.append(output_layer)
            self.dropout_layers.append(dropout_output_layer)

            self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood             
            self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
            
            self.dropout_cross_entropy = self.dropout_layers[-1].categorical_cross_entropy
            self.cross_entropy = self.layers[-1].categorical_cross_entropy

            self.dropout_binary_entropy = self.dropout_layers[-1].binary_cross_entropy
            self.binary_entropy = self.layers[-1].binary_cross_entropy
            
            self.L1 = self.L1 + self.layers[-1].L1
            self.L2 = self.L2 + self.layers[-1].L2

        else:
            if verbose is True:
                print "           -->        initializing max-margin layer with " + str(n_out) + " class predictors and " + str(n_in) + " input units."
            if len(params) < count + 1:
                dropout_output_layer = SVMLayer(
                    input=next_dropout_layer_input,
                    n_in=n_in  , n_out=n_out,rng =rng )
    
                output_layer = SVMLayer(input = next_layer_input,
                                        W=dropout_output_layer.W ,
                                        b=dropout_output_layer.b,
                                        n_in = n_in,
                                        n_out = n_out, rng= rng)
            else:
                dropout_output_layer = SVMLayer(
                    input=next_dropout_layer_input,
                    n_in=n_in, n_out=n_out, rng = rng,
                    W = params[count] if copy_from_old[layer_counter] is True else None,
                    b = params[count+1] if copy_from_old[layer_counter] is True else None)
    
                output_layer = SVMLayer(input = next_layer_input,
                                        W=dropout_output_layer.W,
                                        b=dropout_output_layer.b,
                                        n_in = n_in,
                                        n_out = n_out,
                                        rng = rng)

            self.layers.append(output_layer)
            self.dropout_layers.append(dropout_output_layer)

            self.dropout_hinge_loss = self.dropout_layers[-1].svm_cost
            self.hinge_loss = self.layers[-1].svm_cost
        
        self.errors = self.layers[-1].errors
        self.predicts = self.layers[-1].y_pred
        
        self.params = []
        count = 0
        for layer in self.dropout_layers:
            if freeze[count] is False:
                for param in layer.params:
                    self.params.append (param)
            elif verbose is True:
                print "           -->        freezing post convolutional layer " + str(count + 1)          
                                                            
            count = count + 1
        if svm_flag is True:
            self.probabilities = self.layers[-1].output  
        else:
            self.probabilities = self.layers[-1].probabilities            

       
# From theano tutorials
class Conv2DPoolLayer(object):
    """Pool Layer of a convolutional network .. taken from the theano tutorials"""

    def __init__(self, rng, input,
                         filter_shape,
                          image_shape,
                           poolsize,
                           pooltype,                           
                            stride,
                             pad,
                             max_out,
                              maxout_size,
                               activation,
                                W = None,
                                 b = None,
                                  alpha = None,
                                   batch_norm = False,
                                    p = 0.5 ,
                                     maxrandpool_p = 0,
                                     verbose = True):
                            
        batchsize  = image_shape[0]
        channels   = image_shape[1] 
        width      = image_shape[3]
        height     = image_shape[2]
        
        if pad == 0:
            border_mode = 'valid'
            conv_out_height = height - filter_shape[2] + 1
            conv_out_width  = width - filter_shape[3] + 1
        elif pad == 1:
            border_mode = 'full'
            conv_out_height = height + filter_shape[2] - 1
            conv_out_width = width  + filter_shape[2] - 1
        elif pad == 2:
            border_mode = 'same'
            conv_out_height = height 
            conv_out_width = width   
                     
        else:
            print "... unrecognized padding mode in convolutional layer"
            sys.exit(0)
                    
        if pooltype == 0:
            next_height = int ( floor(conv_out_height/ float(stride[0])) ) 
            next_width = int ( floor(conv_out_width / float(stride[1])) )  
        else: 
            next_height = int ( ceil (conv_out_height / float(poolsize[0] * stride[0]) ) )
            next_width = int ( ceil (conv_out_width / float(poolsize[1] * stride[1]) ) )    
                  
        kern_shape = int(floor(filter_shape[0]/maxout_size))       
        output_size = ( batchsize, kern_shape, next_height , next_width )
        srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
        if verbose is True:
            print "           -->        initializing 2D convolutional layer with " + str(filter_shape[0])  + " kernels"
            print "                                  ....... kernel size [" + str(filter_shape[2]) + " X " + str(filter_shape[3]) +"]"
            print "                                  ....... pooling size [" + str(poolsize[0]) + " X " + str(poolsize[1]) + "]"
            print "                                  ....... stride size [" + str(stride[0]) + " X " + str(stride[1]) + "]"            
            print "                                  ....... maxout size [" + str(maxout_size) + "]"
            print "                                  ....... input size ["  + str(image_shape[2]) + " " + str(image_shape[3]) + "]"
            print "                                  ....... input number of feature maps is " +str(image_shape[1]) 
            print "                                  ....... output size is [" + str(next_height ) + " X " + str(next_width ) + "]"
        self.input = input
        assert image_shape[1] == filter_shape[1]
        
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        if W is None:
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size =filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        else: 
            self.W = W 
        # the bias is a 1D tensor -- one bias per output feature map
        if b is None:
            b_values = numpy.zeros((filter_shape[0]), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b
        
                 
        if alpha is None:
            alpha_values = numpy.ones((filter_shape[0]), dtype=theano.config.floatX)
            self.alpha = theano.shared(value=alpha_values, borrow = True)
        else:
            self.alpha = alpha   
             
        # convolve input feature maps with filters     
        #if fast_conv is False:  
        if not border_mode =='same' :
            conv_out = conv.conv2d(
                input = self.input,
                filters = self.W,
                subsample = stride,
                filter_shape = filter_shape,
                image_shape = image_shape,
                border_mode = border_mode
                )
        else:
            conv_out = conv2D_border_mode_same(
                input = self.input,
                filters = self.W,
                subsample = stride,
                image_shape = image_shape
             )

        # downsample each feature map individually, using maxpooling
        #if fast_conv is False:
        
        if poolsize == (1,1):
            pool_out = conv_out 
            
        elif pooltype == 0:
            pool_out = max_pool_2d_same_size(
                input=conv_out,
                patch_size=poolsize,
                )
                
        elif pooltype == 1:
            pool_out = pool_2d(
                input=conv_out,
                ds=poolsize,
                ignore_border = False,
                mode = 'max'
                )
                
        elif pooltype == 2:                
            pool_out = meanpool(
                input = conv_out,
                ds = poolsize,
                ignore_border = False,                
                )             
                         
        elif pooltype == 3: 
            p = maxrandpool_p
            assert p < poolsize[0]*poolsize[1] - 1         
            pool_out = maxrandpool(
                input = conv_out,
                ds = poolsize,
                p = p,                               
                ignore_border = False
                )
                            
        # self.co = conv_out
        # self.po = pool_out
        # The above statements are used for debugging and probing purposes. They have no use and can be commented out.
        # During debugging while in pdb inside a terminal from the lenet module's train function code, use functions such as :
        #    co = theano.function ( inputs = [index], outputs = conv_layers[0].co, givens = {x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size]})
        #    po = theano.function ( inputs = [index], outputs = conv_layers[0].po, givens = {x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size]})
        #    ip = theano.function ( inputs = [index], outputs = conv_layers[0].input, givens = {x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size]})
        #    op = theano.function ( inputs = [index], outputs = conv_layers[0].output, givens = {x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size]})                              
        #To debug the layers....
        if batch_norm is True:
            mean = pool_out.mean( (0,2,3), keepdims = True )
            std = pool_out.std( (0,2,3), keepdims = True )
            
            std += 0.001 # To avoid divide by zero like fudge_factor
        
            self.pool_out = pool_out - mean
            self.output = pool_out * ( self.alpha.dimshuffle('x', 0, 'x', 'x') / std ) + self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            self.output = pool_out + self.b.dimshuffle('x', 0, 'x', 'x')
            
        self.output = Maxout(x = self.output, maxout_size = maxout_size, max_out_flag = max_out, dimension = 2)                                               
        self.output = activation(self.output)
        
        # store parameters of this layer
        self.params = [self.W, self.b] 
        if batch_norm is True: 
            self.params.append(self.alpha)
        self.output_size = [next_height, next_width, kern_shape]
    
class DropoutConv2DPoolLayer(Conv2DPoolLayer):
    def __init__(self, rng, input,
                             filter_shape,
                              image_shape,
                               poolsize,
                               pooltype,
                               stride,
                               pad,
                               max_out,
                               maxout_size,
                                activation,
                                 W = None,
                                  b = None,
                                  alpha = None,
                                  batch_norm = False,
                                 verbose = True,  
                                  p = 0.5,
                                  maxrandpool_p = 0,):
        super(DropoutConv2DPoolLayer, self).__init__(
                 rng = rng,
                    input = input, 
                    filter_shape = filter_shape,
                     image_shape = image_shape,
                      poolsize = poolsize,
                      pooltype = pooltype,
                      stride = stride,
                      pad = pad,
                      max_out = max_out,
                      maxout_size = maxout_size,
                       activation = activation,
                        W = W, 
                        b = b,
                        alpha = alpha,
                        batch_norm = batch_norm,
                        maxrandpool_p = maxrandpool_p,
                        verbose = False)
                        
        self.output = _dropout_from_layer(rng, self.output, p=p)                        
        
class ConvolutionalLayers (object):

    def __init__ ( self,
            rng,
            input,
            input_size, 
            mean_subtract,
            nkerns,
            filter_size,
            pooling_size,
            pooling_type,
            maxrandpool_p,
            cnn_activations,
            conv_stride_size,
            conv_pad,
            cnn_dropout_rates,
            batch_norm,
            retrain_params,
            init_params,            
            max_out,
            cnn_maxout,
            verbose = True                    
        ):
        
        first_layer_input = input[0]
        mean_sub_input    = input[1] 
        stack_size = 1
        # Create first convolutional - pooling layers 
        self.activity = []       # to record Cnn activities 
        self.weights = []    
        self.conv_layers = []         
        self.dropout_conv_layers = [] 
        
        if retrain_params is not None:
            copy_from_old = retrain_params [ "copy_from_old" ]
            freeze_layers = retrain_params [ "freeze" ]
        else:
            freeze_layers = [] 
            for i in xrange(len(nkerns)):
                freeze_layers.append ( False )              
            
        # if no retrain specified but if init params are given, make default as copy all params.             
            if init_params is not None:
                copy_from_old = []
                for i in xrange(len(nkerns)):
                    copy_from_old.append ( True ) 
        # This could be pushed into the forloop, there is no need to have a seperate writing of the same code for just the first layer.                                        
        filt_size = filter_size[0]
        pool_size = pooling_size[0]
        pool_type = pooling_type[0]
        stride    = conv_stride_size[0]
        pad       = conv_pad[0]
        batch_norm_layer = batch_norm[0]
        maxrandp = maxrandpool_p[0] 
        
        if retrain_params is not None:
            curr_copy = copy_from_old[0] 
            
            if curr_copy is True:
                curr_init_weights = init_params[0]
                curr_init_bias    = init_params[1]
                
                if batch_norm_layer is True:
                    curr_init_alpha    = init_params[2]
                else:
                    curr_init_alpha    = None
            else:
                curr_init_weights = None
                curr_init_bias = None
                curr_init_alpha = None                    
                
        if init_params is None:
            curr_init_weights = None
            curr_init_bias = None 
            curr_init_alpha = None
                
        if max_out > 0:  

                max_out_size = cnn_maxout[0]                
        else: 
            max_out_size = 1

        next_in = [ input_size[0], input_size[1], input_size[2] ]
        stack_size = 1 
        param_counter = 0 

        if len(filt_size) == 2:                        
            self.dropout_conv_layers.append ( 
                            DropoutConv2DPoolLayer(
                                    rng = rng,
                                    input = first_layer_input if mean_subtract is False else mean_sub_input,
                                    image_shape=(input_size[3], next_in[2] , next_in[0], next_in[1]),
                                    filter_shape=(nkerns[0], next_in[2] , filt_size[0], filt_size[1]),
                                    poolsize = pool_size,
                                    pooltype = pool_type,
                                    maxrandpool_p = maxrandp,
                                    pad = pad,
                                    stride = stride,
                                    max_out = max_out,
                                    maxout_size = max_out_size,
                                    activation = cnn_activations[0],
                                    W = None if curr_init_weights is None else curr_init_weights,
                                    b = None if curr_init_bias is None else curr_init_bias, 
                                    batch_norm = batch_norm_layer,
                                    alpha = None if curr_init_alpha is None else curr_init_alpha,
                                    p = cnn_dropout_rates[0],  
                                    verbose = verbose                               
                                        ) ) 
            self.conv_layers.append ( 
                            Conv2DPoolLayer(
                                    rng = rng,
                                    input = first_layer_input if mean_subtract is False else mean_sub_input,
                                    image_shape=(input_size[3], next_in[2] , next_in[0], next_in[1]),
                                    filter_shape=(nkerns[0], next_in[2] , filt_size[0], filt_size[1]),
                                    poolsize = pool_size,
                                    pooltype = pool_type,
                                    maxrandpool_p = maxrandp,                                    
                                    stride = stride,
                                    pad = pad,
                                    max_out = max_out,
                                    maxout_size = max_out_size,
                                    activation = cnn_activations[0],
                                    W = self.dropout_conv_layers[-1].params[0] * (1 - cnn_dropout_rates[0]) ,
                                    b = self.dropout_conv_layers[-1].params[1],
                                    batch_norm = batch_norm_layer,
                                    alpha = self.dropout_conv_layers[-1].alpha,
                                        ) )  
            next_in = self.conv_layers[-1].output_size
                                                                                                                           
        else:
            print "!! as of now Samosa is only capable of 2D conv layers."                               
            sys.exit()
            
        self.activity.append ( self.conv_layers[-1].output.dimshuffle(0,2,3,1) )
        self.weights.append ( self.conv_layers[-1].W )
        
        # Create the rest of the convolutional - pooling layers in a loop
        param_counter = param_counter + 2  
        if batch_norm_layer is True:
            param_counter = param_counter + 1
        for layer in xrange(len(nkerns)-1):   
            
            filt_size = filter_size[layer+1]
            pool_size = pooling_size[layer+1]
            pool_type = pooling_type[layer+1]
            maxrandp  = maxrandpool_p[layer+1]            
            stride    = conv_stride_size[layer +1 ]
            pad       = conv_pad[layer + 1]
            batch_norm_layer = batch_norm [layer + 1]
            if retrain_params is not None:
                curr_copy = copy_from_old[layer + 1] 
                if curr_copy is True:
                    curr_init_weights = init_params[param_counter]
                    curr_init_bias    = init_params[param_counter + 1]
                    if batch_norm_layer is True:
                        curr_init_alpha = init_params[param_counter + 2]   
                    else:
                        curr_init_alpha = None
                else:
                    curr_init_weights  = None
                    curr_init_bias = None
                    curr_init_alpha = None          
            if init_params is None:
                curr_init_weights = None
                curr_init_bias = None
                curr_init_alpha = None
                    
            if max_out > 0:
                max_out_size = cnn_maxout[layer+1]
            else:
                max_out_size = 1 

            if len(filt_size) == 2:         
                self.dropout_conv_layers.append ( 
                                DropoutConv2DPoolLayer(
                                    rng = rng,
                                    input = self.dropout_conv_layers[layer].output,        
                                    image_shape=(input_size[3], next_in[2], next_in[0], next_in[1]),
                                    filter_shape=(nkerns[layer+1], next_in[2], filt_size[0], filt_size[1]),
                                    poolsize=pool_size,
                                    pooltype = pool_type,
                                    maxrandpool_p = maxrandp,                                    
                                    stride = stride,
                                    pad = pad, 
                                    max_out = max_out,
                                    maxout_size = max_out_size,
                                    activation = cnn_activations[layer+1],
                                    W = None if curr_init_weights is None else curr_init_weights ,
                                    b = None if curr_init_bias is None else curr_init_bias ,
                                    batch_norm = batch_norm_layer,
                                    alpha = None if curr_init_alpha is None else curr_init_alpha ,
                                    p = cnn_dropout_rates[layer+1]                                                                                                        
                                        ) )
                                                
                self.conv_layers.append ( 
                                Conv2DPoolLayer(
                                    rng = rng,
                                    input = self.conv_layers[layer].output,        
                                    image_shape=(input_size[3], next_in[2], next_in[0], next_in[1]),
                                    filter_shape=(nkerns[layer+1], next_in[2], filt_size[0], filt_size[1]),
                                    poolsize = pool_size,
                                    pooltype = pool_type,
                                    maxrandpool_p = maxrandp,                                    
                                    stride = stride,
                                    pad = pad,
                                    max_out = max_out,
                                    maxout_size = max_out_size,
                                    activation = cnn_activations[layer+1],
                                    W = self.dropout_conv_layers[-1].params[0] * (1 - cnn_dropout_rates[layer + 1]),
                                    b = self.dropout_conv_layers[-1].params[1],
                                    batch_norm = batch_norm_layer, 
                                    alpha = self.dropout_conv_layers[-1].alpha,
                                    verbose = verbose
                                        ) )                                                       
                                            
                next_in = self.conv_layers[-1].output_size                            
            else:
                print "!! as of now Samosa is only capable of 2D conv layers."                               
                sys.exit()
            self.weights.append ( self.conv_layers[-1].W )
            self.activity.append( self.conv_layers[-1].output.dimshuffle(0,2,3,1) )

            param_counter = param_counter + 2    
            if batch_norm_layer is True:
                param_counter = param_counter + 1  
                
        self.params = []
        count = 0
        for layer in self.dropout_conv_layers:
            if freeze_layers[count] is False:
                self.params = self.params + layer.params
            elif verbose is True:
                print "           -->        freezing convolutional layer " + str(count +  1)  
            count = count + 1 
        
        self.output_size = next_in
                
    def returnOutputSizes(self):
        return self.output_size
        
    def returnActivity(self):
        return self.activity
        
class optimizer(object):
    def __init__(self, params, objective, optimization_params, verbose = True):
        
        self.mom_start                       = optimization_params [ "mom" ][0]
        self.mom_end                         = optimization_params [ "mom" ][1]
        self.mom_epoch_interval              = optimization_params [ "mom" ][2]
        self.mom_type                        = optimization_params [ "mom_type" ]
        self.initial_learning_rate           = optimization_params [ "learning_rate" ][0]  
        self.ft_learning_rate                = optimization_params [ "learning_rate" ][1]          
        self.learning_rate_decay             = optimization_params [ "learning_rate" ][2] 
        self.optim_type                      = optimization_params [ "optim_type" ]   
        self.l1_reg                          = optimization_params [ "reg" ][0]
        self.l2_reg                          = optimization_params [ "reg" ][1]
        self.objective                       = optimization_params [ "objective" ]               
                        
        if verbose is True:
            print "... estimating gradients"

        gradients = []      
        for param in params: 
            gradient = T.grad( objective ,param)
            gradients.append ( gradient )
        epoch = T.scalar('epoch')
        # TO DO: Try implementing Adadelta also. 
        # Compute momentum for the current epoch
        
        mom = ifelse(epoch <= self.mom_epoch_interval,
            self.mom_start*(1.0 - epoch/self.mom_epoch_interval) + self.mom_end*(epoch/self.mom_epoch_interval),
            self.mom_end)   
        
        # learning rate
        self.eta = theano.shared(numpy.asarray(self.initial_learning_rate,dtype=theano.config.floatX))
        # accumulate gradients for adagrad
         
        grad_acc =[]
        grad_acc2= []
        for param in params:
            eps = numpy.zeros_like(param.get_value(borrow=True), dtype=theano.config.floatX)   
            grad_acc.append(theano.shared(eps, borrow=True))
            grad_acc2.append(theano.shared(eps, borrow=True))
    
        # accumulate velocities for momentum
        velocities = []
        for param in params:
            velocity = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,dtype=theano.config.floatX))
            velocities.append(velocity)
         
        # create updates for each combination of stuff 
        updates = OrderedDict()
        print_flag = False
                               
        timestep = theano.shared(numpy.asarray(0., dtype=theano.config.floatX))
        delta_t = timestep + 1
        b1=0.9                       # for ADAM
        b2=0.999                     # for ADAM
        a = T.sqrt (   1-  b2  **  delta_t )   /   (   1   -   b1  **  delta_t )     # for ADAM
        fudge_factor = 1e-7
                
        if verbose is True:
            print "... building back prop network" 
        for velocity, gradient, acc , acc2, param in zip(velocities, gradients, grad_acc, grad_acc2, params):        
            if self.optim_type == 1:
    
                """ Adagrad implemented from paper:
                John Duchi, Elad Hazan, and Yoram Singer. 2011. Adaptive subgradient methods
                for online learning and stochastic optimization. JMLR
                """
    
                current_acc = acc + T.sqr(gradient) # Accumulates Gradient 
                updates[acc] = current_acc          # updates accumulation at timestamp

            elif self.optim_type == 2:
                """ Tieleman, T. and Hinton, G. (2012):
                Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
                Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)"""
                
                rms_rho                         = 0.9    
                current_acc = rms_rho * acc + (1 - rms_rho) * T.sqr(gradient) 
                updates[acc] = current_acc

                
            elif self.optim_type == 0:
                current_acc = 1.


            
            elif self.optim_type == 3:
                """ Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014)."""  
                if not self.mom_type == -1: 
                    self.mom_type = - 1
                    if verbose is True:
                        print "... ADAM doesn't need explicit momentum. Momentum is removed."                                 

                current_acc2 = b1 * acc2 + (1-b1) * gradient
                current_acc = b2 * acc + (1-b2) * T.sqr( gradient )                
                updates[acc2] = current_acc2
                updates[acc] = current_acc
                
            if self.mom_type == -1:
                updates[velocity] = a * current_acc2 / (T.sqrt(current_acc) + fudge_factor)
                
            elif self.mom_type == 0:               # no momentum
                updates[velocity] = - (self.eta / T.sqrt(current_acc + fudge_factor)) * gradient                                            
               
            elif self.mom_type == 1:       # if polyak momentum    
    
                """ Momentum implemented from paper:  
                Polyak, Boris Teodorovich. "Some methods of speeding up the convergence of iteration methods." 
                USSR Computational Mathematics and Mathematical Physics 4.5 (1964): 1-17.
    
                Adapted from Sutskever, Ilya, Hinton et al. "On the importance of initialization and momentum in deep learning." 
                Proceedings of the 30th international conference on machine learning (ICML-13). 2013.
                equation (1) and equation (2)"""   
    
                updates[velocity] = mom * velocity - (1.-mom) * ( self.eta / T.sqrt(current_acc+ fudge_factor))  * gradient                             
    
            elif self.mom_type == 2:             # Nestrov accelerated gradient 
    
                """Nesterov, Yurii. "A method of solving a convex programming problem with convergence rate O (1/k2)."
                Soviet Mathematics Doklady. Vol. 27. No. 2. 1983.
                Adapted from https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/ 
    
                Instead of using past params we use the current params as described in this link
                https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,"""
      
                updates[velocity] = mom * velocity - (1.-mom) * ( self.eta / T.sqrt(current_acc + fudge_factor))  * gradient                                 
                updates[param] = mom * updates[velocity] 
    
            else:
                if print_flag is False:
                    print_flag = True
                    print "!! Unrecognized mometum type, switching to no momentum."
                updates[velocity] = -( self.eta / T.sqrt(current_acc+ fudge_factor) ) * gradient                                              
                
            stepped_param  = param + updates[velocity]
            if self.mom_type == 2:
                stepped_param = stepped_param + updates[param]

            column_norm = True #This I don't fully understand if its needed after BN is implemented.
            if param.get_value(borrow=True).ndim == 2 and column_norm is True:
    
                """ constrain the norms of the COLUMNs of the weight, according to
                https://github.com/BVLC/caffe/issues/109 """
                fudge_factor = 1e-7
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(15))
                scale = desired_norms / (fudge_factor + col_norms)
                updates[param] = stepped_param * scale
    
            else:            
                updates[param] = stepped_param
                
        if self.optim_type == 3:
            updates[timestep] = delta_t       
                     
        self.epoch = epoch
        self.mom = mom 
        self.updates = updates                                               
