#!/usr/bin/python

import numpy
from random import randint
from math import floor, ceil

# Theano Packages
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.ifelse import ifelse
from theano.tensor.signal.downsample import DownsampleFactorMax
from theano.tensor.nnet.conv3d2d import conv3d


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
    
### identity 
def Identity(x):
    return(x)   

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

   
   
def maxpool_3D(input, ds, ignore_border=False):
   
    #input.dimshuffle (0, 2, 1, 3, 4)   # convert to make video in back. 
    # no need to reshuffle. 
    if input.ndim < 3:
        raise NotImplementedError('max_pool_3d requires a dimension >= 3')

    # extract nr dimensions
    vid_dim = input.ndim
    # max pool in two different steps, so we can use the 2d implementation of 
    # downsamplefactormax. First maxpool frames as usual. 
    # Then maxpool the time dimension. Shift the time dimension to the third 
    # position, so rows and cols are in the back


    # extract dimensions
    frame_shape = input.shape[-2:]
    
    # count the number of "leading" dimensions, store as dmatrix
    batch_size = T.prod(input.shape[:-2])
    batch_size = T.shape_padright(batch_size,1)
    
    # store as 4D tensor with shape: (batch_size,1,height,width)
    new_shape = T.cast(T.join(0, batch_size,
                                        T.as_tensor([1,]), 
                                        frame_shape), 'int32')
    input_4D = T.reshape(input, new_shape, ndim=4)

    # downsample mini-batch of videos in rows and cols
    op = DownsampleFactorMax((ds[1],ds[2]), ignore_border)          # so second and third dimensions of ds are for height and width
    output = op(input_4D)
    # restore to original shape                                     
    outshape = T.join(0, input.shape[:-2], output.shape[-2:])
    out = T.reshape(output, outshape, ndim=input.ndim)

    # now maxpool time
    # output (time, rows, cols), reshape so that time is in the back
    shufl = (list(range(vid_dim-3)) + [vid_dim-2]+[vid_dim-1]+[vid_dim-3])
    input_time = out.dimshuffle(shufl)
    # reset dimensions
    vid_shape = input_time.shape[-2:]
    
    # count the number of "leading" dimensions, store as dmatrix
    batch_size = T.prod(input_time.shape[:-2])
    batch_size = T.shape_padright(batch_size,1)
    
    # store as 4D tensor with shape: (batch_size,1,width,time)
    new_shape = T.cast(T.join(0, batch_size,
                                        T.as_tensor([1,]), 
                                        vid_shape), 'int32')
    input_4D_time = T.reshape(input_time, new_shape, ndim=4)
    # downsample mini-batch of videos in time
    op = DownsampleFactorMax((1,ds[0]), ignore_border)            # Here the time dimension is downsampled. 
    outtime = op(input_4D_time)
    # output 
    # restore to original shape (xxx, rows, cols, time)
    outshape = T.join(0, input_time.shape[:-2], outtime.shape[-2:])
    shufl = (list(range(vid_dim-3)) + [vid_dim-1]+[vid_dim-3]+[vid_dim-2])
    #rval = T.reshape(outtime, outshape, ndim=input.ndim).dimshuffle(shufl)
    return T.reshape(outtime, outshape, ndim=input.ndim).dimshuffle(shufl)
    #rval.dimshuffle ( 0, 2, 1, 3, 4 )                      
    # return rval 
    
             
# SVM layer from the discussions in this group
# https://groups.google.com/forum/#!msg/theano-users/on4D16jqRX8/IWGa-Gl07g0J
class SVMLayer(object):

    def __init__(self, input, n_in, n_out, W=None, b=None):


        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(value=numpy.zeros((n_in, n_out),
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
   

    def __init__(self, input, n_in, n_out, W=None, b=None ):

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                    value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
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
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]
        self.probabilities = T.log(self.p_y_given_x)

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
                
        self.output = (lin_output if activation is None else activation(lin_output))

        if max_out == 1:  # Do maxout network.
            maxout_out = None        
            for i in xrange(maxout_size):
                temp = self.output[:,i::maxout_size]                                   
                if maxout_out is None:                                              
                    maxout_out = temp                                                  
                else:                                                               
                    maxout_out = T.maximum(maxout_out, temp)  
            self.output = maxout_out   
             
        elif max_out == 2:  # Do meanout network.
            maxout_out = None                                                       
            for i in xrange(maxout_size):                                            
                temp = self.output[:,i::maxout_size]                                   
                if maxout_out is None:                                              
                    maxout_out = temp                                                  
                else:                                                               
                    maxout_out = (maxout_out*(i+1)+temp)/(i+2)   
            self.output = maxout_out    
            
        elif max_out == 3: # Do mixout network.
            maxout_out = None
            maxout_mean = None
            maxout_max  = None 
            for i in xrange(maxout_size):                                            
                temp = self.output[:,i::maxout_size]                                   
                if maxout_mean is None:                                              
                    maxout_mean = temp
                    maxout_max = temp
                    maxout_out = temp
                else:                                                               
                    maxout_mean = (maxout_out*(i+1)+temp)/(i+2) 
                    maxout_max = T.maximum(maxout_out, temp) 
                    
            lambd      = srng.uniform( maxout_mean.shape, low=0.0, high=1.0)
            maxout_out = lambd * maxout_max + (1 - lambd) * maxout_mean
            self.output = maxout_out
            
        # parameters of the model
        if batch_norm is True:
            self.params = [self.W, self.b, self.alpha]
        else:
            if use_bias:
                self.params = [self.W, self.b]
            else:
                self.params = [self.W]
                
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

        self.dropout_L1 = theano.shared(0)
        self.dropout_L2 = theano.shared(0)
        self.L1 = theano.shared(0)
        self.L2 = theano.shared(0)
        
        count = 0
        if len(layer_sizes) > 2:
            for n_in, n_out in weight_matrix_sizes[:-1]:
                if max_out > 0: 
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
                                            alpha = params[count+1] if batch_norm is True else None)
    
            
                                                        
                self.dropout_layers.append(next_dropout_layer)
                next_dropout_layer_input = self.dropout_layers[-1].output
                self.dropout_L1 = self.dropout_L1 + abs(self.dropout_layers[-1].W).sum()  + abs(self.dropout_layers[-1].alpha**2).sum()
                self.dropout_L2 = self.dropout_L2 + (self.dropout_layers[-1].W**2).sum()  + abs(self.dropout_layers[-1].alpha**2).sum()
    
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
                self.L1 = self.L1 + abs(self.layers[-1].W).sum()  + abs(self.layers[-1].alpha).sum()
                self.L2 = self.L2 + (self.layers[-1].W**2).sum()   + abs(self.layers[-1].alpha**2).sum()
                if max_out > 0:
                    prev_maxout_size = maxout_rates [ layer_counter ]                   
                layer_counter += 1
                
                count = count + 2 
                if batch_norm is True:
                    count = count + 1 
            # Set up the output layer
            n_in, n_out = weight_matrix_sizes[-1] 
            n_in = n_in / prev_maxout_size 
            
        else: 
            n_in, n_out = weight_matrix_sizes[-1]
        # Again, reuse paramters in the dropout output.
    
        if svm_flag is False:
            if verbose is True:
                print "           -->        initializing regression layer with " + str(n_out) + " output units and " + str(n_in) + " input units"
            if not len(params) < count + 1:      
      
                dropout_output_layer = LogisticRegression(
                    input=next_dropout_layer_input,
                    n_in=n_in, n_out=n_out,
                    W = params[count] if copy_from_old[-1] is True else None,
                    b = params[count+1] if copy_from_old[-1] is True else None)
        
                output_layer = LogisticRegression(
                    input=next_layer_input,
                    # scale the weight matrix W with (1-p)
                    W=dropout_output_layer.W * (1 - dropout_rates[layer_counter]),
                    b=dropout_output_layer.b,
                    n_in=n_in, n_out=n_out)
            else:
                dropout_output_layer = LogisticRegression(
                    input=next_dropout_layer_input,
                    n_in=n_in, n_out=n_out
                    )
        
                output_layer = LogisticRegression(
                    input=next_layer_input,
                    # scale the weight matrix W with (1-p)
                    n_in=n_in, n_out=n_out,
                    W=dropout_output_layer.W * (1 - dropout_rates[layer_counter]),
                    b=dropout_output_layer.b
                    )

            self.layers.append(output_layer)
            self.dropout_layers.append(dropout_output_layer)

            self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood             
            self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
            
            self.dropout_cross_entropy = self.dropout_layers[-1].categorical_cross_entropy
            self.cross_entropy = self.layers[-1].categorical_cross_entropy

            self.dropout_binary_entropy = self.dropout_layers[-1].binary_cross_entropy
            self.binary_entropy = self.layers[-1].binary_cross_entropy
            
            self.dropout_L1 = self.dropout_L1 + abs(self.dropout_layers[-1].W).sum() 
            self.dropout_L2 = self.dropout_L2 + abs(self.dropout_layers[-1].W**2).sum()

            self.L1 = self.L1 + abs(self.layers[-1].W).sum() 
            self.L2 = self.L2 + abs(self.layers[-1].W**2).sum()

        else:
            if verbose is True:
                print "           -->        initializing max-margin layer with " + str(n_out) + " class predictors and " + str(n_in) + " input units."
            if len(params) < count + 1:
                dropout_output_layer = SVMLayer(
                    input=next_dropout_layer_input,
                    n_in=n_in  , n_out=n_out )
    
                output_layer = SVMLayer(input = next_layer_input,
                                        W=dropout_output_layer.W ,
                                        b=dropout_output_layer.b,
                                        n_in = n_in,
                                        n_out = n_out)
            else:
                dropout_output_layer = SVMLayer(
                    input=next_dropout_layer_input,
                    n_in=n_in, n_out=n_out, 
                    W = params[count] if copy_from_old[layer_counter] is True else None,
                    b = params[count+1] if copy_from_old[layer_counter] is True else None)
    
                output_layer = SVMLayer(input = next_layer_input,
                                        W=dropout_output_layer.W,
                                        b=dropout_output_layer.b,
                                        n_in = n_in,
                                        n_out = n_out)

            self.layers.append(output_layer)
            self.dropout_layers.append(dropout_output_layer)

            self.dropout_hinge_loss = self.dropout_layers[-1].svm_cost
            self.hinge_loss = self.layers[-1].svm_cost

        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        
        self.dropout_errors = self.dropout_layers[-1].errors
        self.errors = self.layers[-1].errors

        self.predicts_dropouts = self.layers[-1].y_pred
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
                            stride,
                             max_out,
                              maxout_size,
                               activation,
                                W = None,
                                 b = None,
                                  alpha = None,
                                   batch_norm = False,
                                    p = 0.5 ,
                                     verbose = True):
                            
        batchsize  = image_shape[0]
        channels   = image_shape[1] 
        width      = image_shape[3]
        height     = image_shape[2]
        
        next_height = int(floor((height - filter_shape[2] + 1))) / poolsize[0]
        next_width = int(floor((width - filter_shape[3] + 1))) / poolsize[1] 
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
            print "                                  ....... output size is [" + str((image_shape[2] - filter_shape[2] + 1 ) / (poolsize[0] * stride[0]) ) + " X " + str((image_shape[3] - filter_shape[3] + 1 ) / (poolsize[1] * stride[1]) ) + "]"
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
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
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
        conv_out = conv.conv2d(
            input = self.input,
            filters = self.W,
            subsample = stride,
            filter_shape = filter_shape,
            image_shape = image_shape
            )

        # downsample each feature map individually, using maxpooling
        #if fast_conv is False:

        pool_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
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
            self.output = activation(pool_out * ( self.alpha.dimshuffle('x', 0, 'x', 'x') / std ) + self.b.dimshuffle('x', 0, 'x', 'x'))
        else:
            self.output = activation(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))              

       
        """ 
            max_out =1 Ian Goodfellow et al. " Maxout Networks " on arXiv. (jmlr)
            
            max_out =2, max_out = 3, Yu, Dingjun, et al. "Mixed Pooling for Convolutional Neural Networks." Rough Sets and Knowledge 
            Technology. Springer International Publishing, 2014. 364-375.

            Same is also implemeted in the MLP layers also.
            
        """
            
        if max_out == 1:  # Do maxout network.
            maxout_out = None        
            for i in xrange(maxout_size):
                temp = self.output[:,i::maxout_size,:,:]                                   
                if maxout_out is None:                                              
                    maxout_out = temp                                                  
                else:                                                               
                    maxout_out = T.maximum(maxout_out, temp)  
            self.output = maxout_out   
             
        elif max_out == 2:  # Do meanout network.
            maxout_out = None                                                       
            for i in xrange(maxout_size):                                            
                temp = self.output[:,i::maxout_size,:,:]                                   
                if maxout_out is None:                                              
                    maxout_out = temp                                                  
                else:                                                               
                    maxout_out = (maxout_out*(i+1)+temp)/(i+2)   
            self.output = maxout_out    
            
        elif max_out == 3: # Do mixout network.
            maxout_out = None
            maxout_mean = None
            maxout_max  = None 
            for i in xrange(maxout_size):                                            
                temp = self.output[:,i::maxout_size,:,:]                                   
                if maxout_mean is None:                                              
                    maxout_mean = temp
                    maxout_max = temp
                    maxout_out = temp
                else:                                                               
                    maxout_mean = (maxout_out*(i+1)+temp)/(i+2) 
                    maxout_max = T.maximum(maxout_out, temp) 
                    
            lambd      = srng.uniform( maxout_mean.shape, low=0.0, high=1.0)
            maxout_out = lambd * maxout_max + (1 - lambd) * maxout_mean
            self.output = maxout_out
                
        # store parameters of this layer
        self.params = [self.W, self.b] if batch_norm is False else [self.W, self.b, self.alpha]
                
class DropoutConv2DPoolLayer(Conv2DPoolLayer):
    def __init__(self, rng, input,
                             filter_shape,
                              image_shape,
                               poolsize,
                               stride,
                               max_out,
                               maxout_size,
                                activation,
                                 W = None,
                                  b = None,
                                  alpha = None,
                                  batch_norm = False,
                                 verbose = True,  
                                  p = 0.5):
        super(DropoutConv2DPoolLayer, self).__init__(
                 rng = rng,
                    input = input, 
                    filter_shape = filter_shape,
                     image_shape = image_shape,
                      poolsize = poolsize,
                      stride = stride,
                      max_out = max_out,
                      maxout_size = maxout_size,
                       activation = activation,
                        W = W, 
                        b = b,
                        alpha = alpha,
                        batch_norm = batch_norm,
                        verbose = False)
                        
        self.output = _dropout_from_layer(rng, self.output, p=p)          
                  
              
# From theano tutorials
class Conv3DPoolLayer(object):
    """Pool Layer of a convolutional network .. taken from the theano tutorials"""

    def __init__(self, rng, input,
                         filter_shape,
                          image_shape,
                           poolsize,
                           stride,
                            max_out,
                             maxout_size,
                              activation,
                               W = None,
                                b = None,
                                 alpha = None,
                                  batch_norm = False,
                                   p = 0.5 ,
                        verbose = True):
                            
        batchsize  = image_shape[0]
        channels   = image_shape[1] 
        stack_size = image_shape[2]
        width      = image_shape[4]
        height     = image_shape[3]

        
        next_height = int(floor((height - filter_shape[3] + 1))) / poolsize[1]
        next_width = int(floor((width - filter_shape[4] + 1))) / poolsize[2] 
        kern_shape = int(floor(filter_shape[0]/maxout_size))     

        #output_size = ( batchsize, bias_shape, next_height , next_width )
        
        srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
        if verbose is True:
            print "           -->        initializing 3D convolutional layer with " + str(filter_shape[0])  + " kernels"
            print "                                  ....... kernel size [" + str(filter_shape[1]) + " X " + str(filter_shape[3]) + " X " + str(filter_shape[4]) +"]"
            print "                                  ....... pooling size [" + str(poolsize[0]) + " X " + str(poolsize[1]) + " X " + str(poolsize[2]) + "]"
            print "                                  ....... stride size [" + str(stride[0]) + " X " + str(stride[1]) + " X " + str(stride[2]) + "]"            
            print "                                  ....... maxout size [" + str(maxout_size) + "]"
            print "                                  ....... input size ["  + str(height)+ " X " + str(width) + "]"
            print "                                  ....... input number of feature maps is " +str(channels) 
            print "                                  ....... output size is [" + str(filter_shape[0] / poolsize[0]) + " X " + str(int(ceil(((float(height - filter_shape[3])  /  stride[1]) +1) / poolsize[1]))) + " X " + str(int(ceil(((float(width - filter_shape[4])  /  stride[2]) + 1) / poolsize[2] + "]"
        self.input = input
        
        assert stride[2] == 1        
        assert stride[1] == 1
        assert stride[0] == 1 
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
            b_values = numpy.zeros((floor(filter_shape[0] / poolsize[0]),), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b
        
                 
        if alpha is None:
            alpha_values = numpy.ones((floor(filter_shape[0] / poolsize[0]),), dtype=theano.config.floatX)
            self.alpha = theano.shared(value=alpha_values, borrow = True)
        else:
            self.alpha = alpha   
             
        # convolve input feature maps with filters        
        conv_out = conv3d(            
            signals=self.input,
            filters=self.W,
            signals_shape=image_shape,
            filters_shape=filter_shape,
        )

        # downsample each feature map individually, using maxpoolig
        #if fast_conv is False:

        # downsample each feature map individually, using maxpooling    
        if poolsize[1] > 1:    
            pool_out =  maxpool_3D(
                    input=conv_out,
                    ds=poolsize           
                )
        else:
            pool_out = conv_out
  
        pool_out = pool_out.sum(axis = 1, keepdims = False) # This will become a summation like what Pascal said happens in the 2D Conv ??        
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
            self.output = activation(pool_out * ( self.alpha.dimshuffle('x', 0, 'x', 'x') / std ) + self.b.dimshuffle('x', 0, 'x', 'x'))
        else:
            self.output = activation(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))              

        """ 
            max_out =1 Ian Goodfellow et al. " Maxout Networks " on arXiv. (jmlr)
            
            max_out =2, max_out = 3, Yu, Dingjun, et al. "Mixed Pooling for Convolutional Neural Networks." Rough Sets and Knowledge 
            Technology. Springer International Publishing, 2014. 364-375.

            Same is also implemeted in the MLP layers also.
            
        """
            
        if max_out == 1:  # Do maxout network.
            maxout_out = None        
            for i in xrange(maxout_size):
                temp = self.output[:,i::maxout_size,:,:]                                   
                if maxout_out is None:                                              
                    maxout_out = temp                                                  
                else:                                                               
                    maxout_out = T.maximum(maxout_out, temp)  
            self.output = maxout_out   
             
        elif max_out == 2:  # Do meanout network.
            maxout_out = None                                                       
            for i in xrange(maxout_size):                                            
                temp = self.output[:,i::maxout_size,:,:]                                   
                if maxout_out is None:                                              
                    maxout_out = temp                                                  
                else:                                                               
                    maxout_out = (maxout_out*(i+1)+temp)/(i+2)   
            self.output = maxout_out    
            
        elif max_out == 3: # Do mixout network.
            maxout_out = None
            maxout_mean = None
            maxout_max  = None 
            for i in xrange(maxout_size):                                            
                temp = self.output[:,i::maxout_size,:,:]                                   
                if maxout_mean is None:                                              
                    maxout_mean = temp
                    maxout_max = temp
                    maxout_out = temp
                else:                                                               
                    maxout_mean = (maxout_out*(i+1)+temp)/(i+2) 
                    maxout_max = T.maximum(maxout_out, temp) 
                    
            lambd      = srng.uniform( maxout_mean.shape, low=0.0, high=1.0)
            maxout_out = lambd * maxout_max + (1 - lambd) * maxout_mean
            self.output = maxout_out
                
        # store parameters of this layer
        self.params = [self.W, self.b]
                
class DropoutConv3DPoolLayer(Conv3DPoolLayer):
    def __init__(self, rng, input,
                             filter_shape,
                              image_shape,
                               poolsize,
                               stride,
                               max_out,
                               maxout_size,
                                activation,
                                 W = None,
                                  b = None,
                                  alpha = None,
                                  batch_norm = False,
                                 verbose = True,  
                                  p = 0.5):
        super(DropoutConv3DPoolLayer, self).__init__(
                 rng = rng,
                    input = input, 
                    filter_shape = filter_shape,
                     image_shape = image_shape,
                      poolsize = poolsize,
                      stride= stride,
                      max_out = max_out,
                      maxout_size = maxout_size,
                       activation = activation,
                        W = W, 
                        b = b,
                        alpha = alpha,
                        batch_norm = batch_norm,
                        verbose = False)
                        
        self.output = _dropout_from_layer(rng, self.output, p=p)          
           