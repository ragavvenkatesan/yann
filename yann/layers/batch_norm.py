"""
TODO:

    * Need to the deconvolutional-unpooling layer.
    * Something is still not good about the convolutional batch norm layer.
"""

from abstract import layer, _dropout
import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet.bn import batch_normalization_train, batch_normalization_test

class batch_norm_layer_2d (layer):
    """
    This class is the typical 2D batchnorm layer. It is called
    by the ``add_layer`` method in network class.

    Args:
        input: An input ``theano.tensor`` variable. Even ``theano.shared`` will work as long as they
               are in the following shape ``mini_batch_size, height, width, channels``
        verbose: similar to the rest of the toolbox.
        input_shape: ``(mini_batch_size, channels, height, width)``
        rng: typically ``numpy.random``.
        borrow: ``theano`` borrow, typicall ``True``.
        input_params: Supply params or initializations from a pre-trained system.
    """
    def __init__ ( self,
                   input,
                   input_shape,
                   id,
                   rng = None,
                   borrow = True,
                   input_params = None,
                   verbose = 2,
                 ):

        super(batch_norm_layer_2d,self).__init__(id = id, type = 'batch_norm', verbose = verbose)
        if verbose >=3:
            print "... Creating batch norm layer"

        if rng is None:
            rng = numpy.random

        # To copy weights previously created or some wierd initializations
        if input_params is not None:
            init_gamma = input_params[0]
            init_beta = input_params[1]
            init_mean = input_params[2]
            init_var = input_params[3]

        channels   = input_shape[1]



        if input_params is None:
            self.gamma = theano.shared(value=numpy.ones((channels,),
                                dtype=theano.config.floatX), name = 'gamma', borrow = borrow)
            self.beta = theano.shared(value=numpy.zeros((channels,),
                                dtype=theano.config.floatX), name = 'beta', borrow=borrow)  
            self.running_mean = theano.shared(
                                value=numpy.zeros((channels,), 
                                dtype=theano.config.floatX), 
                                name = 'population_mean', borrow = borrow)
            self.running_var = theano.shared(
                                value=numpy.ones((channels,),
                                dtype=theano.config.floatX),
                                name = 'population_var', borrow=borrow)                                                                                               
        else:
            self.gamma = init_gamma
            self.beta = init_beta
            self.running_mean = init_mean
            self.running_var = init_var

        """
        Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network
        training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015). """
        self.output,_,_,mean,var = batch_normalization_train(
                                                inputs = input,
                                                gamma = self.gamma,
                                                beta = self.beta,
                                                axes ='spatial',
                                                running_mean = self.running_mean,
                                                running_var = self.running_var )

        mean = theano.tensor.unbroadcast(mean,0)
        var = theano.tensor.unbroadcast(var,0)
        var = var + 0.000001
        self.updates[self.running_mean] = mean
        self.updates[self.running_var] = var

        self.inference = batch_normalization_test (
                                                inputs = input,
                                                gamma = self.gamma,
                                                beta = self.beta,
                                                axes = 'spatial',
                                                mean = self.running_mean,
                                                var = self.running_var )   

        # store parameters of this layer and do some book keeping.
        self.parmas = [self.gamma, self.beta, self.running_mean, self.running_var]
        self.active_params = [self.gamma, self.beta]
        self.input_shape = input_shape
        self.output_shape = input_shape 

class dropout_batch_norm_layer_2d(batch_norm_layer_2d):
    """
    This class is the typical 2D batchnorm layer. It is called
    by the ``add_layer`` method in network class.

    Args:
        input: An input ``theano.tensor`` variable. Even ``theano.shared`` will work as long as they
               are in the following shape ``mini_batch_size, height, width, channels``
        verbose: similar to the rest of the toolbox.
        input_shape: ``(mini_batch_size, channels, height, width)``
        borrow: ``theano`` borrow, typicall ``True``.
        dropout_rate: bernoulli probabilty to dropoutby
        input_params: Supply params or initializations from a pre-trained system.
    """
    def __init__(  self,
                   input,
                   input_shape,
                   id,
                   rng = None,
                   input_params = None,
                   dropout_rate = 0,
                   verbose = 2,
                   ):

        if verbose >=3:
            print "... setting up the dropout layer, just in case."
        if rng is None:
            rng = numpy.random 
        super(dropout_batch_norm_layer_2d, self).__init__(
                                                input = input,
                                                input_shape = input_shape,
                                                id = id,
                                                rng = rng,
                                                borrow = borrow,
                                                input_params = input_params,
                                                verbose = verbose
                                                )
        if not dropout_rate == 0:
            self.output = _dropout(rng = rng,
                                params = self.output,
                                dropout_rate = dropout_rate)
        if verbose >=3:
            print "... Dropped out"
        self.dropout_rate = dropout_rate

class batch_norm_layer_1d (layer):
    """
    This class is the typical 1D batchnorm layer. It is called
    by the ``add_layer`` method in network class.

    Args:
        input: An input ``theano.tensor`` variable. Even ``theano.shared`` will work as long as they
               are in the following shape ``mini_batch_size, height, width, channels``
        verbose: similar to the rest of the toolbox.
        input_shape: ``(mini_batch_size, channels, height, width)``
        rng: typically ``numpy.random``.
        borrow: ``theano`` borrow, typicall ``True``.
        input_params: Supply params or initializations from a pre-trained system.
    """
    def __init__ ( self,
                   input,
                   input_shape,
                   id,
                   rng = None,
                   borrow = True,
                   input_params = None,
                   verbose = 2,
                 ):

        super(batch_norm_layer_1d,self).__init__(id = id, type = 'batch_norm', verbose = verbose)
        if verbose >=3:
            print "... Creating batch norm layer"

        if rng is None:
            rng = numpy.random

        # To copy weights previously created or some wierd initializations
        if input_params is not None:
            init_gamma = input_params[0]
            init_beta = input_params[1]
            init_mean = input_params[2]
            init_var = input_params[3]

        channels   = input_shape[1]

        if input_params is None:
            self.gamma = theano.shared(value=numpy.ones((channels,),
                                dtype=theano.config.floatX), name = 'gamma', borrow = borrow)
            self.beta = theano.shared(value=numpy.zeros((channels,),
                                dtype=theano.config.floatX), name = 'beta', borrow=borrow)  
            self.running_mean = theano.shared(
                                value=numpy.zeros((channels,), 
                                dtype=theano.config.floatX), 
                                name = 'population_mean', borrow = borrow)
            self.running_var = theano.shared(
                                value=numpy.ones((channels,),
                                dtype=theano.config.floatX),
                                name = 'population_var', borrow=borrow)                                                                                               
        else:
            self.gamma = init_gamma
            self.beta = init_beta
            self.running_mean = init_mean
            self.running_var = init_var

        """
        Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network
        training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015). """
        self.output,_,_,mean,var = batch_normalization_train(
                                                inputs = input,
                                                gamma = self.gamma,
                                                beta = self.beta,
                                                running_mean = self.running_mean,
                                                running_var = self.running_var )

        mean = theano.tensor.unbroadcast(mean,0)
        var = theano.tensor.unbroadcast(var,0)
        var = var + 0.0000001 
        self.updates[self.running_mean] = mean
        self.updates[self.running_var] = var

        self.inference = batch_normalization_test (
                                                inputs = input,
                                                gamma = self.gamma,
                                                beta = self.beta,
                                                mean = self.running_mean,
                                                var = self.running_var )   

        # store parameters of this layer and do some book keeping.
        self.parmas = [self.gamma, self.beta, self.running_mean, self.running_var]
        self.active_params = [self.gamma, self.beta]
        self.input_shape = input_shape
        self.output_shape = input_shape 

class dropout_batch_norm_layer_1d(batch_norm_layer_2d):
    """
    This class is the typical 1D batchnorm layer. It is called
    by the ``add_layer`` method in network class.

    Args:
        input: An input ``theano.tensor`` variable. Even ``theano.shared`` will work as long as they
               are in the following shape ``mini_batch_size, height, width, channels``
        verbose: similar to the rest of the toolbox.
        input_shape: ``(mini_batch_size, channels, height, width)``
        borrow: ``theano`` borrow, typicall ``True``.
        dropout_rate: bernoulli probabilty to dropoutby
        input_params: Supply params or initializations from a pre-trained system.
    """
    def __init__(  self,
                   input,
                   input_shape,
                   id,
                   rng = None,
                   input_params = None,
                   dropout_rate = 0,
                   verbose = 2,
                   ):

        if verbose >=3:
            print "... setting up the dropout layer, just in case."
        if rng is None:
            rng = numpy.random 
        super(dropout_batch_norm_layer_1d, self).__init__(
                                                input = input,
                                                input_shape = input_shape,
                                                id = id,
                                                rng = rng,
                                                borrow = borrow,
                                                input_params = input_params,
                                                verbose = verbose
                                                )
        if not dropout_rate == 0:
            self.output = _dropout(rng = rng,
                                params = self.output,
                                dropout_rate = dropout_rate)
        if verbose >=3:
            print "... Dropped out"
        self.dropout_rate = dropout_rate

if __name__ == '__main__':
    pass
