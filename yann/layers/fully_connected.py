from abstract import layer, _activate, _dropout
import numpy
import theano
import theano.tensor as T

class dot_product_layer (layer):
    """
    This class is the typical neural hidden layer and batch normalization layer. It is called 
    by the ``add_layer`` method in network class.

    Args:
        input: An input ``theano.tensor`` variable. Even ``theano.shared`` will work as long as they
               are in the following shape ``mini_batch_size, height, width, channels``
        verbose: similar to the rest of the toolbox.
        num_neurons: number of neurons in the layer
        input_shape: ``(mini_batch_size, input_size)``
        batch_norm: If provided will be used, default is ``False``.        
        rng: typically ``numpy.random``.
        borrow: ``theano`` borrow, typicall ``True``.                                        
        activation: String, takes options that are listed in :mod:`activations` Needed for
                    layers that use activations.
                    Some activations also take support parameters, for instance ``maxout``
                    takes maxout type and size, ``softmax`` takes an option temperature.
                    Refer to the module :mod:`activations` to know more.
        input_params: Supply params or initializations from a pre-trained system.

    Notes:
        Use ``dot_product_layer.output`` and ``dot_product_layer.output_shape`` from this class.
        ``L1`` and ``L2`` are also public and can also can be used for regularization.
        The class also has in public ``w``, ``b`` and ``alpha``
        which are also a list in ``params``, another property of this class.  
    """

    def __init__ (self,
                  input,
                  num_neurons,
                  input_shape,
                  id,
                  rng = None,
                  input_params = None,
                  borrow = True,
                  activation = 'relu',
                  batch_norm = True,
                  verbose = 2 ):
        super(dot_product_layer,self).__init__(id = id, type = 'dot_product', verbose = verbose)
        if verbose >= 3:
            print "... Creating dot product layer"

        if rng is None:
            rng = numpy.random

        if input_params is None:
            w_values = numpy.asarray(0.01 * rng.standard_normal(
                size=(input_shape[1], num_neurons)), dtype=theano.config.floatX)
            if activation == 'sigmoid':
                w_values*=4 
            self.w = theano.shared(value=w_values, name='w')
            b_values = numpy.zeros((num_neurons,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b')
            if batch_norm is True:
                alpha_values = numpy.ones((num_neurons,), dtype = theano.config.floatX)
                self.alpha = theano.shared(value = alpha_values, name = 'alpha')
            
        else:
            self.w = input_params[0]
            self.b = input_params[1]
            if batch_norm is True:
                self.alpha = input_params[2]  

        dot_product = T.dot(input, self.w)
                
        if batch_norm is True:
            std = dot_product.std( 0 )
            mean = dot_product.mean( 0 )
            std += 0.001 # To avoid divide by zero like fudge_factor
        
            dot_product = dot_product - mean 
            dot_product = dot_product * ( self.alpha / std ) 
            
        dot_product = dot_product  + self.b
        dot_product_shp = (input_shape[0], num_neurons)
        self.output, self.output_shape = _activate (x= dot_product,
                                            activation = activation,
                                            input_size = dot_product_shp,
                                            verbose = verbose,
                                            dimension = 1)   
            
        # parameters of the model
        if batch_norm is True:
            self.params = [self.w, self.b, self.alpha]
        else:
            self.params = [self.w, self.b]
                
        self.L1 = abs(self.w).sum()  
        if batch_norm is True: self.L1 = self.L1 + abs(self.alpha).sum()
        self.L2 = (self.w**2).sum()  
        if batch_norm is True: self.L2 = self.L2 + (self.alpha**2).sum()
        
        if verbose >=3: 
            print "... Dot Product layer is created with output shape " + str(self.output_shape)        

        self.num_neurons = num_neurons
        self.activation = activation
        self.batch_norm = batch_norm

class dropout_dot_product_layer (dot_product_layer):
    """
    This class is the typical dropout neural hidden layer and batch normalization layer. Called 
    by the ``add_layer`` method in network class.

    Args:
        input: An input ``theano.tensor`` variable. Even ``theano.shared`` will work as long as they
               are in the following shape ``mini_batch_size, height, width, channels``
        verbose: similar to the rest of the toolbox.
        num_neurons: number of neurons in the layer
        input_shape: ``(mini_batch_size, input_size)``
        batch_norm: If provided will be used, default is ``False``. 
        rng: typically ``numpy.random``.
        borrow: ``theano`` borrow, typicall ``True``.                                        
        activation: String, takes options that are listed in :mod:`activations` Needed for
                    layers that use activations.
                    Some activations also take support parameters, for instance ``maxout``
                    takes maxout type and size, ``softmax`` takes an option temperature.
                    Refer to the module :mod:`activations` to know more.
        input_params: Supply params or initializations from a pre-trained system.
        dropout_rate: ``0.5`` is the default.

    Notes:
        Use ``dropout_dot_product_layer.output`` and ``dropout_dot_product_layer.output_shape`` from
        this class. ``L1`` and ``L2`` are also public and can also can be used for regularization.
        The class also has in public ``w``, ``b`` and ``alpha``
        which are also a list in ``params``, another property of this class.     
    """
    def __init__ (self,
                  input,
                  num_neurons,
                  input_shape,
                  id,
                  dropout_rate = 0.5,
                  rng = None,
                  input_params = None,
                  borrow = True,
                  activation = 'relu',
                  batch_norm = True,
                  verbose = 2 ):

        if verbose >= 3:
            print "... set up the dropout dot product layer"
        if rng is None:
            rng = numpy.random            
        super(dropout_dot_product_layer, self).__init__ (
                                        input = input,
                                        num_neurons = num_neurons,
                                        input_shape = input_shape,
                                        id = id,
                                        rng = rng,
                                        input_params = input_params,
                                        borrow = borrow,
                                        activation = activation,
                                        batch_norm = batch_norm,
                                        verbose = verbose 
                                        )
        if not dropout_rate == 0:            
            self.output = _dropout(rng = rng,
                                params = self.output,
                                dropout_rate = dropout_rate)                                          
        self.dropout_rate = dropout_rate
        if verbose >=3: 
            print "... Dropped out"

if __name__ == '__main__':
    pass  