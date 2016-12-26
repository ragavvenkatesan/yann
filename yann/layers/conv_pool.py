from abstract import layer, _activate, _dropout
import numpy
import theano
import theano.tensor as T
from yann.core.conv import convolver_2d
from yann.core.pool import pooler_2d

class conv_pool_layer_2d (layer):
    """
    This class is the typical 2D convolutional pooling and batch normalizationlayer. It is called 
    by the ``add_layer`` method in network class.

    Args:
        input: An input ``theano.tensor`` variable. Even ``theano.shared`` will work as long as they
               are in the following shape ``mini_batch_size, height, width, channels``
        verbose: similar to the rest of the toolbox.
        nkerns: number of neurons in the layer
        input_shape: ``(mini_batch_size, channels, height, width)``
        filter_shape: (<int>,<int>)
        pool_size: Subsample size, default is ``(1,1)``.
        pool_type: Refer to :mod:`pool` for details. {'max', 'sum', 'mean', 'max_same_size'}
        batch_norm: If provided will be used, default is ``False``. 
        border_mode: Refer to ``border_mode`` variable in ``yann.core.conv``, module `conv`        
        stride: tuple ``(int , int)``. Used as convolution stride. Default ``(1,1)``
        rng: typically ``numpy.random``.
        borrow: ``theano`` borrow, typicall ``True``.                                        
        activation: String, takes options that are listed in :mod:`activations` Needed for
                    layers that use activations.
                    Some activations also take support parameters, for instance ``maxout``
                    takes maxout type and size, ``softmax`` takes an option temperature.
                    Refer to the module :mod:`activations` to know more.
        input_params: Supply params or initializations from a pre-trained system.

    Notes:
        Use ``conv_pool_layer_2d.output`` and ``conv_pool_layer_2d.output_shape`` from this class.
        ``L1`` and ``L2`` are also public and can also can be used for regularization.
        The class also has in public ``w``, ``b`` and ``alpha``
        which are also a list in ``params``, another property of this class.  
    """
    # Why am I not using **kwargs here ? - I don't want to allow arbitrary forms for this function,
    # I want a particular form with various default inputs, ergo. 
    def __init__ ( self,                    
                   input,
                   nkerns,
                   input_shape,   
                   id,                
                   filter_shape = (3,3),                   
                   poolsize = (2,2),
                   pooltype = 'max',
                   batch_norm = False,                   
                   border_mode = 'valid',  
                   stride = (1,1),
                   rng = None,
                   borrow = True,
                   activation = 'relu',
                   input_params = None,                   
                   verbose = 2,
                 ):

        super(conv_pool_layer_2d,self).__init__(id = id, type = 'conv_pool', verbose = verbose)                
        if verbose >=3: 
            print "... Creating conv pool layer"

        if rng is None:
            rng = numpy.random

        # To copy weights previously created or some wierd initializations
        if input_params is not None:
            init_w = input_params[0]
            init_b = input_params[1]
            if batch_norm is True:
                init_alpha = input_params[2]

        mini_batch_size  = input_shape[0]
        channels   = input_shape[1] 
        width      = input_shape[3]
        height     = input_shape[2]
        # srng = RandomStreams(rng.randint(1,2147462579))
        # Initialize the parameters of this layer.
        w_shp = (nkerns, channels, filter_shape[0], filter_shape[1])        
        if input_params is None:
            # I have no idea what this is all about. Saw this being used in theano tutorials, 
            # I am doing the same thing.
            fan_in = filter_shape[0]*filter_shape[1]
            fan_out = filter_shape[0]*filter_shape[1] / numpy.prod(poolsize)        
            w_bound = numpy.sqrt(6. / (fan_in + fan_out))          
            self.w = theano.shared(value=
                   numpy.asarray(rng.uniform(low=-w_bound, high=w_bound, size =w_shp),
                                    dtype=theano.config.floatX ), borrow=borrow, name ='w' )
            self.b = theano.shared(value=numpy.zeros((w_shp[0]), dtype=theano.config.floatX),
                                     name = 'b', borrow=borrow)  
            self.alpha = theano.shared(value=numpy.ones((w_shp[0]), 
                                 dtype=theano.config.floatX), name = 'alpha', borrow = borrow)                                                                                                                    
        else:
            self.w = init_w
            self.b = init_b
            if batch_norm is True:
                self.alpha = init_alpha

        # Perform the convolution part
        convolver  = convolver_2d (
                        input = input,
                        filters = self.w,
                        subsample = stride,
                        filter_shape = w_shp,
                        image_shape = input_shape,
                        border_mode = border_mode,
                        verbose = verbose
                           )        

        conv_out = convolver.out
        conv_out_shp = (mini_batch_size, nkerns, convolver.out_shp[0], convolver.out_shp[1])   

        self.conv_out = conv_out
        if not poolsize == (1,1):
             pooler = pooler_2d( 
                                input = conv_out,
                                img_shp = conv_out_shp,
                                mode = pooltype, 
                                ds = poolsize,
                                verbose = verbose
                            )
             pool_out = pooler.out
             pool_out_shp = pooler.out_shp           
        else:
            pool_out = conv_out
            pool_out_shp = conv_out_shp
        """
        Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network 
        training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015). """
        if batch_norm is True:
            mean = pool_out.mean( (0,2,3), keepdims = True )
            std = pool_out.std( (0,2,3), keepdims = True )            
            std += 0.001 # To avoid divide by zero like fudge factor        
            pool_out = pool_out - mean
            # use one bias for both batch norm and regular bias.
            batch_norm_out = pool_out * ( self.alpha.dimshuffle('x', 0, 'x', 'x') / std ) + \
                                                        self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            batch_norm_out = pool_out + self.b.dimshuffle('x', 0, 'x', 'x')            
        
        batch_norm_out_shp = pool_out_shp        
        self.output, self.output_shape = _activate (x= batch_norm_out,
                                            activation = activation,
                                            input_size = batch_norm_out_shp,
                                            verbose = verbose,
                                            dimension = 2)
         
        # store parameters of this layer and do some book keeping.
        self.params = [self.w, self.b] 
        if batch_norm is True: 
            self.params.append(self.alpha)        

        self.L1 = abs(self.w).sum() 
        if batch_norm is True : self.L1 = self.L1 + abs(self.alpha).sum()
        self.L2 = (self.w**2).sum() 
        if batch_norm is True: self.L2 = self.L2 + (self.alpha**2).sum()

        # Just doing this for print_layer method to use. 
        self.nkerns = nkerns
        self.filter_shape = filter_shape
        self.poolsize = poolsize
        self.stride = stride
        self.input_shape = input_shape
        self.num_neurons = nkerns
        self.activation = activation
        self.batch_norm = batch_norm

    def print_layer(self, prefix = " " , nest = False, last = True):
        """
        Print information about the layer
        """
        prefix = super(conv_pool_layer_2d, self).print_layer(prefix = prefix, 
                                                             nest = nest, 
                                                             last = last,
                                                             )
        print self.prefix_entry + " filter size [" + str(self.filter_shape[0]) + " X " + \
                                                                str(self.filter_shape[1]) +"]"
        print self.prefix_entry + " pooling size [" + str(self.poolsize[0]) + " X " + \
                                                                    str(self.poolsize[1]) + "]"
        print self.prefix_entry + " stride size [" + str(self.stride[0]) + " X " + \
                                                                    str(self.stride[1]) + "]"            
        print self.prefix_entry + " input shape ["  + str(self.input_shape[2]) + " " + \
                                                                str(self.input_shape[3]) + "]"
        print self.prefix_entry + " input number of feature maps is " +  str(self.input_shape[1]) 
        print self.prefix_entry + "-----------------------------------"
                                                             
        return prefix

class dropout_conv_pool_layer_2d(conv_pool_layer_2d):    
    """
    This class is the typical 2D convolutional pooling and batch normalizationlayer. It is called 
    by the ``add_layer`` method in network class.

    Args:
        input: An input ``theano.tensor`` variable. Even ``theano.shared`` will work as long as they
               are in the following shape ``mini_batch_size, height, width, channels``
        verbose: similar to the rest of the toolbox.
        nkerns: number of neurons in the layer
        input_shape: ``(mini_batch_size, channels, height, width)``
        filter_shape: (<int>,<int>)
        pool_size: Subsample size, default is ``(1,1)``.
        pool_type: Refer to :mod:`pool` for details. {'max', 'sum', 'mean', 'max_same_size'}
        batch_norm: If provided will be used, default is ``False``. 
        border_mode: Refer to ``border_mode`` variable in ``yann.core.conv``, module 
                        :mod:`conv`              
        stride: tuple ``(int , int)``. Used as convolution stride. Default ``(1,1)``
        rng: typically ``numpy.random``.
        borrow: ``theano`` borrow, typicall ``True``.                                        
        activation: String, takes options that are listed in :mod:`activations` Needed for
                    layers that use activations.
                    Some activations also take support parameters, for instance ``maxout``
                    takes maxout type and size, ``softmax`` takes an option temperature.
                    Refer to the module :mod:`activations` to know more.
        input_params: Supply params or initializations from a pre-trained system.

    Notes:
        Use ``conv_pool_layer_2d.output`` and ``conv_pool_layer_2d.output_shape`` from this class.
        ``L1`` and ``L2`` are also public and can also can be used for regularization.
        The class also has in public ``w``, ``b`` and ``alpha``
        which are also a list in ``params``, another property of this class.  
    """
    def __init__(  self, 
                   input,                   
                   nkerns,
                   input_shape,      
                   id, 
                   dropout_rate = 0.5,            
                   filter_shape = (3,3),                   
                   poolsize = (2,2),
                   pooltype = 'max',
                   batch_norm = True,                   
                   border_mode = 'valid',  
                   stride = (1,1),
                   rng = None,
                   borrow = True,
                   activation = 'relu',
                   input_params = None,                   
                   verbose = 2,
                   ):

        if verbose >=3:
            print "... setting up the dropout layer, just in case."
        if rng is None:
            rng = numpy.random                        
        super(dropout_conv_pool_layer_2d, self).__init__(
                                                input = input,
                                                nkerns = nkerns,
                                                input_shape = input_shape,    
                                                id = id,               
                                                filter_shape = filter_shape,                   
                                                poolsize = poolsize,
                                                pooltype = pooltype,
                                                batch_norm = batch_norm,                   
                                                border_mode = border_mode,  
                                                stride = stride,
                                                rng = rng,
                                                borrow = borrow,
                                                activation = activation,
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
