"""
TODO:

    * LSTM / GRN layers
    * A concatenate layer
    * A Merge layer that is going to sum / average two layer activations.
    * An Embed layer that is going to create a new embedding space for two layer's activations to
      project on to the same space and minimize its distances. 
    * An error layer that produces the error between two layers. (use errors.py in core.)
      - This can be used to generate images back such as in the case of auto-encoders.    
"""
import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# from theano.tensor.shared_randomstreams import RandomStreams
# The above import is an experimental code. Not sure if it works perfectly, but I have no doubt 
# yet.
from yann.core import activations
from yann.core.conv import convolver_2d
from yann.core.pool import pooler_2d

class layer(object):
    """
    Prototype for what a layer should look like. Every layer should inherit from this. This is 
    a template class do not use this directly, you need to use a specific type of layer which 
    again will be called by ``yann.network.network.add_layer``

    Args:
        id: String
        origin: String id
        type: string- ``'classifier'``, ``'dot-product'``, ``'objective'``, ``'conv_pool'``,
              ``'input'`` .. .

    Notes:
        Use ``self.type``, ``self.origin``, self.destination``, ``self.output``, 
            ``self.output_shape`` for outside calls and purposes.
    """

    def __init__(self, id, type, verbose = 2):
        self.id = id
        self.type = type
        self.origin = []  # this and destination will be added from outside. 
        self.destination = [] # only None for now during initialization. 
        self.output = None
        self.output_shape = None
        # Every layer must have these four properties.
        if verbose >= 3:
            print "... Initializing a new layer " + self.id + " of type " + self.type        

    def print_layer(self, prefix = " ", nest = True, last = True):
        """
        Print information about the layer
        
        Args:
            nest: If True will print the tree from here on. If False it will print only this
                layer.
            prefix: Is what prefix you want to add to the network print command.
        """       
        prefix_entry = prefix
        
        if last is True:
            prefix += "         "
        else:
            prefix +=  "|        " 
        prefix_entry +=   "|- " 
                
        print prefix_entry + "-----------------"
        print prefix_entry + " id: " + self.id
        print prefix_entry + " type: " + self.type
        print prefix_entry, 
        print self.output_shape
        print prefix_entry + "-----------------"

        if nest is False:
            print prefix_entry + " origin: " + self.origin
            print prefix_entry + " destination: " + self.destination      

        if self.type == 'conv_pool':
            self.prefix_entry = prefix_entry
            self.prefix = prefix
        
        return prefix

def _dropout(rng, params, dropout_rate):
    """
    dropout thanks to misha denil 
    https://github.com/mdenil/dropout    
    """
    srng = RandomStreams(rng.randint(1,2147462468), use_cuda=None)
    # I have raised this issue with the theano guys, use_cuda = True is creating a duplicate 
    # process in the GPU.
    mask = srng.binomial(n=1, p=1-dropout_rate, size=params.shape, dtype = theano.config.floatX )
    output = params * mask
    return output

def _activate (x, activation, input_size, verbose = 2, **kwargs):
    """
    This function is used to produce activations for the outputs of any type of layer.
    
    Args:
        
        x: input tensor.
        activation: Refer to the ``add_layer`` method.
        input_size: supply the size of the inputs. 
        verbose: typical toolbox verbose
        dimension: used only for maxout. Give the dimension on which to maxout.

    Returns:

        tuple: ``(out, out_shp)``

    """
    if verbose >=3: 
        print "... Setting up activations"

    # some activations like maxouts are supplied with special support parameters    
    if type(activation) is tuple:
        if activation[0] == 'maxout':
            maxout_size = activation[2]
            maxout_type = activation[1]
            out, kern_shp = activations.Maxout(x = x, 
                                            maxout_size = maxout_size,
                                            input_size = input_size,
                                            type = maxout_type,
                                            dimension = kwargs["dimension"] ) 
            out_shp = input_size
            out_shp [1] = kern_shp
        if activation[0] == 'relu':
            relu_leak = activation[1]
            out = activations.ReLU (x = x, alpha = relu_leak)
            out_shp = input_size
        if activation[0] == 'softmax':
            temperature = activation[1]
            out = activations.Softmax (x=x,temp = temperature)
            out_shp = input_size
    else:
        if activation == 'relu':
            out = activations.ReLU (x=x)
        elif activation == 'abs':
            out = activations.Abs (x=x)
        elif activation == 'sigmoid':
            out = activations.Sigmoid( x=x)
        elif activation == 'tanh':
            out = activations.Tanh (x=x)
        elif activation == 'softmax':
            out = activations.Softmax (x=x)
        elif activation == 'squared':
            out = activations.Squared (x=x)
        out_shp = input_size

    if verbose >=3: 
        print "... Activations are setup"

    return (out, out_shp)

class input_layer (layer):
    """
    reshapes it into images. This is needed because yann dataset module assumes data in 
    vectorized image formats as used in mnist - theano tutorials. 

    This class also creates a branch between a mean subtracted and non-mean subtracted input. 
    It always assumes as default to use the non-mean subtracted input but if ``mean_subtract``
    flag is provided, it will use the other option. 

    Args: 
        x: ``theano.tensor`` variable with rows are vectorized images.
        y: ``theano.tensor`` variable
        one_hot_y:``theano.tensor`` variable       
        batch_size: Number of images in the data variable.
        height: Height of each image.
        width: Width of each image.
        id: Supply a layer id
        channels: Number of channels in each image.
        mean_subtract: Defauly is ``False``.
        verbose: Similar to all of the toolbox.

    Notes: 
        Use ``input_layer.output`` to continue onwards with the network
        ``input_layer.output_shape`` will tell you the output size.
        Use ``input_layer.x``, ``input_layer.y`` and ``input_layer_one_hot_y`` tensors
        for connections.

    """
    def __init__ (  self,                 
                    batch_size, 
                    x,
                    id = -1,                    
                    height=28,
                    width=28,
                    channels=1,
                    mean_subtract = False,
                    verbose = 2):
           
        if verbose >=3:
            print "... Creating the input layer"

        super(input_layer,self).__init__(id = id, type = 'input', verbose = verbose)
        data_feeder = x.reshape((batch_size, height, width, channels)).dimshuffle(0,3,1,2)
                                # the dim shuffle makes it batch_size, channels height width order
        mean_subtracted_data_feeder = data_feeder - data_feeder.mean()

        self.output = mean_subtracted_data_feeder if mean_subtract is True else data_feeder    
        self.output_shape = (batch_size, channels, height, width)
        
        if verbose >=3: 
            print "... Input layer is created with output shape " + str(self.output_shape)

class dropout_input_layer (input_layer):
    """
    Creates a new input_layer. The layer doesn't do much except to take the networks' x and 
    reshapes it into images. This is needed because yann dataset module assumes data in 
    vectorized image formats as used in mnist - theano tutorials. 

    This class also creates a branch between a mean subtracted and non-mean subtracted input. 
    It always assumes as default to use the non-mean subtracted input but if ``mean_subtract``
    flag is provided, it will use the other option. 

    Args: 
        x: ``theano.tensor`` variable with rows are vectorized images.
          if None, will create a new one.
        batch_size: Number of images in the data variable.
        height: Height of each image.
        width: Width of each image.
        channels: Number of channels in each image.
        mean_subtract: Defauly is ``False``.
        verbose: Similar to all of the toolbox.

    Notes: 
        Use ``input_layer.output`` to continue onwards with the network
        ``input_layer.output_shape`` will tell you the output size.

    """
    def __init__ (  self, 
                    batch_size, 
                    id,
                    x,                    
                    dropout_rate = 0.5,
                    height=28,
                    width=28,
                    channels=1,
                    mean_subtract = False,
                    rng = None,
                    verbose = 2):  

        if verbose >= 3:
            print "... set up the dropout input layer"
        if rng is None:
            rng = numpy.random
        super(dropout_input_layer, self).__init__ (
                                            x = x,
                                            batch_size = batch_size, 
                                            id = id,
                                            height=height,
                                            width=width,
                                            channels=channels,
                                            mean_subtract = mean_subtract,
                                            verbose = verbose
                                        )
        if not dropout_rate == 0:            
            self.output = _dropout(rng = rng,
                                params = self.output,
                                dropout_rate = dropout_rate)  

        if verbose >=3: 
            print "... Dropped out"

class classifier_layer (layer):
    """
    This class is the typical classifier layer. It should be called 
    by the ``add_layer`` method in network class.

    Args:
        input: An input ``theano.tensor`` variable. Even ``theano.shared`` will work as long as they
               are in the following shape ``batch_size, height, width, channels``
        verbose: similar to the rest of the toolbox.
        input_shape: ``(batch_size, features)``
        num_classes: number of classes to classify into
        filter_shape: (<int>,<int>)
        rng: typically ``numpy.random``.
        borrow: ``theano`` borrow, typicall ``True``.   
        rng: typically ``numpy.random``.                                             
        activation: String, takes options that are listed in :mod:`activations` Needed for
                    layers that use activations.
                    Some activations also take support parameters, for instance ``maxout``
                    takes maxout type and size, ``softmax`` takes an option temperature.
                    Refer to the module :mod:`activations` to know more.
                    Default is 'softmax'
        input_params: Supply params or initializations from a pre-trained system.

    Notes:
        Use ``classifier_layer.output`` and ``classifier_layer.output_shape`` from this class.
        ``L1`` and ``L2`` are also public and can also can be used for regularization.
        The class also has in public ``w``, ``b`` and ``alpha`` which are also a list in ``params``, 
        another property of this class.  
    """
    def __init__ (  self,
                    input,
                    input_shape,     
                    id,               
                    num_classes = 10,
                    rng = None,
                    input_params = None,
                    borrow = True,
                    activation = 'softmax',
                    verbose = 2
                    ):
        
        super(classifier_layer,self).__init__(id = id, type = 'classifier', verbose = verbose)    

        if rng is None:
            rng = numpy.random

        if verbose >=3: 
            print "... Creating classifier layer"
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.input = input
        # To copy weights previously created or some wierd initializations
        if input_params is not None:
            self.w = input_params[0]
            self.b = input_params[1]            
        else:
            self.w = theano.shared ( value=numpy.asarray(0.01 * rng.standard_normal( 
                                     size=(input_shape[1], num_classes)), 
                                     dtype=theano.config.floatX), name='w' ,borrow = borrow)                    
            self.b = theano.shared( value=numpy.zeros((num_classes,),
                                        dtype=theano.config.floatX), 
                                     name='b' ,borrow = borrow)
        
        self.fit = T.dot(input, self.w) + self.b
        self.p_y_given_x, softmax_shp = _activate ( x =  self.fit,
                                                    activation = activation,
                                                    input_size = num_classes,
                                                    verbose = verbose,
                                                    dimension = 2 )
        
        # compute prediction as class whose probability is maximal in symbolic form
        self.predictions = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.L1 = abs(self.w).sum()
        self.L2 = (self.w ** 2).sum()
        self.params = [self.w, self.b] 
        self.probabilities = T.log(self.p_y_given_x)
        self.output = self.p_y_given_x 
        self.output_shape = (input_shape[0], num_classes)

        if verbose >=3: 
            print "... Classifier layer is created with output shape " + str(self.output_shape)
            
    def _negative_log_likelihood(self, y):
        """
        Negative log-likelihood cost of the classifier layer.
        Do not use this directly, use the ``loss`` method instead.

        Args:

            y: datastreamer's ``y`` variable, that has the lables. 

        Returns:

            theano variable: Negative log-likelihood
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) 
        
    def _categorical_cross_entropy( self, y ):
        """
        Categorical cross-entropy of the classifier layer.
        Do not use this directly, use the ``loss`` method instead.

        Args:

            y: datastreamer's ``y`` variable, that has the lables. 

        Returns:

            theano variable: categorical_cross_entropy
        """        
        return T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x,y))

    def _binary_cross_entropy ( self, y ):
        """
        Binary cross entropy of the classifier layer.
        Do not use this directly, use the ``loss`` method instead.

        Args:

            y: datastreamer's ``y`` variable, that has the lables. 

        Returns:

            theano variable: Binary cross entropy
        """        
        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x,y))
        
    def errors(self, y):
        """
        This function returns a count of wrong predictions.

        Args:

            y: datastreamer's ``y`` variable, that has the lables. 

        Returns:

            theano variable: number of wrong predictions. 
        """      
        if y.ndim != self.predictions.ndim:
            raise TypeError('y should have the same shape as self.predictions',
                ('y', target.type, 'predictions', self.predictions.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return T.sum(T.neq(self.predictions, y))   
        else:
            raise NotImplementedError()

    def _hinge(self, u):
        """
        Do not use this directly, use the ``loss`` method that uses ``_hinge_loss`` method to call.
        """        
        return T.maximum(0, 1 - u)

    def _hinge_loss(self, y):
        """
        Hinge loss cost of the classifier layer.
        Do not use this directly, use the ``loss`` method instead.

        Args:

            y1: datastreamer's ``y1`` variable, that has the lables.  Use ``svm_flag`` in 
                datastreamer setup.

        Returns:

            theano variable: Hinge loss.
        """        
        margin = y * self.fit
        return self._hinge(margin).mean(axis=0).sum()

    def loss(self, y, type):
        """
        This method will return the cost function of the classifier layer. This can be used by the 
        optimizer module for instance to acquire a symbolic loss function.

        Args: 
            y: symbolic ``theano.ivector`` variable of labels to calculate loss from.
            type: options 'nll' - negative log likelihood,
                          'cce' - categorical cross entropy,
                          'bce' - binary cross entropy,
                          'hinge' - max-margin hinge loss. 

        Returns:
            theano symbolic variable: loss value.
        """
        if type == 'nll':
            return self._negative_log_likelihood( y = y )
        elif type == 'cce':
            return self._categorical_cross_entropy( y = y )
        elif type == 'bce':
            return self._binary_cross_entropy( y = y )
        elif type == 'hinge':
            return self._hinge_loss( y = y )
        else:
            raise Exception("Classifier layer does not support " + type + " loss")

class dot_product_layer (layer):
    """
    This class is the typical neural hidden layer and batch normalization layer. It is called 
    by the ``add_layer`` method in network class.

    Args:
        input: An input ``theano.tensor`` variable. Even ``theano.shared`` will work as long as they
               are in the following shape ``batch_size, height, width, channels``
        verbose: similar to the rest of the toolbox.
        num_neurons: number of neurons in the layer
        input_shape: ``(batch_size, input_size)``
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
        srng = RandomStreams(rng.randint(1,2147462579))

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

class dropout_dot_product_layer (dot_product_layer):
    """
    This class is the typical dropout neural hidden layer and batch normalization layer. Called 
    by the ``add_layer`` method in network class.

    Args:
        input: An input ``theano.tensor`` variable. Even ``theano.shared`` will work as long as they
               are in the following shape ``batch_size, height, width, channels``
        verbose: similar to the rest of the toolbox.
        num_neurons: number of neurons in the layer
        input_shape: ``(batch_size, input_size)``
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
        if verbose >=3: 
            print "... Dropped out"

class conv_pool_layer_2d (layer):
    """
    This class is the typical 2D convolutional pooling and batch normalizationlayer. It is called 
    by the ``add_layer`` method in network class.

    Args:
        input: An input ``theano.tensor`` variable. Even ``theano.shared`` will work as long as they
               are in the following shape ``batch_size, height, width, channels``
        verbose: similar to the rest of the toolbox.
        nkerns: number of neurons in the layer
        input_shape: ``(batch_size, channels, height, width)``
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

        batch_size  = input_shape[0]
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
                                    dtype=theano.config.floatX ), borrow=borrow )
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
        conv_out_shp = (batch_size, nkerns, convolver.out_shp[0], convolver.out_shp[1])   

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

    def print_layer(self, prefix = " " , nest = False, last = True):
        """
        Print information about the layer
        """
        prefix = super(conv_pool_layer_2d, self).print_layer(prefix = prefix, 
                                                             nest = nest, 
                                                             last = last,
                                                             )
        print self.prefix_entry + "...  2D conv-pool layer with " + str(self.nkerns)  + \
                                                                                      " kernels"
        print self.prefix_entry + "...  kernel size [" + str(self.filter_shape[0]) + " X " + \
                                                                str(self.filter_shape[1]) +"]"
        print self.prefix_entry + "...  pooling size [" + str(self.poolsize[0]) + " X " + \
                                                                    str(self.poolsize[1]) + "]"
        print self.prefix_entry + "...  stride size [" + str(self.stride[0]) + " X " + \
                                                                    str(self.stride[1]) + "]"            
        print self.prefix_entry + "...  input shape ["  + str(self.input_shape[2]) + " " + \
                                                                str(self.input_shape[3]) + "]"
        print self.prefix_entry + "...  input number of feature maps is " + \
                                                                    str(self.input_shape[1]) 
        print self.prefix_entry + "...  output shape  [" + str(self.output_shape[2] ) + " X "+ \
                                                            str(self.output_shape[3] ) + "]"  
        return prefix

class dropout_conv_pool_layer_2d(conv_pool_layer_2d):    
    """
    This class is the typical 2D convolutional pooling and batch normalizationlayer. It is called 
    by the ``add_layer`` method in network class.

    Args:
        input: An input ``theano.tensor`` variable. Even ``theano.shared`` will work as long as they
               are in the following shape ``batch_size, height, width, channels``
        verbose: similar to the rest of the toolbox.
        nkerns: number of neurons in the layer
        input_shape: ``(batch_size, channels, height, width)``
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

class objective_layer(layer):
    """
    This class is an objective layer. It just has a wrapper for loss function. 
    I need this because I am making objective as a loss layer. 

    Args:
        loss: ``yann.network.layers.classifier_layer.loss()`` method.        
        labels: ``theano.shared`` variable of labels. 
        objective: depends on what is the classifier layer being used. Each have their own 
                   options.
        input_shape: ``(batch_size, predictions)``
        L1: Symbolic weight of the L1 added together
        L2: Sumbolic L2 of the weights added together
        l1_coeff: Coefficient to weight L1 by.
        l2_coeff: Coefficient to weight L2 by.
        verbose: Similar to the rest of the toolbox.

    TODO:

        The loss method needs to change in input. 

    Notes:
        Use ``objective_layer.output`` and from this class.    
    """
    def __init__(   self, 
                    loss,
                    labels,
                    objective,
                    id,
                    input_shape,
                    L1 = None,
                    L2 = None,
                    l1_coeff = 0.001,
                    l2_coeff = 0.001,
                    verbose = 2):
        """
        Refer to the class description
        """        
        super(objective_layer,self).__init__(id = id, type = 'objective', verbose = verbose)                        
        if verbose >=3:
            print "... creating the objective_layer"
        self.output = loss(y = labels, type = objective)
        if L1 is not None:            
            self.output = self.output + l1_coeff * L1 
        if L2 is not None:
            self.output = self.output + l2_coeff * L2
        self.output_shape = (1,)
        if verbose >=3:
            print "... Objective_layer is created with output shape " + str(self.output_shape)

if __name__ == '__main__':
    pass  