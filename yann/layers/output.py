from abstract import layer, _activate, _dropout
import numpy
import theano
import theano.tensor as T

class classifier_layer (layer):
    """
    This class is the typical classifier layer. It should be called 
    by the ``add_layer`` method in network class.

    Args:
        input: An input ``theano.tensor`` variable. Even ``theano.shared`` will work as long as they
               are in the following shape ``mini_batch_size, height, width, channels``
        verbose: similar to the rest of the toolbox.
        input_shape: ``(mini_batch_size, features)``
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
        self.num_neurons = num_classes
        self.activation = activation
        self.dropout_rate = 0
        self.batch_norm = False

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

class objective_layer(layer):
    """
    This class is an objective layer. It just has a wrapper for loss function. 
    I need this because I am making objective as a loss layer. 

    Args:
        loss: ``yann.network.layers.classifier_layer.loss()`` method.        
        labels: ``theano.shared`` variable of labels. 
        objective: depends on what is the classifier layer being used. Each have their own 
                   options.
        input_shape: ``(mini_batch_size, predictions)``
        L1: Symbolic weight of the L1 added together
        L2: Sumbolic L2 of the weights added together
        l1_coeff: Coefficient to weight L1 by.
        l2_coeff: Coefficient to weight L2 by.
        verbose: Similar to the rest of the toolbox.

    Todo:
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