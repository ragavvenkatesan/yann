from abstract import layer, _activate, _dropout
import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet.bn import batch_normalization_train, batch_normalization_test

class dot_product_layer (layer):
    """
    This class is the typical neural hidden layer and batch normalization layer. It is called
    by the ``add_layer`` method in network class.

    Args:
        input: An input ``theano.tensor`` variable. Even ``theano.shared`` will work as long as they
               are in the following shape ``mini_batch_size, height, width, channels``
        verbose: similar to the rest of the toolbox.
        num_neurons: number of neurons in the layer
        input_shape: ``(mini_batch_size, input_size)`` theano shared
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

        create = False
        if input_params is None:
            create = True
        elif input_params[0] is None:
            create = True
        if create is True:
            w_values = numpy.asarray(0.01 * rng.standard_normal(
                size=(input_shape[1], num_neurons)), dtype=theano.config.floatX)
            if activation == 'sigmoid':
                w_values*=4
            self.w = theano.shared(value=w_values, name='weights')
        else:
            self.w = input_params[0]

        create = False
        if input_params is None:
            create = True
        elif input_params[1] is None:
            create = True
        if create is True:
            b_values = numpy.zeros((num_neurons,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='bias')
        else:
            self.b = input_params[1]

        if batch_norm is True:
            create = False
            if input_params is None:
                create = True
            elif input_params[2] is None:
                create = True
            if create is True:
                gamma_values = numpy.ones((1,num_neurons), dtype = theano.config.floatX)
                self.gamma = theano.shared(value = gamma_values, name = 'gamma')
                beta_values = numpy.zeros((1,num_neurons), dtype=theano.config.floatX)
                self.beta = theano.shared(value=beta_values, name='beta')     
                self.running_mean = theano.shared(
                                    value=numpy.zeros((1,num_neurons), 
                                    dtype=theano.config.floatX), 
                                    name = 'population_mean', borrow = borrow)
                self.running_var = theano.shared(
                                    value=numpy.ones((1,num_neurons),
                                    dtype=theano.config.floatX),
                                    name = 'population_var', borrow=borrow)                                                           
            else:
                self.gamma = input_params[2]
                self.beta = input_params[3]
                self.running_mean = input_params [4]
                self.running_var = input_params [5]

        linear_fit = T.dot(input, self.w) + self.b

        if batch_norm is True:                               
            batch_norm_out,_,_,mean,var = batch_normalization_train (
                                                  inputs = linear_fit,
                                                  gamma = self.gamma,
                                                  beta = self.beta,
                                                  running_mean = self.running_mean,
                                                  running_var = self.running_var) 

            mean = theano.tensor.unbroadcast(mean,0)
            var = theano.tensor.unbroadcast(var,0)
            self.updates[self.running_mean] = mean
            self.updates[self.running_var] = var + 0.001

            batch_norm_inference = batch_normalization_test (inputs = linear_fit,
                                                            gamma = self.gamma,
                                                            beta = self.beta,
                                                            mean = self.running_mean,
                                                            var = self.running_var  )
        else:
            batch_norm_out = linear_fit
            batch_norm_inference = batch_norm_out

        batch_norm_shp = (input_shape[0], num_neurons)
        self.output, self.output_shape = _activate (x= batch_norm_out,
                                            activation = activation,
                                            input_size = batch_norm_shp,
                                            verbose = verbose,
                                            dimension = 1)

        self.inference, _ = _activate (x= batch_norm_out,
                                            activation = activation,
                                            input_size = batch_norm_shp,
                                            verbose = verbose,
                                            dimension = 1)

        # parameters of the model
        if batch_norm is True:
            self.params = [self.w, self.b, self.gamma, self.beta, 
                                    self.running_mean, self.running_var]
            self.active_params = [self.w, self.b, self.gamma, self.beta]                                    
        else:
            self.params = [self.w, self.b]
            self.active_params = [self.w, self.b]
            
        self.L1 = abs(self.w).sum()
        # if batch_norm is True: self.L1 = self.L1 + abs(self.gamma).sum()
        self.L2 = (self.w**2).sum()
        # if batch_norm is True: self.L2 = self.L2 + (self.gamma**2).sum()

        """
        Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network
        training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015). """

        if verbose >=3:
            print "... Dot Product layer is created with output shape " + str(self.output_shape)

        self.num_neurons = num_neurons
        self.activation = activation
        self.batch_norm = batch_norm

    def get_params (self , borrow = True, verbose = 2):
        """
        This method returns the parameters of the layer in a numpy ndarray format.

        Args:
            borrow : Theano borrow, default is True.
            verbose: As always

        Notes:
            This is a slow method, because we are taking the values out of GPU. Ordinarily, I should
            have used get_value( borrow = True ), but I can't do this because some parameters are
            theano.tensor.var.TensorVariable which needs to be run through eval.
        """
        out = []

        for p in self.params:
            try:
                out.append(p.get_value(borrow = borrow))
            except:
                out.append(p.eval())
        return out

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