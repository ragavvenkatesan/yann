"""
Todo:

    * LSTM / GRN layers
    * An Embed layer that is going to create a new embedding space for two layer's activations to
      project on to the same space and minimize its distances.
"""

import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# from theano.tensor.shared_randomstreams import RandomStreams
# The above import is an experimental code. Not sure if it works perfectly, but I have no doubt
# yet.
from yann.core import activations
import numpy
from collections import OrderedDict

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
        Use ``self.type``, ``self.origin``, self.destination``, ``self.output``, ``self.inference``
            ``self.output_shape`` for outside calls and purposes.
    """

    def __init__(self, id, type, verbose = 2):
        self.id = id
        self.type = type
        self.origin = []  # this and destination will be added from outside.
        self.destination = [] # only None for now during initialization.
        self.output = None  # this is the output during training
        self.inference = None # this is the output during testing (batch norm made this happen)
        self.params = None
        self.active_params = None
        self.output_shape = None
        self.num_neurons = None
        self.activation = 'identity'
        self.dropout_rate = 0
        self.batch_norm = False
        self.active = True # this flag is setup by the add_layer module
        # Every layer must have these properties.
        self.updates = OrderedDict() # layer specific updates. 
        if verbose >= 3:
            print("... Initializing a new layer " + self.id + " of type " + self.type)

    def print_layer(self, prefix = " ", nest = True, last = True, verbose = 2):
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
        prefix_entry +=   "|-"

        print(prefix_entry)
        print(prefix_entry)
        print(prefix_entry)
        print(prefix_entry + " id: " + self.id)
        print(prefix_entry + "=================------------------")
        print(prefix_entry + " type: " + self.type)
        print(prefix_entry + " output shape: " + str(self.output_shape))
        if self.batch_norm is True :
            print prefix_entry + " batch norm is ON"
        elif self.type == 'dot_product' or \
             self.type == 'hidden' or  \
             self.type == 'mlp' or  \
             self.type == 'fully_connected' or \
             self.type == 'conv_pool' or \
             self.type == 'convolution':
            print prefix_entry + " batch norm is OFF"        
        print(prefix_entry + "-----------------------------------")

        if nest is False:
            print(prefix_entry + " origin: " + self.origin)
            print(prefix_entry + " destination: " + self.destination)

        if self.type == 'conv_pool' or 'convolution' or 'deconvolution' or 'deconv' or 'conv':
            self.prefix_entry = prefix_entry
            self.prefix = prefix

        return prefix

    def _graph_attributes(self, verbose = 2):
        """
        This is an internal function that returns attributes as a dictionary so that I can add
        it to the networkx graph output.
        """
        out = {}
        out["id"] = self.id
        if not self.output_shape is None:
            out["output_shape"] = self.output_shape
        else:
            out["output_shape"] = "N/A"
        if not self.num_neurons is None:
            out["num_neurons"] = self.num_neurons
        else:
            out["num_neurons"] = "N/A"
        if type(self.activation) is tuple:
            out["activation"] = self.activation[0]
        else:
            out["activation"] = self.activation
        out["dropout_rate"] = self.dropout_rate
        out["batch_norm"] = self.batch_norm
        out["origin"] = self.origin
        out["type"] = self.type
        return out

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
                out.append(numpy.asarray(p.eval()))
        return out

def _dropout(rng, params, dropout_rate, verbose = 2):
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
        print("... Setting up activations")

    # some activations like maxouts are supplied with special support parameters
    if type(activation) is tuple:
        if activation[0] == 'maxout':
            maxout_size = activation[2]
            maxout_type = activation[1]
            out, out_shp = activations.Maxout(x = x,
                                            maxout_size = maxout_size,
                                            input_size = input_size,
                                            type = maxout_type,
                                            dimension = kwargs["dimension"] )
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
        print("... Activations are setup")

    return (out, out_shp)

if __name__ == '__main__':
    pass
