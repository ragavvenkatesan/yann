from abstract import layer
from yann.core.errors import rmse, l2, l1, cross_entropy
import numpy

class merge_layer (layer):
    """
    This is a merge layer. This takes two layer inputs and produces an error between them if the 
    ``type`` argument supplied was ``'error'``. It does other things too accordingly.

    Args: 
        x: a list of inputs (lenght must be two basically)
        input_shape: List of the shapes of all inputs.
        type: ``'error'`` creates an error layer.
              other options are ``'sum'`` and ``'concatenate'``
        error: If the type was ``'error'``, then this variable is used.
               options include, ``'rmse'``, ``'l2'``, ``'l1'``,``'cross_entropy'``.

    """
    def __init__ (  self,                 
                    x,
                    input_shape,
                    id = -1,
                    type = 'error',                        
                    error = 'rmse',
                    verbose = 2):
           
        if type == 'error':
            if verbose >=3:
                print "... Creating the merge layer"

            super(merge_layer,self).__init__(id = id, type = 'merge', verbose = verbose)

            if error == 'rmse':
                error = rmse
            elif error == 'l2':
                error = l2
            elif error == 'l1':
                error = l1
            elif error == 'cross_entropy':
                error = cross_entropy

            if len(x) > 2:
                raise Exception ( "Use merge layer for merging only two layers. If you want \
                                    to merge more than one you may use this layer more than once" )

            self.output = error(x[0], x[1])
            self.output_shape = (1,)

            if len(input_shape) == 2:
                self.num_neurons = self.output_shape[-1]
            elif len(input_shape) == 4:
                self.num_neurons = self.output_shape[1]

            self.generation = x[0] # I'm basically assuming that 0 is the generation in case 
                                   # this was an auto encoder network.

            if verbose >=3: 
                print "... Merge layer is created with output shape " + str(self.output_shape)
        
            
        elif type == 'sum':
            self.output = x[0] + x[1]
            self.output_shape = input_shape[0]

        elif type == 'concatenate':
            self.output = T.concatenate([x[0],x[1]], axis = 1)
                

    def loss(self, type = None):
        """
        This method will return the cost function of the merge layer. This can be used by the 
        optimizer module for instance to acquire a symbolic loss function that tries to minimize
        the distance between the two layers.

        Args: 
            y: symbolic ``theano.ivector`` variable of labels to calculate loss from.
            type: options 
                           None - Simple error
                           'log' - log loss

        Returns:
            theano symbolic variable: loss value.
        """
        if type is None:
            return self.output

        elif type == 'log':
            return T.log(self.output)

class dropout_merge_layer (merge_layer):
    """
    This is a dropotu merge layer. This takes two layer inputs and produces an error between them 
    if the ``type`` argument supplied was ``'error'``. It does other things too accordingly.

    Args: 
        x: a list of inputs (lenght must be two basically)
        input_shape: List of the shapes of all inputs.
        type: ``'error'`` creates an error layer.
              other options are ``'sum'`` and ``'concatenate'``
        error: If the type was ``'error'``, then this variable is used.
               options include, ``'rmse'``, ``'l2'``, ``'l1'``,``'cross_entropy'``.

    """
    def __init__ (  self, 
                    x,      
                    input_shape,              
                    id = -1,
                    error = 'rmse',
                    verbose = 2):  

        if verbose >= 3:
            print "... set up the dropout merge layer"
        if rng is None:
            rng = numpy.random
        super(dropout_merge_layer, self).__init__ (
                                            x = x,
                                            input_shape = input_shape,
                                            error = error,
                                            id = id,
                                            verbose = verbose
                                        )
        dropout_rate = 0
        if not dropout_rate == 0:            
            self.output = _dropout(rng = rng,
                                params = self.output,
                                dropout_rate = dropout_rate)  

        if verbose >=3: 
            print "... Dropped out"

if __name__ == '__main__':
    pass  