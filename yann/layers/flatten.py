from abstract import layer
import theano.tensor as T

class flatten_layer (layer):
    """
    This is a flatten layer. This takes a square layer and flatten it.

    Args:
        input: output of some layer.
        id: id of the layer
        verbose: as usual

    """
    def __init__ (  self,
                    input,
                    input_shape,
                    id = -1,
                    verbose = 2):
        super(flatten_layer,self).__init__(id = id, type = 'flatten', verbose = verbose)
        self.output = input.flatten(2)
        self.output_shape = (input_shape[0], input_shape[1] * input_shape[2] * input_shape[3])
        self.inference = self.output

class unflatten_layer (layer):
    """
    This is an unflatten layer. This takes a flattened input and unflattens it.

    Args:
        input: output of some layer.
        shape: shape to unflatten.
        id: id of the layer
        verbose: as usual.

    """
    def __init__ (  self,
                    input,
                    shape,
                    input_shape,
                    id = -1,
                    verbose = 2):
        super(unflatten_layer,self).__init__(id = id, type = 'unflatten', verbose = verbose)
        self.output = T.reshape(input, (input_shape[0], shape[2], shape[0], shape[1]))
        self.output_shape = (input_shape[0], shape[2], shape[0], shape[1])
        self.inference = self.output

if __name__ == '__main__':
    pass
