from abstract import layer

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
        mini_batch_size: Number of images in the data variable.
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
                    mini_batch_size,
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
        data_feeder = x.reshape((mini_batch_size, height, width, channels)).dimshuffle(0,3,1,2)
                                # the dim shuffle makes it mini_batch_size, channels height width order
        mean_subtracted_data_feeder = data_feeder - data_feeder.mean()

        self.output = mean_subtracted_data_feeder if mean_subtract is True else data_feeder
        self.output_shape = (mini_batch_size, channels, height, width)

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
        mini_batch_size: Number of images in the data variable.
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
                    mini_batch_size,
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
                                            mini_batch_size = mini_batch_size,
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

if __name__ == '__main__':
    pass
