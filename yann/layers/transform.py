from abstract import layer, _activate, _dropout
import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class rotate_layer (layer):
    """
    This is a rotate layer. This takes a layer and an angle (rotation normalized in [0,1]) as input
    and rotates the batch of images by the specified rotation parameter.

    Args:
        input: An input ``theano.tensor`` variable. Even ``theano.shared`` will work as long as they
               are in the following shape ``mini_batch_size, height, width, channels``
        verbose: similar to the rest of the toolbox.
        input_shape: ``(mini_batch_size, input_size)``
        angle: value from [0,1]
        borrow: ``theano`` borrow, typicall ``True``.   
        input_params: Supply params or initializations from a pre-trained system.
    """

    def __init__ (self,
                  input,
                  input_shape,
                  id,
                  angle = None,
                  borrow = True,
                  verbose = 2 ):
        super(rotate_layer,self).__init__(id = id, type = 'rotate', verbose = verbose)
        srng = RandomStreams(rng.randint(1,2147462468), use_cuda=None)
        if verbose >= 3:
            print "... Creating rotate layer"

        if len(input) == 4:
            if verbose >= 3:
                print "... Creating the rotate layer"

            if angle is None:
                angle = srng.uniform(size = input_shape[0], low =limits[0], high =limits[1], 
                                                dtype = theano.config.floatX )

            theta = numpy.zeros((input_shape[0],2,3),dtype='float32')
            theta[:,0,0] = numpy.cos(angle[:,0]*180)
            theta[:,0,1] = -numpy.sin(angle[:,0]*180)
            theta[:,1,0] = numpy.sin(angle[:,0]*180)
            theta[:,1,1] = numpy.cos(angle[:,0]*180)
            theta = theta.reshape((input_shape[0], 6))

            self.output = self._transform_affine(theta, input)
            self.output_shape = self.output.shape
            self.angle = angle

    def _transform_affine(self, theta, input):
        num_batch, num_channels, height, width = input.shape
        theta = T.reshape(theta, (-1, 2, 3))

        # grid of (x_t, y_t, 1)
        out_height = T.cast(height, 'int64')
        out_width = T.cast(width, 'int64')
        grid = _meshgrid(out_height, out_width)

        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = T.dot(theta, grid)
        x_s = T_g[:, 0]
        y_s = T_g[:, 1]
        x_s_flat = x_s.flatten()
        y_s_flat = y_s.flatten()

        # dimshuffle input to  (bs, height, width, channels)
        input_dim = input.dimshuffle(0, 2, 3, 1)
        input_transformed = _interpolate(
            input_dim, x_s_flat, y_s_flat,
            out_height, out_width)

        output = T.reshape(
            input_transformed, (num_batch, out_height, out_width, num_channels))
        output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
        return output

    def _interpolate(self, im, x, y, out_height, out_width):
        # *_f are floats
        num_batch, height, width, channels = im.shape
        height_f = T.cast(height, theano.config.floatX)
        width_f = T.cast(width, theano.config.floatX)

        # clip coordinates to [-1, 1]
        x = T.clip(x, -1, 1)
        y = T.clip(y, -1, 1)

        # scale coordinates from [-1, 1] to [0, width/height - 1]
        x = (x + 1) / 2 * (width_f - 1)
        y = (y + 1) / 2 * (height_f - 1)

        # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
        # we need those in floatX for interpolation and in int64 for indexing. for
        # indexing, we need to take care they do not extend past the image.
        x0_f = T.floor(x)
        y0_f = T.floor(y)
        x1_f = x0_f + 1
        y1_f = y0_f + 1
        x0 = T.cast(x0_f, 'int64')
        y0 = T.cast(y0_f, 'int64')
        x1 = T.cast(T.minimum(x1_f, width_f - 1), 'int64')
        y1 = T.cast(T.minimum(y1_f, height_f - 1), 'int64')

        # The input is [num_batch, height, width, channels]. We do the lookup in
        # the flattened input, i.e [num_batch*height*width, channels]. We need
        # to offset all indices to match the flat version
        dim2 = width
        dim1 = width*height
        base = T.repeat(
            T.arange(num_batch, dtype='int64')*dim1, out_height*out_width)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels for all samples
        im_flat = im.reshape((-1, channels))
        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        # calculate interpolated values
        wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')
        wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')
        wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')
        wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')
        output = T.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)
        return output

    def _linspace(self, start, stop, num):
        # Theano linspace. Behaves similar to np.linspace
        start = T.cast(start, theano.config.floatX)
        stop = T.cast(stop, theano.config.floatX)
        num = T.cast(num, theano.config.floatX)
        step = (stop-start)/(num-1)
        return T.arange(num, dtype=theano.config.floatX)*step+start

    def _meshgrid(self, height, width):
        # This function is the grid generator.
        # It is equivalent to the following numpy code:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        # It is implemented in Theano instead to support symbolic grid sizes.
        # Note: If the image size is known at layer construction time, we could
        # compute the meshgrid offline in numpy instead of doing it dynamically
        # in Theano. However, it hardly affected performance when we tried.
        x_t = T.dot(T.ones((height, 1)),
                    _linspace(-1.0, 1.0, width).dimshuffle('x', 0))
        y_t = T.dot(_linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                    T.ones((1, width)))

        x_t_flat = x_t.reshape((1, -1))
        y_t_flat = y_t.reshape((1, -1))
        ones = T.ones_like(x_t_flat)
        grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
        return grid


class dropout_rotate_layer (rotate_layer):
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
                  input_shape,
                  id,
                  rng = None,
                  dropout_rate = 0.5,
                  angle = None,
                  borrow = True,
                  verbose = 2):

        if verbose >= 3:
            print "... set up the dropout rotate layer"
        if rng is None:
            rng = numpy.random            
        super(dropout_rotate_layer, self).__init__ (
                                        input = input,
                                        input_shape = input_shape,
                                        id = id,
                                        borrow = borrow,
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