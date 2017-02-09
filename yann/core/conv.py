"""
``yann.core.conv.py`` is one file that contains all the convolution operators.
It contains two functions for performing either 2d convolution (``conv2d``) or 3d convolution
(``conv3d``).

These functions shall be called by every convolution layer from ``yann.layers.py``

TODO:
    * Add 3D convolution support from theano.
    * Add Masked convolution support.
"""
from theano.tensor.nnet import conv2d

class convolver_2d(object):
    """
    function that performs convolution

    This function basically performs convolution. Deliberately written as a funciton and
    not as a class. The reason for writing this as a method and not a class is because I
    don't want to probe on to this method too much. These ouputs can be probed using the
    convolution layer if needed. This keeps things simple.

    Args:
        input:  This variable should either ``thenao.tensor4`` (``theano.matrix``
                reshaped also works) variable or an output from a pervious layer which is
                a ``theano.tensor4`` convolved with a ``theano.shared``. The input should
                be of shape ``(batchsize, channels, height, width)``. For those who have
                tried ``pylearn2`` or such, this is called bc01 format.

        fitlers: This variable should be ``theano.shared`` variables of filter weights
                 could even be a filter bank. ``filters`` should be of shape  ``(nchannels,
                 nkerns, filter_height, filter_width)``. ``nchannles`` is the number of input \
                 channels and ``nkerns`` is the number of kernels or output channels.

        subsample: Stride Tuple of ``(int, int)``.

        filter_shape: This variable should be a tuple or an array:
                        ``[nkerns, nchannles, filter_height, filter_width]``

        image_shape: This variable should a tuple or an array:
                    ``[batchsize, channels, height, width]``
                    ``image_shape[1]`` must be equal to ``filter_shape[1]``

        border_mode: The input to this can be either ``'same'`` or other theano defaults


    Notes:
        * ``conv2d.out`` output, Output that could be provided as
          output to the next layer or to other convolutional layer options.
          The size of the outut depends on border mode and subsample
          operation performed.

        * ``conv2d.out_shp``: (``int``, ``int``), A tuple (height, width) of all feature maps

        The options for ``border_mode`` input which at the moment of writing this doc are

        * ``'valid'`` - apply filter wherever it completely overlaps with the
          input. Generates output of shape ``input shape - filter shape + 1``

        * ``'full'``- apply filter wherever it partly overlaps with the input.
          Generates output of shape ``input shape + filter shape - 1``

        * ``'half'``: pad input with a symmetric border of ``filter rows // 2``
          rows and ``filter columns // 2`` columns, then perform a valid
          convolution. For filters with an odd number of rows and columns, this
          leads to the output shape being equal to the input shape.
        * ``<int>``: pad input with a symmetric border of zeros of the given
          width, then perform a valid convolution.
        * ``(<int1>, <int2>)``: pad input with a symmetric border of ``int1``
          rows and ``int2`` columns, then perform a valid convolution.

        Refer to `theano documentation's convolution page
        <http://deeplearning.net/software/theano/library/tensor/nnet/conv.html>`_
        for more details on this.
        Basically cuDNN is used for ``same`` because at the moment of writing
        this funciton, ``theano.conv2d`` doesn't support``same`` convolutions
        on the GPU. For everything else, ``theano`` default will be used.

    TODO:
        Implement ``border_mode = 'same'`` for libgpuarray backend. As of now only supports
        CUDA backend.

    """

    def __init__ (  self,
                    input,
                    filters,
                    subsample,
                    filter_shape,
                    image_shape,
                    border_mode = 'valid',
                    verbose = 1
            ):

        if not image_shape[1] == filter_shape[1]:
            raise Exception ("input_shape[1] and filter_shape[1] must match")

        if verbose >=3 :
            print "... creating convolution operator"

        if border_mode == 'same':               # this is used typically by VGG.
            _out_height = image_shape[2]
            _out_width = image_shape[3]
            x = (filter_shape[2] - 1) // 2
            y = (filter_shape[3] - 1) // 2
            border_mode = (x,y)

            from theano.sandbox import cuda
            if not cuda.dnn.dnn_available():
                raise Exception ("cuDNN is needed for this type of convolution.")

            else:
                if subsample[0] == 1 and subsample[1] == 1:
                    self.out = cuda.dnn.dnn_conv (
                            img = input ,
                            kerns = filters,
                            border_mode= border_mode,
                            conv_mode = 'cross'
                            )
                else:
                    self.out = cuda.dnn.dnn_conv (
                            img = input ,
                            kerns = filters,
                            border_mode= border_mode,
                            subsample = subsample,
                            conv_mode = 'cross'
                            )
        else:
            if border_mode == 'full':
                _out_height = image_shape[2] + filter_shape[2] - 1
                _out_width = image_shape[3]  + filter_shape[2] - 1

            else:
                _out_height = image_shape[2] - filter_shape[2] + 1
                _out_width  = image_shape[3] - filter_shape[3] + 1

                _out_height = int(_out_height/subsample[0])
                _out_width = int(_out_width/subsample[1])

            if subsample[0] == 1 and subsample[1] == 1:
                self.out = conv2d (
                            input = input,
                            filters = filters,
                            input_shape = image_shape,
                            filter_shape = filter_shape,
                            border_mode = border_mode,
                            )
            else:
                self.out = conv2d (
                            input = input,
                            filters = filters,
                            input_shape = image_shape,
                            filter_shape = filter_shape,
                            subsample = subsample,
                            border_mode = border_mode,
                            )
        self.out_shp = (_out_height, _out_width)

if __name__ == '__main__':
    pass
