"""
TODO:
    * Need to support max_rand_pool and rand_pool
"""

from theano.tensor.signal.pool import Pool as DownsampleFactorMax
from theano.tensor.signal.pool import pool_2d
from theano.tensor.signal.pool import max_pool_2d_same_size
from theano.tensor.nnet.neighbours import images2neibs
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from math import floor
import numpy

# Implemented this before the theano update that brought the mean to pool, this has no use now.
def _meanpool ( input, ds, ignore_border = False ):
    """ provide mean pooling """
    out_shp = (input.shape[0], input.shape[1], input.shape[2]/ds[0], input.shape[3]/ds[1])
    neib = images2neibs(input, neib_shape = ds ,
                                mode = 'valid' if ignore_border is False else 'ignore_borders')
    pooled_vectors = neib.mean( axis = - 1 )
    return T.reshape(pooled_vectors, out_shp, ndim = 4 )

# This needs some work to be done on it.
def _maxrandpool ( input, ds, p, ignore_border = False ):
    """ provide random pooling among the top 'p' sorted outputs p = 0 is maxpool """
    rng = numpy.random.RandomState(24546)
    out_shp = (input.shape[0], input.shape[1], input.shape[2]/ds[0], input.shape[3]/ds[1])
    srng = RandomStreams(rng.randint(2147462579))
    pos = srng.random_integers(size=(1,1), low = ds[0]*ds[1]-1-p, high = ds[0]*ds[1]-1)
    neib = images2neibs(input, neib_shape = ds ,
                                mode = 'valid' if ignore_border is False else 'ignore_borders')
    neib = neib.sort(axis = -1)
    pooled_vectors = neib[:,pos]
    return T.reshape(pooled_vectors, out_shp, ndim = 4 )

class pooler_2d(object):
    """ #pragma: no cover
    function that performs pooling

    Args:
        input:  This variable should either ``thenao.tensor4`` (``theano.matrix``
                reshaped also works) variable or an output from a pervious layer which is
                a ``theano.tensor4`` convolved with a ``theano.shared``.
                The input should be of shape ``(batchsize, channels, height, width)``.
                For those who have tried pylearn2 or such, this is called bc01 format.

        img_shp: This variable should a tuple or an array: ``[batchsize, channels, height, width]``

        ds: tuple of pool sizes for rows and columns. ``(pool height, pool width)``

        mode: {'max', 'sum', 'mean', 'max_same_size'}
                Operation executed on each window. `max` and `sum`

                - if ``max_same_size``  we do maxpooling with output is the same size.
                - if ``max``  we do do maxpooling with output being downsampled.
                  Output size will be ``(batchsize, channels, height/ds[0], width/ds[1])``.
                - if ``mean`` we do do meanpooling with output being downsampled.
                  Output size will be ``(batchsize, channels, height/ds[0], width/ds[1])``.
                - if ``sum`` we do do sum pooling with output being downsampled.
                  Output size will be ``(batchsize, channels, height/ds[0], width/ds[1])``.

        ignore_border: (default is ``False``) Consider `theano's documentation
                        <http://deeplearning.net/software/theano/library/tensor/signal/pool.html>`_.
                        It is directly supplied to theano's pool module.
    """
    def __init__ (      self,
                        input,
                        ds,
                        img_shp,
                        mode = 'max',
                        ignore_border = True,
                        verbose = 1
                ):

        if verbose >=3:
            print("... Creating pooling operator")

        if mode == 'max_same_size':      # output same size but maxpool with zeros for non outputs
            self.out = max_pool_2d_same_size(
                            input=input,
                            patch_size=ds,
                            )
            _out_height = img_shp[2]
            _out_width  = img_shp[3]

        elif mode == 'max' or mode == 'sum':    # normal maxpool
            self.out = pool_2d(
                input=input,
                ws=ds,
                ignore_border = ignore_border,
                mode = mode
                )
            _out_height = int(floor(img_shp[2] / float(ds[0])))
            _out_width  = int(floor(img_shp[3] / float(ds[1])))

        elif mode == 'mean':
            self.out = pool_2d(
                input=input,
                ws=ds,
                ignore_border = ignore_border,
                mode = 'average_exc_pad'
                )
        # ignore_border has some issue. False seems to pull things off GPU.
        self.out_shp = (img_shp[0], img_shp[1], _out_height, _out_width)

if __name__ == '__main__':#pragma: no cover
    pass
