#Pool "mean" code has issues _out_height and _out_width are not referenced in the code
import unittest
import numpy
import theano
from yann.core.pool import pooler_2d as p

try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch

class TestPool(unittest.TestCase):
    def setUp(self):
        self.verbose = 3
        self.input_shape = (1,1,10,10)
        self.input_ndarray = numpy.random.rand(1,1,10,10)
        self.input_tensor = theano.shared(self.input_ndarray)
        self.pool_size = (2,2)
        self.max_size = (1,1,10/2,10/2)

    @patch('yann.core.pool.max_pool_2d_same_size')
    def test1_pool_layer_2d_max_pool(self, mock_max_pool):
        mock_max_pool.return_value = self.input_ndarray
        self.pool_layer_2d = p(
                            input = self.input_tensor,
                            ds = self.pool_size,
                            img_shp = self.input_shape,
                            mode='max_same_size',
                            ignore_border=True,
                            verbose = self.verbose
        )
        self.assertEqual(self.pool_layer_2d.out_shp,self.input_shape)
        self.assertTrue(numpy.allclose(self.pool_layer_2d.out,self.input_ndarray))

    @patch('yann.core.pool.pool_2d')
    def test2_pool_layer_2d_max(self, mock_max_pool):
        mock_max_pool.return_value = self.input_ndarray
        self.pool_layer_2d = p(
                            input = self.input_tensor,
                            ds = self.pool_size,
                            img_shp = self.input_shape,
                            mode='max',
                            ignore_border=True,
                            verbose = self.verbose
        )
        self.assertEqual(self.pool_layer_2d.out_shp,self.max_size)
        self.assertTrue(numpy.allclose(self.pool_layer_2d.out,self.input_ndarray))

    @patch('yann.core.pool.pool_2d')
    def test3_pool_layer_2d_sum(self, mock_max_pool):
        mock_max_pool.return_value = self.input_ndarray
        self.pool_layer_2d = p(
                            input = self.input_tensor,
                            ds = self.pool_size,
                            img_shp = self.input_shape,
                            mode='sum',
                            ignore_border=True,
                            verbose = self.verbose
        )
        self.assertEqual(self.pool_layer_2d.out_shp,self.max_size)
        self.assertTrue(numpy.allclose(self.pool_layer_2d.out,self.input_ndarray))

    # @patch('yann.core.pool.pool_2d')
    # def test4_pool_layer_2d_mean(self, mock_max_mean):
    #     mock_max_mean.return_value = self.input_ndarray
    #     self.pool_layer_2d = p(
    #                         input = self.input_tensor,
    #                         ds = self.pool_size,
    #                         img_shp = self.input_shape,
    #                         mode='mean',
    #                         ignore_border=True,
    #                         verbose = self.verbose
    #     )
    #     self.assertEqual(self.pool_layer_2d.out_shp,self.input_shape)
    #     self.assertTrue(numpy.allclose(self.pool_layer_2d.out,self.input_ndarray))
