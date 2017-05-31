#Internal methods are not tested, as calls are linear
import unittest
import numpy
import theano
from yann.layers.transform import rotate_layer as rl
from yann.layers.transform import dropout_rotate_layer as drl
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch

class TestRandom(unittest.TestCase):
    def setUp(self):
        self.verbose = 3
        self.channels = 1
        self.mean_subtract = False
        self.rng = None
        self.borrow = True
        self.input_shape = (1,1,3,3)
        self.input_ndarray = numpy.random.rand(1,1,3,3)
        self.output_dropout_ndarray= numpy.zeros((1,1,3,3))
        self.output_test = numpy.zeros((1,1,3,3))
        self.output_train = numpy.ones((1,1,3,3))
        self.input_tensor = theano.shared(self.input_ndarray)
        self.gamma = theano.shared(value=numpy.ones((self.channels,),dtype=theano.config.floatX), name = 'gamma', borrow = self.borrow)
        self.beta = theano.shared(value=numpy.zeros((self.channels,),dtype=theano.config.floatX), name = 'beta', borrow=self.borrow)
        self.running_mean = theano.shared(value=numpy.zeros((self.channels,), dtype=theano.config.floatX), name = 'population_mean', borrow = self.borrow)
        self.running_var = theano.shared(value=numpy.ones((self.channels,),dtype=theano.config.floatX), name = 'population_var', borrow=self.borrow)
        self.input_params = (self.gamma, self.beta, self.running_mean, self.running_var)
        self.rotate_layer_name = "rl"
        self.dropout_rotate_layer_name = "drl"
        self.dropout_rate = 1
        self.default_param_value = [1.]
        self.custom_param_value = [1., 1.,1.]
        self.classes = 3
        self.output_shape = (self.input_shape[0], self.classes)
        self.sample = numpy.ones((1,1,2,2))
        self.input_params_all = (self.sample,self.sample)


    def test1_rotate(self):
        self.rotate_layer = rl(
                  input = self.input_tensor,
                  input_shape = self.input_shape,
                  id = self.rotate_layer_name,
                  angle = None,
                  borrow = True,
                  verbose = self.verbose
        )
        self.assertEqual(self.rotate_layer.id,self.rotate_layer_name)
        self.assertEqual(self.rotate_layer.output_shape,self.input_shape)

    @patch('yann.layers.transform._dropout')
    def test2_dropout_rotate_layert(self,mock_dropout):
        mock_dropout.return_value = self.input_ndarray
        self.dropout_rotate_layer = drl(
                  input = self.input_tensor,
                  input_shape = self.input_shape,
                  id = self.dropout_rotate_layer_name,
                  dropout_rate= self.dropout_rate,
                  verbose = self.verbose)
        self.assertTrue(numpy.allclose(self.dropout_rotate_layer.output, self.input_ndarray))
        self.assertEqual(self.dropout_rotate_layer.output_shape,self.input_ndarray.shape)
        self.assertEqual(self.dropout_rotate_layer.id,self.dropout_rotate_layer_name)


