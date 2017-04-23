import unittest
import numpy
import theano
from yann.layers.random import random_layer as rl
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

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
        self.random_layer_name = "rl"
        self.dropout_rate = 1
        self.default_param_value = [1.]
        self.custom_param_value = [1., 1.,1.]
        self.classes = 3
        self.output_shape = (self.input_shape[0], self.classes)
        self.sample = numpy.ones((1,1,2,2))
        self.input_params_all = (self.sample,self.sample)
        rng = numpy.random
        self.rs = RandomStreams(rng.randint(1,2147462468))

    @patch('yann.layers.random.RandomStreams')
    @patch('yann.layers.random.RandomStreams.binomial')
    def test1_random_binomial(self,mock_binomial,mock_random_streams):
        mock_random_streams.return_value = self.rs
        mock_binomial.return_value = self.input_ndarray
        self.random_layer = rl(
                    num_neurons=10,
                    id = self.random_layer_name,
                    distribution = 'binomial',
                    options = {},
                    verbose = self.verbose
        )
        self.assertEqual(self.random_layer.id,self.random_layer_name)
        self.assertTrue(numpy.allclose(self.random_layer.output,self.input_ndarray))


    @patch('yann.layers.random.RandomStreams.binomial')
    def test2_random_binomial_p(self,mock_binomial):
        mock_binomial.return_value = self.input_ndarray
        self.random_layer = rl(
                    num_neurons=10,
                    id = self.random_layer_name,
                    distribution = 'binomial',
                    options = {'p':0.5},
                    verbose = self.verbose
        )
        self.assertEqual(self.random_layer.id,self.random_layer_name)
        self.assertTrue(numpy.allclose(self.random_layer.output,self.input_ndarray))

    @patch('yann.layers.random.RandomStreams.uniform')
    def test3_random_uniform(self,mock_uniform):
        mock_uniform.return_value = self.input_ndarray
        self.random_layer = rl(
                    num_neurons=10,
                    id = self.random_layer_name,
                    distribution = 'uniform',
                    options = {},
                    verbose = self.verbose
        )
        self.assertEqual(self.random_layer.id,self.random_layer_name)
        self.assertTrue(numpy.allclose(self.random_layer.output,self.input_ndarray))


    @patch('yann.layers.random.RandomStreams.uniform')
    def test4_random_uniform_limits(self,mock_uniform):
        mock_uniform.return_value = self.input_ndarray
        self.random_layer = rl(
                    num_neurons=10,
                    id = self.random_layer_name,
                    distribution = 'uniform',
                    options = {'limits':(0,1)},
                    verbose = self.verbose
        )
        self.assertEqual(self.random_layer.id,self.random_layer_name)
        self.assertTrue(numpy.allclose(self.random_layer.output,self.input_ndarray))

    @patch('yann.layers.random.RandomStreams.normal')
    def test5_random_gaussian(self,mock_normal):
        mock_normal.return_value = self.input_ndarray
        self.random_layer = rl(
                    num_neurons=10,
                    id = self.random_layer_name,
                    distribution = 'gaussian',
                    options = {},
                    verbose = self.verbose
        )
        self.assertEqual(self.random_layer.id,self.random_layer_name)
        self.assertTrue(numpy.allclose(self.random_layer.output,self.input_ndarray))

    @patch('yann.layers.random.RandomStreams.normal')
    def test6_random_normal(self,mock_normal):
        mock_normal.return_value = self.input_ndarray
        self.random_layer = rl(
                    num_neurons=10,
                    id = self.random_layer_name,
                    distribution = 'normal',
                    options = {},
                    verbose = self.verbose
        )
        self.assertEqual(self.random_layer.id,self.random_layer_name)
        self.assertTrue(numpy.allclose(self.random_layer.output,self.input_ndarray))


    @patch('yann.layers.random.RandomStreams.normal')
    def test7_random_normal_mu_sigma(self,mock_normal):
        mock_normal.return_value = self.input_ndarray
        self.random_layer = rl(
                    num_neurons=10,
                    id = self.random_layer_name,
                    distribution = 'normal',
                    options = {'mu':0,'sigma':1},
                    verbose = self.verbose
        )
        self.assertEqual(self.random_layer.id,self.random_layer_name)
        self.assertTrue(numpy.allclose(self.random_layer.output,self.input_ndarray))

    @patch('yann.layers.random.RandomStreams.normal')
    def test7_random_gaussian_mu_sigma(self,mock_normal):
        mock_normal.return_value = self.input_ndarray
        self.random_layer = rl(
                    num_neurons=10,
                    id = self.random_layer_name,
                    distribution = 'gaussian',
                    options = {'mu':0,'sigma':1},
                    verbose = self.verbose
        )
        self.assertEqual(self.random_layer.id,self.random_layer_name)
        self.assertTrue(numpy.allclose(self.random_layer.output,self.input_ndarray))
