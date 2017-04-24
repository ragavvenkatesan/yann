import unittest
import numpy
import theano
from yann.layers.fully_connected import dot_product_layer as dpl
from yann.layers.fully_connected import dropout_dot_product_layer as ddpl
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch


class TestFullyConnected(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(0)
        self.verbose = 3
        self.channels = 1
        self.fully_connected_layer_name = "fc"
        self.fully_connected_layer_name_dropout_layer_name = "dfc"
        self.dropout_rate = 1
        self.rng = None
        self.mean_subtract = False
        self.input_shape = (1,1,10,10)
        self.input_ndarray = numpy.random.rand(1,1,10,10)
        self.output_dropout_ndarray= numpy.zeros((1,1,10,10))
        self.output_test = numpy.zeros((1,1,10,10))
        self.output_train = numpy.ones((1,1,10,10))
        self.input_tensor = theano.shared(self.input_ndarray)
        self.activation = "sigmoid"
        self.neurons = 10
        self.borrow = True
        self.sample = numpy.ones((1,1,2,2))
        self.gamma = theano.shared(value=numpy.ones((self.channels,),dtype=theano.config.floatX), name = 'gamma', borrow = self.borrow)
        self.beta = theano.shared(value=numpy.zeros((self.channels,),dtype=theano.config.floatX), name = 'beta', borrow=self.borrow)
        self.running_mean = theano.shared(value=numpy.zeros((self.channels,), dtype=theano.config.floatX), name = 'population_mean', borrow = self.borrow)
        self.running_var = theano.shared(value=numpy.ones((self.channels,),dtype=theano.config.floatX), name = 'population_var', borrow=self.borrow)
        self.input_params = (None,None,None,None,None,self.running_var)
        self.input_params_all = (self.sample,self.sample,self.gamma, self.beta, self.running_mean, self.running_var)

    @patch('yann.layers.fully_connected._activate')
    def test1_fully_connected_layer_all_false(self,mock_activate):
        mock_activate.return_value = (self.input_ndarray,self.input_shape)
        self.fclayer = dpl(
                input = self.input_tensor,
                num_neurons= self.neurons,
                input_shape= (1,1,10,10),
                id= self.fully_connected_layer_name,
                rng= self.rng,
                input_params= None,
                borrow= self.borrow,
                activation=self.activation,
                batch_norm= False,
                verbose = self.verbose)
        self.assertTrue(numpy.allclose(self.fclayer.output, self.input_ndarray))
        self.assertTrue(numpy.allclose(self.fclayer.inference, self.input_ndarray))
        self.assertEqual(self.fclayer.output_shape,self.input_ndarray.shape)
        self.assertEqual(self.fclayer.id,self.fully_connected_layer_name)
        self.assertEqual(self.fclayer.activation,self.activation)
        self.assertEqual(self.fclayer.num_neurons,self.neurons)

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.fully_connected._activate')
    @patch('yann.layers.fully_connected.batch_normalization_test')
    @patch('yann.layers.fully_connected.batch_normalization_train')
    def test2_fully_connected_layer_bn_true(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_activate,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_activate.return_value = (self.input_ndarray,self.input_shape)
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.fclayer = dpl(
                input = self.input_tensor,
                num_neurons= self.neurons,
                input_shape= (1,1,10,10),
                id= self.fully_connected_layer_name,
                rng= self.rng,
                input_params= None,
                borrow= self.borrow,
                activation=self.activation,
                batch_norm= True,
                verbose = self.verbose)
        self.assertTrue(numpy.allclose(self.fclayer.output, self.input_ndarray))
        self.assertTrue(numpy.allclose(self.fclayer.inference, self.input_ndarray))
        self.assertEqual(self.fclayer.output_shape,self.input_ndarray.shape)
        self.assertEqual(self.fclayer.id,self.fully_connected_layer_name)
        self.assertEqual(self.fclayer.activation,self.activation)
        self.assertEqual(self.fclayer.num_neurons,self.neurons)

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.fully_connected._activate')
    @patch('yann.layers.fully_connected.batch_normalization_test')
    @patch('yann.layers.fully_connected.batch_normalization_train')
    def test3_fully_connected_layer_bn_true(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_activate,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_activate.return_value = (self.input_ndarray,self.input_shape)
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.fclayer = dpl(
                input = self.input_tensor,
                num_neurons= self.neurons,
                input_shape= (1,1,10,10),
                id= self.fully_connected_layer_name,
                rng= self.rng,
                input_params= self.input_params,
                borrow= self.borrow,
                activation=self.activation,
                batch_norm= True,
                verbose = self.verbose)
        self.assertTrue(numpy.allclose(self.fclayer.output, self.input_ndarray))
        self.assertTrue(numpy.allclose(self.fclayer.inference, self.input_ndarray))
        self.assertEqual(self.fclayer.output_shape,self.input_ndarray.shape)
        self.assertEqual(self.fclayer.id,self.fully_connected_layer_name)
        self.assertEqual(self.fclayer.activation,self.activation)
        self.assertEqual(self.fclayer.num_neurons,self.neurons)

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.fully_connected._activate')
    @patch('yann.layers.fully_connected.batch_normalization_test')
    @patch('yann.layers.fully_connected.batch_normalization_train')
    def test4_fully_connected_layer_bn_true_param_all(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_activate,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_activate.return_value = (self.input_ndarray,self.input_shape)
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.fclayer = dpl(
                input = self.input_tensor,
                num_neurons= self.neurons,
                input_shape= (1,1,10,10),
                id= self.fully_connected_layer_name,
                rng= self.rng,
                input_params= self.input_params_all,
                borrow= self.borrow,
                activation=self.activation,
                batch_norm= True,
                verbose = self.verbose)
        self.assertTrue(numpy.allclose(self.fclayer.output, self.input_ndarray))
        self.assertTrue(numpy.allclose(self.fclayer.inference, self.input_ndarray))
        self.assertEqual(self.fclayer.output_shape,self.input_ndarray.shape)
        self.assertEqual(self.fclayer.id,self.fully_connected_layer_name)
        self.assertEqual(self.fclayer.activation,self.activation)
        self.assertEqual(self.fclayer.num_neurons,self.neurons)


    @patch('yann.layers.fully_connected._dropout')
    def test5_fully_connected_layer_drop_out(self,mock_dropout):
        mock_dropout.return_value = self.input_ndarray
        self.fcdlayer = ddpl(
                input = self.input_tensor,
                num_neurons= self.neurons,
                input_shape= (1,1,10,10),
                id= self.fully_connected_layer_name_dropout_layer_name,
                rng= self.rng,
                input_params=None,
                borrow= self.borrow,
                activation=self.activation,
                dropout_rate= self.dropout_rate,
                batch_norm= False,
                verbose = self.verbose)
        self.assertTrue(numpy.allclose(self.fcdlayer.output, self.input_ndarray))
        self.assertEqual(self.fcdlayer.id,self.fully_connected_layer_name_dropout_layer_name)
        self.assertEqual(self.fcdlayer.dropout_rate,self.dropout_rate)



