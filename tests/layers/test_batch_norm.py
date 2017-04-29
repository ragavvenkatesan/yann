import unittest
import numpy
import theano
from yann.layers.batch_norm import batch_norm_layer_2d as bn
from yann.layers.batch_norm import dropout_batch_norm_layer_2d as dbn
from yann.layers.batch_norm import batch_norm_layer_1d as bn1
from yann.layers.batch_norm import dropout_batch_norm_layer_1d as dbn1
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch



class TestBtachNorm2d(unittest.TestCase):
    def setUp(self):
        self.verbose = 3
        self.channels = 1
        self.mean_subtract = False
        self.rng = None
        self.borrow = True
        self.input_shape = (1,1,10,10)
        self.input_ndarray = numpy.random.rand(1,1,10,10)
        self.output_dropout_ndarray= numpy.zeros((1,1,10,10))
        self.output_test = numpy.zeros((1,1,10,10))
        self.output_train = numpy.ones((1,1,10,10))
        self.input_tensor = theano.shared(self.input_ndarray)
        self.gamma = theano.shared(value=numpy.ones((self.channels,),dtype=theano.config.floatX), name = 'gamma', borrow = self.borrow)
        self.beta = theano.shared(value=numpy.zeros((self.channels,),dtype=theano.config.floatX), name = 'beta', borrow=self.borrow)
        self.running_mean = theano.shared(value=numpy.zeros((self.channels,), dtype=theano.config.floatX), name = 'population_mean', borrow = self.borrow)
        self.running_var = theano.shared(value=numpy.ones((self.channels,),dtype=theano.config.floatX), name = 'population_var', borrow=self.borrow)
        self.input_params = (self.gamma, self.beta, self.running_mean, self.running_var)
        self.batch_norm_layer_name = "bn"
        self.batch_norm_layer_name_val = "bnv"
        self.dropout_batch_norm_layer_2 ="dbn"
        self.batch_norm_layer_name_1 = "bn1"
        self.batch_norm_layer_name_val_1 = "bnv1"
        self.dropout_batch_norm_layer_1 ="dbn1"
        self.dropout_rate = 1
        self.default_param_value = [1.]
        self.custom_param_value = [1., 1.,1.]

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.batch_norm.batch_normalization_test')
    @patch('yann.layers.batch_norm.batch_normalization_train')
    def test1_batch_norm_layer_2d(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.batch_norm_layer_2d = bn(
                            input = self.input_tensor,
                            id = self.batch_norm_layer_name,
                            input_shape = self.input_shape,
                            rng = self.rng,
                            borrow = self.borrow,
                            input_params = None,
                            verbose = self.verbose
        )
        self.assertEqual(self.batch_norm_layer_2d.id,self.batch_norm_layer_name)
        self.assertEqual(self.batch_norm_layer_2d.input_shape,self.input_shape)
        self.assertEqual(self.batch_norm_layer_2d.output_shape,self.input_shape)
        self.assertTrue(numpy.allclose(self.batch_norm_layer_2d.output,self.output_train))
        self.assertTrue(numpy.allclose(self.batch_norm_layer_2d.inference,self.output_test))

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.batch_norm.batch_normalization_test')
    @patch('yann.layers.batch_norm.batch_normalization_train')
    def test2_batch_norm_layer_2d_with_values(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.batch_norm_layer_2d_val = bn(
                            input = self.input_tensor,
                            id = self.batch_norm_layer_name_val,
                            input_shape = self.input_shape,
                            rng = self.rng,
                            borrow = self.borrow,
                            input_params = self.input_params,
                            verbose = self.verbose
        )

        self.assertEqual(self.batch_norm_layer_2d_val.id,self.batch_norm_layer_name_val)
        self.assertEqual(self.batch_norm_layer_2d_val.input_shape,self.input_shape)
        self.assertEqual(self.batch_norm_layer_2d_val.output_shape,self.input_shape)
        self.assertTrue(numpy.allclose(self.batch_norm_layer_2d_val.output,self.output_train))
        self.assertTrue(numpy.allclose(self.batch_norm_layer_2d_val.inference,self.output_test))

    @patch('yann.layers.batch_norm._dropout')
    def test3_dropout_batch_norm_layer_2d(self,mock_dropout):
        mock_dropout.return_value = self.input_ndarray
        self.dropout_batch_norm_layer_2d = dbn(
                            input = self.input_tensor,
                            id = self.dropout_batch_norm_layer_2,
                            input_shape = self.input_shape,
                            rng = self.rng,
                            input_params = None,
                            dropout_rate= self.dropout_rate,
                            verbose = self.verbose
        )
        self.assertTrue(numpy.allclose(self.dropout_batch_norm_layer_2d.output, self.input_ndarray))
        self.assertEqual(self.dropout_batch_norm_layer_2d.output_shape,self.input_ndarray.shape)
        self.assertEqual(self.dropout_batch_norm_layer_2d.id,self.dropout_batch_norm_layer_2)

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.batch_norm.batch_normalization_test')
    @patch('yann.layers.batch_norm.batch_normalization_train')
    def test4_batch_norm_layer_1d(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.batch_norm_layer_1d = bn1(
                            input = self.input_tensor,
                            id = self.batch_norm_layer_name_1,
                            input_shape = self.input_shape,
                            rng = self.rng,
                            borrow = self.borrow,
                            input_params = None,
                            verbose = self.verbose
        )
        self.assertEqual(self.batch_norm_layer_1d.id,self.batch_norm_layer_name_1)
        self.assertEqual(self.batch_norm_layer_1d.input_shape,self.input_shape)
        self.assertEqual(self.batch_norm_layer_1d.output_shape,self.input_shape)
        self.assertTrue(numpy.allclose(self.batch_norm_layer_1d.output,self.output_train))
        self.assertTrue(numpy.allclose(self.batch_norm_layer_1d.inference,self.output_test))


    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.batch_norm.batch_normalization_test')
    @patch('yann.layers.batch_norm.batch_normalization_train')
    def test5_batch_norm_layer_1d_with_values(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.batch_norm_layer_1d = bn1(
                            input = self.input_tensor,
                            id = self.batch_norm_layer_name_val_1,
                            input_shape = self.input_shape,
                            rng = self.rng,
                            borrow = self.borrow,
                            input_params = self.input_params,
                            verbose = self.verbose
        )
        self.assertEqual(self.batch_norm_layer_1d.id,self.batch_norm_layer_name_val_1)
        self.assertEqual(self.batch_norm_layer_1d.input_shape,self.input_shape)
        self.assertEqual(self.batch_norm_layer_1d.output_shape,self.input_shape)
        self.assertTrue(numpy.allclose(self.batch_norm_layer_1d.output,self.output_train))
        self.assertTrue(numpy.allclose(self.batch_norm_layer_1d.inference,self.output_test))


    @patch('yann.layers.batch_norm._dropout')
    def test6_dropout_batch_norm_layer_1d(self,mock_dropout):
        mock_dropout.return_value = self.input_ndarray
        self.dropout_batch_norm_layer_1d = dbn1(
                            input = self.input_tensor,
                            id = self.dropout_batch_norm_layer_1,
                            input_shape = self.input_shape,
                            rng = self.rng,
                            input_params = None,
                            dropout_rate= self.dropout_rate,
                            verbose = self.verbose,
                            borrow = self.borrow
        )
        self.assertTrue(numpy.allclose(self.dropout_batch_norm_layer_1d.output, self.input_ndarray))
        self.assertEqual(self.dropout_batch_norm_layer_1d.output_shape,self.input_ndarray.shape)
        self.assertEqual(self.dropout_batch_norm_layer_1d.id,self.dropout_batch_norm_layer_1)


