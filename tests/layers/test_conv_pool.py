import unittest
import numpy
import theano
from yann.layers.conv_pool import conv_pool_layer_2d as cl
from yann.layers.conv_pool import dropout_conv_pool_layer_2d as dcl
from yann.layers.conv_pool import deconv_layer_2d as dl
from yann.layers.conv_pool import dropout_deconv_layer_2d as ddl
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch



class TestConvPool(unittest.TestCase):
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
        self.w = theano.shared(value=
                               numpy.asarray(0.01 * numpy.random.standard_normal(size=(1,1,3,3)),
                                             dtype=theano.config.floatX), borrow=True,
                               name='filterbank')
        self.b = theano.shared(value=numpy.zeros((10,), dtype=theano.config.floatX),
                               name='bias', borrow=True)
        self.input_params = (self.w, self.b,self.gamma, self.beta, self.running_mean, self.running_var)
        self.input_params_none = (None, None,None, None, None, self.running_var)
        self.conv_pool_layer_2d_name = "cl"
        self.dropout_conv_pool_layer_2d_name = "dcl"
        self.deconv_pool_layer_2d_name = "dl"
        self.dropout_deconv_pool_layer_2d_name = "ddl"
        self.dropout_rate = 1
        self.default_param_value = [1.]
        self.custom_param_value = [1., 1.,1.]
        self.pool_size_mismatch_exception_msg = " Unpool operation not yet supported be deconv layer"
        self.activation_tuple_exception_msg = "Deconvolution layer does not support maxout activation"

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.conv_pool._activate')
    @patch('yann.layers.conv_pool.batch_normalization_test')
    @patch('yann.layers.conv_pool.batch_normalization_train')
    def test1_conv_pool_layer_2d(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_activate,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_activate.return_value = (self.input_ndarray, self.input_shape)
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.conv_pool_layer_2d = cl(
                            input = self.input_tensor,
                            id = self.conv_pool_layer_2d_name,
                            input_shape = self.input_shape,
                            nkerns=10,
                            verbose = self.verbose,
                            batch_norm = True
        )
        self.assertEqual(self.conv_pool_layer_2d.id,self.conv_pool_layer_2d_name)
        self.assertEqual(self.conv_pool_layer_2d.output_shape,self.input_shape)
        self.assertTrue(numpy.allclose(self.conv_pool_layer_2d.output,self.input_ndarray))
        self.assertTrue(numpy.allclose(self.conv_pool_layer_2d.inference,self.input_ndarray))

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.conv_pool._activate')
    @patch('yann.layers.conv_pool.batch_normalization_test')
    @patch('yann.layers.conv_pool.batch_normalization_train')
    def test2_conv_pool_layer_2d_ip_none(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_activate,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_activate.return_value = (self.input_ndarray, self.input_shape)
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.conv_pool_layer_2d = cl(
                            input = self.input_tensor,
                            id = self.conv_pool_layer_2d_name,
                            input_shape = self.input_shape,
                            nkerns=10,
                            verbose = self.verbose,
                            input_params= self.input_params_none,
                            batch_norm = True
        )
        self.assertEqual(self.conv_pool_layer_2d.id,self.conv_pool_layer_2d_name)
        self.assertEqual(self.conv_pool_layer_2d.output_shape,self.input_shape)
        self.assertTrue(numpy.allclose(self.conv_pool_layer_2d.output,self.input_ndarray))
        self.assertTrue(numpy.allclose(self.conv_pool_layer_2d.inference,self.input_ndarray))

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.conv_pool._activate')
    @patch('yann.layers.conv_pool.batch_normalization_test')
    @patch('yann.layers.conv_pool.batch_normalization_train')
    def test3_conv_pool_layer_2d_ip_vals(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_activate,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_activate.return_value = (self.input_ndarray, self.input_shape)
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.conv_pool_layer_2d = cl(
                            input = self.input_tensor,
                            id = self.conv_pool_layer_2d_name,
                            input_shape = self.input_shape,
                            nkerns=10,
                            verbose = self.verbose,
                            input_params= self.input_params,
                            batch_norm = True
        )
        self.assertEqual(self.conv_pool_layer_2d.id,self.conv_pool_layer_2d_name)
        self.assertEqual(self.conv_pool_layer_2d.output_shape,self.input_shape)
        self.assertTrue(numpy.allclose(self.conv_pool_layer_2d.output,self.input_ndarray))
        self.assertTrue(numpy.allclose(self.conv_pool_layer_2d.inference,self.input_ndarray))

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.conv_pool._activate')
    @patch('yann.layers.conv_pool.batch_normalization_test')
    @patch('yann.layers.conv_pool.batch_normalization_train')
    def test4_conv_pool_layer_2d_no_bn(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_activate,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_activate.return_value = (self.input_ndarray, self.input_shape)
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.conv_pool_layer_2d = cl(
                            input = self.input_tensor,
                            id = self.conv_pool_layer_2d_name,
                            input_shape = self.input_shape,
                            nkerns=10,
                            verbose = self.verbose,
                            batch_norm = False
        )
        self.assertEqual(self.conv_pool_layer_2d.id,self.conv_pool_layer_2d_name)
        self.assertEqual(self.conv_pool_layer_2d.output_shape,self.input_shape)
        self.assertTrue(numpy.allclose(self.conv_pool_layer_2d.output,self.input_ndarray))
        self.assertTrue(numpy.allclose(self.conv_pool_layer_2d.inference,self.input_ndarray))

    @patch('yann.layers.conv_pool._dropout')
    def test5_dropout_conv_pool_layer_2d_layer(self,mock_dropout):
        mock_dropout.return_value = self.input_ndarray
        self.dcplayer = dcl(
                dropout_rate = self.dropout_rate,
                id = self.dropout_conv_pool_layer_2d_name,
                input = self.input_tensor,
                input_shape = self.input_shape,
                nkerns=10,
                verbose = self.verbose)
        self.assertTrue(numpy.allclose(self.dcplayer.output, self.input_ndarray))
        self.assertEqual(self.dcplayer.id,self.dropout_conv_pool_layer_2d_name)

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.conv_pool._activate')
    @patch('yann.layers.conv_pool.batch_normalization_test')
    @patch('yann.layers.conv_pool.batch_normalization_train')
    def test6_deconv_pool_layer_2d(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_activate,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_activate.return_value = (self.input_ndarray, self.input_shape)
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.deconv_pool_layer_2d = dl(
                            input = self.input_tensor,
                            id = self.deconv_pool_layer_2d_name,
                            input_shape = self.input_shape,
                            output_shape= self.input_shape,
                            nkerns=10,
                            verbose = self.verbose,
                            batch_norm = True
        )
        self.assertEqual(self.deconv_pool_layer_2d.id,self.deconv_pool_layer_2d_name)
        self.assertEqual(self.deconv_pool_layer_2d.output_shape,self.input_shape)
        self.assertTrue(numpy.allclose(self.deconv_pool_layer_2d.output,self.input_ndarray))
        self.assertTrue(numpy.allclose(self.deconv_pool_layer_2d.inference,self.input_ndarray))

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.conv_pool._activate')
    @patch('yann.layers.conv_pool.batch_normalization_test')
    @patch('yann.layers.conv_pool.batch_normalization_train')
    def test7_deconv_pool_layer_2d_ip_none(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_activate,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_activate.return_value = (self.input_ndarray, self.input_shape)
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.deconv_pool_layer_2d = dl(
                            input = self.input_tensor,
                            id = self.deconv_pool_layer_2d_name,
                            input_shape = self.input_shape,
                            output_shape=self.input_shape,
                            nkerns=10,
                            verbose = self.verbose,
                            input_params= self.input_params_none,
                            batch_norm = True
        )
        self.assertEqual(self.deconv_pool_layer_2d.id,self.deconv_pool_layer_2d_name)
        self.assertEqual(self.deconv_pool_layer_2d.output_shape,self.input_shape)
        self.assertTrue(numpy.allclose(self.deconv_pool_layer_2d.output,self.input_ndarray))
        self.assertTrue(numpy.allclose(self.deconv_pool_layer_2d.inference,self.input_ndarray))

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.conv_pool._activate')
    @patch('yann.layers.conv_pool.batch_normalization_test')
    @patch('yann.layers.conv_pool.batch_normalization_train')
    def test8_deconv_pool_layer_2d_ip_vals(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_activate,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_activate.return_value = (self.input_ndarray, self.input_shape)
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.deconv_pool_layer_2d = dl(
                            input = self.input_tensor,
                            id = self.deconv_pool_layer_2d_name,
                            input_shape = self.input_shape,
                            output_shape=self.input_shape,
                            nkerns=10,
                            verbose = self.verbose,
                            input_params= self.input_params_none,
                            batch_norm = True
        )
        self.assertEqual(self.deconv_pool_layer_2d.id,self.deconv_pool_layer_2d_name)
        self.assertEqual(self.deconv_pool_layer_2d.output_shape,self.input_shape)
        self.assertTrue(numpy.allclose(self.deconv_pool_layer_2d.output,self.input_ndarray))
        self.assertTrue(numpy.allclose(self.deconv_pool_layer_2d.inference,self.input_ndarray))

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.conv_pool._activate')
    @patch('yann.layers.conv_pool.batch_normalization_test')
    @patch('yann.layers.conv_pool.batch_normalization_train')
    def test9_deconv_pool_layer_2d_no_bn(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_activate,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_activate.return_value = (self.input_ndarray, self.input_shape)
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.deconv_pool_layer_2d = dl(
                            input = self.input_tensor,
                            id = self.deconv_pool_layer_2d_name,
                            input_shape = self.input_shape,
                            output_shape=self.input_shape,
                            nkerns=10,
                            verbose = self.verbose,
                            input_params= self.input_params_none,
                            batch_norm = True
        )
        self.assertEqual(self.deconv_pool_layer_2d.id,self.deconv_pool_layer_2d_name)
        self.assertEqual(self.deconv_pool_layer_2d.output_shape,self.input_shape)
        self.assertTrue(numpy.allclose(self.deconv_pool_layer_2d.output,self.input_ndarray))
        self.assertTrue(numpy.allclose(self.deconv_pool_layer_2d.inference,self.input_ndarray))

    @patch('yann.layers.conv_pool._dropout')
    def test10_dropout_deconv_pool_layer_2d_layer(self,mock_dropout):
        mock_dropout.return_value = self.input_ndarray
        self.ddcplayer = ddl(
                dropout_rate = self.dropout_rate,
                id = self.dropout_deconv_pool_layer_2d_name,
                input = self.input_tensor,
                input_shape = self.input_shape,
                output_shape=self.input_shape,
                nkerns=10,
                verbose = self.verbose)
        self.assertTrue(numpy.allclose(self.ddcplayer.output, self.input_ndarray))
        self.assertEqual(self.ddcplayer.id,self.dropout_deconv_pool_layer_2d_name)

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.conv_pool._activate')
    @patch('yann.layers.conv_pool.batch_normalization_test')
    @patch('yann.layers.conv_pool.batch_normalization_train')
    def test11_conv_pool_layer_2d_ip_vals(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_activate,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_activate.return_value = (self.input_ndarray, self.input_shape)
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.conv_pool_layer_2d = cl(
                            input = self.input_tensor,
                            id = self.conv_pool_layer_2d_name,
                            input_shape = self.input_shape,
                            nkerns=10,
                            verbose = self.verbose,
                            input_params= self.input_params,
                            poolsize= (1,1),
                            batch_norm = True
        )
        self.assertEqual(self.conv_pool_layer_2d.id,self.conv_pool_layer_2d_name)
        self.assertEqual(self.conv_pool_layer_2d.output_shape,self.input_shape)
        self.assertTrue(numpy.allclose(self.conv_pool_layer_2d.output,self.input_ndarray))
        self.assertTrue(numpy.allclose(self.conv_pool_layer_2d.inference,self.input_ndarray))

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.conv_pool._activate')
    @patch('yann.layers.conv_pool.batch_normalization_test')
    @patch('yann.layers.conv_pool.batch_normalization_train')
    def test12_conv_pool_layer_print_layer(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_activate,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_activate.return_value = (self.input_ndarray, self.input_shape)
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.layer = cl(
                            input = self.input_tensor,
                            id = self.conv_pool_layer_2d_name,
                            input_shape = self.input_shape,
                            nkerns=10,
                            verbose = self.verbose,
                            batch_norm = True
        )
        self.attributes = self.layer._graph_attributes()
        self.layer.output_shape = self.input_shape
        self.layer.origin = "input"
        self.layer.destination = "classifier"
        self.layer.batch_norm = False
        self.layer.filter_shape = (1,1)
        self.layer.input_shape = (1,1,10,10)
        self.layer.poolsize = (1,1)
        self.layer.stride = (1,1)
        self.layer.print_layer(prefix=" ", nest=False, last=False)
        self.assertTrue(len(self.layer.prefix) > 0)

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.conv_pool._activate')
    @patch('yann.layers.conv_pool.batch_normalization_test')
    @patch('yann.layers.conv_pool.batch_normalization_train')
    def test13_deconv_pool_layer_2d_ip_vals(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_activate,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_activate.return_value = (self.input_ndarray, self.input_shape)
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.deconv_pool_layer_2d = dl(
                            input = self.input_tensor,
                            id = self.deconv_pool_layer_2d_name,
                            input_shape = self.input_shape,
                            output_shape=self.input_shape,
                            nkerns=10,
                            verbose = self.verbose,
                            input_params= self.input_params,
                            batch_norm = True
        )
        self.assertEqual(self.deconv_pool_layer_2d.id,self.deconv_pool_layer_2d_name)
        self.assertEqual(self.deconv_pool_layer_2d.output_shape,self.input_shape)
        self.assertTrue(numpy.allclose(self.deconv_pool_layer_2d.output,self.input_ndarray))
        self.assertTrue(numpy.allclose(self.deconv_pool_layer_2d.inference,self.input_ndarray))

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.conv_pool._activate')
    @patch('yann.layers.conv_pool.batch_normalization_test')
    @patch('yann.layers.conv_pool.batch_normalization_train')
    def test14_deconv_pool_layer_2d_pool_size_mismatch_exception(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_activate,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_activate.return_value = (self.input_ndarray, self.input_shape)
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        try:
            self.deconv_pool_layer_2d = dl(
                                input = self.input_tensor,
                                id = self.deconv_pool_layer_2d_name,
                                input_shape = self.input_shape,
                                output_shape=self.input_shape,
                                nkerns=10,
                                verbose = self.verbose,
                                input_params= self.input_params,
                                poolsize= (2,2),
                                batch_norm = True
            )
            self.assertEqual(True,False)
        except Exception,c:
            self.assertEqual(c.message,self.pool_size_mismatch_exception_msg)

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.conv_pool._activate')
    @patch('yann.layers.conv_pool.batch_normalization_test')
    @patch('yann.layers.conv_pool.batch_normalization_train')
    def test15_deconv_pool_layer_2d_activation_tuple_exception(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_activate,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_activate.return_value = (self.input_ndarray, self.input_shape)
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        try:
            self.deconv_pool_layer_2d = dl(
                            input = self.input_tensor,
                            id = self.deconv_pool_layer_2d_name,
                            input_shape = self.input_shape,
                            output_shape=self.input_shape,
                            nkerns=10,
                            verbose = self.verbose,
                            input_params= self.input_params,
                            batch_norm = False,
                            activation= ('maxout','RelU')
        )
            self.assertEqual(True,False)
        except Exception,c:
            print(c.message)
            self.assertEqual(c.message,self.activation_tuple_exception_msg)

    @patch('theano.tensor.unbroadcast')
    @patch('yann.layers.conv_pool._activate')
    @patch('yann.layers.conv_pool.batch_normalization_test')
    @patch('yann.layers.conv_pool.batch_normalization_train')
    def test15_deconv_pool_layer_2d_print_layer(self,mock_batch_normalization_train,mock_batch_normalization_test,mock_activate,mock_unbroadcast):
        mock_unbroadcast.return_value = 1
        mock_activate.return_value = (self.input_ndarray, self.input_shape)
        mock_batch_normalization_train.return_value = (self.output_train,1,1,1,1)
        mock_batch_normalization_test.return_value =self.output_test
        self.layer = dl(
                            input = self.input_tensor,
                            id = self.deconv_pool_layer_2d_name,
                            input_shape = self.input_shape,
                            output_shape=self.input_shape,
                            nkerns=10,
                            verbose = self.verbose,
                            input_params= self.input_params,
                            batch_norm = True
        )
        self.attributes = self.layer._graph_attributes()
        self.layer.output_shape = self.input_shape
        self.layer.origin = "input"
        self.layer.destination = "classifier"
        self.layer.batch_norm = False
        self.layer.filter_shape = (1,1)
        self.layer.input_shape = (1,1,10,10)
        self.layer.poolsize = (1,1)
        self.layer.stride = (1,1)
        self.layer.print_layer(prefix=" ", nest=False, last=False)
        self.assertTrue(len(self.layer.prefix) > 0)
