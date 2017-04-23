#objective layer class has error: method 'loss' and string 'loss' has conflict
import unittest
import numpy
import theano
from yann.layers.output import classifier_layer as cl
from yann.layers.output import objective_layer as ol
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch

class TestOutput(unittest.TestCase):
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
        self.classifier_layer_name = "cl"
        self.output_layer_name = "ol"
        self.dropout_rate = 1
        self.default_param_value = [1.]
        self.custom_param_value = [1., 1.,1.]
        self.classes = 3
        self.output_shape = (self.input_shape[0], self.classes)
        self.sample = numpy.ones((1,1,2,2))
        self.input_params_all = (self.sample,self.sample)

    @patch('yann.layers.output._activate')
    def test1_output(self,mock_activate):
        mock_activate.return_value = (self.input_ndarray,self.input_shape)
        self.output_layer = cl(
                            input = self.input_tensor,
                            id = self.classifier_layer_name,
                            num_classes= self.classes,
                            input_shape = self.input_shape,
                            rng = self.rng,
                            borrow = self.borrow,
                            input_params = None,
                            verbose = self.verbose
        )
        self.assertEqual(self.output_layer.id,self.classifier_layer_name)
        self.assertEqual(self.output_layer.output_shape,self.output_shape)
        self.assertTrue(numpy.allclose(self.output_layer.output, self.input_ndarray))
        self.assertTrue(numpy.allclose(self.output_layer.inference, self.input_ndarray))


    @patch('yann.layers.output._activate')
    def test2_output_params(self,mock_activate):
        mock_activate.return_value = (self.input_ndarray,self.input_shape)
        self.output_layer = cl(
                            input = self.input_tensor,
                            id = self.classifier_layer_name,
                            num_classes= self.classes,
                            input_shape = self.input_shape,
                            rng = self.rng,
                            borrow = self.borrow,
                            input_params = self.input_params_all,
                            verbose = self.verbose
        )
        self.assertEqual(self.output_layer.id,self.classifier_layer_name)
        self.assertEqual(self.output_layer.output_shape,self.output_shape)
        self.assertTrue(numpy.allclose(self.output_layer.output, self.input_ndarray))
        self.assertTrue(numpy.allclose(self.output_layer.inference, self.input_ndarray))

    @patch('yann.layers.output.T.mean')
    @patch('yann.layers.output._activate')
    def test3_loss_nll(self,mock_activate,mock_mean):
        mock_activate.return_value = (self.input_ndarray,self.input_shape)
        mock_mean.return_value = 1
        self.output_layer = cl(
                            input = self.input_tensor,
                            id = self.classifier_layer_name,
                            num_classes= self.classes,
                            input_shape = self.input_shape,
                            rng = self.rng,
                            borrow = self.borrow,
                            input_params = self.input_params_all,
                            verbose = self.verbose
        )
        self.assertEqual(self.output_layer.loss(theano.tensor.ivector('s'),'nll'),-1)

    @patch('yann.layers.output.T.nnet.categorical_crossentropy')
    @patch('yann.layers.output.T.mean')
    @patch('yann.layers.output._activate')
    def test4_loss_cce(self,mock_activate,mock_mean,mock_entropy):
        mock_entropy.return_value = 1
        mock_activate.return_value = (self.input_ndarray,self.input_shape)
        mock_mean.return_value = 1
        self.output_layer = cl(
                            input = self.input_tensor,
                            id = self.classifier_layer_name,
                            num_classes= self.classes,
                            input_shape = self.input_shape,
                            rng = self.rng,
                            borrow = self.borrow,
                            input_params = self.input_params_all,
                            verbose = self.verbose
        )
        self.assertEqual(self.output_layer.loss(theano.tensor.ivector('s'),'cce'),1)



    @patch('yann.layers.output.T.mean')
    @patch('yann.layers.output._activate')
    def test5_loss_cce(self,mock_activate,mock_mean):
        mock_activate.return_value = (self.input_ndarray,self.input_shape)
        mock_mean.return_value = 1
        self.output_layer = cl(
                            input = self.input_tensor,
                            id = self.classifier_layer_name,
                            num_classes= self.classes,
                            input_shape = self.input_shape,
                            rng = self.rng,
                            borrow = self.borrow,
                            input_params = self.input_params_all,
                            verbose = self.verbose
        )
        self.assertEqual(self.output_layer.loss(theano.tensor.ivector('s'),'bce'),1)

    @patch('yann.layers.output.T.mean')
    @patch('yann.layers.output.T.maximum')
    @patch('yann.layers.output._activate')
    def test6_loss_hinge(self,mock_activate,mock_max,mock_mean):
        mock_mean.return_value = 1
        mock_activate.return_value = (self.input_ndarray,self.input_shape)
        mock_max.return_value = self.output_dropout_ndarray
        self.output_layer = cl(
                            input = self.input_tensor,
                            id = self.classifier_layer_name,
                            num_classes= self.classes,
                            input_shape = self.input_shape,
                            rng = self.rng,
                            borrow = self.borrow,
                            input_params = self.input_params_all,
                            verbose = self.verbose
        )
        self.assertEqual(self.output_layer.loss(self.input_ndarray,'hinge'),0)

