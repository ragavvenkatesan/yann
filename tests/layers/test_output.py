#objective layer class has error: method 'loss' and string 'loss' has conflict
import unittest
import numpy
import theano
import theano.tensor as T
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


    @patch('yann.layers.output.T.sum')
    @patch('yann.layers.output.T.mean')
    @patch('yann.layers.output.T.maximum')
    @patch('yann.layers.output._activate')
    def test7_error(self,mock_activate,mock_max,mock_mean, mock_sum):
        mock_sum.return_value = 1
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

        self.output_layer.predictions = theano.tensor.ivector('s')
        self.assertEqual(self.output_layer.errors(self.output_layer.predictions),1)


    @patch('yann.layers.output.T.sum')
    @patch('yann.layers.output.T.mean')
    @patch('yann.layers.output.T.maximum')
    @patch('yann.layers.output._activate')
    def test8_error_exception(self,mock_activate,mock_max,mock_mean, mock_sum):
        mock_sum.return_value = 1
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

        self.output_layer.predictions = theano.tensor.ivector('s')
        try:
            self.output_layer.errors(theano.tensor.iscalar('s'))
            self.assertEqual(True, False)
        except TypeError,e:
            self.assertEqual(True,True)

    @patch('yann.layers.output.T.sum')
    @patch('yann.layers.output.T.mean')
    @patch('yann.layers.output.T.maximum')
    @patch('yann.layers.output._activate')
    def test9_error_exception(self,mock_activate,mock_max,mock_mean, mock_sum):
        mock_sum.return_value = 1
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

        self.output_layer.predictions = theano.tensor.scalar('s')
        try:
            self.output_layer.errors(theano.tensor.scalar('s'))
            self.assertEqual(True, False)
        except NotImplementedError,e:
            self.assertEqual(True,True)


    @patch('yann.layers.output.T.mean')
    @patch('yann.layers.output.T.maximum')
    @patch('yann.layers.output._activate')
    def test10_loss_exception(self,mock_activate,mock_max,mock_mean):
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
        try:
            self.output_layer.loss(self.input_ndarray,'hinge1')
            self.assertEqual(True, False)
        except Exception,e:
            self.assertEqual(True,True)

    @patch('yann.layers.output.T.mean')
    @patch('yann.layers.output.T.maximum')
    @patch('yann.layers.output._activate')
    def test11_get_params(self,mock_activate,mock_max,mock_mean):
        self.val = T.dot(self.input_tensor,self.input_tensor)
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
        self.output_layer.params = [MockParam(), MockParam()]
        out =  self.output_layer.get_params(borrow=True,verbose=self.verbose)
        self.assertTrue(numpy.allclose(out,[1,1]))

class MockParam():
    def get_value(self,borrow = True):
        return 1
