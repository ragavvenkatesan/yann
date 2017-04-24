import unittest
import numpy
import theano
from yann.layers.abstract import layer as l,_dropout,_activate
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch


class TestAbstract(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(0)
        self.verbose = 3
        self.abstract_layer_name = "input"
        self.dropout_rate = 1
        self.rng = None
        self.mean_subtract = False
        self.input_shape = (1,1,10,10)
        self.input_ndarray = numpy.random.rand(1,1,10,10)
        self.output_dropout_ndarray= numpy.zeros((1,1,10,10))
        self.input_tensor = theano.shared(self.input_ndarray)
        self.exception_msg = "'NoneType' object is not iterable"
        self.rng = numpy.random
        self.rs = RandomStreams(self.rng.randint(1,2147462468))
        self.sample_input = numpy.ones((1, 1, 3, 3))

    def test1_abstract_layer(self):
        self.layer = l(
                id = self.abstract_layer_name,
                type= "input",
                verbose = self.verbose)
        self.attributes = self.layer._graph_attributes()
        self.assertEqual(self.attributes['id'], self.abstract_layer_name)

    def test2_abstract_layer_exception(self):
        try:
            self.layer = l(
                id = self.abstract_layer_name,
                type= "input",
                verbose = self.verbose)
            self.params = self.layer.get_params()
        except Exception,c:
            self.assertEqual(c.message,self.exception_msg)

    @patch('yann.layers.abstract.RandomStreams')
    @patch('yann.layers.abstract.RandomStreams.binomial')
    def test3_abstract_layer_dropout(self,mock_binomial,mock_random_streams):
        mock_random_streams.return_value = self.rs
        mock_binomial.return_value = self.sample_input
        self.out = _dropout(rng = self.rng,
                                   params = self.sample_input,
                                   dropout_rate = self.dropout_rate)
        self.assertTrue(numpy.allclose(self.out,self.sample_input))


    @patch('yann.layers.abstract.activations.ReLU')
    def test4_abstract_layer_activate_relu(self,mock_relu):
        mock_relu.return_value = self.input_ndarray
        self.out,self.out_shape = _activate(self.input_ndarray,"relu",self.input_shape,self.verbose)
        self.assertTrue(numpy.allclose(self.out,self.input_ndarray))

    @patch('yann.layers.abstract.activations.Abs')
    def test5_abstract_layer_activate_abs(self,mock_abs):
        mock_abs.return_value = self.input_ndarray
        self.out,self.out_shape = _activate(self.input_ndarray,"abs",self.input_shape,self.verbose)
        self.assertTrue(numpy.allclose(self.out,self.input_ndarray))

    @patch('yann.layers.abstract.activations.Sigmoid')
    def test6_abstract_layer_activate_sigmoid(self,mock_sigmoid):
        mock_sigmoid.return_value = self.input_ndarray
        self.out,self.out_shape = _activate(self.input_ndarray,"sigmoid",self.input_shape,self.verbose)
        self.assertTrue(numpy.allclose(self.out,self.input_ndarray))


    @patch('yann.layers.abstract.activations.Tanh')
    def test7_abstract_layer_activate_tanh(self,mock_tanh):
        mock_tanh.return_value = self.input_ndarray
        self.out,self.out_shape = _activate(self.input_ndarray,"tanh",self.input_shape,self.verbose)
        self.assertTrue(numpy.allclose(self.out,self.input_ndarray))

    @patch('yann.layers.abstract.activations.Softmax')
    def test8_abstract_layer_activate_softmax(self,mock_softmax):
        mock_softmax.return_value = self.input_ndarray
        self.out,self.out_shape = _activate(self.input_ndarray,"softmax",self.input_shape,self.verbose)
        self.assertTrue(numpy.allclose(self.out,self.input_ndarray))

    @patch('yann.layers.abstract.activations.Squared')
    def test9_abstract_layer_activate_squared(self,mock_squared):
        mock_squared.return_value = self.input_ndarray
        self.out,self.out_shape = _activate(self.input_ndarray,"squared",self.input_shape,self.verbose)
        self.assertTrue(numpy.allclose(self.out,self.input_ndarray))


    @patch('yann.layers.abstract.activations.Maxout')
    def test10_abstract_layer_activate_maxout_tuple(self,mock_maxout):
        mock_maxout.return_value = (self.input_ndarray,self.input_shape)
        self.out,self.out_shape = _activate(self.input_ndarray,("maxout","type",self.input_shape),self.input_shape,self.verbose,**{'dimension':(10,10)})
        self.assertTrue(numpy.allclose(self.out,self.input_ndarray))

    @patch('yann.layers.abstract.activations.ReLU')
    def test11_abstract_layer_activate_relu_tuple(self,mock_relu):
        mock_relu.return_value = self.input_ndarray
        self.out,self.out_shape = _activate(self.input_ndarray,("relu",1),self.input_shape,self.verbose)
        self.assertTrue(numpy.allclose(self.out,self.input_ndarray))

    @patch('yann.layers.abstract.activations.Softmax')
    def test12_abstract_layer_activate_softmax_tuple(self,mock_softmax):
        mock_softmax.return_value = self.input_ndarray
        self.out,self.out_shape = _activate(self.input_ndarray,("softmax",1),self.input_shape,self.verbose)
        self.assertTrue(numpy.allclose(self.out,self.input_ndarray))
