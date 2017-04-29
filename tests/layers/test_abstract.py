import unittest
import numpy
import theano
import theano.tensor as T
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


    def test13_abstract_layer_print_layer(self):
        self.layer = l(
                id = self.abstract_layer_name,
                type= "conv",
                verbose = self.verbose)
        self.attributes = self.layer._graph_attributes()
        self.layer.output_shape = self.input_shape
        self.layer.origin =  "input"
        self.layer.destination = "classifier"
        self.layer.batch_norm = True
        self.layer.print_layer(prefix = " ", nest = False, last = True, verbose = self.verbose)
        print (self.layer.prefix)
        self.assertTrue(len(self.layer.prefix)>0)

    def test14_abstract_layer_print_layer(self):
        self.layer = l(
                id = self.abstract_layer_name,
                type= "conv",
                verbose = self.verbose)
        self.attributes = self.layer._graph_attributes()
        self.layer.output_shape = self.input_shape
        self.layer.origin =  "input"
        self.layer.destination = "classifier"
        self.layer.batch_norm = False
        self.layer.print_layer(prefix = " ", nest = False, last = False, verbose = self.verbose)
        print (self.layer.prefix)
        self.assertTrue(len(self.layer.prefix)>0)

    def test15_abstract_layer_print_layer(self):
        self.layer = l(
                id = self.abstract_layer_name,
                type= "convolution",
                verbose = self.verbose)
        self.attributes = self.layer._graph_attributes()
        self.layer.output_shape = self.input_shape
        self.layer.origin =  "input"
        self.layer.destination = "classifier"
        self.layer.batch_norm = False
        self.layer.print_layer(prefix = " ", nest = False, last = False, verbose = self.verbose)
        print (self.layer.prefix)
        self.assertTrue(len(self.layer.prefix)>0)

    def test16_abstract_layer_get_params(self):
        self.layer = l(
                id = self.abstract_layer_name,
                type= "conv",
                verbose = self.verbose)
        self.layer.params = [self.input_tensor,self.input_tensor]
        out = self.layer.get_params(borrow=True,verbose=self.verbose)
        self.assertTrue(numpy.allclose(out,[self.input_ndarray,self.input_ndarray]))

    def test17_abstract_layer_get_params(self):
        self.layer = l(
                id = self.abstract_layer_name,
                type= "conv",
                verbose = self.verbose)
        self.val = T.dot(self.input_tensor,self.input_tensor)
        self.layer.params = [self.val]
        out = self.layer.get_params(borrow=True,verbose=self.verbose)
        self.assertTrue(numpy.allclose(out,self.val.eval()))

    def test18_abstract_layer_get_params(self):
        self.layer = l(
                id = self.abstract_layer_name,
                type= "conv",
                verbose = self.verbose)
        self.val = T.dot(self.input_tensor,self.input_tensor)
        self.layer.params = [self.val]
        self.layer.output_shape = self.input_shape
        self.layer.num_neurons = 10
        self.layer.activation = ('ReLu')
        out = self.layer.get_params(borrow=True,verbose=self.verbose)
        self.assertTrue(numpy.allclose(out,self.val.eval()))
