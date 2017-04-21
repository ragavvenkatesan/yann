import unittest
import numpy
import theano
from yann.layers.input import input_layer as il
from yann.layers.input import dropout_input_layer as dil
from yann.layers.input import tensor_layer as tl
from yann.layers.input import dropout_tensor_layer as dtl
import yann
class TestInput(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(0)
        self.verbose = 3
        self.input_layer_name = "input"
        self.input_dropout_layer_name = "dinput"
        self.input_tensor_layer_name = "tinput"
        self.input_dropout_tensor_layer_name = "dtinput"
        self.dropout_rate = 1
        self.rng = None
        self.mean_subtract = False
        self.input_shape = (1,1,10,10)
        self.input_ndarray = numpy.random.rand(1,1,10,10)
        self.output_dropout_ndarray= numpy.zeros((1,1,10,10))
        self.input_tensor = theano.shared(self.input_ndarray)
    def test1_input_layer(self):
        self.layer = il(
                x = self.input_tensor,
                mini_batch_size = self.input_ndarray.shape[0],
                id = self.input_layer_name,
                height = self.input_ndarray.shape[2],
                width = self.input_ndarray.shape[3],
                channels = self.input_ndarray.shape[1],
                mean_subtract = self.mean_subtract,
                verbose = self.verbose)
        self.assertTrue(numpy.allclose(self.layer.output.eval(), self.input_ndarray))
        self.assertEqual(self.layer.output_shape,self.input_ndarray.shape)
        self.assertEqual(self.layer.id,self.input_layer_name)

    def test2_dropout_input_layert(self):
        self.dlayer = dil(
                x = self.input_tensor,
                mini_batch_size = self.input_ndarray.shape[0],
                id = self.input_dropout_layer_name,
                height = self.input_ndarray.shape[2],
                width = self.input_ndarray.shape[3],
                channels = self.input_ndarray.shape[1],
                mean_subtract = self.mean_subtract,
                dropout_rate= self.dropout_rate,
                verbose = self.verbose)
        self.assertTrue(numpy.allclose(self.dlayer.output.eval(), self.output_dropout_ndarray))
        self.assertEqual(self.dlayer.output_shape,self.input_ndarray.shape)
        self.assertEqual(self.dlayer.id,self.input_dropout_layer_name)

    def test3_tensor_layer(self):
        self.tlayer = tl(
                id = self.input_tensor_layer_name,
                input = self.input_tensor,
                input_shape = self.input_shape,
                verbose = self.verbose)
        self.assertTrue(numpy.allclose(self.tlayer.output.eval(), self.input_ndarray))
        self.assertEqual(self.tlayer.output_shape,self.input_ndarray.shape)
        self.assertEqual(self.tlayer.id,self.input_tensor_layer_name)

    def test4_dropout_tensor_layer(self):
        self.dtlayer = dtl(
                dropout_rate = self.dropout_rate,
                rng = self.rng,
                id = self.input_dropout_tensor_layer_name,
                input = self.input_tensor,
                input_shape = self.input_shape,
                verbose = self.verbose)

        self.assertTrue(numpy.allclose(self.dtlayer.output.eval(), self.output_dropout_ndarray))
        self.assertEqual(self.dtlayer.output_shape,self.input_ndarray.shape)
        self.assertEqual(self.dtlayer.id,self.input_dropout_tensor_layer_name)

