import unittest
import numpy
import theano
from yann.layers.flatten import flatten_layer as fl,unflatten_layer as ufl
class TestInput(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.verbose = 3
        self.flatten_id = "flatten"
        self.unflatten_id = "unflatten"
        self.input_shape = (1,1,3,3)
        self.input_ndarray = numpy.random.rand(1,1,3,3)
        self.output_shape = (3,3,1) #order changes in code
        self.input_tensor = theano.shared(self.input_ndarray)

    def test_flatten_unflatten(self):
        self.flatten = fl(
            input = self.input_tensor,
            input_shape = self.input_shape,
            id = self.flatten_id,
            verbose= self.verbose
        )
        self.unflatten = ufl(
            input = self.flatten.output.eval(),
            input_shape = self.flatten.output_shape,
            id = self.unflatten_id,
            shape = (3,3,1),
            verbose= self.verbose
        )
        self.assertTrue(numpy.allclose(self.unflatten.output.eval(),self.input_ndarray))
