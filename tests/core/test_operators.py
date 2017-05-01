import theano
import theano.tensor as T
import numpy
import unittest
from yann.core.operators import copy_params

class TestOperators(unittest.TestCase):

    def test_copy_params(self):
        numpy.random.seed(0)
        self.verbose = 3
        self.input_ndarray = numpy.random.rand(1, 1, 10, 10)
        self.input_tensor = theano.shared(self.input_ndarray)
        self.output_ndarray = numpy.zeros((1,1,10,10))
        self.output_tensor = theano.shared(self.output_ndarray)
        self.source = [self.input_tensor]
        self.dest =  [self.output_tensor]
        copy_params(source=self.source, destination= self.dest, borrow= True, verbose= self.verbose)
        self.assertTrue(numpy.allclose(self.dest[0].eval(),self.source[0].eval()))
