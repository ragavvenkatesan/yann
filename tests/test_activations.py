import pytest
import numpy as np
import theano.tensor as T
import yann.core.activations as A

class TestActivations:
    @classmethod
    def setup_class(self):
        self.theano_input = T.matrix()
        self.numpy_input = np.random.uniform(-4, 4, (5, 5))  # Create some 5X5 matrix randomly     
    def Abs(self, x): return np.abs(x)
    def ReLU(self, x): return x * (x > 0)
    def Sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def Tanh(self, x): return np.tanh(x)
    def Softmax(self, x): return (np.exp(x).T / np.exp(x).sum(-1)).T
    def Squared(self, x): return x**2       
    
    def test_abs(self):         
        theano_result = A.Abs(self.theano_input).eval({self.theano_input: self.numpy_input})
        np_result = self.Abs(self.numpy_input)
        assert np.allclose(theano_result, np_result)

