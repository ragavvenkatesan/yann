import pytest
import numpy as np
import theano.tensor as T
import yann.core.activations as A

class TestActivations:
    @classmethod
    def setup_class(self):
        self.theano_input = T.matrix()
        self.numpy_input = np.random.uniform(-4, 4, (5, 1))  # Create some 5X5 matrix randomly     
    def Abs(self, x): return np.abs(x)
    def ReLU(self, x): return x * (x > 0)
    def Sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def Tanh(self, x): return np.tanh(x)
    def Softmax(self, x): return (np.exp(x).T / np.exp(x).sum(-1)).T
    def Squared(self, x): return x**2
    def TemperatureSoftmax(self, x, t): return (np.exp(float(x)/t).T / np.exp(float(x)/t).sum(-1)).T           

    def test_abs(self):         
        theano_result = A.Abs(self.theano_input).eval({self.theano_input: self.numpy_input})
        np_result = self.Abs(self.numpy_input)
        assert np.allclose(theano_result, np_result)

    def test_relu(self):         
        theano_result = A.ReLU(self.theano_input).eval({self.theano_input: self.numpy_input})
        np_result = self.ReLU(self.numpy_input)
        assert np.allclose(theano_result, np_result)      

    def test_sigmoid(self):         
        theano_result = A.Sigmoid(self.theano_input).eval({self.theano_input: self.numpy_input})
        np_result = self.Sigmoid(self.numpy_input)
        assert np.allclose(theano_result, np_result)        

    def test_tanh(self):         
        theano_result = A.Tanh(self.theano_input).eval({self.theano_input: self.numpy_input})
        np_result = self.Tanh(self.numpy_input)
        assert np.allclose(theano_result, np_result)    

    def test_softmax(self):         
        theano_result = A.Softmax(self.theano_input).eval({self.theano_input: self.numpy_input})
        np_result = self.Softmax(self.numpy_input)
        assert np.allclose(theano_result, np_result)  

    def test_temperature_softmax(self):         
        theano_result = A.Softmax(self.theano_input, temp = 2 ).eval({self.theano_input: self.numpy_input})
        np_result = self.TemperatereSoftmax(self.numpy_input, t = 2)
        assert np.allclose(theano_result, np_result)                                     

    def test_squared(self):         
        theano_result = A.Squared(self.theano_input).eval({self.theano_input: self.numpy_input})
        np_result = self.Squared(self.numpy_input)
        assert np.allclose(theano_result, np_result)     
