import pytest
import numpy as np
import theano
import theano.tensor as T
import yann.core.activations as A

class TestActivations:
    @classmethod
    def setup_class(self):
        self.theano_input = T.matrix()
        self.numpy_input = np.asarray(np.random.uniform(-4, 4, (5,5)),dtype=theano.config.floatX)  
        # Create some 5X5 matrix randomly     

    def Elu(self, x, alpha = 1): return np.where(x > 0, x, alpha * (np.exp(x) - 1)) 
    def Abs(self, x): return np.abs(x)
    def ReLU(self, x): return x * (x > 0)
    def Sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def Tanh(self, x): return np.tanh(x)
    def Softmax(self, x, temp = 1):
        if temp != 1:
            expo = np.exp(x / float(temp))
            return expo / expo.sum(axis = 1, keepdims=True)
        else:
            return np.exp(x) / np.exp(x).sum(axis = 1, keepdims=True)  
    def Squared(self, x): return x**2

    def test_abs(self):         
        theano_result = A.Abs(self.theano_input).eval({self.theano_input: self.numpy_input})
        np_result = self.Abs(self.numpy_input)
        assert theano_result.shape == np_result.shape                
        assert np.allclose(theano_result, np_result)

    def test_relu(self):         
        theano_result = A.ReLU(self.theano_input).eval({self.theano_input: self.numpy_input})
        np_result = self.ReLU(self.numpy_input)
        assert theano_result.shape == np_result.shape                
        assert np.allclose(theano_result, np_result)      

    def test_elu(self):      
        alpha = np.asarray(np.random.random(1),dtype = theano.config.floatX)                
        theano_result = A.Elu(self.theano_input,alpha).eval({self.theano_input: self.numpy_input})
        np_result = self.Elu(self.numpy_input,alpha)
        assert theano_result.shape == np_result.shape                
        assert np.allclose(theano_result, np_result) 

    def test_sigmoid(self):         
        theano_result = A.Sigmoid(self.theano_input).eval({self.theano_input: self.numpy_input})
        np_result = self.Sigmoid(self.numpy_input)
        assert theano_result.shape == np_result.shape                
        assert np.allclose(theano_result, np_result)        

    def test_tanh(self):         
        theano_result = A.Tanh(self.theano_input).eval({self.theano_input: self.numpy_input})
        np_result = self.Tanh(self.numpy_input)
        assert theano_result.shape == np_result.shape                
        assert np.allclose(theano_result, np_result)    

    def test_softmax(self):         
        theano_result = A.Softmax(self.theano_input).eval({self.theano_input: self.numpy_input})
        np_result = self.Softmax(self.numpy_input)
        assert theano_result.shape == np_result.shape        
        assert np.allclose(theano_result, np_result)  

    def test_temperature_softmax(self):   
        t = np.asarray(np.random.uniform(2, 10, 1),dtype = theano.config.floatX)     
        theano_result = A.Softmax(self.theano_input, 
                                          temp = t ).eval({self.theano_input: self.numpy_input})
        np_result = self.Softmax(self.numpy_input, temp = t)
        assert theano_result.shape == np_result.shape
        assert np.allclose(theano_result, np_result)                                     

    def test_squared(self):         
        theano_result = A.Squared(self.theano_input).eval({self.theano_input: self.numpy_input})
        np_result = self.Squared(self.numpy_input)
        assert theano_result.shape == np_result.shape                
        assert np.allclose(theano_result, np_result)     