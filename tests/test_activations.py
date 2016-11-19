"""
Just a dummy test so that coverage won't be 100%
"""
import unittest

import numpy as np
import theano.tensor as T
import yann.core.activations as A

class TestActivations(unittest.TestCase):
    
    def setUp(self):
        run_activations = ['Abs','ReLU','Sigmoid','Tanh','Softmax','Squared']
        for activation in run_activations:
            setattr(TestPreReqs, 'test_activation_%d' % k, test_activation(activation))

    def Abs(self, x):
        return np.abs(x)

    def ReLU(self, x):
        return x * (x > 0)

    def Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def Tanh(self, x):
        return np.tanh(x)

    def Softmax(self, x):
        return (np.exp(x).T / np.exp(x).sum(-1)).T

    def Squared(self, x):
        return x**2

    def test_activation(self, activiation):
        theano_test_function = getattr(yann.core.activations,activation)
        np_test_function = getattr(self, activation)
        theano_input = T.matrix()
        numpy_input = np.random.uniform(-4, 4, (5, 5))  # Create some 5X5 matrix randomly 
        theano_result = theano_test_function(theano_input).eval({theano_input: numpy_input})
        np_result = np_test_function(X0)
        self.assertTrue np.allclose(theano_result, np_result)

"""
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestImports)
    unittest.TextTestRunner(verbosity=2).run(suite)
"""