import unittest
import numpy as np
import theano
import yann.core.errors as E


class TestErrors(unittest.TestCase):

    def setUp(self):
        self.numpy_input1 = np.array([[1, 2, 3, 4, 5],[1, 2, 3, 4, 5],[1, 2, 3, 4, 5],[1, 2, 3, 4, 5],[1, 2, 3, 4, 5]],dtype=theano.config.floatX)
        self.numpy_input2 = np.array([[5, 2, 8, 4, 10], [5, 2, 3, 4, 5], [7, 2, 3, 4, 5],[1, 2, 3, 4, 5],[1, 2, 3, 4, 5]],dtype=theano.config.floatX)
        self.theano_input1 = theano.shared(self.numpy_input1, broadcastable=(False,False))
        self.theano_input2 = theano.shared(self.numpy_input2, broadcastable=(False,False))

    def test_cross_entropy(self):
        expected_value = -20.9825484349
        theano_result = E.cross_entropy(self.theano_input1,self.theano_input2).eval()
        self.assertEqual(round(theano_result,10), expected_value)

    def test_l1(self):
        expected_value = 24.0
        theano_result = E.l1(self.theano_input1,self.theano_input2).eval()
        self.assertEqual(round(theano_result,1), expected_value)

    def test_rmse(self):
        expected_value = 2.1725560982
        theano_result = E.rmse(self.theano_input1,self.theano_input2).eval()
        self.assertEqual(round(theano_result,10), expected_value)
