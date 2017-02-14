"""
test_errors.py - Unit tests for YANN core error functions
defined in yann/core/errors.py
"""

import unittest
import numpy as np
import theano
import theano.tensor as T
import yann.core.errors as E


class TestActivations(unittest.TestCase):
    """
    """

    def setUp(self):
        """
        """

        self.theano_input = T.lvector()
        # TO-DO: Find out proper numpy_input dimensionality
        self.numpy_input = np.array([1, 2])

    def test_cross_entropy(self):
        expected_array = 'mean'
        theano_result = E.cross_entropy(self.theano_input, self.numpy_input).eval({self.theano_input: self.numpy_input})
        self.assertEqual(theano_result.shape, expected_array.shape)
        self.assertTrue(np.allclose(theano_result, expected_array))
