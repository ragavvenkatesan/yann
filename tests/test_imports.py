"""
test_imports.py - Unit tests for YANN toolbox required imports
"""

import imp
import unittest


class TestImports(unittest.TestCase):

    @unittest.skip("progressbar is a Python2 exclusive requirement")
    def test_progressbar(self):
        self.assertTrue(imp.find_module('progressbar'))

    @unittest.skip("skdata is an optional data import not a requirement")
    def test_skdata(self):
        self.assertTrue(imp.find_module('skdata'))

    def test_scipy(self):
        self.assertTrue(imp.find_module('scipy'))

    def test_numpy(self):
        self.assertTrue(imp.find_module('numpy'))

    def test_theano(self):
        self.assertTrue(imp.find_module('theano'))

    def test_yann(self):
        self.assertTrue(imp.find_module('yann'))
