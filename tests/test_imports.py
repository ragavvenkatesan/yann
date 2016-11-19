"""
Just a dummy test I'm writing to get used to the idea of unittesting. I've never 
written unit tests before, so this is my template for them.
"""
import imp 
import unittest

class TestImports(unittest.TestCase):
    
    def test_progressbar(self):
        self.assertTrue(imp.find_module('progressbar'))

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
"""
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestImports)
    unittest.TextTestRunner(verbosity=2).run(suite)
"""