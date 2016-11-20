import imp 

class TestImports:
    def test_progressbar(self): assert imp.find_module('progressbar')
    def test_skdata(self): assert imp.find_module('skdata')
    def test_scipy(self): assert imp.find_module('scipy')
    def test_numpy(self): assert imp.find_module('numpy')
    def test_theano(self): assert imp.find_module('theano')
    def test_yann(self): assert imp.find_module('yann')
