import imp 

""""
These are not really important to test.
def test_progressbar():
    try:
        imp.find_module('progressbar')
        progressbar_installed = True
    except ImportError:
        progressbar_installed = False
    assert progressbar_installed

def test_skdata():
    try:
        imp.find_module('skdata')
        skdata_installed = True
    except ImportError:
        skdata_installed = False
    assert skdata_installed

def test_scipy():
    try:
        imp.find_module('scipy')
        scipy_installed = True
    except ImportError:
        scipy_installed = False
    assert scipy_installed

"""

def test_numpy():
    try:
        imp.find_module('numpy')
        numpy_installed = True
    except ImportError:
        numpy_installed = False
    assert numpy_installed

def test_theano():
    try:
        imp.find_module('theano')
        theano_installed = True
    except ImportError:
        theano_installed = False
    assert theano_installed

def test_yann():
    try:
        imp.find_module('yann')
        yann_installed = True
    except ImportError:
        yann_installed = False
    assert yann_installed

if __name__ == '__main__':
    pytest.main([__file__])