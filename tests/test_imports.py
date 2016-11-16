import imp 

def test_progressbar():
    assert imp.find_module('progressbar')

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

def test_numpy():
    try:
        imp.find_module('numpy')
        numpy_installed = True
    except ImportError:
        numpy_installed = False
    assert numpy_installed

def test_theano():
    assert imp.find_module('theano')


def test_yann():
    try:
        imp.find_module('yann')
        yann_installed = True
    except ImportError:
        yann_installed = False
    assert yann_installed

if __name__ == '__main__':
    pytest.main([__file__])