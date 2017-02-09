import theano.tensor as T

def cross_entropy ( a , b ):
    """
    This function produces a point-wise cross entropy error between ``a`` and ``b``

    Args:
        a: first input
        b: second input

    Returns:
        theano shared variable: Computational graph with the error.
    """
    return T.mean(T.nnet.categorical_crossentropy(a.flatten(2),b.flatten(2)))
    # return T.mean(- T.sum(a * T.log(b) + (1 - a) * T.log(1 - b), axis=1))

def l1 ( a, b ):
    """
    This function produces a point-wise L1 error between ``a`` and ``b``

    Args:
        a: first input
        b: second input

    Returns:
        theano shared variable: Computational graph with the error.
    """
    return T.sum( abs (a - b) )

def rmse ( a,  b ):
    """
    This function produces a point-wise root mean squared error error between ``a`` and ``b``

    Args:
        a: first input
        b: second input

    Returns:
        theano shared variable: Computational graph with the error.
    """
    return T.sqrt(T.mean((a - b) ** 2))
