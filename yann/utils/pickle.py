import cPickle
import theano
from collections import OrderedDict

def pickle(net, filename, verbose = 2):
    """
    This method saves the weights of all the layers.

    Args:
        net: A yann network object
        filename: What is the name of the file to pickle the network as.
        verbose: Blah..
    """
    if verbose >= 3:
        print "... Collecting Parameters"

    params = net.get_params(verbose = verbose)
    if verbose >= 3:
        print "... Dumping netowrk parameters"

    f = open(filename, 'wb')                   
    cPickle.dump(params, f, protocol = cPickle.HIGHEST_PROTOCOL)
    f.close()     

def load(infile, verbose = 2):
    """
    This method loads a pickled network and returns the parameters.

    Args:
        infile: Filename of the network pickled by this pickle method.
    Returns:
        params: A dictionary of parameters.
    """
    if verbose >= 2:
        print ".. Loading the network."

    params = OrderedDict()
    if verbose >= 3:
        print "... Loading netowrk parameters"
    params_np = cPickle.load( open( infile, "rb" ) )
    return shared_params (params_np)

def shared_params (params, verbose = 2):
    """
    This will convert a loaded set of parameters to shared variables that could be 
    passed as ``input_params`` to the ``add_layer`` method.

    Args:
        params: List from ``get_params`` method.
    """
    if verbose >= 3:
        print "... Convering parameters to shared parameters"

    for lyr in params:
        shared_list = list()
        for p in params[lyr]:
            ps = theano.shared( value = p, borrow=True )
            shared_list.append(ps)
        params [lyr] = shared_list                     
    return params