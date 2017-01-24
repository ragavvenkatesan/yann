def copy_params (source, destination, borrow = True, verbose = 2):
    """
    Internal function that copies paramters maintaining theano shared nature. 

    Args:
        source: Source
        destination: destination

    Notes: 
        Was using deep copy to do this. This seems faster. But can I use ``theano.clone`` ?
    """
    if verbose >=3:
        print "... Copying paramters"

    for src, dst in zip(source, destination):

        if verbose>=3:
            print "... source shape: " + str(src.get_value(borrow = True).shape)
            print "... destination shape: " + str(dst.get_value(borrow = True).shape)            
        dst.set_value ( src.get_value (borrow = borrow))