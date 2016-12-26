def copy_params (source, destination, borrow = True):
    """
    Internal function that copies paramters maintaining theano shared nature. 

    Args:
        source: Source
        destination: destination

    Notes: 
        Was using deep copy to do this. This seems faster. But can I use ``theano.clone`` ?
    """
    for src, dst in zip(source, destination):
        dst.set_value ( src.get_value (borrow = borrow))