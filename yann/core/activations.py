
import theano.tensor as T
from math import floor
    
#### Exponential linear unit
def Elu(x, alpha = 1):
    """ 
    Exponential Linear Units. 
    
    Applies point-wise ela to the input supplied. ``alpha`` is defualt to ``0``. 
    Supplying a value to ``alpha`` would make this a leay Elu. 

    Notes: 
        Reference :Clevert, Djork-Arné, Thomas Unterthiner, and Sepp Hochreiter. "Fast and accurate 
         deep network learning by exponential linear units (elus)." arXiv preprint arXiv:1511.07289 
         (2015).
    
    Args:
        x: could be a ``theano.tensor`` or a ``theano.shared`` or ``numpy`` arrays or 
            ``python lists``.             
        alpha: should be a ``float``. Default is ``1``.
        
    Returns: 
        same as input: returns a point-wise rectified output.        
    """ 
    y = T.nnet.elu(x,alpha = alpha)
    return(y)

#### rectified linear unit
def ReLU(x, alpha = 0):
    """ 
    Rectified Linear Units. 
    
    Applies point-wise rectification to the input supplied. ``alpha`` is defualt to ``0``. 
    Supplying a value to ``alpha`` would make this a leay ReLU. 

    Notes:
        Reference: Nair, Vinod, and Geoffrey E. Hinton. "Rectified linear units improve restricted
         boltzmann machines." Proceedings of the 27th International Conference on Machine Learning 
         (ICML-10). 2010.
    
    Args:
        x: could be a ``theano.tensor`` or a ``theano.shared`` or ``numpy`` arrays or 
            ``python lists``.             
        alpha: should be a ``float``.
        
    Returns: 
        same as input: returns a point-wise rectified output.        
    """ 
    y = T.nnet.relu(x,alpha = alpha)
    return(y)
    
#### sigmoid
def Sigmoid(x):
    """ 
    Sigmoid Units. 

    Applies point-wise sigmoid  to the input supplied.  
    
    Args:
        x: could be a ``theano.tensor`` or a ``theano.shared`` or ``numpy`` arrays or 
            ``python lists``.             
    Returns: 
        same as input: returns a point-wise sigmoid output of the same shape as the input.
    """ 
    y = T.nnet.sigmoid(x)
    return(y)
    
#### tanh
def Tanh(x):
    """
    Tanh Units. 

    Applies point-wise hyperbolic tangent  to the input supplied.
    
    Args:
        x: could be a ``theano.tensor`` or a ``theano.shared`` or ``numpy`` arrays or 
                    ``python lists``.             
    Returns: 
        same as input: returns a point-wise hyperbolic tangent output.
    """
    y = T.tanh(x)
    return(y)
    
#### softmax
def Softmax(x, temp = 1):
    """ 
    Softmax Units. 

    Applies row-wise softmax  to the input supplied.
            
    Args:
        x: could be a ``theano.tensor`` or a ``theano.shared`` or ``numpy`` arrays or 
            ``python lists``.
        temp: temperature of type ``float``. Mainly used during distillation, normal 
                softmax prefer ``T=1``. 
    Notes:
        Refer [3] for details.
    
        .. [#]  Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in
                a neural network." arXiv preprint arXiv:1503.02531 (2015).       
                        
    Returns: 
        same as input: returns a row-wise softmax output of the same shape as the input.
    """
    if temp != 1:
        expo = T.exp(x / float(temp)) # at this moment this is mini_batch_size X num_classes.
        normalizer = T.sum(expo,axis=1,keepdims=True)  # at this moment this is mini_batch_size X 1.
        T.addbroadcast(normalizer,1)    
        return expo / normalizer
    else:
        return T.nnet.softmax(x)               
   
### Abs    
def Abs(x):
    """ 
    Absolute value Units. 
    
    Applies point-wise absolute value  to the input supplied.
    
    Args:
        x: could be a ``theano.tensor`` or a ``theano.shared`` or ``numpy`` arrays or 
            ``python lists``.             
    Returns: 
        same as input: returns a absolute output of the same shape as the input.
    """ 
    return(abs(x))
 
### Squared
def Squared(x):
    """ 
    Squared Units. 

    Applies point-wise squaring to the input supplied.
    
    Args:
        x: could be a ``theano.tensor`` or a ``theano.shared`` or ``numpy`` arrays or 
            ``python lists``.             
    Returns: 
        same as input: returns a squared output of the same shape as the input.
    """
    return(x ** 2)    

### maxouts   
def _max1d_stride (x,i,stride):
    """ 
    max units. These are used interally by the maxout activations.  

    Applies strided maximum to the input supplied on the second dimension.        
    
    Args:
        x: could be a ``theano.tensor`` or a ``theano.shared`` or ``numpy`` arrays or 
            ``python lists``. size of the argument must strictly be of two dimensions that 
            is windowed runnable through ``stride``.
            Second dimension must be the channels to maxout from
        i: is a range variable. It tells from which index the maxing must begin from
        stride: strides of the maxouts (default is maxoutsize)

    Returns: 
        same as input: returns the max of that range.
    """ 
    return x[:,i::stride]
    
def _max2d_stride (x,i,stride):
    """ 
    max units. These are used interally by the maxout activations.  
    
    Applies strided maximum to the input supplied on the second dimension.        
    
    Args:
        x: could be a ``theano.tensor`` or a ``theano.shared`` or ``numpy`` arrays or 
            ``python lists``. size of the argument must strictly be of four dimensions that
            is windowed runnable through ``stride``. 
            Second dimension must be the channels to maxout from
            
        i: is a range variable. It tells from which index the maxing must begin from
        stride: strides of the maxouts (default is maxoutsize)

    Returns: 
        same as input: returns the max of that range.
    """ 
    return x[:,i::stride,:,:]
    

def Maxout(x, maxout_size, input_size, type = 'maxout', dimension = 1):
    """ 
    Function performs the maxout activation.
    You can import all these functions and supply the fuctions as arguments to functions 
    that use ``activation`` variable as an input. Refer to the mnist example in the 
    modelzoo for how to do this.

    Args:

        x: could be a ``theano.tensor`` or a ``theano.shared`` or ``numpy`` arrays or 
            ``python lists``. Size of the argument must strictly be windowed runnable 
            through ``stride``. Second dimension must be the channels to maxout from
                    
        maxout_size: is the size of the window to stride through
        
        input_size: is number of nodes in the input
        dimension: If ``1`` perform MLP layer maxout, input must be two dimensional. 
                   If ``2`` perform CNN layer maxout, input must be four dimensional. 
                    
        type: If ``maxout`` perform, [1]         
              If ``meanout`` or ``mixedout`` perform, meanout or mixed out respectively 
              from [2] 
                        
    .. [#]  Yu, Dingjun, et al. "Mixed Pooling for Convolutional Neural Networks." Rough
            Sets and Knowledge Technology. Springer International Publishing, 
            2014. 364-375.
    .. [#]  Ian Goodfellow et al. " Maxout Networks " on arXiv. (jmlr).                        
            
    Returns:
    
        ``theano.tensor4``:  1. ``theano.tensor4`` output, Output that could be provided 
                                    as output to the next layer or to other convolutional 
                                    layer options. the size of the output depends on border
                                    mode and subsample operation performed.  
            
                                2. ``tuple``, Number of feature maps after maxout is applied
    """   
    if dimension == 1:
        maxing = _max1d_stride
        output_shape = (input_size[0], input_size[1]/maxout_size)   
    elif dimension == 2:
        maxing = _max2d_stride
        output_shape = (input_size[0], input_size[1]/maxout_size, input_size[2], input_size[3])
        
    if type == 'maxout':  # Do maxout network.
        maxout_out = None        
        for i in xrange(maxout_size):
            temp = maxing(x,i,maxout_size)                                   
            if maxout_out is None:                                              
                maxout_out = temp                                                  
            else:                                                               
                maxout_out = T.maximum(maxout_out, temp)  
        output = maxout_out 
                      
    elif type == 'meanout':  # Do meanout network.
        maxout_out = None                                                       
        for i in xrange(maxout_size):                                            
            temp = maxing(x,i,maxout_size)                                   
            if maxout_out is None:                                              
                maxout_out = temp                                                  
            else:                                                               
                maxout_out = (maxout_out*(i+1)+temp)/(i+2)   
        output = maxout_out

    elif type == 'mixedout': # Do mixout network.
        maxout_out = None
        maxout_mean = None
        maxout_max  = None 
        for i in xrange(maxout_size):                                            
            temp = maxing(x,i,maxout_size)                                   
            if maxout_mean is None:                                              
                maxout_mean = temp
                maxout_max = temp
                maxout_out = temp
            else:                                                               
                maxout_mean = (maxout_out*(i+1)+temp)/(i+2) 
                maxout_max = T.maximum(maxout_out, temp)                 
        lambd      = srng.uniform( maxout_mean.shape, low=0.0, high=1.0)
        maxout_out = lambd * maxout_max + (1 - lambd) * maxout_mean
        output = maxout_out   
    
    return (output, output_shape)               
    
if __name__ == '__main__':
    pass             