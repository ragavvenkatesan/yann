"""
Notes:
    This code contains three methods. 
        1. A modern reincarnation of LeNet5 for MNIST.
        2. The same Lenet with batchnorms
            2.a. Batchnorm before activations.
            2.b. Batchnorm after activations.

    All these methods are setup for MNIST dataset.

Todo:

    Add detailed comments.
"""

from yann.network import network
from yann.utils.graph import draw_network

def lenet5 ( dataset= None, verbose = 1 ):             
    """
    This function is a demo example of lenet5 from the infamous paper by Yann LeCun. 
    This is an example code. You should study this code rather than merely run it.  
    
    Warning:
        This is not the exact implementation but a modern re-incarnation.

    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.
    """
    optimizer_params =  {        
                "momentum_type"       : 'nesterov',             
                "momentum_params"     : (0.65, 0.97, 30),      
                "optimizer_type"      : 'rmsprop',                
                "id"                  : "main"
                        }

    dataset_params  = {
                            "dataset"   : dataset,
                            "svm"       : False, 
                            "n_classes" : 10,
                            "id"        : 'data'
                      }

    visualizer_params = {
                    "root"       : 'lenet5',
                    "frequency"  : 1,
                    "sample_size": 144,
                    "rgb_filters": True,
                    "debug_functions" : False,
                    "debug_layers": False,  # Since we are on steroids this time, print everything.
                    "id"         : 'main'
                        }       

    # intitialize the network
    net = network(   borrow = True,
                     verbose = verbose )                       
    
    # or you can add modules after you create the net.
    net.add_module ( type = 'optimizer',
                     params = optimizer_params, 
                     verbose = verbose )

    net.add_module ( type = 'datastream', 
                     params = dataset_params,
                     verbose = verbose )

    net.add_module ( type = 'visualizer',
                     params = visualizer_params,
                     verbose = verbose 
                    )
    # add an input layer 
    net.add_layer ( type = "input",
                    id = "input",
                    verbose = verbose, 
                    datastream_origin = 'data', # if you didnt add a dataset module, now is 
                                                 # the time. 
                    mean_subtract = False )
    
    # add first convolutional layer
    net.add_layer ( type = "conv_pool",
                    origin = "input",
                    id = "conv_pool_1",
                    num_neurons = 20,
                    filter_size = (5,5),
                    pool_size = (2,2),
                    activation = 'relu',
                    # regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_1",
                    id = "conv_pool_2",
                    num_neurons = 50,
                    filter_size = (3,3),
                    pool_size = (2,2),
                    activation = 'relu',
                    # regularize = True,
                    verbose = verbose
                    )      


    net.add_layer ( type = "dot_product",
                    origin = "conv_pool_2",
                    id = "dot_product_1",
                    num_neurons = 1250,
                    activation = 'relu',
                    # regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "dot_product_1",
                    id = "dot_product_2",
                    num_neurons = 1250,                    
                    activation = 'relu',  
                    # regularize = True,    
                    verbose = verbose
                    ) 
    
    net.add_layer ( type = "classifier",
                    id = "softmax",
                    origin = "dot_product_2",
                    num_classes = 10,
                    # regularize = True,
                    activation = 'softmax',
                    verbose = verbose
                    )

    net.add_layer ( type = "objective",
                    id = "obj",
                    origin = "softmax",
                    objective = "nll",
                    datastream_origin = 'data', 
                    regularization = (0.0001, 0.0001),                
                    verbose = verbose
                    )
                    
    learning_rates = (0.05, .0001, 0.001)  
    net.pretty_print()  
    # draw_network(net.graph, filename = 'lenet.png')   

    net.cook()

    net.train( epochs = (20, 20), 
               validate_after_epochs = 1,
               training_accuracy = True,
               learning_rates = learning_rates,               
               show_progress = True,
               early_terminate = True,
               patience = 2,
               verbose = verbose)

    net.test(verbose = verbose)
    
# Advaned versions of the CNN
def lenet_maxout_batchnorm_before_activation ( dataset= None, verbose = 1 ):
    """
    This is a version with nesterov momentum and rmsprop instead of the typical sgd. 
    This also has maxout activations for convolutional layers, dropouts on the last
    convolutional layer and the other dropout layers and this also applies batch norm
    to all the layers.  The batch norm is applied by using the ``batch_norm = True`` parameters
    in all layers. This batch norm is applied before activation as is used in the original 
    version of the paper. So we just spice things up and add a bit of steroids to 
    :func:`lenet5`.  This also introduces a visualizer module usage.

    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.    
    """
    optimizer_params =  {        
                "momentum_type"       : 'nesterov',             
                "momentum_params"     : (0.75, 0.95, 30),      
                "optimizer_type"      : 'rmsprop',                
                "id"                  : "main"
                        }

    dataset_params  = {
                            "dataset"   : dataset,
                            "svm"       : False, 
                            "n_classes" : 10,
                            "id"        : 'data'
                    }

    visualizer_params = {
                    "root"       : 'lenet_on_steroids',
                    "frequency"  : 1,
                    "sample_size": 32,
                    "rgb_filters": True,
                    "debug_functions" : False,
                    "debug_layers": False,  # Since we are on steroids this time, print everything.
                    "id"         : 'main'
                        }                      

    net = network(   borrow = True,
                     verbose = verbose )                       
    
    net.add_module ( type = 'optimizer',
                     params = optimizer_params, 
                     verbose = verbose )

    net.add_module ( type = 'datastream', 
                     params = dataset_params,
                     verbose = verbose )

    net.add_module ( type = 'visualizer',
                     params = visualizer_params,
                     verbose = verbose )

    net.add_layer ( type = "input",
                    id = "input",
                    verbose = verbose, 
                    origin = 'data' )
    
    net.add_layer ( type = "conv_pool",
                    origin = "input",
                    id = "conv_pool_1",
                    num_neurons = 40,
                    filter_size = (5,5),
                    pool_size = (2,2),
                    activation = ('maxout', 'maxout', 2),
                    batch_norm = True,           
                    regularize = True,                             
                    verbose = verbose
                    )

    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_1",
                    id = "conv_pool_2",
                    num_neurons = 100,
                    filter_size = (3,3),
                    pool_size = (2,2),
                    activation = ('maxout', 'maxout', 2),
                    batch_norm = True,
                    regularize = True,                    
                    verbose = verbose
                    )      
        
    net.add_layer ( type = "dot_product",
                    origin = "conv_pool_2",
                    id = "dot_product_1",
                    num_neurons = 1250,
                    activation = 'relu',
                    dropout_rate = 0.5,
                    batch_norm = True,
                    regularize = True,                                        
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "dot_product_1",
                    id = "dot_product_2",
                    num_neurons = 1250,                    
                    activation = 'relu',
                    dropout_rate = 0.5,
                    regularize = True,  
                    batch_norm = True,                                      
                    verbose = verbose
                    ) 
    
    net.add_layer ( type = "classifier",
                    id = "softmax",
                    origin = "dot_product_2",
                    num_classes = 10,
                    regularize = True,                    
                    activation = 'softmax',
                    verbose = verbose
                    )

    net.add_layer ( type = "objective",
                    id = "obj",
                    origin = "softmax",
                    objective = "nll",
                    regularization = (0.0001, 0.0001),                                    
                    datastream_origin = 'data', 
                    verbose = verbose
                    )

    learning_rates = (0.05, 0.001, 0.0001)  

    net.cook(  )
    #draw_network(net.graph, filename = 'lenet.png')    
    net.pretty_print()
    
    net.train( epochs = (4, 4), 
               validate_after_epochs = 1,
               visualize_after_epochs = 1,
               training_accuracy = True,
               learning_rates = learning_rates,               
               show_progress = True,
               early_terminate = True,
               verbose = verbose)

    net.test(verbose = verbose)

def lenet_maxout_batchnorm_after_activation ( dataset= None, verbose = 1 ):
    """
    This is a version with nesterov momentum and rmsprop instead of the typical sgd. 
    This also has maxout activations for convolutional layers, dropouts on the last
    convolutional layer and the other dropout layers and this also applies batch norm
    to all the layers. The difference though is that we use the ``batch_norm`` layer to apply
    batch norm that applies batch norm after the activation fo the previous layer.
    So we just spice things up and add a bit of steroids to 
    :func:`lenet5`.  This also introduces a visualizer module usage.

    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.    
    """
    optimizer_params =  {        
                "momentum_type"       : 'nesterov',             
                "momentum_params"     : (0.75, 0.95, 30),      
                "optimizer_type"      : 'rmsprop',                
                "id"                  : "main"
                        }

    dataset_params  = {
                            "dataset"   : dataset,
                            "svm"       : False, 
                            "n_classes" : 10,
                            "id"        : 'data'
                    }

    visualizer_params = {
                    "root"       : 'lenet_bn_after',
                    "frequency"  : 1,
                    "sample_size": 32,
                    "rgb_filters": True,
                    "debug_functions" : False,
                    "debug_layers": False,  # Since we are on steroids this time, print everything.
                    "id"         : 'main'
                        }                      

    net = network(   borrow = True,
                     verbose = verbose )                       
    
    net.add_module ( type = 'optimizer',
                     params = optimizer_params, 
                     verbose = verbose )

    net.add_module ( type = 'datastream', 
                     params = dataset_params,
                     verbose = verbose )

    net.add_module ( type = 'visualizer',
                     params = visualizer_params,
                     verbose = verbose )

    net.add_layer ( type = "input",
                    id = "input",
                    verbose = verbose, 
                    origin = 'data' )
    
    net.add_layer ( type = "conv_pool",
                    origin = "input",
                    id = "conv_pool_1",
                    num_neurons = 40,
                    filter_size = (5,5),
                    pool_size = (2,2),
                    activation = ('maxout', 'maxout', 2),
                    batch_norm = False,           
                    regularize = True,                             
                    verbose = verbose
                    )

    net.add_layer ( type = 'batch_norm',
                    origin = 'conv_pool_1',
                    id = 'batch_norm_after_cp_1',
                    )

    net.add_layer ( type = "convolution",
                    origin = "batch_norm_after_cp_1",
                    id = "conv_pool_2",
                    num_neurons = 100,
                    filter_size = (3,3),
                    pool_size = (2,2),
                    activation = ('maxout', 'maxout', 2),
                    batch_norm = False,
                    regularize = True,                    
                    verbose = verbose
                    )      
        
    net.add_layer ( type = 'batchnorm',
                    origin = 'conv_pool_2',
                    id = 'batch_norm_after_cp_2',
                    )

    net.add_layer ( type = "dot_product",
                    origin = "batch_norm_after_cp_2",
                    id = "dot_product_1",
                    num_neurons = 1250,
                    activation = 'relu',
                    dropout_rate = 0.5,
                    batch_norm = False,
                    regularize = True,                                        
                    verbose = verbose
                    )

    net.add_layer ( type = 'batchnorm',
                    origin = 'dot_product_1',
                    id = 'batch_norm_after_dp_1',
                    )

    net.add_layer ( type = "dot_product",
                    origin = "batch_norm_after_dp_1",
                    id = "dot_product_2",
                    num_neurons = 1250,                    
                    activation = 'relu',
                    dropout_rate = 0.5,
                    regularize = True,  
                    batch_norm = False,                                      
                    verbose = verbose
                    ) 
    
    net.add_layer ( type = 'batch_norm',
                origin = 'dot_product_2',
                id = 'batch_norm_after_dp_2',
                )

    net.add_layer ( type = "classifier",
                    id = "softmax",
                    origin = "batch_norm_after_dp_2",
                    num_classes = 10,
                    regularize = True,                    
                    activation = 'softmax',
                    verbose = verbose
                    )

    net.add_layer ( type = "objective",
                    id = "obj",
                    origin = "softmax",
                    objective = "nll",
                    regularization = (0.0001, 0.0001),                                    
                    datastream_origin = 'data', 
                    verbose = verbose
                    )

    learning_rates = (0.05, 0.001, 0.0001)  

    net.cook( )
    draw_network(net.graph, filename = 'lenet.png')    
    net.pretty_print()
    
    net.train( epochs = (4, 4), 
               validate_after_epochs = 1,
               visualize_after_epochs = 1,
               training_accuracy = True,
               learning_rates = learning_rates,               
               show_progress = True,
               early_terminate = True,
               verbose = verbose)

    net.test(verbose = verbose)

    ## Boiler Plate ## 
if __name__ == '__main__':
    import sys
    dataset = None  
    if len(sys.argv) > 1:
        if sys.argv[1] == 'create_dataset':
            from yann.special.datasets import cook_cifar10 
            from yann.special.datasets import cook_mnist
            
            data = cook_cifar10 (verbose = 2)
            # data = cook_mnist()        
            dataset = data.dataset_location()
        else:
            dataset = sys.argv[1]
    else:
        print "provide dataset"
    
    if dataset is None:
        print " creating a new dataset to run through"
        from yann.special.datasets import cook_cifar10 
        from yann.special.datasets import cook_mnist 
        
        data = cook_cifar10 (verbose = 2)
        # data = cook_mnist()
        dataset = data.dataset_location()

    lenet5 ( dataset, verbose = 2 )
    lenet_maxout_batchnorm_before_activation (dataset, verbose = 2)
    lenet_maxout_batchnorm_after_activation (dataset, verbose = 2)