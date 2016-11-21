from yann.network import network

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
                "momentum_type"       : 'false',             
                                        # false, polyak, nesterov
                "momentum_params"     : (0.5, 0.95, 30),      
                    # (mom_start, momentum_end, momentum_end_epoch)                                                           
                "regularization"      : (0.00, 0.00),       
                        # l1_coeff, l2_coeff, decisiveness (optional)                                
                "optimizer_type"      : 'sgd',                
                                        # sgd, adagrad, rmsprop, adam 
                "id"                  : "main"
                        }

    dataset_params  = {
                            "dataset"   : dataset,
                            "svm"       : False, 
                            "n_classes" : 10,
                            "id"        : 'data'
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
                    activation = 'tanh',
                    verbose = verbose
                    )

    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_1",
                    id = "conv_pool_2",
                    num_neurons = 50,
                    filter_size = (3,3),
                    pool_size = (2,2),
                    activation = 'tanh',
                    verbose = verbose
                    )      
        
    net.add_layer ( type = "dot_product",
                    origin = "conv_pool_2",
                    id = "dot_product_1",
                    num_neurons = 800,
                    activation = 'tanh',
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "dot_product_1",
                    id = "dot_product_2",
                    num_neurons = 800,                    
                    activation = 'tanh',
                    verbose = verbose
                    ) 
    
    net.add_layer ( type = "classifier",
                    id = "softmax",
                    origin = "dot_product_2",
                    num_classes = 10,
                    activation = 'softmax',
                    verbose = verbose
                    )

    net.add_layer ( type = "objective",
                    id = "obj",
                    origin = "softmax",
                    objective = "nll",
                    datastream_origin = 'data', 
                    verbose = verbose
                    )
    # objective provided by classifier layer               
    # nll-negative log likelihood, 
    # cce-categorical cross entropy, 
    # bce-binary cross entropy,
    # hinge-hinge loss 
    learning_rates = (0, 0.1, 0.01, 0.001, 0.0001)  
    # (annealing, initial_learning_rate, ... )
    # net.pretty_print()  # this will print out the network.
    net.cook( optimizer = 'main',
              objective_layer = 'obj',
              datastream = 'data',
              classifier = 'softmax',
              verbose = verbose
              )

    net.train( epochs = (10, 10, 10, 10), 
               validate_after_epochs = 1,
               training_accuracy = True,
               learning_rates = learning_rates,               
               show_progress = True,
               early_terminate = True,
               verbose = verbose)

    net.test(verbose = verbose)

# Advaned version of the CNN
def lenet_on_steroids ( dataset= None, verbose = 1 ):             
    """
    This function is a demo example of lenet5 from the infamous paper by Yann LeCun. 
    This is an example code. You should study this code rather than merely run it.  
    This is a version with nesterov momentum and rmsprop instead of the typical sgd. 
    This also has maxout activations for convolutional layers, dropouts on the last
    convolutional layer and the other dropout layers and this also applies batch norm
    to all the layers.  So we just spice things up and add a bit of steroids to 
    :func:`lenet5`.  This also introduces a visualizer module usage.

    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.    
    """
    optimizer_params =  {        
                "momentum_type"       : 'nesterov',             
                "momentum_params"     : (0.5, 0.95, 30),      
                "regularization"      : (0.0001, 0.0001),       
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
                    "root"       : '.',
                    "frequency"  : 1,
                    "sample_size": 32,
                    "rgb_filters": False,
                    "debug_functions" : True,
                    "debug_layers": True,   # Since we are on steroids this time, print everything.
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
                    datastream_origin = 'data', # if you didnt add a dataset module, now is 
                                                 # the time. 
                    mean_subtract = False )
    
    net.add_layer ( type = "conv_pool",
                    origin = "input",
                    id = "conv_pool_1",
                    num_neurons = 20,
                    filter_size = (5,5),
                    pool_size = (2,2),
                    activation = ('maxout', 'maxout', 2),
                    batch_norm = True,                                        
                    verbose = verbose
                    )

    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_1",
                    id = "conv_pool_2",
                    num_neurons = 50,
                    filter_size = (3,3),
                    pool_size = (2,2),
                    activation = ('maxout', 'maxout', 2),
                    batch_norm = True,
                    dropout_rate = 0.5,
                    verbose = verbose
                    )      
        
    net.add_layer ( type = "dot_product",
                    origin = "conv_pool_2",
                    id = "dot_product_1",
                    num_neurons = 800,
                    activation = 'relu',
                    batch_norm = True,
                    dropout_rate = 0.5,
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "dot_product_1",
                    id = "dot_product_2",
                    num_neurons = 800,                    
                    activation = 'relu',
                    batch_norm = True,
                    dropout_rate = 0.5,
                    verbose = verbose
                    ) 
    
    net.add_layer ( type = "classifier",
                    id = "softmax",
                    origin = "dot_product_2",
                    num_classes = 10,
                    activation = 'softmax',
                    verbose = verbose
                    )

    net.add_layer ( type = "objective",
                    id = "obj",
                    origin = "softmax",
                    objective = "cce",
                    datastream_origin = 'data', 
                    verbose = verbose
                    )

    learning_rates = (0.001, 0.01, 0.001, 0.0001, 0.00001)  

    net.cook( optimizer = 'main',
              objective_layer = 'obj',
              datastream = 'data',
              classifier = 'softmax',
              verbose = verbose
              )

    net.train( epochs = (20, 20, 10, 5), 
               validate_after_epochs = 1,
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
            from yann.utils.dataset import cook_mnist  
            data = cook_mnist (verbose = 2)
            dataset = data.dataset_location()
        else:
            dataset = sys.argv[1]
    else:
        print "provide dataset"
    
    if dataset is None:
        print " creating a new dataset to run through"
        from yann.utils.dataset import cook_mnist  
        data = cook_mnist (verbose = 2)
        dataset = data.dataset_location()

    #lenet5 ( dataset, verbose = 2 )
    lenet_on_steroids (dataset, verbose = 2)
     

