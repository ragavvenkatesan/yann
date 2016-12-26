"""
TODO: 

    Testing module does not work still 
    
"""

from yann.network import network

def autoencoder ( dataset= None, verbose = 1 ):             
    """
    This function is a demo example of a sparse autoencoder. 
    This is an example code. You should study this code rather than merely run it.  

    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.
    """
    optimizer_params =  {        
                "momentum_type"       : 'nesterov',             
                                        # false, polyak, nesterov
                "momentum_params"     : (0.5, 0.95, 30),      
                    # (mom_start, momentum_end, momentum_end_epoch)                                                           
                "regularization"      : (0.000, 0.001),       
                        # l1_coeff, l2_coeff, decisiveness (optional)                                
                "optimizer_type"      : 'rmsprop',                
                                        # sgd, adagrad, rmsprop, adam 
                "id"                  : "main"
                        }

    dataset_params  = {
                            "dataset"   : dataset,
                            "type"      : 'x',
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
                    mean_subtract = True )

    net.add_layer ( type = "flatten",
                    origin = "input",
                    id = "flatten",
                    verbose = verbose
                    )


    net.add_layer ( type = "dot_product",
                    origin = "flatten",
                    id = "encoder_1",
                    num_neurons = 128,
                    activation = 'relu',
                    batch_norm = True,
                    dropout_rate = 0.5,
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "encoder_1",
                    id = "encoder_2",
                    num_neurons = 32,
                    activation = 'relu',
                    batch_norm = True,
                    dropout_rate = 0.5,
                    verbose = verbose
                    )                    

    net.add_layer ( type = "dot_product",
                    origin = "encoder_2",
                    id = "decoder_1",
                    num_neurons = 64,
                    activation = 'relu',
                    batch_norm = True,
                    dropout_rate = 0.5,
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "decoder_1",
                    id = "decoder_2",
                    num_neurons = 784,
                    activation = 'relu',
                    batch_norm = True,
                    dropout_rate = 0.5,                    
                    verbose = verbose
                    )                    

    net.add_layer ( type = "unflatten",
                    origin = "decoder_2",
                    id = "unflatten",
                    shape = (28,28,1),
                    verbose = verbose
                    )

    net.add_layer ( type = "merge",
                    origin = ("input","unflatten"),
                    id = "merge",
                    layer_type = "error",
                    error = "l2",
                    verbose = verbose)

    net.add_layer ( type = "objective",
                    origin = "merge",
                    id = "obj",
                    objective = None,
                    layer_type = 'generator',
                    verbose = verbose
                    )

    learning_rates = (0.05, 0.1, 0.01)  

    net.cook( optimizer = 'main',
              objective_layer = 'obj',
              datastream = 'data',
              generator = 'merge',
              learning_rates = learning_rates,
              verbose = verbose
              )

    from yann.utils.graph import draw_network
    draw_network(net.graph, filename = 'autoencoder.png')    
    net.pretty_print()

    net.train( epochs = (70, 30), 
               validate_after_epochs = 1,
               training_accuracy = True,
               show_progress = True,
               early_terminate = True,
               verbose = verbose)
               
    net.test( show_progress = True,
               verbose = verbose)
                           
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

    autoencoder ( dataset, verbose = 2 )