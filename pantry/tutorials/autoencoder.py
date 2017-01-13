"""
TODO:

    Need a validation and testing thats better than just measuring rmse. Can't find something great.
    
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
    dataset_params  = {
                            "dataset"   : dataset,
                            "type"      : 'x',
                            "id"        : 'data'
                    }

    visualizer_params = {
                    "root"       : '.',
                    "frequency"  : 1,
                    "sample_size": 32,
                    "rgb_filters": False,
                    "debug_functions" : False,
                    "debug_layers": True,  
                    "id"         : 'main'
                        }  
                      
    # intitialize the network
    net = network(   borrow = True,
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
                    mean_subtract = True )

    net.add_layer ( type = "flatten",
                    origin = "input",
                    id = "flatten",
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "flatten",
                    id = "encoder",
                    num_neurons = 64,
                    activation = 'tanh',
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "encoder",
                    id = "decoder",
                    num_neurons = 784,
                    activation = 'tanh',
                    input_params = [net.layers['encoder'].w.T, None],
                    # Use the same weights but transposed for decoder. 
                    learnable = False,                    
                    # because we don't want to learn the weights of somehting already used in 
                    # an optimizer, when reusing the weights, always use learnable as False                    
                    verbose = verbose
                    )           

    net.add_layer ( type = "unflatten",
                    origin = "decoder",
                    id = "unflatten",
                    shape = (28,28,1),
                    verbose = verbose
                    )

    net.add_layer ( type = "merge",
                    origin = ("input","unflatten"),
                    id = "merge",
                    layer_type = "error",
                    error = "rmse",
                    verbose = verbose)

    net.add_layer ( type = "objective",
                    origin = "merge",
                    id = "obj",
                    objective = None,
                    layer_type = 'generator',
                    verbose = verbose
                    )

    learning_rates = (0, 0.1, 0.01)  

    net.cook( objective_layer = 'obj',
              datastream = 'data',
              generator = 'merge',
              learning_rates = learning_rates,
              verbose = verbose
              )

    from yann.utils.graph import draw_network
    # draw_network(net.graph, filename = 'autoencoder.png')    
    # net.pretty_print()

    net.train( epochs = (10, 10), 
               validate_after_epochs = 1,
               training_accuracy = True,
               show_progress = True,
               early_terminate = True,
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