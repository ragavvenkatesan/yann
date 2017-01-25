"""
Implementation from 

Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, 
Aaron Courville, and Yoshua Bengio. "Generative adversarial nets." In Advances in Neural Information
 Processing Systems, pp. 2672-2680. 2014.
"""
from yann.special.gan import gan
from theano import tensor as T

def simple_gan ( dataset= None, verbose = 1 ):             
    """
    This function is a demo example of a sparse autoencoder. 
    This is an example code. You should study this code rather than merely run it.  

    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.
    """
    optimizer_params =  {        
                "momentum_type"       : 'nesterov',             
                "momentum_params"     : (0.65, 0.95, 30),      
                "regularization"      : (0.000, 0.000),       
                "optimizer_type"      : 'rmsprop',                
                "id"                  : "main"
                        }

    dataset_params  = {
                            "dataset"   : dataset,
                            "type"      : 'xy',
                            "id"        : 'data'
                    }

    visualizer_params = {
                    "root"       : '.',
                    "frequency"  : 1,
                    "sample_size": 225,
                    "rgb_filters": False,
                    "debug_functions" : False,
                    "debug_layers": True,  
                    "id"         : 'main'
                        }  
                      
    # intitialize the network
    net = gan (      borrow = True,
                     verbose = verbose )                       
    
    net.add_module ( type = 'datastream', 
                     params = dataset_params,
                     verbose = verbose )    
    
    net.add_module ( type = 'visualizer',
                     params = visualizer_params,
                     verbose = verbose 
                    ) 

    #z - latent space created by random gaussian layer
    net.add_layer(type = 'random',
                        id = 'z',
                        num_neurons = (500,128), 
                        distribution = 'normal',
                        mu = 0,
                        sigma = 1,
                        verbose = verbose)
    
    #x - inputs come from dataset 1 X 784
    net.add_layer ( type = "input",
                    id = "x",
                    verbose = verbose, 
                    datastream_origin = 'data', # if you didnt add a dataset module, now is 
                                                 # the time. 
                    mean_subtract = False )

    #G(z) contains params theta_g - 100 X 784 - creates images of 1 X 784
    net.add_layer ( type = "dot_product",
                    origin = "z",
                    id = "G(z)",
                    num_neurons = 784,
                    activation = 'tanh',
                    verbose = verbose
                    )

    #D(x) - Contains params theta_d - 784 X 256 - first layer of D, creates features 1 X 256. 
    net.add_layer ( type = "dot_product",
                    id = "D(x)",
                    origin = "x",
                    num_neurons = 512,
                    activation = ('maxout','maxout',2),
                    verbose = verbose
                    )

    #D(x) - Contains the same params theta_d from above
    #784 X 256 - Takes inputs from G(z) and outputs features 1 X 256. 
    net.add_layer ( type = "dot_product",
                    id = "D(G(z))",
                    origin = "G(z)",
                    num_neurons = 512,
                    input_params = net.dropout_layers["D(x)"].params, # must be the same params, 
                                                        # this way it remains the same network.
                    activation = ('maxout','maxout',2),
                    verbose = verbose
                    )

    #C(D(G(z))) fake - the classifier for fake/real that always predicts fake 
    net.add_layer ( type = "dot_product",
                    id = "fake",
                    origin = "D(G(z))",
                    num_neurons = 1,
                    activation = 'sigmoid',
                    verbose = verbose
                    )

    #C(D(x)) - This is the opposite of C(D(G(z))), real
    net.add_layer ( type = "dot_product",
                    id = "real",
                    origin = "D(x)",
                    num_neurons = 1,
                    input_params = net.dropout_layers["fake"].params, # Again share their parameters
                    activation = 'sigmoid',
                    verbose = verbose
                    )

    #C(D(x)) - This is the opposite of C(D(G(z))), real
    net.add_layer ( type = "classifier",
                    id = "softmax",
                    origin = "D(x)",
                    num_classes = 10,
                    activation = 'softmax',
                    verbose = verbose
                    )


    # objective layers 
    # fake objective 
    net.add_layer ( type = "objective",
                    id = "fake_obj",
                    origin = "fake", # this is useless anyway.
                    layer_type = 'value',
                    objective = -T.mean(T.log(1-net.layers['fake'].output)),
                    datastream_origin = 'data', 
                    verbose = verbose
                    )

    #real objective 
    net.add_layer ( type = "objective",
                    id = "real_obj",
                    origin = "real", # this is useless anyway.
                    layer_type = 'value',
                    objective = -T.mean(T.log(net.layers['real'].output)),
                    datastream_origin = 'data', 
                    verbose = verbose
                    )                

    #softmax objective.
    net.add_layer ( type = "objective",
                    id = "classifier_obj",
                    origin = "softmax",
                    objective = "cce",
                    datastream_origin = 'data', 
                    verbose = verbose
                    )

    from yann.utils.graph import draw_network
    draw_network(net.graph, filename = 'gan.png')    
    net.pretty_print()
    
    net.cook (  objective_layers = ["classifier_obj","real_obj","fake_obj"],
                optimizer_params = optimizer_params,
                generator_layers = ["G(z)"], 
                discriminator_layers = ["D(x)"],
                classifier_layers = ["D(x)","softmax"],
                softmax_layer = "softmax",
                verbose = verbose )
                    
    learning_rates = (0.05, 0.1)  

    net.train( epochs = (30), 
               k = 1,  # refer to Ian Goodfellow's paper Algorithm 1.
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
            from yann.special.datasets import cook_mnist  
            data = cook_mnist (verbose = 2)
            dataset = data.dataset_location()
        else:
            dataset = sys.argv[1]
    else:
        print "provide dataset"
    
    if dataset is None:
        print " creating a new dataset to run through"
        from yann.special.datasets import cook_mnist  
        data = cook_mnist (verbose = 2)
        dataset = data.dataset_location()

    simple_gan ( dataset, verbose = 2 )