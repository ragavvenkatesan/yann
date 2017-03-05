"""
Implementation from 

Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, 
Aaron Courville, and Yoshua Bengio. "Generative adversarial nets." In Advances in Neural Information
 Processing Systems, pp. 2672-2680. 2014.
"""
from yann.special.gan import gan 
from theano import tensor as T 

def cook_mnist(  verbose = 1, **kwargs):
    """
	Wrapper to cook mnist dataset specifically for the gan. Will take as input,

	Args:
        save_directory: which directory to save the cooked dataset onto.
        dataset_parms: default is the dictionary. Refer to :mod:`setup_dataset`		
        preprocess_params: default is the dictionary. Refer to :mod:`setup_dataset`

    Notes:
        By default, this will create a dataset that is not mean-subtracted.
	"""
    from yann.utils.dataset import setup_dataset

    if not 'data_params' in kwargs.keys():

        data_params = {
        "source"             : 'skdata',                                   
        "name"               : 'mnist',    
        "location"			 : '',                                      
        "mini_batch_size"    : 100,                                     
        "mini_batches_per_batch" : (500, 100, 100), 
        "batches2train"      : 1,                                      
        "batches2test"       : 1,                                      
        "batches2validate"   : 1,                                        
        "height"             : 28,                                       
        "width"              : 28,                                       
        "channels"           : 1  }    

    else:
        data_params = kwargs['data_params']

    if not 'preprocess_params' in kwargs.keys():

        # parameters relating to preprocessing.
        preprocess_params = { 
                "normalize"     : True,
                "ZCA"           : False,
                "grayscale"     : False,
                "zero_mean" 	: True,
            }
    else:
        preprocess_params = kwargs['preprocess_params']

    if not 'save_directory' in kwargs.keys():
        save_directory = '_datasets'
    else:
        save_directory = kwargs ['save_directory']

    dataset = setup_dataset(dataset_init_args = data_params,
            save_directory = save_directory,
            preprocess_init_args = preprocess_params,
            verbose = 3)
    return dataset

def shallow_gan ( dataset= None, verbose = 1 ):             
    """
    This function is a demo example of a generative adversarial network. 
    This is an example code. You should study this code rather than merely run it.  

    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.
    """
    optimizer_params =  {        
                "momentum_type"       : 'polyak',             
                "momentum_params"     : (0.65, 0.9, 50),      
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

    #z - latent space created by random layer
    net.add_layer(type = 'random',
                        id = 'z',
                        num_neurons = (100,32), 
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

    net.add_layer ( type = "dot_product",
                    origin = "z",
                    id = "G(z)",
                    num_neurons = 784,
                    activation = 'tanh',
                    verbose = verbose
                    )  # This layer is the one that creates the images.
        
    #D(x) - Contains params theta_d creates features 1 X 800. 
    net.add_layer ( type = "dot_product",
                    id = "D(x)",
                    origin = "x",
                    num_neurons = 800,
                    activation = 'relu',
                    regularize = True,                                                         
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    id = "D(G(z))",
                    origin = "G(z)",
                    input_params = net.dropout_layers["D(x)"].params, 
                    num_neurons = 800,
                    activation = 'relu',
                    regularize = True,
                    verbose = verbose
                    )


    #C(D(x)) - This is the opposite of C(D(G(z))), real
    net.add_layer ( type = "dot_product",
                    id = "real",
                    origin = "D(x)",
                    num_neurons = 1,
                    activation = 'sigmoid',
                    verbose = verbose
                    )

    #C(D(G(z))) fake - the classifier for fake/real that always predicts fake 
    net.add_layer ( type = "dot_product",
                    id = "fake",
                    origin = "D(G(z))",
                    num_neurons = 1,
                    activation = 'sigmoid',
                    input_params = net.dropout_layers["real"].params, # Again share their parameters                    
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
    # discriminator objective 
    net.add_layer (type = "tensor",
                    input =  - 0.5 * T.mean(T.log(net.layers['real'].output)) - \
                                  0.5 * T.mean(T.log(1-net.layers['fake'].output)),
                    input_shape = (1,),
                    id = "discriminator_task"
                    )

    net.add_layer ( type = "objective",
                    id = "discriminator_obj",
                    origin = "discriminator_task",
                    layer_type = 'value',
                    objective = net.dropout_layers['discriminator_task'].output,
                    datastream_origin = 'data', 
                    verbose = verbose
                    )
    #generator objective 
    net.add_layer (type = "tensor",
                    input =  - 0.5 * T.mean(T.log(net.layers['fake'].output)),
                    input_shape = (1,),
                    id = "objective_task"
                    )
    net.add_layer ( type = "objective",
                    id = "generator_obj",
                    layer_type = 'value',
                    origin = "objective_task",
                    objective = net.dropout_layers['objective_task'].output,
                    datastream_origin = 'data', 
                    verbose = verbose
                    )   

    #softmax objective.    
    net.add_layer ( type = "objective",
                    id = "classifier_obj",
                    origin = "softmax",
                    objective = "nll",
                    layer_type = 'discriminator',
                    datastream_origin = 'data', 
                    verbose = verbose
                    )
    
    from yann.utils.graph import draw_network
    draw_network(net.graph, filename = 'gan.png')    
    net.pretty_print()
    
    net.cook (  objective_layers = ["classifier_obj", "discriminator_obj", "generator_obj"],
                optimizer_params = optimizer_params,
                discriminator_layers = ["D(x)"],
                generator_layers = ["G(z)"], 
                classifier_layers = ["D(x)", "softmax"],                                                
                softmax_layer = "softmax",
                game_layers = ("fake", "real"),
                verbose = verbose )
                    
    learning_rates = (0.05, 0.01 )  

    net.train( epochs = (20), 
               k = 2,  
               pre_train_discriminator = 3,
               validate_after_epochs = 1,
               visualize_after_epochs = 1,
               training_accuracy = True,
               show_progress = True,
               early_terminate = True,
               verbose = verbose)
                           
    return net


def deep_gan (dataset, verbose = 1 ):
    """
    This function is a demo example of a generative adversarial network. 
    This is an example code. You should study this code rather than merely run it.  

    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.

    Returns:
        net: A Network object.

    Notes:
        This is not setup properly therefore does not learn at the moment. This network here mimics
        Ian Goodfellow's original code and implementation for MNIST adapted from his source code:
        https://github.com/goodfeli/adversarial/blob/master/mnist.yaml .It might not be a perfect 
        replicaiton, but I tried as best as I could.
    """
    if verbose >=2:
        print (".. Creating a GAN network")

    optimizer_params =  {        
                "momentum_type"       : 'polyak',             
                "momentum_params"     : (0.5, 0.7, 20),      
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

    #z - latent space created by random layer
    net.add_layer(type = 'random',
                        id = 'z',
                        num_neurons = (100,10), 
                        distribution = 'normal',
                        mu = 0,
                        sigma = 1,
                        # limits = (0,1),
                        verbose = verbose)
    
    #x - inputs come from dataset 1 X 784
    net.add_layer ( type = "input",
                    id = "x",
                    verbose = verbose, 
                    datastream_origin = 'data', # if you didnt add a dataset module, now is 
                                                # the time. 
                    mean_subtract = False )

    # Generator layers
    net.add_layer ( type = "dot_product",
                    origin = "z",
                    id = "G1",
                    num_neurons = 1200,
                    activation = 'relu',
                    # batch_norm = True,
                    verbose = verbose
                    ) 

    net.add_layer ( type = "dot_product",
                    origin = "G1",
                    id = "G2",
                    num_neurons = 1200,
                    activation = 'relu',
                    # batch_norm = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "G2",
                    id = "G(z)",
                    num_neurons = 784,
                    activation = 'tanh',
                    verbose = verbose
                    )  # This layer is the one that creates the images.
        
    #D(x) - Contains params theta_d creates features 1 X 800. 
    # Discriminator Layers
    net.add_layer ( type = "unflatten",
                    origin = "G(z)",
                    id = "G(z)-unflattened",
                    shape = (28,28),
                    verbose = verbose )


    net.add_layer ( type = "dot_product",
                    id = "D1-x",
                    origin = "x",
                    num_neurons = 1200,
                    activation = ('maxout','maxout',5),
                    regularize = True,  
                    # batch_norm = True,
                    # dropout_rate = 0.5,                                                       
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    id = "D1-z",
                    origin = "G(z)-unflattened",
                    input_params = net.dropout_layers["D1-x"].params, 
                    num_neurons = 1200,
                    activation = ('maxout','maxout',5),
                    regularize = True,
                    # batch_norm = True,
                    # dropout_rate = 0.5,                       
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    id = "D2-x",
                    origin = "D1-x",
                    num_neurons = 1200,
                    activation = ('maxout','maxout',5),
                    regularize = True,       
                    # batch_norm = True,
                    # dropout_rate = 0.5,                                                                         
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    id = "D2-z",
                    origin = "D1-z",
                    input_params = net.dropout_layers["D2-x"].params, 
                    num_neurons = 1200,
                    activation = ('maxout','maxout',5),
                    regularize = True,
                    # dropout_rate = 0.5,          
                    # batch_norm = True,                    
                    verbose = verbose
                    )

    #C(D(x)) - This is the opposite of C(D(G(z))), real
    net.add_layer ( type = "dot_product",
                    id = "D(x)",
                    origin = "D2-x",
                    num_neurons = 1,
                    activation = 'sigmoid',
                    verbose = verbose
                    )

    #C(D(G(z))) fake - the classifier for fake/real that always predicts fake 
    net.add_layer ( type = "dot_product",
                    id = "D(G(z))",
                    origin = "D2-z",
                    num_neurons = 1,
                    activation = 'sigmoid',
                    input_params = net.dropout_layers["D(x)"].params,                   
                    verbose = verbose
                    )

    
    #C(D(x)) - This is the opposite of C(D(G(z))), real
    net.add_layer ( type = "classifier",
                    id = "softmax",
                    origin = "D2-x",
                    num_classes = 10,
                    activation = 'softmax',
                    verbose = verbose
                )
    
    # objective layers 
    # discriminator objective 
    net.add_layer (type = "tensor",
                    input =  - 0.5 * T.mean(T.log(net.layers['D(x)'].output)) - \
                                0.5 * T.mean(T.log(1-net.layers['D(G(z))'].output)),
                    input_shape = (1,),
                    id = "discriminator_task"
                    )

    net.add_layer ( type = "objective",
                    id = "discriminator_obj",
                    origin = "discriminator_task",
                    layer_type = 'value',
                    objective = net.dropout_layers['discriminator_task'].output,
                    datastream_origin = 'data', 
                    verbose = verbose
                    )
    #generator objective 
    net.add_layer (type = "tensor",
                    input =  - 0.5 * T.mean(T.log(net.layers['D(G(z))'].output)),
                    input_shape = (1,),
                    id = "objective_task"
                    )
    net.add_layer ( type = "objective",
                    id = "generator_obj",
                    layer_type = 'value',
                    origin = "objective_task",
                    objective = net.dropout_layers['objective_task'].output,
                    datastream_origin = 'data', 
                    verbose = verbose
                    )   
    
    #softmax objective.    
    net.add_layer ( type = "objective",
                    id = "classifier_obj",
                    origin = "softmax",
                    objective = "nll",
                    layer_type = 'discriminator',
                    datastream_origin = 'data', 
                    verbose = verbose
                    )
    
    # from yann.utils.graph import draw_network
    # draw_network(net.graph, filename = 'gan.png')    
    net.pretty_print()
    
    net.cook (  objective_layers = ["classifier_obj", "discriminator_obj", "generator_obj"],
                optimizer_params = optimizer_params,
                discriminator_layers = ["D1-x","D2-x"],
                generator_layers = ["G1","G2","G(z)"], 
                classifier_layers = ["D1-x","D2-x","softmax"],                                                
                softmax_layer = "softmax",
                game_layers = ("D(x)", "D(G(z))"),
                verbose = verbose )
                    
    learning_rates = (0.00004, 0.001 )  

    net.train( epochs = (20), 
            k = 1, 
            pre_train_discriminator = 0,
            validate_after_epochs = 1,
            visualize_after_epochs = 1,
            training_accuracy = True,
            show_progress = True,
            early_terminate = True,
            verbose = verbose)

    return net
if __name__ == '__main__':
    import sys
    dataset = None  
    if len(sys.argv) > 1:
        if sys.argv[1] == 'create_dataset':
            data = cook_mnist (verbose = 2)
            dataset = data.dataset_location()
        else:
            dataset = sys.argv[1]
    else:
        print "provide dataset"
    
    if dataset is None:
        print " creating a new dataset to run through"
        data = cook_mnist (verbose = 2)
        dataset = data.dataset_location()

    #net = shallow_gan ( dataset, verbose = 2 )
    net = deep_gan ( dataset, verbose = 2 )    