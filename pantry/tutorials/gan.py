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
        "location"			: '',                                      
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
                "zero_mean" 	: False,
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

def mlgan ( dataset= None, verbose = 1 ):             
    """
    This function is a demo example of a generative adversarial network. 
    This is an example code. You should study this code rather than merely run it.  

    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.
    """
    optimizer_params =  {        
                "momentum_type"       : 'polyak',             
                "momentum_params"     : (0.5, 0.7, 50),      
                "regularization"      : (0.000, 0.0001),       
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
                        num_neurons = (100,64), 
                        distribution = 'normal',
                        mu = 0,
                        sigma = 1,
                        limits = (0,1),
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
                    id = "G1",
                    num_neurons = 1200,
                    activation = 'relu',
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "G1",
                    id = "G2",
                    num_neurons = 1200,
                    activation = 'relu',
                    verbose = verbose
                    )                     
    
    net.add_layer ( type = "dot_product",
                    origin = "G2",
                    id = "G(z)",
                    num_neurons = 784,
                    activation = 'sigmoid',
                    verbose = verbose
                    )  # This layer is the one that creates the images.
        
    #D(x) - Contains params theta_d creates features 1 X 800. 
    net.add_layer ( type = "dot_product",
                    id = "D1",
                    origin = "x",
                    num_neurons = 1200,
                    activation = ('maxout','maxout',5),   
                    regularize = True,                                                         
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    id = "Dz1",
                    origin = "G(z)",
                    input_params = net.dropout_layers["D1"].params, 
                    num_neurons = 1200,
                    activation = ('maxout','maxout',5),
                    regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    id = "D(x)",
                    origin = "D1",
                    num_neurons = 1200,
                    activation = ('maxout','maxout',5),
                    regularize = True,                    
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    id = "D(G(z))",
                    origin = "Dz1",
                    input_params = net.dropout_layers["D(x)"].params,   
                    num_neurons = 1200,
                    activation = ('maxout','maxout',5),
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
    # fake objective 
    net.add_layer ( type = "objective",
                    id = "fake_obj",
                    origin = "fake", # this is useless anyway.
                    layer_type = 'value',
                    objective = T.mean(T.log(net.layers['fake'].output)),
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
                    objective = "nll",
                    layer_type = 'discriminator',
                    datastream_origin = 'data', 
                    verbose = verbose
                    )
    
    # from yann.utils.graph import draw_network
    # draw_network(net.graph, filename = 'gan.png')    
    # net.pretty_print()
    
    net.cook (  objective_layers = ["classifier_obj","real_obj","fake_obj"],
                optimizer_params = optimizer_params,
                classifier_layers = ["D1", "D(x)", "softmax"],                                
                discriminator_layers = ["D1","D(x)"],
                generator_layers = ["G1","G(z)"], 
                softmax_layer = "softmax",
                verbose = verbose )
                    
    learning_rates = (0.04, 0.001, 0.0001 )  

    net.train( epochs = (50, 50 ), 
               k = 5,  
               pre_train_discriminator = 0,
               validate_after_epochs = 1,
               visualize_after_epochs = 1,
               training_accuracy = True,
               show_progress = True,
               early_terminate = True,
               verbose = verbose)
                           
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

    mlgan ( dataset, verbose = 2 )