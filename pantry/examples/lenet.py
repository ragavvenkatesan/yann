#!/usr/bin/python
import sys, os
sys.path.insert(0, os.getcwd())

from yann.network import network

def lenet5 ( verbose = 1 ):             

    optimizer_params =  {        
                "momentum_type"       : 'polyak',             
                                        # false, polyak, nesterov
                "momentum_params"     : (0.9, 0.95, 30),      
                    # (mom_start, momentum_end, momentum_end_epoch)                                                           
                "regularization"      : (0.0001, 0.0001),       
                        # l1_coeff, l2_coeff, decisiveness (optional)                                
                "optimizer_type"      : 'rmsprop',                
                                        # sgd, adagrad, rmsprop, adam 
                "id"                  : "main"
                        }

    dataset_params  = {
                            "dataset"   :  "_datasets/_dataset_71367",
                            "svm"       :  False, 
                            "n_classes" : 10,
                            "id"        : 'mnist'
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
                    datastream_origin = 'mnist', # if you didnt add a dataset module, now is 
                                                 # the time. 
                    dropout_rate = 0,
                    mean_subtract = True )
    
    # add first convolutional layer
    
    net.add_layer ( type = "conv_pool",
                    origin = "input",
                    id = "conv_pool_1",
                    num_neurons = 20,
                    filter_size = (5,5),
                    pool_size = (2,2),
                    batch_norm = True,
                    activation = 'relu',
                    verbose = verbose
                    )

    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_1",
                    id = "conv_pool_2",
                    num_neurons = 50,
                    filter_size = (3,3),
                    pool_size = (2,2),
                    batch_norm = True,
                    dropout_rate = 0.5,
                    activation = 'relu',
                    verbose = verbose
                    )
    
    net.add_layer ( type = "dot_product",
                    origin = "conv_pool_2",
                    id = "dot_product_1",
                    num_neurons = 800,
                    batch_norm = True,
                    dropout_rate = 0.5,
                    activation = 'relu',
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "dot_product_1",
                    id = "dot_product_2",
                    num_neurons = 800,                    
                    batch_norm = True,
                    dropout_rate = 0.5,
                    activation = 'relu',
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
                    id = "nll",
                    origin = "softmax",
                    verbose = verbose
                    )
    # objective provided by classifier layer               
    # nll-negative log likelihood, 
    # cce-categorical cross entropy, 
    # bce-binary cross entropy,
    # hinge-hinge loss 
    learning_rates = (0.01, 0.05, 0.001)  
    # (initial_learning_rate, annealing, ft_learnint_rate)

    net.cook( optimizer = 'main',
              objective_layer = 'nll',
              datastream = 'mnist',
              classifier = 'softmax',
              learning_rates = learning_rates,
              verbose = verbose
              )

    net.train( epochs = (20, 20), 
               ft_learning_rate = 0.001,
               validate_after_epochs = 20,
               training_accuracy = True,
               show_progress = True,
               early_terminate = True,
               verbose = verbose)

    net.test(verbose = verbose)


## Boiler Plate ## 
if __name__ == '__main__':
        
    # prepare_dataset (verbose = 3)
    lenet5 ( verbose = 2 ) 

