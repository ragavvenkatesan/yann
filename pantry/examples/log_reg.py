#!/usr/bin/python
import sys, os
# assuming the the code begins with yann root folder.

sys.path.insert(0, os.getcwd())
from yann.network import network

def log_reg ( verbose ):            

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
    net = network( verbose = verbose )                       
    
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
                    mean_subtract = True )
    
    net.add_layer ( type = "classifier",
                    id = "softmax",
                    origin = "input",
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
               validate_after_epochs = 1,
               training_accuracy = True,
               show_progress = True,
               early_terminate = True,
               verbose = verbose)
    net.test( show_progress = True,
               verbose = verbose)


## Boiler Plate ## 
if __name__ == '__main__':
        
    # prepare_dataset (verbose = 3)
    log_reg ( verbose = 2 ) 

