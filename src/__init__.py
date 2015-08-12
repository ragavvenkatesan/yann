#!/usr/bin/python
from lenet import network
from cnn import ReLU, Sigmoid, Softmax, Tanh
import sys
import pdb

def samosa( 
                    arch_params,
                    optimization_params ,
                    data_params, 
                    filename_params,
                    visual_params,
                    n_epochs = 200,
                    validate_after_epochs = 1,
                    verbose = False, 
           ):
              
    # Initialize the network class.            

    net = network( data_params = data_params, filename_params = filename_params, random_seed = arch_params ["random_seed"], verbose = verbose )
    # running init_data in the __init__ of the class so that with the initiation of the class itself the 
    # first set of data is also loaded. 
    # Just checking as a way to see if the intended self.dataset is indeed loaded.
    net.init_data ()      
    net.build_network( arch_params = arch_params, optimization_params = optimization_params, verbose = verbose)
    net.create_dirs ( visual_params = visual_params)              
    net.train( n_epochs = n_epochs, validate_after_epochs = validate_after_epochs, verbose = verbose )
    net.test ( verbose = True )
                       
                    
    ## Boiler Plate ## 
if __name__ == '__main__':
                                                                                            # for epoch in [0, mom_epoch_interval] the momentum increases linearly
    optimization_params = {
                            "mom_start"                         : 0.5,                      # from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
                            "mom_end"                           : 0.98,
                            "mom_interval"                      : 100,
                            "mom_type"                          : 1,                         # if mom_type = 1 , classical momentum if mom_type = 0, no momentum, if mom_type = 2 Nesterov's accelerated gradient momentum 
                            "initial_learning_rate"             : 0.01,                      # Learning rate at the start
                            "learning_rate_decay"               : 0.98, 
                            "l1_reg"                            : 0.0001,                     # regularization coeff for the last logistic layer and MLP layers
                            "l2_reg"                            : 0.0001,                     # regularization coeff for the last logistic layer and MLP layers
                            "ada_grad"                          : False,
                            "rms_prop"                          : True,
                            "rms_rho"                           : 0.9,                      # implement rms_prop with this rho
                            "rms_epsilon"                       : 1e-6,                     # implement rms_prop with this epsilon
                            "fudge_factor"                      : 1e-7,                     # Just to avoid divide by zero, but google even advocates trying '1'
                            "objective"                         : 1,                        # objective = 0 is negative log likelihood, objective = 2 is categorical cross entropy                                                       
                            }

    filename_params = { 
                        "results_file_name"     : "../results/results_mnist.txt",        # Files that will be saved down on completion Can be used by the parse.m file
                        "error_file_name"       : "../results/error_mnist.txt",
                        "cost_file_name"        : "../results/cost_mnist.txt",
                        "confusion_file_name"   : "../results/confusion_mnist.txt",
                        "network_save_name"     : "../results/network_mnist.pkl.gz "
                    }        
        
    data_params = {
                   "type"               : 'skdata',                                    # Options: 'pkl', 'skdata' , 'mat' for loading pkl files, mat files for skdata files.
                   "loc"                : 'mnist',                                          # location for mat or pkl files, which data for skdata files. Skdata will be downloaded and used from '~/.skdata/'
                   "batch_size"         : 500,                                      # For loading and for Gradient Descent Batch Size
                   "load_batches"       : -1, 
                   "batches2train"      : 100,                                      # Number of training batches.
                   "batches2test"       : 20,                                       # Number of testing batches.
                   "batches2validate"   : 20,                                       # Number of validation batches
                   "height"             : 28,                                       # Height of each input image
                   "width"              : 28,                                       # Width of each input image
                   "channels"           : 1                                         # Number of channels of each input image 
                  }

    arch_params = {
                    # Decay of Learninig rate after each epoch of SGD
                    "squared_filter_length_limit"       : 15,   
                    "mlp_activations"                   : [ ReLU ],           # Activations of MLP layers Options: ReLU, Sigmoid, Tanh
                    "cnn_activations"                   : [ ReLU, ReLU],           # Activations for CNN layers Options: ReLU,  
                    "dropout"                           : True,                     # Flag for dropout / backprop                    
                    "column_norm"                       : True,
                    "dropout_rates"                     : [ 0.5 , 0.5],             # Rates of dropout. Use 0 is backprop.
                    "nkerns"                            : [ 20 , 50  ],               # Number of feature maps at each CNN layer
                    "outs"                              : 10,                       # Number of output nodes ( must equal number of classes)
                    "filter_size"                       : [  5 , 5],                # Receptive field of each CNN layer
                    "pooling_size"                      : [  2 , 2 ],                # Pooling field of each CNN layer
                    "num_nodes"                         : [  500 ],                # Number of nodes in each MLP layer
                    "use_bias"                          : True,                     # Flag for using bias                   
                    "random_seed"                       : 23455,                    # Use same seed for reproduction of results.
                    "svm_flag"                          : False                     # True makes the last layer a SVM

                 }

    visual_params = {
                        "visualize_flag"        : False,
                        "visualize_after_epochs": 1,
                        "n_visual_images"       : 20,
                        "display_flag"          : False          # haven't tested this feature due to opencv-libgtk trouble.
                    }                                             

    samosa(
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    data_params             = data_params, 
                    filename_params         = filename_params,
                    visual_params           = visual_params,
                    n_epochs                = 200,
                    validate_after_epochs   = 1,
                    verbose                 = True,                                                # True prints in a lot of intermetediate steps, False keeps it to minimum.
                )
