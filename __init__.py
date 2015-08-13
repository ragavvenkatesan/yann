#!/usr/bin/python
from samosa.lenet import network
from samosa.cnn import ReLU, Sigmoid, Softmax, Tanh
import sys
import pdb
from samosa.util import load_network

def run_cnn( 
                    arch_params,
                    optimization_params ,
                    data_params, 
                    filename_params,
                    visual_params,
                    n_epochs = 200,
                    validate_after_epochs = 1,
                    verbose = False, 
           ):
              
    net = network(  
                     filename_params = filename_params,
                     random_seed = arch_params ["random_seed"],
                     verbose = verbose )                     
    net.init_data (data_params = data_params )          
    
    net.build_network(   arch_params = arch_params,
                         optimization_params = optimization_params,
                         verbose = verbose)
    net.create_dirs ( visual_params = visual_params)    
                  
    net.train( n_epochs = n_epochs, 
                 validate_after_epochs = validate_after_epochs,
                 verbose = verbose )    
   
   
    """                          
    net.save_network ()     

    params_loaded, arch_params_loaded = load_network (filename_params ["network_save_name"] ,
                                        data_params = False, 
                                        optimization_params = False)    
    
    new_net = network( 
                     data_params = data_params,
                     filename_params = filename_params,
                     random_seed = arch_params ["random_seed"],
                     verbose = verbose )  
    
    new_net.init_data () 
    
    new_net.build_network (
                           arch_params = arch_params_loaded,
                           optimization_params = optimization_params,
                           init_params = params_loaded,
                           verbose = verbose )    
                           
                           
                           
    new_net.test ( verbose = verbose )
    """                   
                    
    ## Boiler Plate ## 
if __name__ == '__main__':
                                                                                            
    optimization_params = {
                            "mom_start"                         : 0.5,                      
                            "mom_end"                           : 0.98,
                            "mom_interval"                      : 100,
                            "mom_type"                          : 1,                         
                            "initial_learning_rate"             : 0.01,                     
                            "learning_rate_decay"               : 0.98, 
                            "l1_reg"                            : 0.0001,                     
                            "l2_reg"                            : 0.0001,                    
                            "ada_grad"                          : False,
                            "rms_prop"                          : True,
                            "rms_rho"                           : 0.9,                      
                            "rms_epsilon"                       : 1e-6,                     
                            "fudge_factor"                      : 1e-7,                    
                            "objective"                         : 1,                        
                            }

    filename_params = { 
                        "results_file_name"     : "../results/results_mnist.txt",      
                        "error_file_name"       : "../results/error_mnist.txt",
                        "cost_file_name"        : "../results/cost_mnist.txt",
                        "confusion_file_name"   : "../results/confusion_mnist.txt",
                        "network_save_name"     : "../results/network_mnist.pkl.gz "
                    }        
        
    data_params = {
                   "type"               : 'skdata',                                   
                   "loc"                : 'mnist',                                          
                   "batch_size"         : 500,                                     
                   "load_batches"       : -1, 
                   "batches2train"      : 100,                                      
                   "batches2test"       : 20,                                      
                   "batches2validate"   : 20,                                       
                   "height"             : 28,                                       
                   "width"              : 28,                                       
                   "channels"           : 1                                        
                  }

    arch_params = {
                    
                    "squared_filter_length_limit"       : 15,   
                    "mlp_activations"                   : [ ReLU ],          
                    "cnn_activations"                   : [ ReLU, ReLU],             
                    "dropout"                           : True,                                        
                    "column_norm"                       : True,
                    "dropout_rates"                     : [ 0.5 , 0.5],            
                    "nkerns"                            : [ 20 , 50  ],              
                    "outs"                              : 10,                      
                    "filter_size"                       : [  (5, 5) , (5, 5) ],               
                    "pooling_size"                      : [  (2, 2) , (2, 2) ],               
                    "num_nodes"                         : [  500 ],                
                    "use_bias"                          : True,                                      
                    "random_seed"                       : 23455,                   
                    "svm_flag"                          : False                    

                 }

    visual_params = {
                        "visualize_flag"        : True,
                        "visualize_after_epochs": 1,
                        "n_visual_images"       : 20,
                        "display_flag"          : False          
                    }                                             

    run_cnn(
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    data_params             = data_params, 
                    filename_params         = filename_params,
                    visual_params           = visual_params,
                    n_epochs                = 200,
                    validate_after_epochs   = 1,
                    verbose                 = True,                                                
                )
