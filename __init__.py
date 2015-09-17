#!/usr/bin/python
from samosa.lenet import network
from samosa.cnn import ReLU, Sigmoid, Softmax, Tanh
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
              
    net = network(  filename_params = filename_params,
                     random_seed = arch_params ["random_seed"],
                     verbose = verbose )    
    net.init_data (data_params = data_params , outs = arch_params["outs"])              
    net.build_network(   arch_params = arch_params,
                         optimization_params = optimization_params,
                         verbose = verbose)
    net.create_dirs ( visual_params = visual_params)                      
    net.train( n_epochs = n_epochs, 
                 validate_after_epochs = validate_after_epochs,
                 verbose = verbose )          
    net.test( verbose = verbose )
   
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
                            "mom_end"                           : 0.99,
                            "mom_interval"                      : 500,
                            "mom_type"                          : 1,                         
                            "initial_learning_rate"             : 0.01,                     
                            "learning_rate_decay"               : 0.998, 
                            "l1_reg"                            : 0.0,                     
                            "l2_reg"                            : 0.0,                    
                            "ada_grad"                          : False,
                            "rms_prop"                          : True,
                            "rms_rho"                           : 0.9,                      
                            "rms_epsilon"                       : 1e-6,                     
                            "fudge_factor"                      : 1e-7,                    
                            "objective"                         : 1,                        
                            }

    filename_params = { 
                        "results_file_name"     : "../results/results.txt",      
                        "error_file_name"       : "../results/errortxt",
                        "cost_file_name"        : "../results/cost.txt",
                        "confusion_file_name"   : "../results/confusion.txt",
                        "network_save_name"     : "../results/network.pkl.gz "
                    }        
        
    data_params = {
                   "type"               : 'skdata',                                   
                   "loc"                : 'cifar10',                                          
                   "batch_size"         : 100,                                     
                   "load_batches"       : -1, 
                   "batches2train"      : 400,                                      
                   "batches2test"       : 100,                                      
                   "batches2validate"   : 100,                                        
                   "height"             : 32,                                       
                   "width"              : 32,                                       
                   "channels"           : 3                                        
                  }

    arch_params = {
                    
                    "squared_filter_length_limit"       : 15,   
                    "mlp_activations"                   : [ ReLU, Softmax ],
                    "cnn_dropout"                       : True,
                    "mlp_dropout"                       : True,
                    "mlp_dropout_rates"                 : [ 0.2 , 0.5],
                    "num_nodes"                         : [ 1200 ],                                     
                    "outs"                              : 10,                                                                                                                               
                    "svm_flag"                          : False,                                       
                    "cnn_activations"                   : [ ReLU, ReLU, ReLU ],             
                    "batch_norm"                        : True,
                    "nkerns"                            : [ 48, 64, 128 ],              
                    "filter_size"                       : [ (5, 5), (5 ,5), (3, 3)],
                    "pooling_size"                      : [ (1, 1), (2, 2), (2, 2)],                                   
                    "cnn_maxout"                        : [ 2, 2, 2],
                    "mlp_maxout"                        : [ 2 ],
                    "cnn_dropout_rates"                 : [ 0.1, 0.2, 0.2],
                    "random_seed"                       : 23455, 
                    "max_out"                           : 1

                 }

    visual_params = {
                        "visualize_flag"        : True,
                        "visualize_after_epochs": 1,
                        "n_visual_images"       : 20,
                        "display_flag"          : False,
                        "color_filter"          : True         
                    }                                             

    run_cnn(
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    data_params             = data_params, 
                    filename_params         = filename_params,          
                    visual_params           = visual_params, 
                    validate_after_epochs   = 1,
                    n_epochs                = 3000,
                    verbose                 = False,                                                
                )