#!/usr/bin/python
from samosa.build import network
from samosa.core import ReLU, Sigmoid, Softmax, Tanh, Identity
from samosa.util import load_network
from samosa.dataset import setup_dataset

def run_cnn( 
                    arch_params,
                    optimization_params ,
                    dataset, 
                    filename_params,
                    visual_params,
                    n_epochs = 50,
                    ft_epochs = 200, 
                    validate_after_epochs = 1,
                    verbose = False, 
           ):            
    net = network(  filename_params = filename_params,
                     random_seed = arch_params ["random_seed"],
                     verbose = verbose )   
               
    net.init_data ( dataset, outs = arch_params["outs"], verbose = verbose )      
    
    net.build_network(   arch_params = arch_params,
                         optimization_params = optimization_params,
                         verbose = verbose)
                         
    net.create_dirs ( visual_params = visual_params )   
                       
    net.train( n_epochs = n_epochs, 
                ft_epochs = ft_epochs,
                 validate_after_epochs = validate_after_epochs,
                 verbose = verbose )          
    net.test( verbose = verbose )
   
                              
    net.save_network ()     
    
    """
    # Use the following commands to load a pre-trained network and just show testing
    # performance. 
    params_loaded, arch_params_loaded = load_network (filename_params ["network_save_name"] ,
                                        data_params = False, 
                                        optimization_params = False)   
 
    test_net = network( 
                     data_params = data_params,
                     filename_params = filename_params,
                     random_seed = arch_params ["random_seed"],
                     verbose = verbose )  
    
    test_net.init_data ( dataset )   
    test_net.build_network (
                           arch_params = arch_params_loaded,
                           optimization_params = optimization_params,
                           init_params = params_loaded,
                           verbose = verbose )    
                   
    test_net.test ( verbose = verbose )
    """                   
                    
                   
 ## Boiler Plate ## 
if __name__ == '__main__':
             
    filename_params = { 
                        "results_file_name"     : "../results/results.txt",      
                        "error_file_name"       : "../results/errortxt",
                        "cost_file_name"        : "../results/cost.txt",
                        "confusion_file_name"   : "../results/confusion.txt",
                        "network_save_name"     : "../results/network.pkl.gz "
                    }
    visual_params = {
                        "visualize_flag"        : True,
                        "visualize_after_epochs": 1,
                        "n_visual_images"       : 49,
                        "display_flag"          : False,
                        "color_filter"          : True         
                    }   
                                                                                                                            
    optimization_params = {
                            "mom_start"                         : 0.5,                      
                            "mom_end"                           : 0.65,
                            "mom_interval"                      : 50,
                            "mom_type"                          : 1,                         
                            "initial_learning_rate"             : 0.01,
		 	                "ft_learning_rate"                  : 0.001,    
                            "learning_rate_decay"               : 0.995,
                            "l1_reg"                            : 0.000,                     
                            "l2_reg"                            : 0.000,                    
                            "ada_grad"                          : False,
                            "rms_prop"                          : True,
                            "rms_rho"                           : 0.9,                      
                            "rms_epsilon"                       : 1e-7,                     
                            "fudge_factor"                      : 1e-7,                    
                            "objective"                         : 0,    
                            # for some reason, cross-entropy some times produces NaNs ... need to debug.                     
                            }        

    arch_params = {
                    
                    "squared_filter_length_limit"       : 15,   
                    "mlp_activations"                   : [ ReLU, ReLU ],
                    "cnn_dropout"                       : True,
                    "mlp_dropout"                       : True,
                    "mlp_dropout_rates"                 : [ 0.5,  0.5,  0.5 ],
                    "num_nodes"                         : [ 1200, 1200  ],                                     
                    "outs"                              : 102,                                                                                                                               
                    "svm_flag"                          : False,                                       
                    "cnn_activations"                   : [ Identity, Identity, Identity, Identity, Identity ],             
                    "batch_norm"                        : True,
                    "nkerns"                            : [ 48,       96,       96,      128,    96,     ],              
                    "filter_size"                       : [ (5, 5 ),  (5, 5) , (5, 5), (5, 5), (3, 3)    ],
                    "pooling_size"                      : [ (1, 1 ),  (1, 1),  (2, 2) ,(1 ,1), (1, 1)   ],
                    # somethign wrong with stride... use only (1,1) for now !!                         
                    "conv_stride_size"                  : [ (1, 1 ),  (1, 1),  (1, 1) ,(1 ,1), (1, 1)  ],
                    "cnn_maxout"                        : [ 2,         2,        2,     2,      2    ],                    
                    "mlp_maxout"                        : [ 1  , 1  ],
                    "cnn_dropout_rates"                 : [ 0.1,       0.2,      0.2,     0.3,    0.3   ],
                    "random_seed"                       : 23455, 
                    "mean_subtract"                     : False,
                    "max_out"                           : 1 

                 }                                          

    run_cnn(
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = "_datasets/_dataset_91882", 
                    filename_params         = filename_params,          
                    visual_params           = visual_params, 
                    validate_after_epochs   = 1,
                    n_epochs                = 50,
                    ft_epochs               = 200, 
                    verbose                 = False,                                                
                )
