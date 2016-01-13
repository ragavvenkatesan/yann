#!/usr/bin/python
from samosa.build import network
from samosa.core import ReLU, Sigmoid, Softmax, Tanh, Identity
from samosa.util import load_network
from samosa.dataset import setup_dataset
import os

import pdb

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
               
    net.init_data ( dataset = dataset , outs = arch_params ["outs"], verbose = verbose )      
    
    net.build_network(   arch_params = arch_params,
                         optimization_params = optimization_params,
                         retrain_params = None,
                         verbose = verbose)
                         
    net.create_dirs ( visual_params = visual_params )   
                       
    net.train( n_epochs = n_epochs, 
                ft_epochs = ft_epochs,
                 validate_after_epochs = validate_after_epochs,
                 verbose = verbose )          
    net.test( verbose = verbose )
   
                              
    net.save_network ()   
             
                           
    
## Boiler Plate ## 
if __name__ == '__main__':
             
    if os.path.isfile('dump.txt'):
        f = open('dump.txt', 'a')
    else:
        f = open('dump.txt', 'w')
        f.close()
        f.open ('dump.txt','a')
        
    f.write("... main net")
    # run the base CNN as usual.              
    filename_params = { 
                        "results_file_name"     : "../results/results.txt",      
                        "error_file_name"       : "../results/error.txt",
                        "cost_file_name"        : "../results/cost.txt",
                        "confusion_file_name"   : "../results/confusion.txt",
                        "network_save_name"     : "../results/network.pkl.gz "
                    }
                    
    visual_params = {
                        "visualize_flag"        : False,
                        "visualize_after_epochs": 1,
                        "n_visual_images"       : 81,
                        "display_flag"          : False,
                        "color_filter"          : False         
                    }   
                                                                                                                            
    optimization_params = {
                            "mom_start"                         : 0.5,                      
                            "mom_end"                           : 0.85,
                            "mom_interval"                      : 100,
                            "mom_type"                          : 1,                         
                            "initial_learning_rate"             : 0.01,
			                "ft_learning_rate"                  : 0.0001,    
                            "learning_rate_decay"               : 0.005,
                            "l1_reg"                            : 0.000,                     
                            "l2_reg"                            : 0.000,                    
                            "ada_grad"                          : False,
                            "rms_prop"                          : False,
                            "rms_rho"                           : 0.9,                      
                            "rms_epsilon"                       : 1e-7,                     
                            "fudge_factor"                      : 1e-7,                    
                            "objective"                         : 1,    
                            }        

    arch_params = {
                    
                    "squared_filter_length_limit"       : 15,   
                    "mlp_activations"                   : [ ReLU, ReLU ],
                    "cnn_dropout"                       : False,
                    "mlp_dropout"                       : False,
                    "mlp_dropout_rates"                 : [ 0.5, 0.5, 0.5 ],
                    "num_nodes"                         : [ 1200, 1200 ],                                     
                    "outs"                              : 10,                                                                                                                               
                    "svm_flag"                          : False,                                       
                    "cnn_activations"                   : [ ReLU      , ReLU ,  ReLU    ],             
                    "cnn_batch_norm"                    : [ True      , True,   True    ],
                    "mlp_batch_norm"                    : False,
                    "nkerns"                            : [     ],              
                    "filter_size"                       : [ ( 5, 5 ) , (5, 5 ), (3, 3)  ],
                    "pooling_size"                      : [ ( 2, 2 ) , (1, 1 ), (1, 1)  ],
                    "conv_stride_size"                  : [ ( 1, 1 ) , (1, 1 ), (1, 1)  ],
                    "cnn_maxout"                        : [  1,         1 ,       1,    ],                    
                    "mlp_maxout"                        : [  1 , 1   ],
                    "cnn_dropout_rates"                 : [ 0.5,        0.5     , 0.5   ],
                    "random_seed"                       : 23455, 
                    "mean_subtract"                     : False,
                    "max_out"                           : 0 

                 }                          

    # other loose parameters. 
    n_epochs = 2
    validate_after_epochs = 1
    ft_epochs = 2
    verbose = True 
    
    run_cnn(
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = "_datasets/_dataset_65953", 
                    filename_params         = filename_params,          
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,                                                
                )
                
 
    pdb.set_trace()                             