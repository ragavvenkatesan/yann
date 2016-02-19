#!/usr/bin/python
from samosa.cnn import cnn_mlp
from samosa.core import ReLU, Sigmoid, Softmax, Tanh, Abs, Squared
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
                    n_epochs = 200,
                    ft_epochs = 100, 
                    validate_after_epochs = 1,
                    verbose = False, 
           ):            
               
    net = cnn_mlp(   filename_params = filename_params,
                     arch_params = arch_params,
                     optimization_params = optimization_params,
                     retrain_params = None,
                     init_params = None,
                     verbose =verbose ) 
    net.init_data ( dataset = dataset , outs = arch_params ["outs"], visual_params = visual_params, verbose = verbose )                                 
    net.build_network(verbose = verbose)                               
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
                        "visualize_flag"        : True,
                        "visualize_after_epochs": 1,
                        "n_visual_images"       : 81,
                        "display_flag"          : False,
                        "color_filter"          : True         
                    }   
                                                                                                                            
    optimization_params = {
                            "mom_start"                         : 0.5,                      
                            "mom_end"                           : 0.99,
                            "mom_interval"                      : 100,
                            "mom_type"                          : 2,                         
                            "initial_learning_rate"             : 0.01,
                            "ft_learning_rate"                  : 0.0001,    
                            "learning_rate_decay"               : 0.005,
                            "l1_reg"                            : 0.000,                     
                            "l2_reg"                            : 0.000,                    
                            "ada_grad"                          : False,
                            "rms_prop"                          : True,
                            "rms_rho"                           : 0.9,                      
                            "rms_epsilon"                       : 1e-7,                     
                            "fudge_factor"                      : 1e-7,                    
                            "objective"                         : 0,   
                            }        

    arch_params = {
                    
                    "mlp_activations"                   : [ ReLU ],
                    "cnn_dropout"                       : True,
                    "mlp_dropout"                       : True,
                    "mlp_dropout_rates"                 : [ 0.5 , 0.5 ],
                    "num_nodes"                         : [ 800  ],                                     
                    "outs"                              : 10,                                                                                                                               
                    "svm_flag"                          : False,                                       
                    "cnn_activations"                   : [ ReLU,   ReLU  ],             
                    "cnn_batch_norm"                    : [ True,  True ],
                    "mlp_batch_norm"                    : True,
                    "nkerns"                            : [ 20,     50    ],              
                    "filter_size"                       : [ (5,5), (5,5)  ],
                    "pooling_size"                      : [ (2,2), (1,1)  ],
                    "conv_stride_size"                  : [ (1,1), (1,1)  ],
                    "cnn_maxout"                        : [ 1,     1      ],                    
                    "mlp_maxout"                        : [ 1 ],
                    "cnn_dropout_rates"                 : [ 0.5,   0.5    ],
                    "random_seed"                       : 23455,
                    "use_bias"                          : True, 
                    "mean_subtract"                     : True,
                    "max_out"                           : 0
        
                 }                          

    # other loose parameters. 
    n_epochs = 2
    validate_after_epochs = 1
    ft_epochs = 2
    verbose = False 
    
    run_cnn(
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = "_datasets/_dataset_57689", 
                    filename_params         = filename_params,          
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,                                                
                )
                
 
    pdb.set_trace()                             