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
     
# this is for the generality experiments. This retrains by freezing and unfreezing layers diferenty.     
def generality_experiment(                
                    arch_params,
                    optimization_params ,
                    dataset, 
                    original_filename_params,   
                    filename_params_retrain, 
                    retrain_params,     
                    visual_params,    
                    validate_after_epochs = 1,  
                    n_epochs = 50,
                    ft_epochs = 200, 
                    verbose = False                                                                                                                              

                      ):         
                                                                                                     
    params_loaded, arch_params_loaded = load_network (filename_params ["network_save_name"] ,
                                        data_params = False, 
                                        optimization_params = False)   
 
    # retrain is used to do the dataset some wierd experiments.     
    retrain_net = network( 
                     filename_params = filename_params_retrain,
                     random_seed = arch_params ["random_seed"],
                     verbose = verbose )  
                                        
    retrain_net.init_data ( dataset = dataset , outs = arch_params ["outs"], verbose = verbose )      

    retrain_net.build_network (
                           arch_params = arch_params_loaded,
                           optimization_params = optimization_params,
                           init_params = params_loaded,
                           retrain_params = retrain_params,
                           verbose = verbose )    

    retrain_net.create_dirs ( visual_params = visual_params )   
                               
    retrain_net.train ( n_epochs = n_epochs, 
                        ft_epochs = ft_epochs,
                         validate_after_epochs = validate_after_epochs,
                         verbose = verbose)                                             
    
    retrain_net.test ( verbose = verbose )   
    
    retrain_net.save_network ()                       
                           
    
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
                            "rms_prop"                          : True,
                            "rms_rho"                           : 0.9,                      
                            "rms_epsilon"                       : 1e-7,                     
                            "fudge_factor"                      : 1e-7,                    
                            "objective"                         : 1,    
                            }        

    arch_params = {
                    
                    "squared_filter_length_limit"       : 15,   
                    "mlp_activations"                   : [ ReLU ],
                    "cnn_dropout"                       : False,
                    "mlp_dropout"                       : True,
                    "mlp_dropout_rates"                 : [ 0.5, 0.5 ],
                    "num_nodes"                         : [ 1600 ],                                     
                    "outs"                              : 10,                                                                                                                               
                    "svm_flag"                          : False,                                       
                    "cnn_activations"                   : [ ReLU      , ReLU ,  ReLU    ],             
                    "cnn_batch_norm"                    : [ True      , True,   True    ],
                    "mlp_batch_norm"                    : True,
                    "nkerns"                            : [     20  ,     20    , 50    ],              
                    "filter_size"                       : [ ( 5, 5 ) , (5, 5 ), (3, 3)  ],
                    "pooling_size"                      : [ ( 2, 2 ) , (1, 1 ), (1, 1)  ],
                    "conv_stride_size"                  : [ ( 1, 1 ) , (1, 1 ), (1, 1)  ],
                    "cnn_maxout"                        : [  1,         1 ,       1,    ],                    
                    "mlp_maxout"                        : [  1    ],
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
                    dataset                 = "_datasets/_dataset_48205", 
                    filename_params         = filename_params,          
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,                                                
                )
                
                
                
                
                
                
    ## Generality experiments begin from here on. Don't bother about these if all you 
    # want to do is just run the Samosa Toolbox. 
    
                
    # All frozen
    """
    retrain_dataset = "_datasets/_dataset_48205"
    arch_params ["outs"]  = 10 
    
    print "... running all frozen"
    f.write("... running all frozen ")
    retrain_params = {
                        "copy_from_old"     : [ True, True, True, True, False ],
                        "freeze"            : [ True, True, True, True, False ]
                    }                


    filename_params_retrain = { 
                        "results_file_name"     : "../results/results_retrain_0.txt",      
                        "error_file_name"       : "../results/error_retrain_0.txt",
                        "cost_file_name"        : "../results/cost_retrain_0.txt",
                        "confusion_file_name"   : "../results/confusion_retrain_0.txt",
                        "network_save_name"     : "../results/network_retrain_0.pkl.gz "
                    }                    
                                       
    generality_experiment( 
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = retrain_dataset,
                    original_filename_params= filename_params, 
                    filename_params_retrain = filename_params_retrain,  
                    retrain_params          = retrain_params,        
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,
                      )                       
                      
                      
                      
    # one Unfrozen 
    f.write("... running one unfrozen ")
    retrain_params = {
                        "copy_from_old"     : [ True, True, True, True, False ],
                        "freeze"            : [ True, True, True, False, False ]
                    }                


    filename_params_retrain = { 
                        "results_file_name"     : "../results/results_retrain_1.txt",      
                        "error_file_name"       : "../results/error_retrain_1.txt",
                        "cost_file_name"        : "../results/cost_retrain_1.txt",
                        "confusion_file_name"   : "../results/confusion_retrain_1.txt",
                        "network_save_name"     : "../results/network_retrain_1.pkl.gz "
                    }                    
                                       
    generality_experiment( 
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = retrain_dataset,
                    original_filename_params= filename_params, 
                    filename_params_retrain = filename_params_retrain,  
                    retrain_params          = retrain_params,        
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,
                      )                       
                      
                      
                      
                      
                      
                      
    # two Unfrozen 
    print "... running two unfrozen"
    f.write("... running two unfrozen ")
    retrain_params = {
                        "copy_from_old"     : [ True, True, True, True, False ],
                        "freeze"            : [ True, True, False, False, False ]
                    }                


    filename_params_retrain = { 
                        "results_file_name"     : "../results/results_retrain_2.txt",      
                        "error_file_name"       : "../results/error_retrain_2.txt",
                        "cost_file_name"        : "../results/cost_retrain_2.txt",
                        "confusion_file_name"   : "../results/confusion_retrain_2.txt",
                        "network_save_name"     : "../results/network_retrain_2.pkl.gz "
                    }                    
                                     
    generality_experiment( 
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = retrain_dataset,
                    original_filename_params= filename_params, 
                    filename_params_retrain = filename_params_retrain,  
                    retrain_params          = retrain_params,        
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,
                      )                            
                      
                      
                      
                      
                      
                      
                      
    # three Unfrozen 
    print "... running three unfrozen"
    f.write("... running three unfrozen ")    
    retrain_params = {
                        "copy_from_old"     : [ True, True, True, True, False ],
                        "freeze"            : [ True, False, False, False, False ]
                    }                


    filename_params_retrain = { 
                        "results_file_name"     : "../results/results_retrain_3.txt",      
                        "error_file_name"       : "../results/error_retrain_3.txt",
                        "cost_file_name"        : "../results/cost_retrain_3.txt",
                        "confusion_file_name"   : "../results/confusion_retrain_3.txt",
                        "network_save_name"     : "../results/network_retrain_3.pkl.gz "
                    }                    
                    
    arch_params ["outs"]  = 10                    
    generality_experiment( 
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = retrain_dataset,
                    original_filename_params= filename_params, 
                    filename_params_retrain = filename_params_retrain,  
                    retrain_params          = retrain_params,        
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,
                      )                                                  
                      

                      
                      
                      
                      
                      
                      
    # four Unfrozen 
    print "... running four unfrozen"
    f.write("... running four unfrozen ")    
    retrain_params = {
                        "copy_from_old"     : [ True, True, True, True, False ],
                        "freeze"            : [ False, False, False, False, False ]
                    }                


    filename_params_retrain = { 
                        "results_file_name"     : "../results/results_retrain_4.txt",      
                        "error_file_name"       : "../results/error_retrain_4.txt",
                        "cost_file_name"        : "../results/cost_retrain_4.txt",
                        "confusion_file_name"   : "../results/confusion_retrain_4.txt",
                        "network_save_name"     : "../results/network_retrain_4.pkl.gz "
                    }                    
                    
    arch_params ["outs"]  = 10                    
    generality_experiment( 
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = retrain_dataset,
                    original_filename_params= filename_params, 
                    filename_params_retrain = filename_params_retrain,  
                    retrain_params          = retrain_params,        
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,
                      )                                                  



    # five Unfrozen 
    print "... running all unfrozen"
    f.write("... running all unfrozen ")    
    retrain_params = {
                        "copy_from_old"     : [ True, True, True, True, False ],
                        "freeze"            : [ False, False, False, False, False ]
                    }                


    filename_params_retrain = { 
                        "results_file_name"     : "../results/results_retrain_6.txt",      
                        "error_file_name"       : "../results/error_retrain_6.txt",
                        "cost_file_name"        : "../results/cost_retrain_6.txt",
                        "confusion_file_name"   : "../results/confusion_retrain_6.txt",
                        "network_save_name"     : "../results/network_retrain_6.pkl.gz "
                    }                    
                    
    arch_params ["outs"]  = 10                    
    generality_experiment( 
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = retrain_dataset,
                    original_filename_params= filename_params, 
                    filename_params_retrain = filename_params_retrain,  
                    retrain_params          = retrain_params,        
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,
                      )                                                    
                      
    """                  
    pdb.set_trace()                             