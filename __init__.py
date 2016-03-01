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
             
    # first things first, initialize the net.
    # This call creates an object net, but also initializes more parameters of the network.
    net = cnn_mlp(   filename_params = filename_params,
                     arch_params = arch_params,
                     optimization_params = optimization_params,
                     retrain_params = None,
                     init_params = None,
                     verbose =verbose ) 
                     
                     
    # load the inputs of the net. 
    # This is needed to create variables like batchsizes and input layer sizes.
    net.init_data ( dataset = dataset , outs = arch_params ["outs"], visual_params = visual_params, verbose = verbose )   
    
     
    # Build network.
    # This function creates both the forward and the backward. 
    # It also creates all the functions for training, testing and validation.
    # It establishes all connections and sets up the optimzier. 
    # build network is the important part of the code where all the action happens.                            
    net.build_network(verbose = verbose)   
    
    
    # this function saves the network.  Just so that you don't have to lose everything.                                 
    net.save_network ()        
    # This function goes through the epochs      
    net.train( n_epochs = n_epochs, 
                ft_epochs = ft_epochs,
                 validate_after_epochs = validate_after_epochs,
                 verbose = verbose )  
    net.save_network ()        
                      
    # this function tests on the dataset
    net.test( verbose = verbose )       
             
    """ 
    # If you want to reload an already trained net follow these steps
    params_loaded, arch_params_loaded = load_network (filename_params ["network_save_name"] ,
                                        data_params = False, 
                                        optimization_params = False) 
    copy_classifier_layer   = True         # also copy the classifer layer too. 
                                           # Remember if the number of labels are now different from when it was trained, you need to False This.
    freeze_classifier_layer = False        # if you are going to fine tune or re-train the softmax layer, avoid this.
    
    freeze_layer_params       = False       # setting this to True will mean that layer will not learn. 
                                            # you can still learn classifier layer, but not other layers.                         
                                                              
    retrain_params = {      # copy all layers that are loaded. False will make that particular layer randomly initialized.
                            "copy_from_old"     : [True] * (len(arch_params_loaded["nkerns"]) + len(arch_params_loaded["num_nodes"])) + [copy_classifier_layer],
                            # Freeze - refer above.
                            "freeze"            : [freeze_layer_params] * (len(arch_params_loaded["nkerns"]) + len(arch_params_loaded["num_nodes"])) + [freeze_classifier_layer]
                     }     
    # initialze the net.                     
    rebuilt_net = cnn_mlp(  filename_params = filename_params_,
                            arch_params = arch_params_loaded,
                            optimization_params = optimization_params,
                            retrain_params = retrain_params,
                            init_params = params_loaded,
                            verbose = verbose   )
    # This now initializes a rebuilt_net. 
    
    # You still need to init its data and build it like how you did previously.
    # If this is an already trained network, you don't need to retrain it you could directly test also. 
    """
## Boiler Plate ## 
if __name__ == '__main__':
             
    if os.path.isfile('dump.txt'):
        f = open('dump.txt', 'a')
    else:
        f = open('dump.txt', 'w')
        
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
                            "mom"                         	    : (0.5, 0.99, 100), # (mom_start, momentum_end, momentum_interval)                     
                            "mom_type"                          : 1,                # 0-no mom, 1-polyak, 2-nestrov          
                            "learning_rate"                     : (0.001,0.0001, 0.05 ),          # (initial_learning_rate, ft_learning_rate, annealing)
                            "reg"                               : (0.000,0.000),    # l1_coeff, l2_coeff                                
                            "optim_type"                        : 3,                # 0-SGD, 1-Adagrad, 2-RmsProp, 3-Adam
                            "objective"                         : 1,                # 0-negative log likelihood, 1-categorical cross entropy, 2-binary cross entropy
                            }        

    arch_params = {
                    "mlp_activations"                   : [ ReLU ],         # activation functions for each mlp layers                
                    "mlp_dropout"                       : True,             # same as above 
                    "mlp_dropout_rates"                 : [ 0.5 , 0.5 ],    # p values for dropout first one for input
                    "num_nodes"                         : [ 800  ],         # number of nodes in mlp layers only    
                    "mlp_batch_norm"                    : True ,            # batch norm for mlp layers           
                    "mlp_maxout"                        : [ 1 ],            # 1 - nothing, >1 maxout by that size.                                               
                    "outs"                              : 10,               # nuber of output nodes in softmax or svm layer                                                                                                                
                    "svm_flag"                          : False,            # toggle between softmax and svm layers                            
                    "cnn_activations"                   : [ ReLU ],         # activation functions for each cnn layer    
                    "cnn_batch_norm"                    : [ True ],         # batch norm for cnn layers 
                    "nkerns"                            : [ 20,   ],        # number of convolutional kernels in each layer       
                    "filter_size"                       : [ (5,5) ],        # filter size of each convoutional filter
                    "pooling_size"                      : [ (2,2) ],        # pooling size after the layer
                    "pooling_type"                      : [ 1,    ],        # 0 - maxpool to same size, 1 - maxpool, 2- average pool, 3-maxrand pool
                    "maxrandpool_p"                     : [ 1,    ],        # p value for maxrand pool, used only if maxrand pool is used.                                                                                            
                    "conv_stride_size"                  : [ (1,1) ],        # stride size of each convolutional layer
                    "conv_pad"                          : [ 0,    ],        # 0 - 'valid' convolution and 1 - ' fully padded convolution' 
                    "cnn_maxout"                        : [ 1,    ],        # 1 - nothing, >1 maxout by that size.             
                    "cnn_dropout"                       : False,            # False for no dropout, True for dropout                    
                    "cnn_dropout_rates"                 : [ 0.5 ],          # p values for dropout, used only if convolutioanl dropouts are True.
                    "random_seed"                       : 23455,            # Just a random seed, different seeds create new type of initializations
                    "mean_subtract"                     : False,            # subtract means of inputs. 
                    "use_bias"                          : True,             # use bias in all layers.       
                    "max_out"                           : 0                 # maxout type. 0-no maxout, 1-maxout, 2-meanout, 3-randout.
        
                 }                          

    # other loose parameters. 
    n_epochs = 2                    # number of epochs to run unless early terminated
    validate_after_epochs = 1       # number of epochs after which to validate.    
    ft_epochs = 2                   # number of epoch to finetune learning with.
    verbose = False                 # if True makes a lot of prints, if False doesn't. 
    
    # code to tutor on how to setup and run. 
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