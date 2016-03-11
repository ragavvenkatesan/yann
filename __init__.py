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
    """
    net = cnn_mlp(   filename_params = filename_params,
                     arch_params = arch_params,
                     optimization_params = optimization_params,
                     retrain_params = None,
                     init_params = None,
                     verbose =verbose ) 
                     
    """
    # Use this to pre-load and build VGG-19 network.
    # Read vgg2samosa for parameter desctiptions.
    model = './dataset/vgg/vgg19.pkl'
    outs = 102
    freeze_layer_params = True
    from samosa.vgg2samosa import load_vgg
    
    net = load_vgg(   
                            model = model,
                            dataset = dataset[0],
                            filename_params = filename_params,
                            optimization_params = optimization_params,
                            freeze_layer_params = freeze_layer_params,
                            visual_params = visual_params,
                            outs = arch_params["outs"],
                            verbose = verbose
                           ) 
    
    # load the inputs of the net. 
    # This is needed to create variables like batchsizes and input layer sizes.
    net.init_data ( dataset = dataset,
                    outs = arch_params ["outs"],
                    visual_params = visual_params,
                    verbose = verbose
                  )   
         
    # Build network.
    # This function creates both the forward and the backward. 
    # It also creates all the functions for training, testing and validation.
    # It establishes all connections and sets up the optimzier. 
    # build network is the important part of the code where all the action happens.                            
    net.build_network (verbose = verbose)   
    
    # This is needed to build the backprop part of the network
    net.build_learner (verbose = verbose)    
    # this function saves the network.  Just so that you don't have to lose everything.                                 
    # net.save_network ()        
    # This function goes through the epochs 
                   
    net.train  ( n_epochs = n_epochs, 
                 ft_epochs = ft_epochs,
                 validate_after_epochs = validate_after_epochs,
                 verbose = verbose
               )  
    net.save_network ()        
                      
    # this function tests on the dataset 
    net.test( verbose = verbose )       
             
    """ 
    # If you want to reload an already trained net follow these steps
    params_loaded, arch_params_loaded = load_network (filename_params ["network_save_name"] ,
                                        data_params = False, 
                                        optimization_params = False) 
    copy_classifier_layer   = True          # also copy the classifer layer too. 
                                            # Remember if the number of labels are now different from when it was trained, you need to False This.
    freeze_classifier_layer = False         # if you are going to fine tune or re-train the softmax layer, avoid this.
    
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
                            "network_save_name"     : "../results/network.pkl "
                    }
                    
    visual_params = {
                            "visualize_flag"        : True,
                            "visualize_after_epochs": 1,
                            "n_visual_images"       : 4,
                            "display_flag"          : False,
                            "color_filter"          : True         
                    }   
                                                                                                                            
    optimization_params = {
        
                            "mom"                         	    : (0.9, 0.95, 30),     # (mom_start, momentum_end, momentum_interval)                     
                            "mom_type"                          : 2,                    # 0-no mom, 1-polyak, 2-nestrov          
                            "learning_rate"                     : (0.01,0.001, 0.05 ),          # (initial_learning_rate, ft_learning_rate, annealing)
                            "reg"                               : (0.000,0.00051),       # l1_coeff, l2_coeff                                
                            "optim_type"                        : 2,                   # 0-SGD, 1-Adagrad, 2-RmsProp, 3-Adam
                            "objective"                         : 1,                   # 0-negative log likelihood, 1-categorical cross entropy, 2-binary cross entropy
                              
                                }       

    arch_params = {                    
                            "mlp_activations"                   : [  ReLU, ReLU, Softmax ],
                            "cnn_dropout"                       : False,
                            "mlp_dropout"                       : True,
                            "mlp_dropout_rates"                 : [ 0.5,  0.5, 0.5],
                            "num_nodes"                         : [ 450, 450 ],                                     
                            "outs"                              : 102,                                                                                                                               
                            "svm_flag"                          : False,                                       
                            "cnn_activations"                   : [ ReLU,   ReLU,   ReLU,    ],             
                            "cnn_batch_norm"                    : False,
                            "mlp_batch_norm"                    : True,
                            "nkerns"                            : [  20,    50,     50,         ],              
                            "filter_size"                       : [ (3,3),  (3,3),  (3,3),       ],
                            "pooling_size"                      : [ (2,2),  (2,2),  (2,2),       ],
                            "conv_pad"                          : [ 0,      0,      0,           ],                            
                            "pooling_type"                      : [ 1,      1,      1,           ],
                            "maxrandpool_p"                     : [ 1,      1,      1,           ],                           
                            "conv_stride_size"                  : [ (1,1),  (1,1),  (1,1),       ],
                            "cnn_maxout"                        : [ 1,      1,      1,           ],                    
                            "mlp_maxout"                        : [ 1,      1,      1,           ],
                            "cnn_dropout_rates"                 : [ 0.5,    0.5,    0.5,         ],
                            "random_seed"                       : 23455,
                            "mean_subtract"                     : False,
                            "use_bias"                          : True,
                            "max_out"                           : 0 
                 }                          

    # other loose parameters. 
    n_epochs = 75                 # number of epochs to run unless early terminated
    validate_after_epochs = 1      # number of epochs after which to validate.    
    ft_epochs = 0                # number of epoch to finetune learning with.
    verbose = False                 # if True makes a lot of prints, if False doesn't. 
    
    # code to tutor on how to setup and run. 
    run_cnn(
                    arch_params             = arch_params,
                    optimization_params     = optimization_params,
                    dataset                 = "_datasets/_dataset_23837", 
                    filename_params         = filename_params,          
                    visual_params           = visual_params, 
                    validate_after_epochs   = validate_after_epochs,
                    n_epochs                = n_epochs,
                    ft_epochs               = ft_epochs, 
                    verbose                 = verbose ,                                                
                )                 
    pdb.set_trace()                             