#!/usr/bin/python
#from samosa.cnn import cnn_mlp
#from samosa.core import ReLU
import os
import pdb
import cPickle

import theano, numpy 

from samosa.cnn import cnn_mlp
from samosa.core import ReLU, Sigmoid, Softmax, Tanh, Abs, Squared
from samosa.util import load_network

class vgg_net(cnn_mlp):
    def init_data (self, dataset, outs, visual_params = None, verbose = False):
                
        super (vgg_net, self).init_data ( dataset = dataset, outs = outs, visual_params = visual_params, verbose = verbose )
        assert self.height == 224 and self.width == 224 and self.channels == 3   
        # May be instead of throwing an assertion error, I might consider a resizer function.                       
        
        if not self.outs == 1000:
            if verbose is True:
                print "... loading a non imagenet dataset. creating random nodes for the last layer."                        
            self.retrain_params["copy_from_old"][-1] = False
            self.retrain_params["freeze"][-1] = False 
            
def load_vgg(model, dataset, outs, optimization_params, filename_params, visual_params, freeze_layer_params = True, verbose = True):

    if (not os.path.isfile(model)):
            from six.moves import urllib
            origin = (
                'https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl'
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin,model)
    
    f = open(model)
    vgg_loaded = cPickle.load(f)
    vgg_params = vgg_loaded['param values']
    
    arch_params = {
                    "mlp_activations"                   : [ ReLU, ReLU, ReLU ],                     
                    "cnn_dropout"                       : False,
                    "mlp_dropout"                       : True,
                    "mlp_dropout_rates"                 : [ 0.5 , 0.5, 0.5 ],
                    "num_nodes"                         : [ 4096, 4096  ],  
                    "mlp_maxout"                        : [ 1,    1 ],     
                    "mlp_batch_norm"                    : False,                                                                      
                    "outs"                              : 1000,                                                                                                                               
                    "svm_flag"                          : False,                                       
                    "cnn_activations"                   : [ ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU,   ReLU    ],             
                    "cnn_batch_norm"                    : [ False,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False,  False   ],
                    "nkerns"                            : [ 64,     64,     128,    128,    256,    256,    256,    256,    512,    512,    512,    512,    512,    512,    512,    512     ],              
                    "filter_size"                       : [ (3,3), (3,3),   (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3),  (3,3)   ],
                    "pooling_size"                      : [ (1,1), (2,2),   (1,1),  (2,2),  (1,1),  (1,1),  (1,1),  (2,2),  (1,1),  (1,1),  (1,1),  (2,2),  (1,1),  (1,1),  (1,1),  (2,2)   ],
                    "conv_pad"                          : [ 2,     2,       2,      2,      2,      2,      2,      2,       2,      2,      2,      2,     2,      2,      2,      2,      ],                                       
                    "pooling_type"                      : [ 1,     1,       1,      1,      1,      1,      1,      1,       1,      1,      1,      1,     1,      1,      1,      1,      ],
                    "maxrandpool_p"                     : [ 1,     1,       1,      1,      1,      1,      1,      1,       1,      1,      1,      1,     1,      1,      1,      1,      ],                                                                                                    
                    "conv_stride_size"                  : [ (1,1), (1,1),   (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1),  (1,1)   ],
                    "cnn_maxout"                        : [ 1,     1,       1,      1,      1,      1,      1,      1,       1,      1,      1,      1,     1,      1,      1,      1,      ],                    
                    "cnn_dropout_rates"                 : [ 0,     0,       0,      0,      0,      0,      0,      0,       0,      0,      0,      0,     0,      0,      0,      0,      ],
                    "random_seed"                       : 23455, 
                    "mean_subtract"                     : False,
                    "use_bias"                          : True,                    
                    "max_out"                           : 0 
        
                 }                             
                           
    # Should these be input parameters ? 
    copy_classifier_layer   = False
    freeze_classifier_layer = False  
                                                              
    retrain_params = {
                            "copy_from_old"     : [True] * (len(arch_params["nkerns"]) + len(arch_params["num_nodes"])) + [copy_classifier_layer],
                            "freeze"            : [freeze_layer_params] * (len(arch_params["nkerns"]) + len(arch_params["num_nodes"])) + [freeze_classifier_layer]
                     } 
                          
    init_params = []
    for param_values in vgg_loaded['param values']:
        init_params.append(theano.shared(numpy.asarray(param_values, dtype = theano.config.floatX)))       
                                                
    net = vgg_net(   filename_params = filename_params,
                     arch_params = arch_params,
                     optimization_params = optimization_params,
                     retrain_params = retrain_params,
                     init_params = init_params,
                     verbose =verbose    )                                                                               
    return net                                                 

    
if __name__ == '__main__':

    model = './dataset/vgg/vgg19.pkl'
    dataset = "_datasets/_dataset_41982"
    outs = 102
    verbose = True
    freeze_layer_params = True
        
    optimization_params = {
                            "mom"                         	    : (0.5, 0.99, 100), # (mom_start, momentum_end, momentum_interval)                     
                            "mom_type"                          : 1,                # 0-no mom, 1-polyak, 2-nestrov          
                            "learning_rate"                     : (0.001,0.0001, 0.05 ),          # (initial_learning_rate, ft_learning_rate, annealing)
                            "reg"                               : (0.000,0.000),    # l1_coeff, l2_coeff                                
                            "optim_type"                        : 2,                # 0-SGD, 1-Adagrad, 2-RmsProp, 3-Adam
                            "objective"                         : 1,                # 0-negative log likelihood, 1-categorical cross entropy, 2-binary cross entropy
                            }
                                    
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
                        "n_visual_images"       : 36,
                        "display_flag"          : False,
                        "color_filter"          : True         
                    }   
                                        
    load_vgg(   model = model,
                dataset = dataset,
                filename_params = filename_params,
                optimization_params = optimization_params,
                freeze_layer_params = freeze_layer_params,
                visual_params = visual_params,
                outs = outs,
                verbose = verbose
            )   