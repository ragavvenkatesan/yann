#!/usr/bin/python

# General Packages
import os
from collections import OrderedDict
import cPickle, gzip

# Math Packages
import numpy
import cv2
import time
from math import floor, ceil

# Theano Packages
import theano
import theano.tensor as T
from theano.ifelse import ifelse

# CNN code packages
import core
import util
import dataset

# From the Theano Tutorials
def shared_dataset(data_xy, n_classes, borrow=True, svm_flag = True):

    data_x, data_y = data_xy
    
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype='int32'),  borrow=borrow)
                                    
    if svm_flag is True:
        # one-hot encoded labels as {-1, 1}
        #n_classes = len(numpy.unique(data_y))  # dangerous?
        y1 = -1 * numpy.ones((data_y.shape[0], n_classes))
        y1[numpy.arange(data_y.shape[0]), data_y] = 1
        shared_y1 = theano.shared(numpy.asarray(y1,dtype=theano.config.floatX), borrow=borrow)
        
        return shared_x, shared_y, shared_y1
    else:
        return shared_x, shared_y  

class network(object):

    def __init__(  self, random_seed,
                         filename_params, 
                         verbose = False, 
                        ):

        self.results_file_name   = filename_params [ "results_file_name" ]                
        # Files that will be saved down on completion Can be used by the parse.m file
        self.error_file_name     = filename_params [ "error_file_name" ]
        self.cost_file_name      = filename_params [ "cost_file_name"  ]
        self.confusion_file_name = filename_params [ "confusion_file_name" ]
        self.network_save_name   = filename_params [ "network_save_name" ]
        
        self.rng = numpy.random.RandomState(random_seed)  
        self.main_img_visual = True 
       
    def save_network( self ):          # for others use only data_params or optimization_params

        f = gzip.open(self.network_save_name, 'wb')
        for obj in [self.params, self.arch, self.data_struct, self.optim_params]:
            cPickle.dump(obj, f, protocol = cPickle.HIGHEST_PROTOCOL)
        f.close()  
        
    def load_data_init(self, dataset, verbose = False):
        # every dataset will have atleast one batch ..... load that.
        
        f = gzip.open(dataset + '/train/batch_0.pkl.gz', 'rb')
        train_data_x, train_data_y = cPickle.load(f)
        f.close()

        f = gzip.open(dataset + '/test/batch_0.pkl.gz', 'rb')
        test_data_x, test_data_y = cPickle.load(f)
        f.close()
        
        f = gzip.open(dataset + '/valid/batch_0.pkl.gz', 'rb')
        valid_data_x, valid_data_y = cPickle.load(f)
        f.close()
                  
        self.test_set_x, self.test_set_y, self.test_set_y1 = shared_dataset((test_data_x, test_data_y), n_classes = self.outs, svm_flag = True)
        self.valid_set_x, self.valid_set_y, self.valid_set_y1 = shared_dataset((valid_data_x, valid_data_y), self.outs, svm_flag = True)
        self.train_set_x, self.train_set_y, self.train_set_y1 = shared_dataset((train_data_x, train_data_y), self.outs, svm_flag = True)
        
    # this loads up the data_params from a folder and sets up the initial databatch.         
    def init_data ( self, dataset, outs, verbose = False):
        print "... initializing dataset " + dataset 
        f = gzip.open(dataset + '/data_params.pkl.gz', 'rb')
        data_params = cPickle.load(f)
        f.close()
        
        self.outs                = outs
        self.data_struct         = data_params # This command makes it possible to save down
        self.dataset             = dataset
        self.data_type           = data_params [ "type" ]
        self.height              = data_params [ "height" ]
        self.width               = data_params [ "width" ]
        self.batch_size          = data_params [ "batch_size" ]    
        self.load_batches        = data_params [ "load_batches"  ] * self.batch_size
        if self.load_batches < self.batch_size and (self.dataset == "caltech101" or self.dataset == "caltech256"):
            AssertionError("load_batches is improper for this self.dataset " + self.dataset)
        self.batches2train       = data_params [ "batches2train" ]
        self.batches2test        = data_params [ "batches2test" ]
        self.batches2validate    = data_params [ "batches2validate" ] 
        self.channels            = data_params [ "channels" ]
        self.multi_load          = data_params [ "multi_load" ]
        self.n_train_batches     = data_params [ "n_train_batches" ]
        self.n_test_batches      = data_params [ "n_test_batches" ]
        self.n_valid_batches     = data_params [ "n_valid_batches" ] 
        
        self.load_data_init(self.dataset, verbose)

        
    # define the optimzer function 
    def build_network (self, arch_params, optimization_params , retrain_params = None, init_params = None, verbose = True):    
        self.optim_params                    = optimization_params 
        self.mom_start                       = optimization_params [ "mom_start" ]
        self.mom_end                         = optimization_params [ "mom_end" ]
        self.mom_epoch_interval              = optimization_params [ "mom_interval" ]
        self.mom_type                        = optimization_params [ "mom_type" ]
        self.initial_learning_rate           = optimization_params [ "initial_learning_rate" ]  
        self.ft_learning_rate                = optimization_params [ "ft_learning_rate" ]          
        self.learning_rate_decay             = optimization_params [ "learning_rate_decay" ] 
        self.ada_grad                        = optimization_params [ "ada_grad" ]   
        self.fudge_factor                    = optimization_params [ "fudge_factor" ]
        self.l1_reg                          = optimization_params [ "l1_reg" ]
        self.l2_reg                          = optimization_params [ "l2_reg" ]
        self.rms_prop                        = optimization_params [ "rms_prop" ]
        self.rms_rho                         = optimization_params [ "rms_rho" ]
        self.rms_epsilon                     = optimization_params [ "rms_epsilon" ]
        self.objective                       = optimization_params [ "objective" ]        
    
        self.arch                            = arch_params
        self.squared_filter_length_limit     = arch_params [ "squared_filter_length_limit" ]   
        self.mlp_activations                 = arch_params [ "mlp_activations"  ] 
        self.cnn_activations                 = arch_params [ "cnn_activations" ]
        self.cnn_dropout                     = arch_params [ "cnn_dropout"  ]
        self.mlp_dropout                     = arch_params [ "mlp_dropout"  ]
        self.batch_norm                      = arch_params [ "cnn_batch_norm"  ]  
        self.mlp_batch_norm                  = arch_params [ "mlp_batch_norm" ]  
        self.mlp_dropout_rates               = arch_params [ "mlp_dropout_rates" ]
        self.cnn_dropout_rates               = arch_params [ "cnn_dropout_rates" ]
        self.nkerns                          = arch_params [ "nkerns"  ]
        self.outs                            = arch_params [ "outs" ]
        self.filter_size                     = arch_params [ "filter_size" ]
        self.pooling_size                    = arch_params [ "pooling_size" ]
        self.conv_stride_size                = arch_params [ "conv_stride_size" ]
        self.num_nodes                       = arch_params [ "num_nodes" ]
        random_seed                          = arch_params [ "random_seed" ]
        self.svm_flag                        = arch_params [ "svm_flag" ]   
        self.mean_subtract                   = arch_params [ "mean_subtract" ]
        self.max_out                         = arch_params [ "max_out" ] 
        self.cnn_maxout                      = arch_params [ "cnn_maxout" ]   
        self.mlp_maxout                      = arch_params [ "mlp_maxout" ]
       
        if retrain_params is not None:
            self.copy_from_old = retrain_params [ "copy_from_old" ]
            self.freeze_layers = retrain_params [ "freeze" ]
        else:
            self.freeze_layers = [] 
            for i in xrange(len(self.nkerns) + len(self.num_nodes) + 1):
                self.freeze_layers.append ( False )              
            
        # if no retrain specified but if init params are given, make default as copy all params.             
            if init_params is not None:
                self.copy_from_old = []
                for i in xrange(len(self.nkerns) + len(self.num_nodes) + 1):
                    self.copy_from_old.append ( True ) 
                                                      
        if self.ada_grad is True:
            assert self.rms_prop is False
        elif self.rms_prop is True:
            assert self.ada_grad is False
            self.fudge_factor = self.rms_epsilon
       
        print '... building the network'    
        
        
        start_time = time.clock()
        # allocate symbolic variables for the data
        index = T.lscalar('index')  # index to a [mini]batch
        x = T.matrix('x')           # the data is presented as rasterized images
        y = T.ivector('y')          # the labels are presented as 1D vector of [int] 
            
        if self.svm_flag is True:
            y1 = T.matrix('y1')     # [-1 , 1] labels in case of SVM    
     
        first_layer_input = x.reshape((self.batch_size, self.height, self.width, self.channels)).dimshuffle(0,3,1,2)
        mean_sub_input = first_layer_input - first_layer_input.mean()
        
        # whenever we setup data, convert from the above rehsape order//
    
        # Create first convolutional - pooling layers 
        activity = []       # to record Cnn activities 
        self.weights = []
    
        conv_layers = []         
        dropout_conv_layers = [] 
        
        if not self.nkerns == []:
            filt_size = self.filter_size[0]
            pool_size = self.pooling_size[0]
            stride    = self.conv_stride_size[0]
            batch_norm_layer = self.batch_norm[0]
            
            if retrain_params is not None:
                curr_copy = self.copy_from_old[0] 
                
                if curr_copy is True:
                    curr_init_weights = init_params[0]
                    curr_init_bias    = init_params[1]
                    
                    if batch_norm_layer is True:
                        curr_init_alpha    = init_params[2]
                    else:
                        curr_init_alpha    = None
                else:
                    curr_init_weights = None
                    curr_init_bias = None
                    curr_init_alpha = None                    
                    
            if init_params is None:
                curr_init_weights = None
                curr_init_bias = None 
                curr_init_alpha = None
                
        if self.max_out > 0:  
            if self.nkerns == []:
                max_out_size = self.mlp_maxout[0]
            else:
                max_out_size = self.cnn_maxout[0]                
        else: 
            max_out_size = 1

        next_in = [ self.height, self.width, self.channels ]
        stack_size = 1 
        param_counter = 0 
 
        if not self.nkerns == []:     
            if len(filt_size) == 2:                        
                dropout_conv_layers.append ( 
                                core.DropoutConv2DPoolLayer(
                                        rng = self.rng,
                                        input = first_layer_input if self.mean_subtract is False else mean_sub_input,
                                        image_shape=(self.batch_size, self.channels , self.height, self.width),
                                        filter_shape=(self.nkerns[0], self.channels , filt_size[0], filt_size[1]),
                                        poolsize = pool_size,
                                        stride = stride,
                                        max_out = self.max_out,
                                        maxout_size = max_out_size,
                                        activation = self.cnn_activations[0],
                                        W = None if curr_init_weights is None else curr_init_weights,
                                        b = None if curr_init_bias is None else curr_init_bias, 
                                        batch_norm = batch_norm_layer,
                                        alpha = None if curr_init_alpha is None else curr_init_alpha,
                                        p = self.cnn_dropout_rates[0],                                 
                                         ) ) 
                conv_layers.append ( 
                                core.Conv2DPoolLayer(
                                        rng = self.rng,
                                        input = first_layer_input if self.mean_subtract is False else mean_sub_input,
                                        image_shape=(self.batch_size, self.channels , self.height, self.width),
                                        filter_shape=(self.nkerns[0], self.channels , filt_size[0], filt_size[1]),
                                        poolsize = pool_size,
                                        stride = stride,
                                        max_out = self.max_out,
                                        maxout_size = max_out_size,
                                        activation = self.cnn_activations[0],
                                        W = dropout_conv_layers[-1].params[0] * (1 - self.cnn_dropout_rates[0]) ,
                                        b = dropout_conv_layers[-1].params[1],
                                        batch_norm = batch_norm_layer,
                                        alpha = dropout_conv_layers[-1].alpha,
                                        verbose = verbose                                       
                                         ) )  
                next_in[0] = int(floor((ceil( (self.height - filt_size [0])  / float(stride[0])) + 1  ) / pool_size[0] ))       
                next_in[1] = int(floor((ceil( (self.width - filt_size[1]) / float(stride[1]   )) + 1  ) / pool_size[1] ))    
                next_in[2] = self.nkerns[0]  / max_out_size                                                                                                                 
            elif len(filt_size) == 3:
                dropout_conv_layers.append ( 
                                core.DropoutConv3DPoolLayer(
                                        rng = self.rng,
                                        input = first_layer_input if self.mean_subtract is False else mean_sub_input,
                                        image_shape=(self.batch_size, self.channels , stack_size, self.height, self.width),
                                        filter_shape=(self.nkerns[0], filt_size[0] , stack_size, filt_size[1], filt_size[2]),
                                        poolsize=pool_size,      
                                        stride = stride,                                   
                                        max_out = self.max_out,
                                        maxout_size = max_out_size,
                                        activation = self.cnn_activations[0],
                                        W = None if curr_init_weights is None else curr_init_weights,
                                        b = None if curr_init_bias is None else curr_init_bias, 
                                        batch_norm = batch_norm_layer,
                                        alpha = None if curr_init_alpha is None else curr_init_alpha,
                                        p = self.cnn_dropout_rates[0]                             
                                         ) )
                conv_layers.append ( 
                                core.Conv3DPoolLayer(
                                        rng = self.rng,
                                        input = first_layer_input if self.mean_subtract is False else mean_sub_input,
                                        image_shape=(self.batch_size, self.channels , stack_size, self.height, self.width),
                                        filter_shape=(self.nkerns[0], filt_size[0] , stack_size, filt_size[1], filt_size[2]),
                                        poolsize=pool_size,
                                        stride = stride, 
                                        max_out = self.max_out,
                                        maxout_size = max_out_size,                                        
                                        activation = self.cnn_activations[0],
                                        W = dropout_conv_layers[-1].params[0] * (1 - self.cnn_dropout_rates[0]),
                                        b = dropout_conv_layers[-1].params[1],
                                        batch_norm = batch_norm_layer,
                                        alpha = dropout_conv_layers[-1].alpha, 
                                        verbose = verbose
                                         ) )
                                          
                # strides creates a mess in 3D !! 
               
                next_in[0] = int(ceil(((( self.height - filt_size [1])  / float(stride[1])) + 1) / pool_size[1] ))       
                next_in[1] = int(ceil(((( self.width - filt_size[2]) / float(stride[2]  )) + 1) / pool_size[2] ))                                                                                                
                next_in[2] = self.nkerns[0]  / (pool_size[0] * max_out_size * stride[0])

                   
            else:
                print "!! So far Samosa is only capable of 2D and 3D conv layers."                               
                sys.exit()
                
            activity.append ( conv_layers[-1].output.dimshuffle(0,2,3,1) )
            self.weights.append ( conv_layers[-1].W)
    
    
            # Create the rest of the convolutional - pooling layers in a loop
            param_counter = param_counter + 2  
            if batch_norm_layer is True:
                param_counter = param_counter + 1
            for layer in xrange(len(self.nkerns)-1):   
                
                filt_size = self.filter_size[layer+1]
                pool_size = self.pooling_size[layer+1]
                stride    = self.conv_stride_size[layer +1 ]
                batch_norm_layer = self.batch_norm [layer + 1]
                if retrain_params is not None:
                    curr_copy = self.copy_from_old[layer + 1] 
                    if curr_copy is True:
                        curr_init_weights = init_params[param_counter]
                        curr_init_bias    = init_params[param_counter + 1]
                        if batch_norm_layer is True:
                            curr_init_alpha = init_params[param_counter + 2]   
                        else:
                            curr_init_alpha = None
                    else:
                        curr_init_weights  = None
                        curr_init_bias = None
                        curr_init_alpha = None          
                if init_params is None:
                    curr_init_weights = None
                    curr_init_bias = None
                    curr_init_alpha = None
                     
                if self.max_out > 0:
                    max_out_size = self.cnn_maxout[layer+1]
                else:
                    max_out_size = 1 

                if len(filt_size) == 2:
                    dropout_conv_layers.append ( 
                                    core.DropoutConv2DPoolLayer(
                                        rng = self.rng,
                                        input = dropout_conv_layers[layer].output,        
                                        image_shape=(self.batch_size, next_in[2], next_in[0], next_in[1]),
                                        filter_shape=(self.nkerns[layer+1], next_in[2], filt_size[0], filt_size[1]),
                                        poolsize=pool_size,
                                        stride = stride,
                                        max_out = self.max_out,
                                        maxout_size = max_out_size,
                                        activation = self.cnn_activations[layer+1],
                                        W = None if curr_init_weights is None else curr_init_weights ,
                                        b = None if curr_init_bias is None else curr_init_bias ,
                                        batch_norm = batch_norm_layer,
                                        alpha = None if curr_init_alpha is None else curr_init_alpha ,
                                        p = self.cnn_dropout_rates[layer+1]                                                                                                        
                                         ) )
                                                 
                    conv_layers.append ( 
                                    core.Conv2DPoolLayer(
                                        rng = self.rng,
                                        input = conv_layers[layer].output,        
                                        image_shape=(self.batch_size, next_in[2], next_in[0], next_in[1]),
                                        filter_shape=(self.nkerns[layer+1], next_in[2], filt_size[0], filt_size[1]),
                                        poolsize=pool_size,
                                        stride = stride,
                                        max_out = self.max_out,
                                        maxout_size = max_out_size,
                                        activation = self.cnn_activations[layer+1],
                                        W = dropout_conv_layers[-1].params[0] * (1 - self.cnn_dropout_rates[layer + 1]),
                                        b = dropout_conv_layers[-1].params[1],
                                        batch_norm = batch_norm_layer, 
                                        alpha = dropout_conv_layers[-1].alpha,
                                        verbose = verbose
                                         ) )                                                       
                                             
                    next_in[0] = int(floor((ceil( (next_in[0] - filt_size[0] ) / float(stride[0])) + 1 ) / pool_size[0] ))      
                    next_in[1] = int(floor((ceil( (next_in[1]- filt_size[1] ) / float(stride[1])) + 1 ) / pool_size[1] ))
                    next_in[2] = self.nkerns[layer+1] / max_out_size
                    
                elif len(filt_size) == 3:
                    dropout_conv_layers.append ( 
                                    core.DropoutConv3DPoolLayer(
                                        rng = self.rng,
                                        input = dropout_conv_layers[layer].output,        
                                        image_shape=(self.batch_size, next_in[2], stack_size, next_in[0], next_in[1]),
                                        filter_shape=(self.nkerns[layer+1], filt_size[0], stack_size, filt_size[1], filt_size[2]),
                                        poolsize=pool_size,
                                        stride = stride,
                                        max_out = self.max_out,
                                        maxout_size = max_out_size,
                                        activation = self.cnn_activations[layer+1],
                                        W = None if init_params is None else init_params[param_counter    ] ,
                                        b = None if init_params is None else init_params[param_counter + 1] ,
                                        batch_norm = batch_norm_layer,  
                                        alpha = None if curr_init_alpha is None else curr_init_alpha,
                                        p = self.cnn_dropout_rates[layer+1]                                                                                                       
                                         ) )                                                                                             
                    conv_layers.append ( 
                                    core.Conv3DPoolLayer(
                                        rng = self.rng,
                                        input = conv_layers[layer].output,        
                                        image_shape=(self.batch_size, next_in[2], stack_size, next_in[0], next_in[1]),
                                        filter_shape=(self.nkerns[layer+1], filt_size[0], stack_size, filt_size[1], filt_size[2]),
                                        poolsize=pool_size,
                                        stride = stride,
                                        max_out = self.max_out,
                                        maxout_size = max_out_size,
                                        activation = self.cnn_activations[layer+1],
                                        W = dropout_conv_layers[-1].params[0] * (1 - self.cnn_dropout_rates[layer + 1]),
                                        b = dropout_conv_layers[-1].params[1] ,
                                        batch_norm = batch_norm_layer,
                                        alpha = dropout_conv_layers[-1].alpha,
                                        verbose = verbose
                                         ) )   
                                         
                    # please dont use stride for 3D                                                   
                    next_in[0] = int(floor(( next_in[0] - filt_size[1] + 1 ))) / (pool_size[1] * stride[1])    
                    next_in[1] = int(floor(( next_in[1] - filt_size[2] + 1 ))) / (pool_size[2] * stride[2])
                    next_in[2] = self.nkerns[layer+1] / (pool_size[0] * max_out_size * stride[0])    
                                              
                else:
                    print "!! So far Samosa is only capable of 2D and 3D conv layers."                               
                    sys.exit()
                self.weights.append ( conv_layers[-1].W )
                activity.append( conv_layers[-1].output.dimshuffle(0,2,3,1) )

                param_counter = param_counter + 2    
                if batch_norm_layer is True:
                    param_counter = param_counter + 1           
        # Assemble fully connected laters
        if self.nkerns == []: # If there is no convolutional layer 
            fully_connected_input = first_layer_input.flatten(2) if self.mean_subtract is False else mean_sub_input.flatten(2)
            dropout_fully_connected_input = first_layer_input.flatten(2) if self.mean_subtract is False else mean_sub_input.flatten(2)            
        else:
            fully_connected_input = conv_layers[-1].output.flatten(2)
            dropout_fully_connected_input = dropout_conv_layers[-1].output.flatten(2)                
    
        if len(self.num_nodes) > 1 :
            layer_sizes =[]                        
            layer_sizes.append( next_in[0] * next_in[1] * next_in[2] )
            
            for i in xrange(len(self.num_nodes)):
                layer_sizes.append ( self.num_nodes[i] )
            layer_sizes.append ( self.outs )
            
        elif self.num_nodes == [] :
            
            layer_sizes = [ next_in[0] * next_in[1] * next_in[2], self.outs]
        elif len(self.num_nodes) ==  1:
            layer_sizes = [ next_in[0] * next_in[1] * next_in[2], self.num_nodes[0] , self.outs]
     
        assert len(layer_sizes) - 2 == len(self.num_nodes)           # Just checking.
    
        """  Dropouts implemented from paper:
        Srivastava, Nitish, et al. "Dropout: A simple way to prevent neural networks
        from overfitting." The Journal of Machine Learning Research 15.1 (2014): 1929-1958.
        """
        
        MLPlayers = core.MLP( rng = self.rng,
                         input = (fully_connected_input, dropout_fully_connected_input),
                         layer_sizes = layer_sizes,
                         dropout_rates = self.mlp_dropout_rates,
                         maxout_rates = self.mlp_maxout,
                         max_out = self.max_out, 
                         activations = self.mlp_activations,
                         use_bias = True,
                         svm_flag = self.svm_flag,
                         batch_norm = self.mlp_batch_norm, 
                         params = [] if init_params is None else init_params[param_counter:],
                         copy_from_old = self.copy_from_old [len(self.nkerns):] if init_params is not None else None,
                         freeze = self.freeze_layers [ len(self.nkerns):],
                         verbose = verbose)
    
        # create theano functions for evaluating the graph
        # I don't like the idea of having test model only hooked to the test_set_x variable.
        # I would probably have liked to have only one data variable.. but theano tutorials is using 
        # this style, so wth, so will I. 
                                          
        self.test_model = theano.function(
                inputs = [index],
                outputs = MLPlayers.errors(y),
                givens={
                    x: self.test_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                    y: self.test_set_y[index * self.batch_size:(index + 1) * self.batch_size]})
    
        self.validate_model = theano.function(
                inputs = [index],
                outputs = MLPlayers.errors(y),
                givens={
                    x: self.valid_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                    y: self.valid_set_y[index * self.batch_size:(index + 1) * self.batch_size]})
    
        self.prediction = theano.function(
            inputs = [index],
            outputs = MLPlayers.predicts,
            givens={
                    x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size]})
    
        self.nll = theano.function(
            inputs = [index],
            outputs = MLPlayers.probabilities,
            givens={
                x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size]})
    
        # function to return activations of each image
        if not self.nkerns == [] :
            self.activities = theano.function (
                inputs = [index],
                outputs = activity,
                givens = {
                        x: self.train_set_x[index * self.batch_size: (index + 1) * self.batch_size]
                        })
    
        # Compute cost and gradients of the model wrt parameter
        self.params = []
        count = 0
        for layer in dropout_conv_layers:
            if self.freeze_layers[count] is False:
                self.params = self.params + layer.params
            elif verbose is True:
                print "           -->        convolutional layer " + str(count +  1) + " is frozen." 
            count = count + 1 
        self.params = self.params + MLPlayers.params
       
        # Build the expresson for the categorical cross entropy function.
        if self.svm_flag is False:
            if self.objective == 0:
                cost = MLPlayers.negative_log_likelihood( y )
                dropout_cost = MLPlayers.dropout_negative_log_likelihood( y )
            elif self.objective == 1:
                if len(numpy.unique(self.train_set_y.eval())) > 2:
                    cost = MLPlayers.cross_entropy ( y )
                    dropout_cost = MLPlayers.dropout_cross_entropy ( y )
                else:
                    cost = MLPlayers.binary_entropy ( y )
                    dropout_cost = MLPlayers.dropout_binary_entropy ( y )
            else:
                print "!! Objective is not understood, switching to cross entropy"
                cost = MLPlayers.cross_entropy ( y )
                dropout_cost = MLPlayers.dropout_cross_entropy ( y )
    
        else :        
            cost = MLPlayers.hinge_loss( y1 )
            dropout_cost = MLPlayers.hinge_loss( y1 )
            
            
        output = ( dropout_cost + self.l1_reg * MLPlayers.dropout_L1 + self.l2_reg *
                             MLPlayers.dropout_L2 )if self.mlp_dropout else ( cost + self.l1_reg 
                             * MLPlayers.L1 + self.l2_reg * MLPlayers.L2)
    
        if verbose is True:
            print "... estimating gradients"
        gradients = []      
        for param in self.params: 
            gradient = T.grad( output ,param)
            gradients.append ( gradient )
    
        # TO DO: Try implementing Adadelta also. 
        # Compute momentum for the current epoch
        epoch = T.scalar()
        mom = ifelse(epoch <= self.mom_epoch_interval,
            self.mom_start*(1.0 - epoch/self.mom_epoch_interval) + self.mom_end*(epoch/self.mom_epoch_interval),
            self.mom_end)
    
        # learning rate
        self.eta = theano.shared(numpy.asarray(self.initial_learning_rate,dtype=theano.config.floatX))
        # accumulate gradients for adagrad
         
        grad_acc =[]
        for param in self.params:
            eps = numpy.zeros_like(param.get_value(borrow=True), dtype=theano.config.floatX)   
            grad_acc.append(theano.shared(eps, borrow=True))
    
        # accumulate velocities for momentum
        velocities = []
        for param in self.params:
            velocity = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,dtype=theano.config.floatX))
            velocities.append(velocity)
         
        # create updates for each combination of stuff 
        updates = OrderedDict()
        print_flag = False
         
        if verbose is True:
            print "... building back prop network" 
        for velocity, gradient, acc , param in zip(velocities, gradients, grad_acc, self.params):        
            if self.ada_grad is True:
    
                """ Adagrad implemented from paper:
                John Duchi, Elad Hazan, and Yoram Singer. 2011. Adaptive subgradient methods
                for online learning and stochastic optimization. JMLR
                """
    
                current_acc = acc + T.sqr(gradient) # Accumulates Gradient 
                updates[acc] = current_acc          # updates accumulation at timestamp
    
    
            elif self.rms_prop is True:
                """ Tieleman, T. and Hinton, G. (2012):
                Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
                Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)"""
    
                current_acc = self.rms_rho * acc + (1 - self.rms_rho) * T.sqr(gradient) 
                updates[acc] = current_acc
    
            else:
                current_acc = 1
                self.fudge_factor = 0
    
            if self.mom_type == 0:               # no momentum
                updates[velocity] = -(self.eta / T.sqrt(current_acc + self.fudge_factor)) * gradient                                            
               
            elif self.mom_type == 1:       # if polyak momentum    
    
                """ Momentum implemented from paper:  
                Polyak, Boris Teodorovich. "Some methods of speeding up the convergence of iteration methods." 
                USSR Computational Mathematics and Mathematical Physics 4.5 (1964): 1-17.
    
                Adapted from Sutskever, Ilya, Hinton et al. "On the importance of initialization and momentum in deep learning." 
                Proceedings of the 30th international conference on machine learning (ICML-13). 2013.
                equation (1) and equation (2)"""   
    
                updates[velocity] = mom * velocity - (1.-mom) * ( self.eta / T.sqrt(current_acc+ self.fudge_factor))  * gradient                             
    
            elif self.mom_type == 2:             # Nestrov accelerated gradient 
    
                """Nesterov, Yurii. "A method of solving a convex programming problem with convergence rate O (1/k2)."
                Soviet Mathematics Doklady. Vol. 27. No. 2. 1983.
                Adapted from https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/ 
    
                Instead of using past params we use the current params as described in this link
                https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,"""
      
                updates[velocity] = mom * velocity - (1.-mom) * ( self.eta / T.sqrt(current_acc + self.fudge_factor))  * gradient                                 
                updates[param] = mom * updates[velocity] 
    
            else:
                if print_flag is False:
                    print_flag = True
                    print "!! Unrecognized mometum type, switching to no momentum."
                updates[velocity] = -( self.eta / T.sqrt(current_acc+ self.fudge_factor) ) * gradient                                              
                            
    
            if self.mom_type != 2:
                stepped_param  = param + updates[velocity]
            else:
                stepped_param = param + updates[velocity] + updates[param]
            column_norm = True #This I don't fully understand if its needed after BN is implemented.
            if param.get_value(borrow=True).ndim == 2 and column_norm is True:
    
                """ constrain the norms of the COLUMNs of the weight, according to
                https://github.com/BVLC/caffe/issues/109 """
    
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(self.squared_filter_length_limit))
                scale = desired_norms / (1e-7 + col_norms)
                updates[param] = stepped_param * scale
    
            else:            
                updates[param] = stepped_param
    
        if verbose is True:
            print "... building training model" 
        if self.svm_flag is True:
            self.train_model = theano.function(
                    inputs= [index, epoch],
                    outputs = output,
                    updates = updates,
                    givens={
                        x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                        y1: self.train_set_y1[index * self.batch_size:(index + 1) * self.batch_size]},
                    on_unused_input = 'ignore'                    
                        )
        else: 
            self.train_model = theano.function(
                    inputs = [index, epoch],
                    outputs = output,
                    updates = updates,
                    givens={
                        x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                        y: self.train_set_y[index * self.batch_size:(index + 1) * self.batch_size]},
                    on_unused_input='ignore'                    
                        )
    
        self.decay_learning_rate = theano.function(
               inputs=[],          # Just updates the learning rates. 
               updates={self.eta: self.eta -  self.eta * self.learning_rate_decay }
                )
    
        self.momentum_value = theano.function ( 
                            inputs =[epoch],
                            outputs = mom,
                            )
        end_time = time.clock()
        print "...         time taken is " +str(end_time - start_time) + " seconds"
                       
    # this is only for self.multi_load = True type of datasets.. 
    # All datasets are not multi_load enabled. This needs to change ??                         
    # this is only for self.multi_load = True type of datasets.. 
    # All datasets are not multi_load enabled. This needs to change ??    
    
    # from theano tutorials for loading pkl files like what is used in theano tutorials
    def load_data_base( self, batch = 1, type_set = 'train' ):
        # every dataset will have atleast one batch ..... load that.
        
        if type_set == 'train':
            f = gzip.open(self.dataset + '/train/batch_' +str(batch) +'.pkl.gz', 'rb')
        elif type_set == 'valid':
            f = gzip.open(self.dataset + '/valid/batch_' +str(batch) +'.pkl.gz', 'rb')            
        else:
            f = gzip.open(self.dataset + '/test/batch_' +str(batch) +'.pkl.gz', 'rb')
            
        data_x, data_y = cPickle.load(f)
        f.close()
    
        if self.svm_flag is True:
            # one-hot encoded labels as {-1, 1}
            n_classes = len(numpy.unique(data_y))  # dangerous?
            y1 = -1 * numpy.ones((data_y.shape[0], n_classes))
            y1[numpy.arange(data_y.shape[0]), data_y] = 1		
            rval = (data_x, data_y, y1)
        else:   
            rval = (data_x, data_y, data_y)
            
        return rval
                       
    def set_data (self, batch, type_set, verbose = True):
        
        data_x, data_y, data_y1 = self.load_data_base(batch, type_set )             
        
        # If we had used only one datavariable instead of three... this wouldn't have been needed. 
        if type_set == 'train':                             
            self.train_set_x.set_value(data_x ,borrow = True)
            self.train_set_y.set_value(data_y ,borrow = True)                
            if self.svm_flag is True:
                self.train_set_y1.set_value(data_y1, borrow = True)
        elif type_set == 'test':
            self.test_set_x.set_value(data_x ,borrow = True)
            self.test_set_y.set_value(data_y ,borrow = True)                
            if self.svm_flag is True:
                self.test_set_y1.set_value(data_y1, borrow = True)
        else:
            self.valid_set_x.set_value(data_x ,borrow = True)
            self.valid_set_y.set_value(data_y ,borrow = True)                
            if self.svm_flag is True:
                self.valid_set_y1.set_value(data_y1, borrow = True)
        
    def print_net (self, epoch, display_flag = True ):
        # saving down true images.    
        if self.main_img_visual is False:          
            imgs = self.train_set_x.reshape((self.train_set_x.shape[0].eval(),self.height,self.width,self.channels))                                 
            imgs = imgs.eval()[self.visualize_ind]
            loc_im = '../visuals/images/image_'
            imgs = util.visualize(imgs, prefix = loc_im, is_color = self.color_filter if self.channels == 3 else False)    
        self.main_img_visual = True
                
        # visualizing activities.
        activity_now = self.activities(0)     
        for m in xrange(len(self.nkerns)):   #For each layer 
            loc_ac = '../visuals/activities/layer_' + str(m) + "/epoch_" + str(epoch)
            if not os.path.exists(loc_ac):   
                os.makedirs(loc_ac)
            loc_ac = loc_ac + "/filter_"
            current_activity = activity_now[m]
            current_activity = current_activity[self.visualize_ind]                            
            imgs = util.visualize(current_activity, loc_ac, is_color = False)
            
            current_weights = self.weights[m]    # for each layer       
            loc_we = '../visuals/filters/layer_' + str(m) + "/epoch_" + str(epoch)
            if not os.path.exists(loc_we):   
                os.makedirs(loc_we)
            loc_we = loc_we + "/filter_"
            if len(current_weights.shape.eval()) == 5:
                imgs = util.visualize(numpy.squeeze(current_weights.dimshuffle(0,3,4,1,2).eval()), prefix = loc_we, is_color = self.color_filter)
            else:   
                imgs = util.visualize(current_weights.dimshuffle(0,2,3,1).eval(), prefix = loc_we, is_color = self.color_filter)            
            
    # ToDo: should make a results root dir and put in results there ... like root +'/visuals/' 
    def create_dirs( self, visual_params ):  
        
        self.visualize_flag          = visual_params ["visualize_flag" ]
        self.visualize_after_epochs  = visual_params ["visualize_after_epochs" ]
        self.n_visual_images         = visual_params ["n_visual_images" ] 
        self.display_flag            = visual_params ["display_flag" ]
        self.color_filter            = visual_params ["color_filter" ]
        self.shuffle_batch_ind = numpy.arange(self.batch_size)
        numpy.random.shuffle(self.shuffle_batch_ind)
        self.visualize_ind = self.shuffle_batch_ind[0:self.n_visual_images] 
        # create all directories required for saving results and data.
        if self.visualize_flag is True:
            if not os.path.exists('../visuals'):
                os.makedirs('../visuals')                
            if not os.path.exists('../visuals/activities'):
                os.makedirs('../visuals/activities')
                for i in xrange(len(self.nkerns)):
                    os.makedirs('../visuals/activities/layer_'+str(i))
            if not os.path.exists('../visuals/filters'):
                os.makedirs('../visuals/filters')
                for i in xrange(len(self.nkerns)):
                    os.makedirs('../visuals/filters/layer_'+str(i))
            if not os.path.exists('../visuals/images'):
                os.makedirs('../visuals/images')
        if not os.path.exists('../results/'):
            os.makedirs ('../results')
        
        assert self.batch_size >= self.n_visual_images
        
        
    # TRAIN 
    def train(self, n_epochs, ft_epochs, validate_after_epochs, verbose = True):
        print "... training"        
        self.main_img_visual = False
        patience = numpy.inf 
        patience_increase = 2  
        improvement_threshold = 0.995  
        this_validation_loss = []
        best_validation_loss = numpy.inf
        best_iter = 0
        epoch_counter = 0
        early_termination = False
        cost_saved = []
        iteration= 0        
        #self.print_net(epoch = 0, display_flag = self.display_flag)
        start_time_main = time.clock()
        if os.path.isfile('dump.txt'):
            f = open('dump.txt', 'a')
        else:
            f = open('dump.txt', 'w')
        while (epoch_counter < (n_epochs + ft_epochs)) and (not early_termination):
            if epoch_counter == n_epochs:
                print "... fine tuning"
                self.eta.set_value(self.ft_learning_rate)
            epoch_counter = epoch_counter + 1 
            start_time = time.clock() 
            for batch in xrange (self.batches2train):
                if verbose is True:
                    print "...          -> epoch: " + str(epoch_counter) + " batch: " + str(batch+1) + " out of " + str(self.batches2train) + " batches"
    
                if self.multi_load is True:
                    iteration= (epoch_counter - 1) * self.n_train_batches * self.batches2train + batch
                    # Load data for this batch
                    self.set_data ( batch = batch , type_set = 'train', verbose = verbose)
                    for minibatch_index in xrange(self.n_train_batches):
                        if verbose is True:
                            print "...                  ->    mini Batch: " + str(minibatch_index + 1) + " out of "    + str(self.n_train_batches)
                        cost_ij = self.train_model( minibatch_index, epoch_counter)
                        cost_saved = cost_saved + [cost_ij]                        
                else:   
                    iteration= (epoch_counter - 1) * self.n_train_batches + batch
                    cost_ij = self.train_model(batch, epoch_counter)
                    cost_saved = cost_saved +[cost_ij]
           
            if  epoch_counter % validate_after_epochs == 0:  
                validation_losses = 0.   
                if self.multi_load is True:                    
                    for batch in xrange ( self.batches2validate):                       
                        self.set_data ( batch = batch , type_set = 'valid' , verbose = verbose)
                        validation_losses = validation_losses + numpy.sum([[self.validate_model(i) for i in xrange(self.n_valid_batches)]])
                        this_validation_loss = this_validation_loss + [validation_losses]
   
                    print ("...      -> epoch " + str(epoch_counter) + 
                                         ", cost: " + str(numpy.mean(cost_saved[-1*self.n_train_batches:])) +
                                         ",  validation accuracy :" + str(float( self.batch_size * self.n_valid_batches * self.batches2validate - this_validation_loss[-1])*100
                                                                 /(self.batch_size*self.n_valid_batches*self.batches2validate)) +
                                         "%, learning_rate = " + str(self.eta.get_value(borrow=True))+ 
                                         ", momentum = " +str(self.momentum_value(epoch_counter))  +
                                         " -> best thus far ") if this_validation_loss[-1] < best_validation_loss else ("...      -> epoch " + str(epoch_counter) + 
                                         ", cost: " + str(numpy.mean(cost_saved[-1*self.n_train_batches:])) +
                                         ",  validation accuracy :" + str(float( self.batch_size * self.n_valid_batches * self.batches2validate - this_validation_loss[-1])*100
                                                                 /(self.batch_size*self.n_valid_batches*self.batches2validate)) +
                                         "%, learning_rate = " + str(self.eta.get_value(borrow=True))+ 
                                         ", momentum = " +str(self.momentum_value(epoch_counter)))     
                    f.write(("...      -> epoch " + str(epoch_counter) + 
                                         ", cost: " + str(numpy.mean(cost_saved[-1*self.n_train_batches:])) +
                                         ",  validation accuracy :" + str(float( self.batch_size * self.n_valid_batches * self.batches2validate - this_validation_loss[-1])*100
                                                                 /(self.batch_size*self.n_valid_batches*self.batches2validate)) +
                                         "%, learning_rate = " + str(self.eta.get_value(borrow=True))+ 
                                         ", momentum = " +str(self.momentum_value(epoch_counter))  +
                                         " -> best thus far ") if this_validation_loss[-1] < best_validation_loss else ("...      -> epoch " + str(epoch_counter) + 
                                         ", cost: " + str(numpy.mean(cost_saved[-1*self.n_train_batches:])) +
                                         ",  validation accuracy :" + str(float( self.batch_size * self.n_valid_batches * self.batches2validate - this_validation_loss[-1])*100
                                                                 /(self.batch_size*self.n_valid_batches*self.batches2validate)) +
                                         "%, learning_rate = " + str(self.eta.get_value(borrow=True))+ 
                                         ", momentum = " +str(self.momentum_value(epoch_counter))))
                    f.write('\n')
                else: # if not multi_load
                    
                    
                    if numpy.isnan(cost_saved[-1]):
                        print " NAN !! "
                        import pdb
                        pdb.set_trace()
                    validation_losses = [self.validate_model(i) for i in xrange(self.batches2validate)]
                    this_validation_loss = this_validation_loss + [numpy.sum(validation_losses)]
                                            
                    print ("...      -> epoch " + str(epoch_counter) + 
                              ", cost: " + str(cost_saved[-1]) +
                              ",  validation accuracy :" + str(float(self.batch_size*self.batches2validate - this_validation_loss[-1])*100
                                                           /(self.batch_size*self.batches2validate)) + 
                              "%, learning_rate = " + str(self.eta.get_value(borrow=True)) + 
                              ", momentum = " +str(self.momentum_value(epoch_counter)) +
                              " -> best thus far ") if this_validation_loss[-1] < best_validation_loss else ("...      -> epoch " + str(epoch_counter) + 
                              ", cost: " + str(cost_saved[-1]) +
                              ",  validation accuracy :" + str(float(self.batch_size*self.batches2validate - this_validation_loss[-1])*100
                                                           /(self.batch_size*self.batches2validate)) + 
                              "%, learning_rate = " + str(self.eta.get_value(borrow=True)) + 
                              ", momentum = " +str(self.momentum_value(epoch_counter)) )
                    f.write(("...      -> epoch " + str(epoch_counter) + 
                              ", cost: " + str(cost_saved[-1]) +
                              ",  validation accuracy :" + str(float(self.batch_size*self.batches2validate - this_validation_loss[-1])*100
                                                           /(self.batch_size*self.batches2validate)) + 
                              "%, learning_rate = " + str(self.eta.get_value(borrow=True)) + 
                              ", momentum = " +str(self.momentum_value(epoch_counter)) +
                              " -> best thus far ") if this_validation_loss[-1] < best_validation_loss else ("...      -> epoch " + str(epoch_counter) + 
                              ", cost: " + str(cost_saved[-1]) +
                              ",  validation accuracy :" + str(float(self.batch_size*self.batches2validate - this_validation_loss[-1])*100
                                                           /(self.batch_size*self.batches2validate)) + 
                              "%, learning_rate = " + str(self.eta.get_value(borrow=True)) + 
                              ", momentum = " +str(self.momentum_value(epoch_counter)) ) )
                    f.write('\n')
                # improve patience if loss improvement is good enough
                if this_validation_loss[-1] < best_validation_loss *  \
                   improvement_threshold:
                    patience = max(patience, iteration* patience_increase)
                    best_iter = iteration

                best_validation_loss = min(best_validation_loss, this_validation_loss[-1])
            self.decay_learning_rate()    
    
    
            if self.visualize_flag is True and epoch_counter % self.visualize_after_epochs == 0:            
                self.print_net (epoch = epoch_counter, display_flag = self.display_flag)   
            
            end_time = time.clock()
            print "...           time taken for this epoch is " +str((end_time - start_time)) + " seconds"
            
            if patience <= iteration:
                early_termination = True
                break
         
        end_time_main = time.clock()
        print "... time taken for the entire training is " +str((end_time_main - start_time_main)/60) + " minutes"
        f.close()            
        # Save down training stuff
        f = open(self.error_file_name,'w')
        for i in xrange(len(this_validation_loss)):
            f.write(str(this_validation_loss[i]))
            f.write("\n")
        f.close()
    
        f = open(self.cost_file_name,'w')
        for i in xrange(len(cost_saved)):
            f.write(str(cost_saved[i]))
            f.write("\n")
        f.close()
    
    def test(self, verbose = True):
        print "... testing"
        start_time = time.clock()
        wrong = 0
        predictions = []
        class_prob = []
        labels = []
         
        if self.multi_load is False:   
            labels = self.test_set_y.eval().tolist()   
            for mini_batch in xrange(self.batches2test):
                #print ".. Testing batch " + str(mini_batch)
                wrong = wrong + int(self.test_model(mini_batch))                        
                predictions = predictions + self.prediction(mini_batch).tolist()
                class_prob = class_prob + self.nll(mini_batch).tolist()
            print ("...      -> total test accuracy : " + str(float((self.batch_size*self.batches2test)-wrong )*100
                                                         /(self.batch_size*self.batches2test)) + 
                         " % out of " + str(self.batch_size*self.batches2test) + " samples.")
            f = open('dump.txt','a')
            
            f.write(("...      -> total test accuracy : " + str(float((self.batch_size*self.batches2test)-wrong )*100
                                                         /(self.batch_size*self.batches2test)) + 
                         " % out of " + str(self.batch_size*self.batches2test) + " samples."))
            f.write('\n')                         
            f.close()
                        
        else:           
            for batch in xrange(self.batches2test):
                if verbose is True:
                    print "..       --> testing batch " + str(batch)
                # Load data for this batch
                self.set_data ( batch = batch, type_set = 'test' , verbose = verbose)
                labels = labels + self.test_set_y.eval().tolist() 
                for mini_batch in xrange(self.n_test_batches):
                    wrong = wrong + int(self.test_model(mini_batch))   
                    predictions = predictions + self.prediction(mini_batch).tolist()
                    class_prob = class_prob + self.nll(mini_batch).tolist()
             
            print ("...      -> total test accuracy : " + str(float((self.batch_size*self.n_test_batches*self.batches2test)-wrong )*100/
                                                         (self.batch_size*self.n_test_batches*self.batches2test)) + 
                         " % out of " + str(self.batch_size*self.n_test_batches*self.batches2test) + " samples.")
            f = open('dump.txt','a')
            f.write(("...      -> total test accuracy : " + str(float((self.batch_size*self.n_test_batches*self.batches2test)-wrong )*100/
                                                         (self.batch_size*self.n_test_batches*self.batches2test)) + 
                         " % out of " + str(self.batch_size*self.n_test_batches*self.batches2test) + " samples."))
            f.write('\n')                         
            f.close()
        correct = 0 
        confusion = numpy.zeros((self.outs,self.outs), dtype = int)
        for index in xrange(len(predictions)):
            if labels[index] == predictions[index]:
                correct = correct + 1
            confusion[int(predictions[index]),int(labels[index])] = confusion[int(predictions[index]),int(labels[index])] + 1
    
        # Save down data 
        f = open(self.results_file_name, 'w')
        for i in xrange(len(predictions)):
            f.write(str(i))
            f.write("\t")
            f.write(str(labels[i]))
            f.write("\t")
            f.write(str(predictions[i]))
            f.write("\t")
            for j in xrange(self.outs):
                f.write(str(class_prob[i][j]))
                f.write("\t")
            f.write('\n')
        f.close() 

        numpy.savetxt(self.confusion_file_name, confusion, newline="\n")
        print "confusion Matrix with accuracy : " + str(float(correct)/len(predictions)*100) + "%"
        end_time = time.clock()
        print "...         time taken is " +str(end_time - start_time) + " seconds"
                
        if self.visualize_flag is True:    
            print "... saving down the final model's visualizations" 
            self.print_net (epoch = 'final' , display_flag = self.display_flag)     
            