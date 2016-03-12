#!/usr/bin/python

# General Packages
import os
import cPickle
import progressbar
import copy

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
    
class cnn_mlp(object):

    def __init__(  self, filename_params, 
                         optimization_params,
                         arch_params,   
                         retrain_params,
                         init_params,                      
                         verbose = False, 
                        ):

        self.results_file_name   = filename_params [ "results_file_name" ]                
        # Files that will be saved down on completion Can be used by the parse.m file
        self.error_file_name     = filename_params [ "error_file_name" ]
        self.cost_file_name      = filename_params [ "cost_file_name"  ]
        self.confusion_file_name = filename_params [ "confusion_file_name" ]
        self.network_save_name   = filename_params [ "network_save_name" ]

        self.rng = numpy.random  
        self.main_img_visual = True 
        
        self.optim_params                    = optimization_params           
        self.arch                            = arch_params
        self.mlp_activations                 = arch_params [ "mlp_activations"  ] 
        self.cnn_activations                 = arch_params [ "cnn_activations" ]
        self.cnn_dropout                     = arch_params [ "cnn_dropout"  ]
        self.mlp_dropout                     = arch_params [ "mlp_dropout"  ]
        self.cnn_batch_norm                  = arch_params [ "cnn_batch_norm"  ]  
        self.mlp_batch_norm                  = arch_params [ "mlp_batch_norm" ]  
        self.mlp_dropout_rates               = arch_params [ "mlp_dropout_rates" ]
        self.cnn_dropout_rates               = arch_params [ "cnn_dropout_rates" ]
        self.nkerns                          = arch_params [ "nkerns"  ]
        self.outs                            = arch_params [ "outs" ]
        self.filter_size                     = arch_params [ "filter_size" ]
        self.pooling_size                    = arch_params [ "pooling_size" ]
        self.conv_stride_size                = arch_params [ "conv_stride_size" ]
        self.conv_pad                        = arch_params [ "conv_pad" ]        
        self.num_nodes                       = arch_params [ "num_nodes" ]

        self.svm_flag                        = arch_params [ "svm_flag" ]   
        self.mean_subtract                   = arch_params [ "mean_subtract" ]
        self.max_out                         = arch_params [ "max_out" ] 
        self.cnn_maxout                      = arch_params [ "cnn_maxout" ]   
        self.mlp_maxout                      = arch_params [ "mlp_maxout" ]
        self.use_bias                        = arch_params [ "use_bias" ]  
        self.pooling_type                    = arch_params [ "pooling_type"]  
        self.maxrandpool_p                   = arch_params [ "maxrandpool_p"]          

        self.retrain_params = retrain_params
        self.init_params    = init_params 
        
        self.ft_learning_rate = self.optim_params["learning_rate"][1] 
        self.learning_rate_decay = self.optim_params["learning_rate"][2]      
                
    def save_network( self, name = None, verbose = False ):          # for others use only data_params or optimization_params
        if verbose is True:
            print "   dumping network"
        if name is None:
            name = self.network_save_name
        f = open(name, 'wb')                   
        for obj in [self.params, self.arch, self.data_struct, self.optim_params]:
            cPickle.dump(obj, f, protocol = cPickle.HIGHEST_PROTOCOL)
        f.close()  
        
    def load_data_init(self, dataset, verbose = False):
        # every dataset will have atleast one batch ..load that.
        
        f = open(dataset + '/train/batch_0.pkl', 'rb')
        train_data_x, train_data_y = cPickle.load(f)
        f.close()

        f = open(dataset + '/test/batch_0.pkl', 'rb')
        test_data_x, test_data_y = cPickle.load(f)
        f.close()
        
        f = open(dataset + '/valid/batch_0.pkl', 'rb')
        valid_data_x, valid_data_y = cPickle.load(f)
        f.close()
                  
        self.test_set_x, self.test_set_y, self.test_set_y1 = core.shared_dataset((test_data_x, test_data_y), n_classes = self.outs, svm_flag = True)
        self.valid_set_x, self.valid_set_y, self.valid_set_y1 = core.shared_dataset((valid_data_x, valid_data_y), self.outs, svm_flag = True)
        self.train_set_x, self.train_set_y, self.train_set_y1 = core.shared_dataset((train_data_x, train_data_y), self.outs, svm_flag = True)
        
    # this loads up the data_params from a folder and sets up the initial databatch.         
    def init_data ( self, dataset, outs, visual_params = None, verbose = False):
        print "initializing dataset " + dataset 
        f = open(dataset + '/data_params.pkl', 'rb')
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
        if visual_params is not None:
            self.create_dirs ( visual_params = visual_params )       
        
    # define the optimzer function 
    def build_network (self, verbose = True):    
                     
       
        print 'building the network'    
        
        
        start_time = time.clock()
        # allocate symbolic variables for the data       
        self.x = T.matrix('x')           # the data is presented as rasterized images
        self.y = T.ivector('y')          # the labels are presented as 1D vector of [int]                     
        if self.svm_flag is True:
            self.y1 = T.matrix('y1')     # [-1 , 1] labels in case of SVM    
     
        first_layer_input = self.x.reshape((self.batch_size, self.height, self.width, self.channels)).dimshuffle(0,3,1,2)
        mean_sub_input = first_layer_input - first_layer_input.mean()        
                
        next_in = mean_sub_input if self.mean_subtract is True else first_layer_input 
        ###########################################
        # Convolutional layers      
        if not self.nkerns == []: # If there are some convolutional layers             
            self.ConvLayers = core.ConvolutionalLayers (      
                                                    input = next_in,
                                                    rng = self.rng,
                                                    input_size = (self.height, self.width, self.channels, self.batch_size), 
                                                    mean_subtract = self.mean_subtract,
                                                    nkerns = self.nkerns,
                                                    filter_size = self.filter_size,
                                                    pooling_size = self.pooling_size,
                                                    pooling_type = self.pooling_type,
                                                    maxrandpool_p = self.maxrandpool_p,
                                                    cnn_activations = self.cnn_activations,
                                                    conv_stride_size = self.conv_stride_size,
                                                    cnn_dropout_rates = self.cnn_dropout_rates,
                                                    conv_pad = self.conv_pad,
                                                    batch_norm = self.cnn_batch_norm,         
                                                    max_out = self.max_out,
                                                    cnn_maxout = self.cnn_maxout,
                                                    retrain_params = self.retrain_params, # default None
                                                    init_params = self.init_params,                                   
                                                    verbose = verbose          
                                              )                                                                             
        # Assemble fully connected laters
        if self.nkerns == []: # If there is no convolutional layer 
            fully_connected_input = first_layer_input.flatten(2) if self.mean_subtract is False else mean_sub_input.flatten(2)
            dropout_fully_connected_input = first_layer_input.flatten(2) if self.mean_subtract is False else mean_sub_input.flatten(2) 
            next_in = self.height, self.width, self.channels                

        else:
            fully_connected_input = self.ConvLayers.conv_layers[-1].output.flatten(2)
            dropout_fully_connected_input = self.ConvLayers.dropout_conv_layers[-1].output.flatten(2)    
            next_in = self.ConvLayers.returnOutputSizes() 
            
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
        
        ###########################################
        # MLP layers               
        if self.retrain_params is not None:
            self.copy_from_old = self.retrain_params [ "copy_from_old" ]
            self.freeze_layers = self.retrain_params [ "freeze" ]
        else:
            self.freeze_layers = [] 
            for i in xrange(len(self.nkerns) + len(self.num_nodes) + 1):
                self.freeze_layers.append ( False )                          
        # if no retrain specified but if init params are given, make default as copy all params.             
            if self.init_params is not None:
                self.copy_from_old = []
                for i in xrange(len(self.nkerns) + len(self.num_nodes) + 1):
                    self.copy_from_old.append ( True ) 

        if len(self.cnn_batch_norm) == 1:
            param_counter = len(self.nkerns) * 2 if self.cnn_batch_norm[0] is False else len(self.nkerns) * 3  
        else:   
            param_counter = 0         
            for i in xrange(len(self.nkerns)):
                if self.cnn_batch_norm[i] is True:
                    param_counter = param_counter + 3
                else:
                    param_counter = param_counter + 2
        self.MLPlayers = core.MLP  ( 
                                    rng = self.rng,
                                    input = (fully_connected_input, dropout_fully_connected_input),
                                    layer_sizes = layer_sizes,
                                    dropout_rates = self.mlp_dropout_rates,
                                    maxout_rates = self.mlp_maxout,
                                    max_out = self.max_out, 
                                    activations = self.mlp_activations,
                                    use_bias = self.use_bias,
                                    svm_flag = self.svm_flag,
                                    batch_norm = self.mlp_batch_norm, 
                                    cnn_dropped = self.cnn_dropout,
                                    params = [] if self.init_params is None else self.init_params[param_counter:],
                                    copy_from_old = self.copy_from_old [len(self.nkerns):] if self.init_params is not None else None,
                                    freeze = self.freeze_layers [ len(self.nkerns):],
                                    verbose = verbose
                              )
  
        # Compute cost and gradients of the model wrt parameter   
        if self.nkerns == []: 
            self.learnable_params = self.MLPlayers.learnable_params
            self.params           = self.MLPlayers.params 
        else:
            self.learnable_params = self.ConvLayers.learnable_params + self.MLPlayers.learnable_params   
            self.params           = self.ConvLayers.params           + self.MLPlayers.params   
            
        self.probabilities = self.MLPlayers.probabilities
        self.errors = self.MLPlayers.errors
        self.predicts = self.MLPlayers.predicts  
        if not self.nkerns == []:              
            self.activity = self.ConvLayers.activity
        end_time = time.clock()
        print "        time taken to build is " +str(end_time - start_time) + " seconds"
                                   
    def build_cost_function(self, verbose):
        if verbose is True:
            print "building cost function"    
        # Build the expresson for the categorical cross entropy function.
        self.objective = self.optim_params["objective"]
        self.l1_reg = float(self.optim_params["reg"][0])
        self.l2_reg = float(self.optim_params["reg"][1])        
        if self.svm_flag is False:
            if self.objective == 0:
                cost = self.MLPlayers.negative_log_likelihood( self.y )
                dropout_cost = self.MLPlayers.dropout_negative_log_likelihood( self.y )
            elif self.objective == 1:
                if len(numpy.unique(self.train_set_y.eval())) > 2:
                    cost = self.MLPlayers.cross_entropy ( self.y )
                    dropout_cost = self.MLPlayers.dropout_cross_entropy ( self.y )
                else:
                    cost = self.MLPlayers.binary_entropy ( self.y )
                    dropout_cost = self.MLPlayers.dropout_binary_entropy ( self.y )                    
            else:
                print "!! Objective is not understood, switching to cross entropy"
                cost = self.MLPlayers.cross_entropy ( self.y )
                dropout_cost = self.MLPlayers.dropout_cross_entropy ( self.y )    
        else :        
            cost = self.MLPlayers.hinge_loss( self.y1 )
            dropout_cost = self.MLPlayers.hinge_loss( self.y1 )
        output = dropout_cost if self.mlp_dropout else cost
        self.output = output + self.l1_reg * self.MLPlayers.L1 + self.l2_reg * self.MLPlayers.L2  
       
       
    def build_optimizer(self, verbose):                                    
        network_optimizer = core.optimizer( params = self.learnable_params,
                                    objective = self.output,
                                    optimization_params = self.optim_params,
                                  )                                    
        self.eta = network_optimizer.eta
        self.epoch = network_optimizer.epoch
        self.updates = network_optimizer.updates
        self.mom = network_optimizer.mom
    
    def build_network_functions(self, verbose = True):
        # create theano functions for evaluating the graph
        # I don't like the idea of having test model only hooked to the test_set_x variable.
        # I would probably have liked to have only one data variable.. but theano tutorials is using 
        # this style, so wth, so will I.    
        if verbose is True:
            print "creating network functions"
                        
        index = T.lscalar('index')  # index to a [mini]batch           
                                                       
        self.test_model = theano.function(
                inputs = [index],
                outputs = self.errors(self.y),
                givens={
                    self.x: self.test_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                    self.y: self.test_set_y[index * self.batch_size:(index + 1) * self.batch_size]})    
        self.validate_model = theano.function(
                inputs = [index],
                outputs = self.errors(self.y),
                givens={
                    self.x: self.valid_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                    self.y: self.valid_set_y[index * self.batch_size:(index + 1) * self.batch_size]})    
        self.prediction = theano.function(
            inputs = [index],
            outputs = self.predicts,
            givens={
                    self.x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size]})    
        self.nll = theano.function(
            inputs = [index],
            outputs = self.probabilities,
            givens={
                self.x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size]})    
        # function to return activations of each image
        if not self.nkerns == [] :
            activity = self.ConvLayers.returnActivity()
            self.activities = theano.function (
                inputs = [index],
                outputs = self.activity,
                givens = {
                        self.x: self.train_set_x[index * self.batch_size: (index + 1) * self.batch_size]
                        })               
        if verbose is True:
            print "building training model" 
        if self.svm_flag is True:
            self.train_model = theano.function(
                    inputs= [index, self.epoch],
                    outputs = self.output,
                    updates = self.updates,
                    givens={
                        self.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                        self.y1: self.train_set_y1[index * self.batch_size:(index + 1) * self.batch_size]},
                    on_unused_input = 'ignore'                    
                        )
        else: 
            self.train_model = theano.function(
                    inputs = [index, self.epoch],
                    outputs = self.output,
                    updates = self.updates,
                    givens={
                        self.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                        self.y: self.train_set_y[index * self.batch_size:(index + 1) * self.batch_size]},
                    on_unused_input='ignore'                    
                        )    
                        
        self.decay_learning_rate = theano.function(
               inputs=[],          # Just updates the learning rates. 
               updates={self.eta: self.eta -  self.eta * self.learning_rate_decay }
                )    
        self.momentum_value = theano.function ( 
                            inputs =[self.epoch],
                            outputs = self.mom,
                            )                       
        self.training_accuracy = theano.function(
                inputs = [index],
                outputs = self.errors(self.y),
                givens={
                    self.x: self.train_set_x[index * self.batch_size:(index + 1) * self.batch_size],
                    self.y: self.train_set_y[index * self.batch_size:(index + 1) * self.batch_size]})

    def build_learner (self, verbose = True):
        start_time = time.clock()   
        self.build_cost_function(verbose =verbose)     
        self.build_optimizer (verbose = verbose)                                                        
        self.build_network_functions(verbose = verbose)
        end_time = time.clock()
        print "        time taken is " +str(end_time - start_time) + " seconds"
                              
                                                                                                        
    def load_data_base( self, batch = 1, type_set = 'train' ):
        # every dataset will have atleast one batch ..load that.
        
        if type_set == 'train':
            f = open(self.dataset + '/train/batch_' +str(batch) +'.pkl', 'rb')
        elif type_set == 'valid':
            f = open(self.dataset + '/valid/batch_' +str(batch) +'.pkl', 'rb')            
        else:
            f = open(self.dataset + '/test/batch_' +str(batch) +'.pkl', 'rb')            
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
        # If we had used only one datavariable instead of threethis wouldn't have been needed. 
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
        bar = progressbar.ProgressBar(maxval=len(self.nkerns), \
                        widgets=[progressbar.AnimatedMarker(), \
                        ' visualizing ', 
                        ' ', progressbar.Percentage(), \
                        ' ',progressbar.ETA(), \
                        ]).start()        
        for m in xrange(len(self.nkerns)):   #For each layer 
            loc_ac = '../visuals/activities/layer_' + str(m) + "/epoch_" + str(epoch)
            if not os.path.exists(loc_ac):   
                os.makedirs(loc_ac)
            loc_ac = loc_ac + "/filter_"
            current_activity = activity_now[m]
            current_activity = current_activity[self.visualize_ind]                            
            imgs = util.visualize(current_activity, loc_ac, is_color = False)
            
            current_weights = self.ConvLayers.weights[m]    # for each layer       
            loc_we = '../visuals/filters/layer_' + str(m) + "/epoch_" + str(epoch)
            if not os.path.exists(loc_we):   
                os.makedirs(loc_we)
            loc_we = loc_we + "/filter_"
            if len(current_weights.shape.eval()) == 5:
                imgs = util.visualize(numpy.squeeze(current_weights.dimshuffle(0,3,4,1,2).eval()), prefix = loc_we, is_color = self.color_filter)
            else:   
                imgs = util.visualize(current_weights.dimshuffle(0,2,3,1).eval(), prefix = loc_we, is_color = self.color_filter)            
            bar.update(m+1)
        bar.finish()
    # ToDo: should make a results root dir and put in results there like root +'/visuals/' 
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

    # following are some functions that deal with wanting to reset the network parameters.        
    def convert2maxpool(self, verbose):
        count = 0
        pool_temp = self.pooling_type    
        print "rebuilding net with maxpool"            
        self.retrain_params = {
                                "copy_from_old"     : [True] * (len(self.nkerns) + len(self.num_nodes) + 1),
                                "freeze"            : [False] * (len(self.nkerns) + len(self.num_nodes) + 1)
                                } 
        self.init_params = self.params
        self.build_network(verbose = verbose)
        #reset it back
        self.pooling_type = pool_temp             
        
    def convert2meanpool(self, verbose):
        count = 0
        pool_temp = self.pooling_type
        print "rebuilding net with meanpool"            
        self.retrain_params = {
                                "copy_from_old"     : [True] * (len(self.nkerns) + len(self.num_nodes) + 1),
                                "freeze"            : [False] * (len(self.nkerns) + len(self.num_nodes) + 1)
                                } 
        self.init_params = self.params
        self.build_network(verbose = verbose)
        #reset it back
        self.pooling_type = pool_temp             


    def remove_momentum (self, verbose = False):
        if verbose is True:
            print "removing momentum"
        self.optim_params["mom_type"] = 0
        curr_lr = self.eta.get_value( borrow = True)
        ft_lr = self.ft_learning_rate if curr_lr < self.ft_learning_rate else curr_lr
        self.optim_params["learning_rate"] = (curr_lr, ft_lr, self.optim_params["learning_rate"][2])

    def reset_simple_sgd (self, verbose = False):
        if verbose is True:
            print "resetting to sgd"
        self.optim_params["optim_type"] = 0
                
    def setup_ft ( self, params, ft_lr = 0.001, verbose  = False ):
        if verbose is True:
            print "resetting learning rate to :" +str(ft_lr)
        self.reset_params (init_params = params, verbose = verbose)
        self.eta.set_value(ft_lr)
        # self.reset_simple_sgd(verbose =verbose )
        self.remove_momentum(verbose = verbose )
        self.build_optimizer (verbose = verbose)                                                        
        self.build_network_functions(verbose = verbose)
        
    def reset_params (self, init_params, verbose = True):
        if verbose is True:
            print "reseting parameters"
             
        self.init_params = init_params
            
        if self.retrain_params is None:
            self.retrain_params = {      
                                    "copy_from_old"     : [True] * (len(self.nkerns) + len(self.num_nodes) + 1 ),
                                    "freeze"            : [False] * (len(self.nkerns) + len(self.num_nodes) + 1 )
                                    }  
        self.build_network(verbose = verbose)
        self.build_learner(verbose = verbose)
        
    # TRAIN 
    def validate(self, epoch, verbose = True):
        validation_losses = 0.   
        training_losses = 0.
    	f = open('dump.txt', 'a')        
        if self.multi_load is True:                       
            print "   cost                : " + str(numpy.mean(self.cost_saved[-1*self.batches2train*self.n_train_batches:]))
            print "   learning_rate       : " + str(self.eta.get_value(borrow=True))
            print "   momentum            : " + str(self.momentum_value(epoch))
            bar = progressbar.ProgressBar(maxval=self.batches2train + self.batches2validate, \
                        widgets=[progressbar.AnimatedMarker(), \
                        ' validation ', 
                        ' ', progressbar.Percentage(), \
                        ' ',progressbar.ETA(), \
                        ]).start()
            for batch in xrange ( self.batches2validate):  
                self.set_data ( batch = batch , type_set = 'valid' , verbose = verbose)
                validation_losses = validation_losses + numpy.sum([[self.validate_model(i) for i in xrange(self.n_valid_batches)]])
                bar.update(batch + 1)                
            self.this_validation_loss = self.this_validation_loss + [validation_losses]
            for batch in xrange (self.batches2train):
                self.set_data ( batch = batch , type_set = 'train' , verbose = verbose)            
                training_losses = training_losses + numpy.sum([[self.training_accuracy(i) for i in xrange(self.n_train_batches)]])
                bar.update(batch+1 + self.batches2validate)                            
            self.this_training_loss = self.this_training_loss + [training_losses]
            bar.finish()    
            if self.this_validation_loss[-1] < self.best_validation_loss:
                print "   validation accuracy : " + str(float( self.batch_size * self.n_valid_batches * self.batches2validate - self.this_validation_loss[-1])*100 /(self.batch_size*self.n_valid_batches*self.batches2validate)) + " -> best thus far "
            else:
                print "   validation accuracy : " + str(float( self.batch_size * self.n_valid_batches * self.batches2validate - self.this_validation_loss[-1])*100 /(self.batch_size*self.n_valid_batches*self.batches2validate))
                
            if self.this_training_loss[-1] < self.best_training_loss:
                print "   training accuracy   : " + str(float( self.batch_size * self.n_train_batches * self.batches2train - self.this_training_loss[-1])*100 /(self.batch_size*self.n_train_batches*self.batches2train)) + " -> best thus far "
            else:
                print "   training accuracy   : " + str(float( self.batch_size * self.n_train_batches * self.batches2train - self.this_training_loss[-1])*100 /(self.batch_size*self.n_train_batches*self.batches2train))
                 
            f.write ("-> epoch " + str(epoch) +  ", cost : " + str(numpy.mean(self.cost_saved[-1*self.n_train_batches:]))  + "\n")
            f.write ("   validation accuracy : " + str(float( self.batch_size * self.n_valid_batches * self.batches2validate - self.this_validation_loss[-1])*100 /(self.batch_size*self.n_valid_batches*self.batches2validate)) +"\n")
            f.write ("   training accuracy   : " + str(float( self.batch_size * self.n_train_batches * self.batches2train - self.this_training_loss[-1])*100 /(self.batch_size*self.n_train_batches*self.batches2train)) + "\n")                                                    
            
        else: # if not multi_load
            print "   cost                : " + str(numpy.mean(self.cost_saved[-1*self.batches2train:]))
            print "   learning_rate       : " + str(self.eta.get_value(borrow=True))
            print "   momentum            : " +str(self.momentum_value(epoch))   
            
            validation_losses = [self.validate_model(i) for i in xrange(self.batches2validate)]
            self.this_validation_loss = self.this_validation_loss + [numpy.sum(validation_losses)]
            
            training_losses = [self.training_accuracy(i) for i in xrange(self.batches2train)]
            self.this_training_loss = self.this_training_loss + [numpy.sum(training_losses)]     
                   
            if self.this_validation_loss[-1] < self.best_validation_loss:
                print "   validation accuracy : " + str(float(self.batch_size*self.batches2validate - self.this_validation_loss[-1])*100 /(self.batch_size*self.batches2validate)) +  " -> best thus far "
            else:
                print "   validation accuracy : " + str(float(self.batch_size*self.batches2validate - self.this_validation_loss[-1])*100 /(self.batch_size*self.batches2validate))
            if self.this_training_loss[-1] < self.best_training_loss:
                print "   training accuracy   : " + str(float(self.batch_size*self.batches2train - self.this_training_loss[-1])*100 /(self.batch_size*self.batches2train)) +  " -> best thus far "
            else:
                print "   training accuracy   : " + str(float(self.batch_size*self.batches2train - self.this_training_loss[-1])*100 /(self.batch_size*self.batches2train))

            f.write ("-> epoch " + str(epoch) +  ", cost : " + str(self.cost_saved[-1])  + "\n")
            f.write ("   validation accuracy : " + str(float(self.batch_size*self.batches2validate - self.this_validation_loss[-1])*100 /(self.batch_size*self.batches2validate)) +"\n")
            f.write ("   training accuracy   : " + str(float(self.batch_size*self.batches2train - self.this_training_loss[-1])*100 /(self.batch_size*self.batches2train)) + "\n")                                                                
        f.close()    
         
        # Save down training stuff
        f = open(self.error_file_name,'a')
        f.write("\n")        
        f.write(str(self.this_validation_loss[-1])  + "\t" + str(self.this_training_loss[-1]))
        f.close()
        self.save_network('temp.pkl')  
              
    
    def train(self, n_epochs = 200 , ft_epochs = 200 , validate_after_epochs = 1, patience_epochs = 1, verbose = True):
        start_time_main = time.clock()
        self.main_img_visual = False

        if self.multi_load is True:
            patience = patience_epochs 
        else:
            patience = patience_epochs
            
        # wait one validation cycle
        patience_increase = 2  
        improvement_threshold = 0.995 
        self.this_validation_loss = []
        self.best_validation_loss = numpy.inf
        
        self.this_training_loss = []
        self.best_training_loss = numpy.inf
        
        best_iter = 0
        epoch_counter = 0
        early_termination = False
        self.cost_saved = []
        iteration= 0        
                
        best_params = copy.deepcopy(self.params)
        nan_insurance = copy.deepcopy(self.params)
                            
        nan_flag = False
        fine_tune = False

        while (epoch_counter < (n_epochs + ft_epochs)) and (not early_termination):
            start_time = time.clock()         
            if epoch_counter == n_epochs and fine_tune is False and self.eta.get_value(borrow = True) > self.ft_learning_rate:
                print "\n\n"
                print "fine tuning"
                fine_tune = True
                self.setup_ft(best_params, ft_lr = self.ft_learning_rate, verbose =verbose )
            epoch_counter = epoch_counter + 1 
            print "\n"
            print "-> epoch: " +str(epoch_counter)     
            if not verbose is True:         
                batch = 0
                bar = progressbar.ProgressBar(maxval=self.batches2train, \
                        widgets=[ progressbar.AnimatedMarker(), ' ', progressbar.Percentage(), \
                               ' training ', 
                               ' ',progressbar.ETA(), \
                                ]).start()
            for batch in xrange (self.batches2train):
                if verbose is True:
                    print "   batch: " + str(batch+1) + " out of " + str(self.batches2train) + " batches"    
                if self.multi_load is True:
                    # Load data for this batch
                    self.set_data ( batch = batch , type_set = 'train', verbose = verbose)
                    for minibatch_index in xrange(self.n_train_batches):
                        if verbose is True and self.n_train_batches > 1:
                            print "      mini Batch: " + str(minibatch_index + 1) + " out of "    + str(self.n_train_batches)
                        cost_ij = self.train_model( minibatch_index, epoch_counter)
                        if numpy.isnan(cost_ij):
                            print "NAN !! slowing the learning rate by 10 times"
                            self.setup_ft(params = nan_insurance, ft_lr = numpy.asarray(self.eta.get_value(borrow = True)*0.1,dtype=theano.config.floatX),
                                            verbose =verbose )
                            nan_flag = True     
                            break         
                        self.cost_saved = self.cost_saved + [cost_ij]     
                    if verbose is True:
                        print "      cost: " + str(cost_ij)
                else:   
                    cost_ij = self.train_model(batch, epoch_counter)
                    if numpy.isnan(cost_ij):
                        print "NAN !! slowing the learning rate by 10 times"                     
                        nan_flag = True
                        self.setup_ft(params = nan_insurance, ft_lr = numpy.asarray(self.eta.get_value(borrow = True)*0.1,dtype=theano.config.floatX),
                                        verbose =verbose ) 
                        break                 
                    self.cost_saved = self.cost_saved +[cost_ij] 
                    if verbose is True:                    
                        print "      cost: " + str(cost_ij)                    
                if not verbose is True:
                    bar.update(batch+1)
            if not verbose is True:
                bar.finish()

            if nan_flag is False:
            
                if  epoch_counter % validate_after_epochs == 0:              
                    self.validate(epoch = epoch_counter, verbose = verbose)
                    if self.this_validation_loss[-1] < self.best_validation_loss * improvement_threshold:
                        patience = max(patience, epoch_counter* patience_increase)
                        best_iter = iteration
                        nan_insurance = copy.deepcopy(best_params)
                        best_params = copy.deepcopy(self.params) 
                        self.save_network()   
                        best_iter = epoch_counter           
                    self.best_validation_loss = min(self.best_validation_loss, self.this_validation_loss[-1])
                    self.best_training_loss = min(self.best_training_loss, self.this_training_loss[-1])
                    
                if self.visualize_flag is True and epoch_counter % self.visualize_after_epochs == 0 and not self.nkerns == []:            
                    self.print_net (epoch = epoch_counter, display_flag = self.display_flag)            
            
            print "   patience " + str(patience-epoch_counter) + " epochs"                          
            self.decay_learning_rate()  
            if patience < epoch_counter:
                early_termination = True
                if fine_tune is False:
                    print "\n\n"
                    print "fine tuning"
                    fine_tune = True
                    self.setup_ft(params = best_params, ft_lr = numpy.asarray(self.eta.get_value(borrow = True)*0.1,dtype=theano.config.floatX) ,
                                        verbose =verbose )
                    early_termination = False
                else:
                    print "\n\n"
                    print "early stopping"
                    break   
                end_time = time.clock()
                print "   total time taken for this epoch is " +str((end_time - start_time)/60) + " minutes"
            if nan_flag is True:
                nan_flag = False 
                                                                                    
            end_time = time.clock()
            print "   total time taken for this epoch is " +str((end_time - start_time)/60) + " minutes"
        
        self.setup_ft (params = best_params, ft_lr = self.ft_learning_rate, verbose = verbose)     
        end_time_main = time.clock()
        print "   time taken for the entire training is " +str((end_time_main - start_time_main)/3600) + " hours"                
                         
        f = open(self.cost_file_name,'w')
        for i in xrange(len(self.cost_saved)):
            f.write(str(self.cost_saved[i]))
            f.write("\n")
        f.close()  
            
    def test(self, verbose = True):
        print "testing"
        start_time = time.clock()
        wrong = 0
        predictions = []
        class_prob = []
        labels = []
         
        if self.multi_load is False:   
            labels = self.test_set_y.eval().tolist()  
            bar = progressbar.ProgressBar(maxval=self.batches2test, \
                        widgets=[progressbar.AnimatedMarker(), ' ', progressbar.Percentage(), \
                                ' testing ',                        
                                ' ',progressbar.ETA(), \
                                ]).start()              
            for mini_batch in xrange(self.batches2test):
                #print ".. Testing batch " + str(mini_batch)
                wrong = wrong + int(self.test_model(mini_batch))                        
                predictions = predictions + self.prediction(mini_batch).tolist()
                class_prob = class_prob + self.nll(mini_batch).tolist()
                bar.update(mini_batch)
            bar.finish()
            print ("total test accuracy : " + str(float((self.batch_size*self.batches2test)-wrong )*100
                                                         /(self.batch_size*self.batches2test)) + 
                         " % out of " + str(self.batch_size*self.batches2test) + " samples.")
            f = open('dump.txt','a')
            
            f.write(("total test accuracy : " + str(float((self.batch_size*self.batches2test)-wrong )*100
                                                         /(self.batch_size*self.batches2test)) + 
                         " % out of " + str(self.batch_size*self.batches2test) + " samples."))
            f.write('\n')                         
            f.close()
                        
        else: 
            if verbose is False:
                bar = progressbar.ProgressBar(maxval=self.batches2test * self.n_test_batches, \
                        widgets=[progressbar.AnimatedMarker(), ' ', progressbar.Percentage(), \
                               ' testing ',
                               ' ',progressbar.ETA(), \
                                ]).start()                  
            for batch in xrange(self.batches2test):                                           
                if verbose is True:
                    print "   testing batch " + str(batch)
                # Load data for this batch
                self.set_data ( batch = batch, type_set = 'test' , verbose = verbose)
                labels = labels + self.test_set_y.eval().tolist()                   
                for mini_batch in xrange(self.n_test_batches):
                    wrong = wrong + int(self.test_model(mini_batch))   
                    predictions = predictions + self.prediction(mini_batch).tolist()
                    class_prob = class_prob + self.nll(mini_batch).tolist()
                    if verbose is False:
                        bar.update(mini_batch * (batch+1) + 1)
            if verbose is False:
                bar.finish()
             
            print ("total test accuracy : " + str(float((self.batch_size*self.n_test_batches*self.batches2test)-wrong )*100/
                                                         (self.batch_size*self.n_test_batches*self.batches2test)) + 
                         " % out of " + str(self.batch_size*self.n_test_batches*self.batches2test) + " samples.")
            f = open('dump.txt','a')
            f.write(("total test accuracy : " + str(float((self.batch_size*self.n_test_batches*self.batches2test)-wrong )*100/
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
        print "        time taken is " +str(end_time - start_time) + " seconds"
                
        if self.visualize_flag is True and not self.nkerns == []:    
            print "saving down the final model's visualizations" 
            self.print_net (epoch = 'final' , display_flag = self.display_flag)                 