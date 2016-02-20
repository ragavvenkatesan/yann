#!/usr/bin/python

# General Packages
import os
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
        random_seed = arch_params["random_seed"]
        self.rng = numpy.random.RandomState(random_seed)  
        self.main_img_visual = True 
        
        self.optim_params                    = optimization_params           
        self.arch                            = arch_params
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
        self.use_bias                        = arch_params [ "use_bias" ]           

        self.retrain_params = retrain_params
        self.init_params    = init_params 
        
        self.ft_learning_rate = self.optim_params["ft_learning_rate"]
                
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
    def init_data ( self, dataset, outs, visual_params = None, verbose = False):
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
        if visual_params is not None:
            self.create_dirs ( visual_params = visual_params )       
        
    # define the optimzer function 
    def build_network (self, verbose = True):    
                     
       
        print '... building the network'    
        
        
        start_time = time.clock()
        # allocate symbolic variables for the data       
        self.x = T.matrix('x')           # the data is presented as rasterized images
        self.y = T.ivector('y')          # the labels are presented as 1D vector of [int]                     
        if self.svm_flag is True:
            self.y1 = T.matrix('y1')     # [-1 , 1] labels in case of SVM    
     
        first_layer_input = self.x.reshape((self.batch_size, self.height, self.width, self.channels)).dimshuffle(0,3,1,2)
        mean_sub_input = first_layer_input - first_layer_input.mean()        
        
        ###########################################
        # Convolutional layers        
        if not self.nkerns == []: # If there are some convolutional layers... 
            self.ConvLayers = core.ConvolutionalLayers (      
                                                    input = (first_layer_input, mean_sub_input),
                                                    rng = self.rng,
                                                    input_size = (self.height, self.width, self.channels, self.batch_size), 
                                                    mean_subtract = self.mean_subtract,
                                                    nkerns = self.nkerns,
                                                    filter_size = self.filter_size,
                                                    pooling_size = self.pooling_size,
                                                    cnn_activations = self.cnn_activations,
                                                    conv_stride_size = self.conv_stride_size,
                                                    cnn_dropout_rates = self.cnn_dropout_rates,
                                                    batch_norm = self.batch_norm,         
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
                                          
        param_counter = len(self.nkerns) * 2 if self.batch_norm is False else len(self.nkerns) * 3                                          
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
                                    params = [] if self.init_params is None else self.init_params[param_counter:],
                                    copy_from_old = self.copy_from_old [len(self.nkerns):] if self.init_params is not None else None,
                                    freeze = self.freeze_layers [ len(self.nkerns):],
                                    verbose = verbose
                              )
  
        # Compute cost and gradients of the model wrt parameter   
        if self.nkerns == []: 
            self.params = self.MLPlayers.params 
        else:
            self.params = self.ConvLayers.params + self.MLPlayers.params      

        self.probabilities = self.MLPlayers.probabilities
        self.errors = self.MLPlayers.errors
        self.predicts = self.MLPlayers.predicts  
        if not self.nkerns == []:              
            self.activity = self.ConvLayers.activity
        end_time = time.clock()
        print "...         time taken to build is " +str(end_time - start_time) + " seconds"
        if verbose is True:
            print "... building cost function"
        self.build_cost_function()            
        
        if verbose is True:
            print "... creating network functions"                                   
        self.create_network_functions(verbose = verbose)
                                   
    def build_cost_function(self):
        # Build the expresson for the categorical cross entropy function.
        start_time = time.clock()        
        self.objective = self.optim_params["objective"]
        self.l1_reg = self.optim_params["l1_reg"]
        self.l2_reg = self.optim_params["l2_reg"]        
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
        self.output =  dropout_cost if self.mlp_dropout else cost
        self.output = self.output + + self.l1_reg * self.MLPlayers.L1 + self.l2_reg * self.MLPlayers.L2  
                                        
        optimizer = core.optimizer( params = self.params,
                                    objective = self.output,
                                    optimization_params = self.optim_params,
                                  )                                    
        self.eta = optimizer.eta
        self.epoch = optimizer.epoch
        self.updates = optimizer.updates
        self.mom = optimizer.mom
        end_time = time.clock()
        print "...         time taken is " +str(end_time - start_time) + " seconds"        
        
    def create_network_functions(self, verbose = True):
        # create theano functions for evaluating the graph
        # I don't like the idea of having test model only hooked to the test_set_x variable.
        # I would probably have liked to have only one data variable.. but theano tutorials is using 
        # this style, so wth, so will I.        
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
            print "... building training model" 
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
               updates={self.eta: self.eta -  self.eta * self.optim_params["learning_rate_decay"] }
                )    
        self.momentum_value = theano.function ( 
                            inputs =[self.epoch],
                            outputs = self.mom,
                            )                       
                            
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
            
            current_weights = self.ConvLayers.weights[m]    # for each layer       
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
    def train(self, n_epochs = 200 , ft_epochs = 200 , validate_after_epochs = 1, verbose = True):
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
                    
            if self.visualize_flag is True and epoch_counter % self.visualize_after_epochs == 0 and not self.nkerns == []:            
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
                
        if self.visualize_flag is True and not self.nkerns == []:    
            print "... saving down the final model's visualizations" 
            self.print_net (epoch = 'final' , display_flag = self.display_flag)                 