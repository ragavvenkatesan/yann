#!/usr/bin/python

# General Packages
import os
import sys
import time
import pdb
from collections import OrderedDict

# Math Packages
import math
import numpy

# Theano Packages
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.ifelse import ifelse

# CNN code packages
from cnn import ReLU
from cnn import Sigmoid
from cnn import Tanh
from cnn import SVMLayer
from cnn import LogisticRegression
from cnn import HiddenLayer
from cnn import _dropout_from_layer
from cnn import DropoutHiddenLayer
from cnn import MLP
from cnn import LeNetConvPoolLayer
from cnn import write_filters

# Datahandling packages
from loaders import load_data_pkl
from loaders import load_data_mat
from loaders import load_skdata_caltech101
from loaders import load_skdata_mnist
from loaders import load_skdata_cifar10



















def run_cnn(  arch_params,
                    optimization_params ,
                    data_params, 
                    filename_params,
                    verbose = True  ):

    












    

    

    #####################
    # Unpack Variables  #
    #####################

    results_file_name = filename_params [ "results_file_name" ]                # Files that will be saved down on completion Can be used by the parse.m file
    error_file_name   = filename_params [ "error_file_name" ]
    cost_file_name    = filename_params [ "cost_file_name"  ]

    dataset             = data_params [ "loc" ]
    height              = data_params [ "height" ]
    width               = data_params [ "width" ]
    batch_size          = data_params [ "batch_size" ]    
    load_batches        = data_params [ "load_batches"  ] * batch_size
    batches2train       = data_params [ "batches2train" ]
    batches2test        = data_params [ "batches2test" ]
    batches2validate    = data_params [ "batches2validate" ] 
    channels            = data_params [ "channels" ]

    mom_start                       = optimization_params [ "mom_start" ]
    mom_end                         = optimization_params [ "mom_end" ]
    mom_epoch_interval              = optimization_params [ "mom_interval" ]
    mom_type                        = optimization_params [ "mom_type" ]
    initial_learning_rate           = optimization_params [ "initial_learning_rate" ]              
    learning_rate_decay             = optimization_params [ "learning_rate_decay" ]  
    ada_grad_begin                  = optimization_params [ "ada_grad_begin"  ]
    ada_grad_end                    = optimization_params [ "ada_grad_end"  ]
    learning_rate_at_ada_grad_begin = optimization_params [ "learning_rate_at_ada_grad_begin" ] 
    learning_rate_at_ada_grad_end   = optimization_params [ "learning_rate_at_ada_grad_end" ]
    fudge_factor                    = optimization_params [ "fudge_factor" ]
    regularize_l1                   = optimization_params [ "regularize_l1" ]
    regularize_l2                   = optimization_params [ "regularize_l2" ]
    cross_entropy                   = optimization_params [ "cross_entropy"]

    squared_filter_length_limit     = arch_params [ "squared_filter_length_limit" ]   
    n_epochs                        = arch_params [ "n_epochs" ]
    validate_after_epochs           = arch_params [ "validate_after_epochs"  ]
    mlp_activations                 = arch_params [ "mlp_activations"  ] 
    cnn_activations                 = arch_params [ "cnn_activations" ]
    dropout                         = arch_params [ "dropout"  ]
    column_norm                     = arch_params [ "column_norm"  ]    
    dropout_rates                   = arch_params [ "dropout_rates" ]
    nkerns                          = arch_params [ "nkerns"  ]
    outs                            = arch_params [ "outs" ]
    filter_size                     = arch_params [ "filter_size" ]
    pooling_size                    = arch_params [ "pooling_size" ]
    num_nodes                       = arch_params [ "num_nodes" ]
    use_bias                        = arch_params [ "use_bias" ]
    random_seed                     = arch_params [ "random_seed" ]
    svm_flag                        = arch_params [ "svm_flag" ]

    results_file_name   = filename_params[ "results_file_name" ]
    error_file_name     = filename_params[ "error_file_name" ]
    cost_file_name      = filename_params[ "cost_file_name" ]
    


    # Random seed initialization.
    rng = numpy.random.RandomState(random_seed)  

 




















    #################
    # Data Loading  #
    #################
    print "... loading data"

    # load matlab files as dataset.
    if data_params["type"] == 'mat':
        train_data_x, train_data_y, train_data_y1 = load_data_mat(dataset, batch = 1 , type_set = 'train')             
        test_data_x, test_data_y, valid_data_y1 = load_data_mat(dataset, batch = 1 , type_set = 'test')      # Load dataset for first epoch.
        valid_data_x, valid_data_y, test_data_y1 = load_data_mat(dataset, batch = 1 , type_set = 'valid')    # Load dataset for first epoch.

        train_set_x = theano.shared(numpy.asarray(train_data_x, dtype=theano.config.floatX), borrow=True)
        train_set_y = T.cast(theano.shared(numpy.asarray(train_data_y, dtype='int32'), borrow=True), 'int32' )
        train_set_y1 = theano.shared(numpy.asarray(train_data_y1, dtype=theano.config.floatX), borrow=True)

        test_set_x = theano.shared(numpy.asarray(test_data_x, dtype=theano.config.floatX), borrow=True)
        test_set_y = T.cast(theano.shared(numpy.asarray(test_data_y, dtype='int32'), borrow=True) , 'int32' )
        test_set_y1 = theano.shared(numpy.asarray(test_data_y1, dtype=theano.config.floatX), borrow=True)

        valid_set_x = theano.shared(numpy.asarray(valid_data_x, dtype=theano.config.floatX), borrow=True)
        valid_set_y = T.cast(theano.shared(numpy.asarray(valid_data_y, dtype='int32'), borrow=True) , 'int32' )
        valid_set_y1 = theano.shared(numpy.asarray(valid_data_y1, dtype=theano.config.floatX), borrow=True)

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

        multi_load = True

    # load pkl data as is shown in theano tutorials
    elif data_params["type"] == 'pkl':   

        data = load_data_pkl(dataset)
        train_set_x, train_set_y, train_set_y1 = data[0]
        valid_set_x, valid_set_y, valid_set_y1 = data[1]
        test_set_x, test_set_y, test_set_y1 = data[2]

         # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

        n_train_images = train_set_x.get_value(borrow=True).shape[0]
        n_test_images = test_set_x.get_value(borrow=True).shape[0]
        n_valid_images = valid_set_x.get_value(borrow=True).shape[0]

        n_train_batches_all = n_train_images / batch_size 
        n_test_batches_all = n_test_images / batch_size 
        n_valid_batches_all = n_valid_images / batch_size

        if (n_train_batches_all < batches2train) or (n_test_batches_all < batches2test) or (n_valid_batches_all < batches2validate):        # You can't have so many batches.
            print "...  !! Dataset doens't have so many batches. "
            raise AssertionError()

        multi_load = False

    # load skdata ( its a good library that has a lot of datasets)
    elif data_params["type"] == 'skdata':

        if dataset == 'mnist':
            print "... lmporting mnist from skdata"

            data = load_skdata_mnist()
            train_set_x, train_set_y, train_set_y1 = data[0]
            valid_set_x, valid_set_y, valid_set_y1 = data[1]
            test_set_x, test_set_y, test_set_y1 = data[2]

            # compute number of minibatches for training, validation and testing
            n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
            n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
            n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

            n_train_images = train_set_x.get_value(borrow=True).shape[0]
            n_test_images = test_set_x.get_value(borrow=True).shape[0]
            n_valid_images = valid_set_x.get_value(borrow=True).shape[0]

            n_train_batches_all = n_train_images / batch_size 
            n_test_batches_all = n_test_images / batch_size 
            n_valid_batches_all = n_valid_images / batch_size

            if (n_train_batches_all < batches2train) or (n_test_batches_all < batches2test) or (n_valid_batches_all < batches2validate):        # You can't have so many batches.
                print "...  !! Dataset doens't have so many batches. "
                raise AssertionError()

            multi_load = False

        elif dataset == 'cifar10':
            print "... lmporting cifar 10 from skdata"

            data = load_skdata_cifar10()
            train_set_x, train_set_y, train_set_y1 = data[0]
            valid_set_x, valid_set_y, valid_set_y1 = data[1]
            test_set_x, test_set_y, test_set_y1 = data[2]

            # compute number of minibatches for training, validation and testing
            n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
            n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
            n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

            multi_load = False

        elif dataset == 'caltech101':
            print "... importing caltech 101 from skdata"

                # shuffle the data
            total_images_in_dataset = 9144 
            rand_perm = numpy.random.permutation(total_images_in_dataset)  # create a constant shuffle, so that data can be loaded in batchmode with the same random shuffle

            n_train_images = total_images_in_dataset / 3
            n_test_images = total_images_in_dataset / 3
            n_valid_images = total_images_in_dataset / 3 

            n_train_batches_all = n_train_images / batch_size 
            n_test_batches_all = n_test_images / batch_size 
            n_valid_batches_all = n_valid_images / batch_size

            if (n_train_batches_all < batches2train) or (n_test_batches_all < batches2test) or (n_valid_batches_all < batches2validate):        # You can't have so many batches.
                print "...  !! Dataset doens't have so many batches. "
                raise AssertionError()

            train_data_x, train_data_y  = load_skdata_caltech101(batch_size = load_batches, rand_perm = rand_perm, batch = 1 , type_set = 'train' , height = height, width = width)             
            test_data_x, test_data_y  = load_skdata_caltech101(batch_size = load_batches, rand_perm = rand_perm, batch = 1 , type_set = 'test' , height = height, width = width)      # Load dataset for first epoch.
            valid_data_x, valid_data_y  = load_skdata_caltech101(batch_size = load_batches, rand_perm = rand_perm, batch = 1 , type_set = 'valid' , height = height, width = width)    # Load dataset for first epoch.

            train_set_x = theano.shared(train_data_x, borrow=True)
            train_set_y = T.cast(theano.shared(train_data_y, borrow=True), 'int32' )
            
            test_set_x = theano.shared(test_data_x, borrow=True)
            test_set_y = T.cast(theano.shared(test_data_y, borrow=True) , 'int32' )
          
            valid_set_x = theano.shared(valid_data_x, borrow=True)
            valid_set_y = T.cast(theano.shared(valid_data_y, borrow=True) , 'int32' )

            # compute number of minibatches for training, validation and testing
            n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
            n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
            n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

            multi_load = True

    # Just checking as a way to see if the intended dataset is indeed loaded.
    assert height*width*channels == train_set_x.get_value( borrow = True ).shape[1]

























    ######################
    # BUILD NETWORK      #
    ######################
    print '... building the network'    
    start_time = time.clock()
    # allocate symbolic variables for the data
    index = T.lscalar()         # index to a [mini]batch
    x = T.matrix('x')           # the data is presented as rasterized images
    y = T.ivector('y')          # the labels are presented as 1D vector of [int] 

    if svm_flag is True:
        y1 = T.matrix('y1')     # [-1 , 1] labels in case of SVM    

    first_layer_input = x.reshape((batch_size, channels, height, width))

    # Create first convolutional - pooling layers 
    conv_layers=[]
    filt_size = filter_size[0]
    pool_size = pooling_size[0]
    conv_layers.append ( LeNetConvPoolLayer(
                            rng,
                            input = first_layer_input,
                            image_shape=(batch_size, channels , height, width),
                            filter_shape=(nkerns[0], channels , filt_size, filt_size),
                            poolsize=(pool_size, pool_size),
                            activation = cnn_activations[0],
                            verbose = verbose
                             ) )

    # Create the rest of the convolutional - pooling layers in a loop
    next_in_1 = ( height - filt_size + 1 ) / pool_size        
    next_in_2 = ( width - filt_size + 1 ) / pool_size

    for layer in xrange(len(nkerns)-1):   
        filt_size = filter_size[layer+1]
        pool_size = pooling_size[layer+1]
        conv_layers.append ( LeNetConvPoolLayer(
                            rng,
                            input=conv_layers[layer].output,        
                            image_shape=(batch_size, nkerns[layer], next_in_1, next_in_2),
                            filter_shape=(nkerns[layer+1], nkerns[layer], filt_size, filt_size),
                            poolsize=(pool_size, pool_size),
                            activation = cnn_activations[layer+1],
                            verbose = verbose
                             ) )
        next_in_1 = ( next_in_1 - filt_size + 1 ) / pool_size        
        next_in_2 = ( next_in_2 - filt_size + 1 ) / pool_size

    # Assemble fully connected laters 
    fully_connected_input = conv_layers[-1].output.flatten(2)
    if len(dropout_rates) > 2 :
        layer_sizes =[]
        layer_sizes.append( nkerns[-1] * next_in_1 * next_in_2 )
        for i in xrange(len(dropout_rates)-1):
            layer_sizes.append ( num_nodes[i] )
        layer_sizes.append ( outs )
    else :
        layer_sizes = [ nkerns[-1] * next_in_1 * next_in_2, num_nodes[0] , outs]

    assert len(layer_sizes) - 1 == len(dropout_rates)           # Just checking.

    MLPlayers = MLP( rng=rng,
                     input=fully_connected_input,
                     layer_sizes=layer_sizes,
                     dropout_rates=dropout_rates,
                     activations=mlp_activations,
                     use_bias=use_bias,
                     svm_flag = svm_flag,
                     verbose = verbose)

    # Build the expresson for the cross entropy function.
    if svm_flag is False:
        cost = MLPlayers.negative_log_likelihood(y,regularize_l1 = regularize_l1, regularize_l2 = regularize_l2)
        dropout_cost = MLPlayers.dropout_negative_log_likelihood(y,regularize_l1 = regularize_l1, regularize_l2 = regularize_l2)
    else :        
        cost = MLPlayers.negative_log_likelihood(y1,regularize_l1 = regularize_l1, regularize_l2 = regularize_l2)
        dropout_cost = MLPlayers.dropout_negative_log_likelihood(y1,regularize_l1 = regularize_l1, regularize_l2 = regularize_l2)


    test_model = theano.function(
            inputs=[index],
            outputs=MLPlayers.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})

    validate_model = theano.function(
            inputs=[index],
            outputs=MLPlayers.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    prediction = theano.function(
        inputs = [index],
        outputs = MLPlayers.predicts,
        givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size]})

    nll = theano.function(
        inputs = [index],
        outputs = MLPlayers.probabilities,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]})

    params = []
    for layer in conv_layers:
        params = params + layer.params
    params = params + MLPlayers.params

    # Compute gradients of the model wrt parameters

    """  Dropouts implemented from paper:
    Srivastava, Nitish, et al. "Dropout: A simple way to prevent neural networks
    from overfitting." The Journal of Machine Learning Research 15.1 (2014): 1929-1958.
    """

    negative_log_likelihood_output = dropout_cost if dropout else cost
    
    if cross_entropy is True:
        output = T.nnet.binary_crossentropy(negative_log_likelihood_output, y).mean()
    else:
        output = negative_log_likelihood_output

    gradients = []
    for param in params: 
        gradient = T.grad( output , param)
        gradients.append ( gradient )

    """ Adagrad implemented from paper:
    John Duchi, Elad Hazan, and Yoram Singer. 2011. Adaptive subgradient methods
    for online learning and stochastic optimization. JMLR
    """

    # TO DO: Try implementing Adadelta also. 

    # Compute momentum for the current epoch
    epoch = T.scalar()
    mom = ifelse(epoch <= mom_epoch_interval,
        mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
        mom_end)

    grad_acc = []
    for param in params:
        eps = numpy.zeros_like(param.get_value(borrow=True), dtype=theano.config.floatX)   
        grad_acc.append(theano.shared(eps, borrow=True))
            
    eta = theano.shared(numpy.asarray(initial_learning_rate,dtype=theano.config.floatX))

    velocities = []
    for param in params:
        velocity = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,dtype=theano.config.floatX))
        velocities.append(velocity)

    # Update the step direction using momentum
    updates = OrderedDict()

    print_flag = False
    for velocity, gradient, acc , param in zip(velocities, gradients, grad_acc, params):        

        current_acc = acc + T.sqr(gradient)
        updates[acc] = current_acc

        if mom_type == 0:               # no momentum
            updates[velocity] = ifelse( (T.lt(epoch,ada_grad_end) and T.ge(epoch,ada_grad_begin)) , (-1* ( eta / (T.sqrt(current_acc) + fudge_factor ) ) *gradient ), (-1 * eta * gradient ) )

        elif mom_type == 1:       
            """ Momentum implemented from paper:
            Polyak, Boris Teodorovich. "Some methods of speeding up the convergence of iteration methods." 
            USSR Computational Mathematics and Mathematical Physics 4.5 (1964): 1-17.

            Adapted from Sutskever, Ilya, et al. "On the importance of initialization and momentum in deep learning." 
            Proceedings of the 30th international conference on machine learning (ICML-13). 2013.
            equation (1) and equation (2)
            """      
            updates[velocity] = ifelse( (T.lt(epoch,ada_grad_end) and T.ge(epoch,ada_grad_begin)) , (mom * velocity - ( eta / (T.sqrt(current_acc) + fudge_factor ) ) * gradient ) , ( mom * velocity - eta * gradient ) )

        elif mom_type == 2:             # Not yet fully functional
            # """Nesterov, Yurii. "A method of solving a convex programming problem with convergence rate O (1/k2)."
            #  Soviet Mathematics Doklady. Vol. 27. No. 2. 1983.
            #  Adapted from https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/ """
            # if param.get_value(borrow=True).ndim == 2 and column_norm is True:
            #     col_norms = T.sqrt(T.sum(T.sqr(param), axis=0))
            #     desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
            #     scale = desired_norms / (1e-7 + col_norms)
            #     updates[param] = param * scale
            # else:            
            #     updates[param] = param 
            # x = mom * velocity + updates[param] - param
            # updates[velocity] = x
            # updates[param] = mom * x + updates[param]
            print " Choose something else for momentum. This one is incomplete and still has a lot of trouble."
        else:
            if print_flag is False:
                print_flag = True
                print "!! Unrecognized mometum type, switching to no momentum."
            updates[velocity] = ifelse( (T.lt(epoch,ada_grad_end) and T.ge(epoch,ada_grad_begin)) , (-1* ( eta / (T.sqrt(current_acc) + fudge_factor ) ) *gradient ), (-1 * eta * gradient ) )



    for param, velocity in zip(params, velocities):
        if mom_type != 2:           # for Nesterov's update is done in the loop above only. No need to do it here. 
            stepped_param  = param + updates[velocity]
            if param.get_value(borrow=True).ndim == 2 and column_norm is True:

                """ constrain the norms of the COLUMNs of the weight, according to
                 https://github.com/BVLC/caffe/issues/109 """
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
                scale = desired_norms / (1e-7 + col_norms)
                updates[param] = stepped_param * scale
            else:            
                updates[param] = stepped_param

    if svm_flag is True:
        train_model = theano.function(inputs= [index, epoch],
                outputs=output,
                updates=updates,
                givens={
                    x: train_set_x[index * batch_size:(index + 1) * batch_size],
                    y1: train_set_y1[index * batch_size:(index + 1) * batch_size]},
                on_unused_input='ignore'                    
                    )
    else:
        train_model = theano.function(inputs= [index, epoch],
                outputs=output,
                updates=updates,
                givens={
                    x: train_set_x[index * batch_size:(index + 1) * batch_size],
                    y: train_set_y[index * batch_size:(index + 1) * batch_size]},
                on_unused_input='ignore'                    
                    )

    decay_learning_rate = theano.function(
           inputs=[], 
           outputs=eta,                                               # Just updates the learning rates. 
           updates={eta: eta * learning_rate_decay}
            )

    momentum_value = theano.function ( 
                        inputs =[epoch],
                        outputs = mom,
                        )

    end_time = time.clock()
    print "...      -> building complete, took " + str((end_time - start_time)) + " seconds" 
























    ###############
    # TRAIN MODEL #
    ###############

    #pdb.set_trace()
    print "... training"
    start_time = time.clock()

    # early-stopping parameters
    best_validation_loss = numpy.inf
    this_validation_loss = []
    best_iter = 0
    epoch_counter = 0
    done_looping = False
    cost_saved = []
    best_params = None

    while (epoch_counter < n_epochs) and (not done_looping):
        epoch_counter = epoch_counter + 1 
        #print "+++++++++++++++++++++++++++=================|.|=================++++++++++++++++++++++++++"
        if epoch_counter == ada_grad_begin:
            eta.set_value (learning_rate_at_ada_grad_begin)
        if epoch_counter == ada_grad_end:
            eta.set_value (learning_rate_at_ada_grad_end)

        for batch in xrange (batches2train):
            if verbose is True:
                if epoch_counter >= ada_grad_begin and epoch_counter < ada_grad_end:                
                    print "...          -> Epoch: " + str(epoch_counter) + " AdaGrad Batch: " + str(batch+1) + " out of " + str(batches2train) + " batches"
                else:
                    print "...          -> Epoch: " + str(epoch_counter) + " BackProp Batch: " + str(batch+1) + " out of " + str(batches2train) + " batches"

            if multi_load is True:
                # Load data for this batch
                if verbose is True:
                    print "...          -> loading data for new batch"

                if data_params["type"] == 'mat':
                    train_data_x, train_data_y, train_data_y1 = load_data_mat(dataset, batch = batch + 1 , type_set = 'train')             

                elif data_params["type"] == 'skdata':                   
                    if dataset == 'caltech101':

                        train_data_x, train_data_y  = load_skdata_caltech101(batch_size = load_batches, batch = 1 , type_set = 'train', rand_perm = rand_perm, height = height, width = width )
                        # Do not use svm_flag for caltech 101                        
                train_set_x.set_value(train_data_x ,borrow = True)
                train_set_y.set_value(train_data_y ,borrow = True)

                for minibatch_index in xrange(n_train_batches):
                    if verbose is True:
                        print "...                  ->    Mini Batch: " + str(minibatch_index + 1) + " out of "    + str(n_train_batches)                                                             
                        cost_ij = train_model( minibatch_index, epoch_counter) 
                        cost_saved = cost_saved +[cost_ij]
                    
            else:        
                cost_ij = train_model(batch, epoch_counter)
                cost_saved = cost_saved +[cost_ij]

        if  epoch_counter % validate_after_epochs is 0:  
            # Load Validation Dataset here.
            validation_losses = 0.      
            if multi_load is True:
                # Load data for this batch
                for batch in xrange ( batches2test ):
                    if data_params["type"] == 'mat':
                        valid_data_x, valid_data_y, valid_data_y1 = load_data_mat(dataset, batch = batch + 1 , type_set = 'valid')             

                    elif data_params["type"] == 'skdata':                   
                        if dataset == 'caltech101':
          
                            valid_data_x, valid_data_y = load_skdata_caltech101(batch_size = load_batches, batch = 1 , type_set = 'valid' , rand_perm = rand_perm, height = height, width = width )
                            # Do not use svm_flag for caltech 101                    
                    valid_set_x.set_value(valid_data_x,borrow = True)
                    valid_set_y.set_value(valid_data_y,borrow = True)

                    validation_losses = validation_losses + numpy.sum([[validate_model(i) for i in xrange(n_valid_batches)]])

                this_validation_loss = this_validation_loss + [validation_losses]

                if verbose is True:
                    if this_validation_loss[-1] < best_validation_loss :
                        print "...      -> epoch " + str(epoch_counter) + ", cost: " + str(numpy.mean(cost_saved[-1*n_train_batches:])) +",  validaiton accuracy :" + str(float( batch_size * n_valid_batches * batches2validate - this_validation_loss[-1])*100/(batch_size*n_valid_batches*batches2validate)) + "%, learning_rate = " + str(eta.get_value(borrow=True))+  ", momentum = " +str(momentum_value(epoch_counter))  + " -> best thus far " 
                    else :
                        print "...      -> epoch " + str(epoch_counter) + ", cost: " + str(numpy.mean(cost_saved[-1*n_train_batches:])) +",  validaiton accuracy :" + str(float( batch_size * n_valid_batches * batches2validate - this_validation_loss[-1])*100/(batch_size*n_valid_batches*batches2validate)) + "%, learning_rate = " + str(eta.get_value(borrow=True)) +  ", momentum = " +str(momentum_value(epoch_counter)) 
                else:
                    if this_validation_loss[-1] < best_validation_loss :
                        print "...      -> epoch " + str(epoch_counter) + ", cost: " + str(numpy.mean(cost_saved[-1*n_train_batches:])) +",  validaiton accuracy :" + str(float( batch_size * n_valid_batches * batches2validate - this_validation_loss[-1])*100/(batch_size*n_valid_batches*batches2validate)) + "% -> best thus far " 
                    else :
                        print "...      -> epoch " + str(epoch_counter) + ", cost: " + str(numpy.mean(cost_saved[-1*n_train_batches:])) +",  validaiton accuracy :" + str(float( batch_size * n_valid_batches * batches2validate - this_validation_loss[-1])*100/(batch_size*n_valid_batches*batches2validate)) + "%"
            else:

                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = this_validation_loss + [numpy.sum(validation_losses)]
                if verbose is True:
                    if this_validation_loss[-1] < best_validation_loss :                    
                        print "...      -> epoch " + str(epoch_counter) + ", cost: " + str(cost_saved[-1]) +",  validaiton accuracy :" + str(float(batch_size*n_valid_batches - this_validation_loss[-1])*100/(batch_size*n_valid_batches)) + "%, learning_rate = " + str(eta.get_value(borrow=True)) + ", momentum = " +str(momentum_value(epoch_counter)) + " -> best thus far " 
                    else:
                        print "...      -> epoch " + str(epoch_counter) + ", cost: " + str(cost_saved[-1]) +",  validaiton accuracy :" + str(float(batch_size*n_valid_batches - this_validation_loss[-1])*100/(batch_size*n_valid_batches)) + "%, learning_rate = " + str(eta.get_value(borrow=True)) + ", momentum = " +str(momentum_value(epoch_counter)) 
                else:
                    if this_validation_loss[-1] < best_validation_loss :                    
                        print "...      -> epoch " + str(epoch_counter) + ", cost: " + str(cost_saved[-1]) +",  validaiton accuracy :" + str(float(batch_size*n_valid_batches - this_validation_loss[-1])*100/(batch_size*n_valid_batches)) + "% -> best thus far " 
                    else:
                        print "...      -> epoch " + str(epoch_counter) + ", cost: " + str(cost_saved[-1]) +",  validaiton accuracy :" + str(float(batch_size*n_valid_batches - this_validation_loss[-1])*100/(batch_size*n_valid_batches)) + "% "                        

            best_validation_loss = min(best_validation_loss, this_validation_loss[-1])
        new_leanring_rate = decay_learning_rate()    
    end_time = time.clock()
    print "... training complete, took " + str((end_time - start_time)/ 60.) +" minutes"




























    ###############
    # TEST MODEL  #
    ###############
    start_time = time.clock()
    print "... testing"
    wrong = 0
    predictions = []
    class_prob = []
    labels = []

    if multi_load is False:

        labels = test_set_y.eval().tolist()   
        for mini_batch in xrange(batches2test):
            #print ".. Testing batch " + str(mini_batch)
            wrong = wrong + int(test_model(mini_batch))                        
            predictions = predictions + prediction(mini_batch).tolist()
            class_prob = class_prob + nll(mini_batch).tolist()
        print "...      -> Total test accuracy : " + str(float((batch_size*n_test_batches)-wrong )*100/(batch_size*n_test_batches)) + " % out of " + str(batch_size*n_test_batches) + " samples."

    else:
        for batch in xrange(batches2test):
            print ".. Testing batch " + str(batch)
            # Load data for this batch
            if data_params["type"] == 'mat':
                test_data_x, test_data_y, test_data_y1 = load_data_mat(dataset, batch = batch + 1 , type_set = 'test')             

            elif data_params["type"] == 'skdata':                   
                if dataset == 'caltech101':
  
                    test_data_x, test_data_y = load_skdata_caltech101(batch_size = load_batches, batch = 1 , type_set = 'test', rand_perm = rand_perm, height = height, width = width )
            test_set_x.set_value(test_data_x,borrow = True)
            test_set_y.set_value(test_data_y,borrow = True)

            labels = labels + test_set_y.eval().tolist() 
            for mini_batch in xrange(n_test_batches):
                wrong = wrong + int(test_model(mini_batch))   
                predictions = predictions + prediction(mini_batch).tolist()
                class_prob = class_prob + nll(mini_batch).tolist()
         
        print "...      -> Total test accuracy : " + str(float((batch_size*n_test_batches*batches2test)-wrong )*100/(batch_size*n_test_batches*batches2test)) + " % out of " + str(batch_size*n_test_batches*batches2test) + " samples."

    end_time = time.clock()

    correct = 0
    confusion = numpy.zeros((outs,outs), dtype = int)
    for index in xrange(len(predictions)):
        if labels[index] is predictions[index]:
            correct = correct + 1
        confusion[int(predictions[index]),int(labels[index])] = confusion[int(predictions[index]),int(labels[index])] + 1


    # Save down data 
    f = open(results_file_name, 'w')
    for i in xrange(len(predictions)):
        f.write(str(i))
        f.write("\t")
        f.write(str(labels[i]))
        f.write("\t")
        f.write(str(predictions[i]))
        f.write("\t")
        for j in xrange(outs):
            f.write(str(class_prob[i][j]))
            f.write("\t")
        f.write('\n')

    f = open(error_file_name,'w')
    for i in xrange(len(this_validation_loss)):
        f.write(str(this_validation_loss[i]))
        f.write("\n")
    f.close()

    f = open(cost_file_name,'w')
    for i in xrange(len(cost_saved)):
        f.write(str(cost_saved[i]))
        f.write("\n")
    f.close()

        

    f.close()
    end_time = time.clock()
    print "Testing complete, took " + str((end_time - start_time)/ 60.) + " minutes"    
    print "Confusion Matrix with accuracy : " + str(float(correct)/len(predictions)*100)
    print confusion
    print "Done"

    pdb.set_trace()

















    #################
    # Boiler PLate  #
    #################
    
## Boiler Plate ## 
if __name__ == '__main__':
    
    import sys

                                                                                            # for epoch in [0, mom_epoch_interval] the momentum increases linearly
    optimization_params = {
                            "mom_start"                         : 0.5,                      # from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
                            "mom_end"                           : 0.98,
                            "mom_interval"                      : 500,
                            "mom_type"                          : 1,                        # if mom_type = 1 , classical momentum if mom_type = 0, no momentum, if mom_type = 2 Nesterov's accelerated gradient momentum ( not yet done... something is wrong.. refer code)
                            "initial_learning_rate"             : 0.01,                      # Learning rate at the start
                            "learning_rate_at_ada_grad_begin"   : 1,                        # Learning rate once ada_grad starts. Although ada grad can start anywhere, I recommend at the begining.
                            "learning_rate_at_ada_grad_end"     : 0.001,                      # learning rate once ada grad ends and regular SGD begins again. 
                            "learning_rate_decay"               : 0.998, 
                            "regularize_l1"                     : False,                     # only for the last logistic layer
                            "regularize_l2"                     : False,                     # only for the last logistic layer 
                            "cross_entropy"                     : False,                     # use for binary labels only. 
                            "ada_grad_begin"                    : 500,                        # Which epoch to begin ada grad. Don't use adagrad when using dropout. 
                            "ada_grad_end"                      : 2000,                       # Which epoch to end ada grad
                            "fudge_factor"                      : 1e-7,                     # Just to avoid divide by zero, but google even advocates trying '1'                            
                            }



    filename_params ={ 
                        "results_file_name" : "../results/results_caltech101.txt",        # Files that will be saved down on completion Can be used by the parse.m file
                        "error_file_name"   : "../results/error_caltech101.txt",
                        "cost_file_name"    : "../results/cost_caltech101.txt"
                    }        
        
    data_params = {
                   "type"               : 'skdata',                                    # Options: 'pkl', 'skdata' , 'mat' for loading pkl files, mat files for skdata files.
                   "loc"                : 'caltech101',                             # location for mat or pkl files, which data for skdata files. Skdata will be downloaded and used from '~/.skdata/'
                   "batch_size"         : 80,                                      # For loading and for Gradient Descent Batch Size
                   "load_batches"       : 2, 
                   "batches2train"      : 19,                                      # Number of training batches.
                   "batches2test"       : 19,                                       # Number of testing batches.
                   "batches2validate"   : 19,                                       # Number of validation batches
                   "height"             : 128,                                       # Height of each input image
                   "width"              : 128,                                       # Width of each input image
                   "channels"           : 3                                         # Number of channels of each input image 
                  }

    arch_params = {
                       # Decay of Learninig rate after each epoch of SGD
                    "squared_filter_length_limit"       : 15,   
                    "n_epochs"                          : 2000,                      # Total Number of epochs to run before completion (no premature completion)
                    "validate_after_epochs"             : 1,                        # After how many iterations to calculate validation set accuracy ?
                    "mlp_activations"                   : [ ReLU, ReLU , ReLU ],           # Activations of MLP layers Options: ReLU, Sigmoid, Tanh
                    "cnn_activations"                   : [ ReLU, ReLU , ReLU, ReLU, ReLU ],           # Activations for CNN layers Options: ReLU, Sigmoid, Tanh
                    "dropout"                           : True,                    # Flag for dropout / backprop When using dropout, don't use adagrad.
                    "column_norm"                       : False,
                    "dropout_rates"                     : [ 0.5, 0.5 ,0.5 ],             # Rates of dropout. Use 0 is backprop.
                    "nkerns"                            : [ 48 , 128 , 198, 198, 128],              # Number of feature maps at each CNN layer
                    "outs"                              : 102,                       # Number of output nodes ( must equal number of classes)
                    "filter_size"                       : [  11, 7 , 5 , 3 , 3],                # Receptive field of each CNN layer
                    "pooling_size"                      : [  2,  2 , 2 , 1 , 1],                # Pooling field of each CNN layer
                    "num_nodes"                         : [  2098, 2098  ],                # Number of nodes in each MLP layer
                    "use_bias"                          : True,                     # Flag for using bias                   
                    "random_seed"                       : 23455,                    # Use same seed for reproduction of results.
                    "svm_flag"                          : False                     # True makes the last layer a SVM

                 }

    run_cnn(
                    arch_params         = arch_params,
                    optimization_params = optimization_params,
                    data_params         = data_params, 
                    filename_params     = filename_params,
                    verbose             = False                                                  # True prints in a lot of intermetediate steps, False keeps it to minimum.
                )


