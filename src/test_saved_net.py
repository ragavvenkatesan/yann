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
import cv2

# Theano Packages
import theano
import theano.tensor as T
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

# Utilities
from util import visualize
from util import visualize_color_filters
from util import save_network
from util import load_network

# Datahandling packages
from loaders import load_data_pkl
from loaders import load_data_mat
from loaders import load_skdata_caltech101
from loaders import load_skdata_mnist
from loaders import load_skdata_cifar10
from loaders import load_skdata_mnist_noise1
from loaders import load_skdata_mnist_noise2
from loaders import load_skdata_mnist_noise3
from loaders import load_skdata_mnist_noise4
from loaders import load_skdata_mnist_noise5
from loaders import load_skdata_mnist_noise6
from loaders import load_skdata_mnist_bg_images
from loaders import load_skdata_mnist_bg_rand
from loaders import load_skdata_mnist_rotated
from loaders import load_skdata_mnist_rotated_bg



def run ( 
                   filename_params         ,                  
                   verbose                 ,                                                # True prints in a lot of intermetediate steps, False keeps it to minimum.
				   data_params			   ,
                   visual_params           
        ):


    #####################
    # Unpack Variables  #
    #####################
        
        
   
    visualize_flag          = visual_params ["visualize_flag" ]
    n_visual_images         = visual_params ["n_visual_images" ] 
    display_flag            = visual_params ["display_flag" ]           
                
    results_file_name   = filename_params [ "results_file_name" ]                # Files that will be saved down on completion Can be used by the parse.m file
    confusion_file_name = filename_params [ "confusion_file_name" ]
    load_file_name      = filename_params [ "load_file_name" ]
     
    dataset             = data_params [ "loc" ]
    height              = data_params [ "height" ]
    width               = data_params [ "width" ]
    batch_size          = data_params [ "batch_size" ]    
    load_batches        = data_params [ "load_batches"  ] * batch_size
    batches2test        = data_params [ "batches2test" ]
    channels            = data_params [ "channels" ]
             
                
    #################
    # Load Network  #
    #################            
    params, arch_params = load_network(load_file_name)                         
                
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

                
    rng = numpy.random.RandomState(random_seed)  
                                    
    #################
    # Data Loading  #
    #################
    # To DO: Make this a class in the loaders.py so that this doesn't have to be copied into all the future codes. 
    print "... loading data"

    # load matlab files as dataset.
    if data_params["type"] == 'mat':
        train_data_x, train_data_y, train_data_y1 = load_data_mat(dataset, batch = 1 , type_set = 'train')             
        test_data_x, test_data_y, valid_data_y1 = load_data_mat(dataset, batch = 1 , type_set = 'test')      # Load dataset for first epoch.
        valid_data_x, valid_data_y, test_data_y1 = load_data_mat(dataset, batch = 1 , type_set = 'valid')    # Load dataset for first epoch.       
        test_set_x = theano.shared(numpy.asarray(test_data_x, dtype=theano.config.floatX), borrow=True)
        test_set_y = theano.shared(numpy.asarray(test_data_y, dtype='int32'), borrow=True) 
        test_set_y1 = theano.shared(numpy.asarray(test_data_y1, dtype=theano.config.floatX), borrow=True)      

        # compute number of minibatches for training, validation and testing
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

        multi_load = True

    # load pkl data as is shown in theano tutorials
    elif data_params["type"] == 'pkl':   

        data = load_data_pkl(dataset)
        test_set_x, test_set_y, test_set_y1 = data[2]

         # compute number of minibatches for training, validation and testing
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_images = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches_all = n_test_images / batch_size 


        if  (n_test_batches_all < batches2test):        # You can't have so many batches.
            print "...  !! Dataset doens't have so many batches. "
            raise AssertionError()

        multi_load = False

    # load skdata ( its a good library that has a lot of datasets)
    elif data_params["type"] == 'skdata':

        if (dataset == 'mnist' or 
            dataset == 'mnist_noise1' or 
            dataset == 'mnist_noise2' or
            dataset == 'mnist_noise3' or
            dataset == 'mnist_noise4' or
            dataset == 'mnist_noise5' or
            dataset == 'mnist_noise6' or
            dataset == 'mnist_bg_images' or
            dataset == 'mnist_bg_rand' or
            dataset == 'mnist_rotated' or
            dataset == 'mnist_rotated_bg') :

            print "... importing " + dataset + " from skdata"

            func = globals()['load_skdata_' + dataset]
            data = func()
            test_set_x, test_set_y, test_set_y1 = data[2]
            # compute number of minibatches for training, validation and testing
            n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
            n_test_images = test_set_x.get_value(borrow=True).shape[0]
            n_test_batches_all = n_test_images / batch_size 
            if  (n_test_batches_all < batches2test):        # You can't have so many batches.
                print "...  !! Dataset doens't have so many batches. "
                raise AssertionError()

            multi_load = False

        elif dataset == 'cifar10':
            print "... importing cifar 10 from skdata"
            data = load_skdata_cifar10()
            test_set_x, test_set_y, test_set_y1 = data[2]
            # compute number of minibatches for training, validation and testing
            n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
            multi_load = False

        elif dataset == 'caltech101':
            print "... importing caltech 101 from skdata"

                # shuffle the data
            total_images_in_dataset = 9144 
            rand_perm = numpy.random.permutation(total_images_in_dataset)  # create a constant shuffle, so that data can be loaded in batchmode with the same random shuffle
            n_test_images = total_images_in_dataset / 3     
            n_test_batches_all = n_test_images / batch_size 


            if (n_test_batches_all < batches2test):        # You can't have so many batches.
                print "...  !! Dataset doens't have so many batches. "
                raise AssertionError()

            test_data_x, test_data_y  = load_skdata_caltech101(batch_size = load_batches, rand_perm = rand_perm, batch = 1 , type_set = 'test' , height = height, width = width)      # Load dataset for first epoch.           
            test_set_x = theano.shared(test_data_x, borrow=True)
            test_set_y = theano.shared(test_data_y, borrow=True) 
          
            # compute number of minibatches for training, validation and testing
            n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
            multi_load = True
                             
                  

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
    activity = []       # to record Cnn activities 
    weights = []

    conv_layers=[]
    filt_size = filter_size[0]
    pool_size = pooling_size[0]

    count = 0 
    if not nkerns == []: 
        conv_layers.append ( LeNetConvPoolLayer(
                                rng,
                                input = first_layer_input,
                                image_shape=(batch_size, channels , height, width),
                                filter_shape=(nkerns[0], channels , filt_size, filt_size),
                                poolsize=(pool_size, pool_size),
                                activation = cnn_activations[0],
                                W = params[count], b = params[count+1],
                                verbose = verbose
                                 ) )
        count = count + 2
        activity.append ( conv_layers[-1].output )
        weights.append ( conv_layers[-1].filter_img)

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
                                W = params[count], b = params[count+1],
                                verbose = verbose
                                 ) )
            next_in_1 = ( next_in_1 - filt_size + 1 ) / pool_size        
            next_in_2 = ( next_in_2 - filt_size + 1 ) / pool_size
            weights.append ( conv_layers[-1].filter_img )
            activity.append( conv_layers[-1].output )
            count = count + 2 
    # Assemble fully connected laters
    if nkerns == []:
        fully_connected_input = first_layer_input
    else:
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

    """  Dropouts implemented from paper:
    Srivastava, Nitish, et al. "Dropout: A simple way to prevent neural networks
    from overfitting." The Journal of Machine Learning Research 15.1 (2014): 1929-1958.
    """

    MLPlayers = MLP( rng=rng,
                     input=fully_connected_input,
                     layer_sizes=layer_sizes,
                     dropout_rates=dropout_rates,
                     activations=mlp_activations,
                     use_bias = use_bias,
                     svm_flag = svm_flag,
                     params = params[count:],
                     verbose = verbose)

    # Build the expresson for the categorical cross entropy function.
    if svm_flag is False:
        cost = MLPlayers.negative_log_likelihood( y )
        dropout_cost = MLPlayers.dropout_negative_log_likelihood( y )
    else :        
        cost = MLPlayers.negative_log_likelihood( y1 )
        dropout_cost = MLPlayers.dropout_negative_log_likelihood( y1 )

    # create theano functions for evaluating the graphs
    test_model = theano.function(
            inputs=[index],
            outputs=MLPlayers.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})

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



    # function to return activations of each image
    activities = theano.function (
        inputs = [index],
        outputs = activity,
        givens = {
                x: test_set_x[index * batch_size: (index + 1) * batch_size]
                 })
    
      
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
  
                    test_data_x, test_data_y = load_skdata_caltech101(batch_size = load_batches, batch = batch +  1 , type_set = 'test', rand_perm = rand_perm, height = height, width = width )

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
        
    f = open(confusion_file_name, 'w')
    f.write(confusion)

    f.close()
    end_time = time.clock()
    print "Testing complete, took " + str((end_time - start_time)/ 60.) + " minutes"    
    print "Confusion Matrix with accuracy : " + str(float(correct)/len(predictions)*100)
    print confusion
    print "Done"

    pdb.set_trace()


## Boiler Plate ## 
if __name__ == '__main__':
    
    import sys

    filename_params = { 
                        "load_file_name"        : "../dataset/network.pkl.gz",
                        "results_file_name"     : "../results/results.txt",        # Files that will be saved down on completion Can be used by the parse.m file
                        "confusion_file_name"   : "../results/confusion.txt"
                    }        
       
	# if data_params is pickled, these could be just anything it doens't matter.    
    data_params = {
                   "type"               : 'skdata',                                    # Options: 'pkl', 'skdata' , 'mat' for loading pkl files, mat files for skdata files.
                   "loc"                : 'mnist',                             # location for mat or pkl files, which data for skdata files. Skdata will be downloaded and used from '~/.skdata/'
                   "batch_size"         : 500,                                      # For loading and for Gradient Descent Batch Size
                   "load_batches"       : -1, 
                   "batches2test"       : 20,                                       # Number of testing batches.
                   "height"             : 28,                                       # Height of each input image
                   "width"              : 28,                                       # Width of each input image
                   "channels"           : 1                                         # Number of channels of each input image 
                  }
                  
                  
    visual_params = {
                "visualize_flag"        : True,
                "n_visual_images"       : 20,
                "display_flag"          : False
        }


    run(
                   filename_params         = filename_params,                  
                   verbose                 = False,                                                # True prints in a lot of intermetediate steps, False keeps it to minimum.
				   data_params			   = data_params,
                   visual_params           = visual_params
				   		
                )

