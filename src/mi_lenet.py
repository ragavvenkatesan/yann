#!/usr/bin/python

# General Packages
import os
import sys
import time
import pdb
from collections import OrderedDict
from operator import add

# Math Packages
import math
import scipy.io
import gzip
import cPickle
import cv2, cv
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
from cnn import load_data_mat
















	##########################
	# Data loading functions #
	##########################


def setup_data (	data_x, 
					division_x,
					division_y , 
					height,
					width 
				):

	# print "... Assembling Data"
	nImgs = data_x.shape[0]
	stride_x = width/division_y
	stride_y = height/division_x

	out = numpy.zeros((data_x.shape[0],data_x.shape[1]), dtype = theano.config.floatX)
	for img in xrange(nImgs):
		count = 0
		end_length = 0
		# I am transposing because its easy for me to correlate the transpose with my MATLAB setup. 
		# This means now the netwoks number or ordering with change around a bit, but still instances remain the same.
		curr = numpy.reshape(data_x[img,:],[height,width]).T
		for block_x in xrange(division_x):
			for block_y in xrange(division_y):
				length = len(curr[block_x*stride_x : block_x*stride_x+stride_x  , block_y*stride_y : block_y*stride_y+stride_y ].flatten(1))
				out[img, end_length : end_length + length] = curr[block_x*stride_x : block_x*stride_x+stride_x, block_y*stride_y : block_y*stride_y+stride_y].flatten(1)
				end_length = end_length + length
				# print " ... Assembling block " + str(count - 1) + " from " + str(block_x*stride_x) +" to " + str(block_x*stride_x + stride_x) + " in x "
	
	 
	if data_x.ndim != out.ndim:
		print " !! Error in setup data function "
	return out


	def load_data_pkl(dataset, division_x, division_y, height, width):
		''' Loads the dataset

		:type dataset: string
		:param dataset: the path to the dataset (here MNIST)
		'''
		# Download the MNIST dataset if it is not present
		data_dir, data_file = os.path.split(dataset)
		if data_dir == "" and not os.path.isfile(dataset):
			# Check if dataset is in the data directory.
			new_path = os.path.join(
			    os.path.split(__file__)[0],
			    dataset
			)
			if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
				dataset = new_path

		if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
			import urllib
			origin = (
			    'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
			)
			print 'Downloading data from %s' % origin
			urllib.urlretrieve(origin, dataset)

		print '... loading data'

		# Load the dataset
		f = gzip.open(dataset, 'rb')
		train_set, valid_set, test_set = cPickle.load(f)
		f.close()


	def shared_dataset(data_xy, division_x, division_y, height, width,  borrow=True):

		data_x, data_y = data_xy        
		data_x = setup_data ( data_x ,division_x, division_y , height, width )
		shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
		shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX),  borrow=borrow)
		                                 
		# one-hot encoded labels as {-1, 1}
		n_classes = len(numpy.unique(data_y))  # dangerous?
		y1 = -1 * numpy.ones((data_y.shape[0], n_classes))
		y1[numpy.arange(data_y.shape[0]), data_y] = 1
		shared_y1 = theano.shared(numpy.asarray(y1,dtype=theano.config.floatX), borrow=borrow)

		return shared_x, T.cast(shared_y, 'int32'), shared_y1

	test_set_x, test_set_y, test_set_y1 = shared_dataset(test_set, division_x, division_y, height, width)
	valid_set_x, valid_set_y, valid_set_y1 = shared_dataset(valid_set, division_x, division_y, height, width)
	train_set_x, train_set_y, train_set_y1 = shared_dataset(train_set, division_x, division_y, height, width)

	rval = [(train_set_x, train_set_y, train_set_y1), (valid_set_x, valid_set_y, valid_set_y1), (test_set_x, test_set_y, test_set_y1)]
	return rval
























		#################
		# Subnet Class  #
		#################

class sub_net(object):
	"""Multi-Instance Sub_net Creator
	Creates one sub net
    """
	def __init__(self,
				x, 
				y,
				y1,
				height, 
				width, 
				batch_size,
				filter_size, 
				pooling_size, 
				nkerns,
				outs,
				dropout_rates,
				rng,
				activations,
				cnn_activations,				
				use_bias,
				svm_flag,
				num_nodes,
				net_id,
				dropout,				
				eta, 
				mom,
				squared_filter_length_limit,
				ada_grad_flag,
				verbose,				
				):

		# allocate symbolic variables for the data
		# Working with load z
		self.main_input = x[ : , net_id*height*width : net_id*height*width + height*width ]    # wire up inputs. Possible thanks to setup data routine above as data is now re-arranged in this format.

		self.first_layer_input = self.main_input.reshape((batch_size, 1, height, width)) # basically make each into images for convolutions.

	    # Create all convolutional - pooling layers 
		self.conv_layers=[]		# List of convolutional layer objects 
		self.filt_size = filter_size[0]		# first layer filter size
		self.pool_size = pooling_size[0]	# first layer pooling size 
		# create the first layer 

		self.conv_layers.append ( LeNetConvPoolLayer(
		                        rng,
		                        input = self.first_layer_input,
		                        image_shape=(batch_size, 1 , height, width),
		                        filter_shape=(nkerns[0], 1 , self.filt_size, self.filt_size),
		                        poolsize=(self.pool_size, self.pool_size),
		                        activation = cnn_activations[0],
		                        verbose = verbose 
		                         ) )

		self.next_in_1 = ( height - self.filt_size + 1 ) / self.pool_size    	# next layer inputs sizes     
		self.next_in_2 = ( width - self.filt_size + 1 ) / self.pool_size

		# create subsequent convolutional laters using a for loop. 
		for layer in xrange(len(nkerns)-1):   
		    self.filt_size = filter_size[layer+1]
		    self.pool_size = pooling_size[layer+1]
		    self.conv_layers.append ( LeNetConvPoolLayer(
		                        rng,
		                        input=self.conv_layers[layer].output,        
		                        image_shape=(batch_size, nkerns[layer], self.next_in_1, self.next_in_2),
		                        filter_shape=(nkerns[layer+1], nkerns[layer], self.filt_size, self.filt_size),
		                        poolsize=(self.pool_size, self.pool_size),
		                        activation = cnn_activations[layer + 1],
		                        verbose = verbose 

		                         ) )
		    self.next_in_1 = ( self.next_in_1 - self.filt_size + 1 ) / self.pool_size        
		    self.next_in_2 = ( self.next_in_2 - self.filt_size + 1 ) / self.pool_size

		self.fully_connected_input = self.conv_layers[-1].output.flatten(2)		# inpput into the first fully connected mlp layer...

		# Assemble fully connected laters 
		if len(dropout_rates) > 2 :
		    self.layer_sizes =[]
		    self.layer_sizes.append( nkerns[-1] * self.next_in_1 * self.next_in_2 )
		    for i in xrange(len(dropout_rates)-1):
		        self.layer_sizes.append ( num_nodes[i] )
		    self.layer_sizes.append ( outs )
		else :
		    self.layer_sizes = [ nkerns[-1] * self.next_in_1 * self.next_in_2, num_nodes[0] , outs]

		assert len(self.layer_sizes) - 1 == len(dropout_rates)

		# create all the mlp layers using this routine.
		self.MLPlayers = MLP( rng=rng,
		                 input=self.fully_connected_input,
		                 layer_sizes=self.layer_sizes,
		                 dropout_rates=dropout_rates,
		                 activations=activations,
		                 use_bias=use_bias,
		                 svm_flag = svm_flag,
		                 verbose = verbose )

		# Build the expresson for the cost function.
		if svm_flag is False:
		    self.negative_log_likelihood = self.MLPlayers.negative_log_likelihood(y)
		    self.dropout_negative_log_likelihood = self.MLPlayers.dropout_negative_log_likelihood(y)
		else :        
		    self.negative_log_likelihood = self.MLPlayers.negative_log_likelihood(y1)
		    self.dropout_negative_log_likelihood = self.MLPlayers.dropout_negative_log_likelihood(y1)
	 	

		# This section is useful for back prop. Until now only fwd prop is defined. 		
		self.params = []			# list of all parameters of network. 
		for layer in self.conv_layers:
			self.params = self.params + layer.params
		self.params = self.params + self.MLPlayers.params

		# grdient of the entire network... most important step... reason why theano is needed. 
		self.gparams = T.grad(self.dropout_negative_log_likelihood if dropout else self.negative_log_likelihood, self.params)  #Nll is dependedent on the error. Basically -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

		if ada_grad_flag is True:
			self.grad_acc = []
			for param in self.params:
				eps = numpy.zeros_like(param.get_value(borrow=True), dtype=theano.config.floatX)
				self.grad_acc.append(theano.shared(eps, borrow=True))
    
			# create gradient wit momentum parameters.
		self.gparams_mom = []
		for param in self.params:
			gparam_mom = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,dtype=theano.config.floatX))
			self.gparams_mom.append(gparam_mom)

		self.updates = OrderedDict()
		if ada_grad_flag is True:
			fudge_factor = 1e-2  # Numerical Stability to avoid division by 0 
			for gparam_mom, gparam, acc in zip(self.gparams_mom, self.gparams, self.grad_acc):        
				# Misha Denil's original version
				# updates[gparam_mom] = mom * gparam_mom + (1. - mom) * gparam

				# change the update rule to match Hinton's dropout paper
				current_acc = acc + T.sqr(gparam)
				self.updates[gparam_mom] = mom * gparam_mom - (1. - mom) * (eta / (T.sqrt(current_acc) + fudge_factor ) ) * gparam # 
				self.updates[acc] = current_acc
		else:
			for gparam_mom, gparam in zip(self.gparams_mom, self.gparams):	       
				self.updates[gparam_mom] = mom * gparam_mom - (1. - mom) * eta * gparam

		# ... and take a step along that direction
		for param, gparam_mom in zip(self.params, self.gparams_mom):	       
			stepped_param = param + self.updates[gparam_mom]
			if param.get_value(borrow=True).ndim == 2:	            
				col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
				desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
				scale = desired_norms / (1e-7 + col_norms)
				self.updates[param] = stepped_param * scale
			else:
				self.updates[param] = stepped_param

		#self.errors = self.MLPlayers.errors					# count the total wrongs by this net. This is useless . 
		self.y_pred = self.MLPlayers.predicts 				# 0 , 1  - array    Prediction of this net.
		self.probabilities = self.MLPlayers.probabilities  # log(P_y_given_x)- 2d array
		self.output = self.dropout_negative_log_likelihood if dropout else self.negative_log_likelihood 




















		##################
		# Main Fucntion  #
		##################


def mi_lenet_dropout(  
					learning_rate_decay,
                    squared_filter_length_limit,
                    batch_size,
                    mom_params,
                    activations,
                    cnn_activations,
                    dropout,
                    dropout_rates,
                    results_file_name,
                    error_file_name,
                    cost_file_name,
                    use_bias,
					batches2pre_train,                    
                    batches2train,
                    batches2test,
                    batches2validate,
                    random_seed=9648,
                    dataset='mnist.pkl.gz',
                    initial_learning_rate= 1,
                    n_epochs=200,
					validate_after_epochs = 5,                    
                    nkerns=[20, 50],
                    filter_size = [5,5],
                    pooling_size = [2,2],
                    num_nodes =[500, 500],
                    height = 160,
                    width = 236,
                    outs = 10,
                    svm_flag = False,
                    division_x = 3,
                    division_y = 3 ,
                    mat_flag = True ,
                    ada_grad_flag = True,
                    pre_train_epochs = 20,                 
                    verbose = True 
                    ):

	print "           \n\n\n\n"
	# Forced inputs :: !!! ::  .... To be changed in future.  

	# batch_size = 1   # only online learninig supported as of now. 
	# dataset = 'data/mnist.pkl.gz'
	rng = numpy.random.RandomState(random_seed)  





















		#######################
		# Load and setup Data #
		#######################


	if mat_flag is True:

		train_data_x, train_data_y, train_data_y1, train_data_z = load_data_mat(dataset, batch = 1 , type_set = 'train', load_z = True)     
		test_data_x, test_data_y, valid_data_y1, valid_data_z = load_data_mat(dataset, batch = 1 , type_set = 'test' , load_z = True)   # Load dataset for first epoch.
		valid_data_x, valid_data_y, test_data_y1, test_data_z = load_data_mat(dataset, batch = 1 , type_set = 'valid' , load_z = True)   # Load dataset for first epoch.
		
		train_data_x = setup_data ( train_data_x ,division_x, division_y , height, width )
		train_set_x = theano.shared(numpy.asarray(train_data_x, dtype=theano.config.floatX), borrow=True)
		train_set_y = T.cast(theano.shared(numpy.asarray(train_data_y, dtype='int32'), borrow=True), 'int32')
		train_set_y1 = theano.shared(numpy.asarray(train_data_y1, dtype=theano.config.floatX), borrow=True)
		train_set_z = theano.shared(numpy.asarray(train_data_z, dtype=theano.config.floatX), borrow=True)


		test_data_x = setup_data ( test_data_x ,division_x, division_y , height, width )
		test_set_x  = theano.shared(numpy.asarray(test_data_x, dtype=theano.config.floatX), borrow=True)
		test_set_y = T.cast(theano.shared(numpy.asarray(test_data_y, dtype='int32'), borrow=True) , 'int32' )
		test_set_y1 = theano.shared(numpy.asarray(test_data_y1, dtype=theano.config.floatX), borrow=True)
		test_set_z = theano.shared(numpy.asarray(test_data_z, dtype=theano.config.floatX), borrow=True)


		valid_data_x = setup_data ( valid_data_x ,division_x, division_y , height, width )
		valid_set_x = theano.shared(numpy.asarray(valid_data_x, dtype=theano.config.floatX), borrow=True)
		valid_set_y = T.cast(theano.shared(numpy.asarray(valid_data_y, dtype='int32'), borrow=True) , 'int32' )
		valid_set_y1 = theano.shared(numpy.asarray(valid_data_y1, dtype=theano.config.floatX), borrow=True)
		valid_set_z = theano.shared(numpy.asarray(valid_data_z, dtype=theano.config.floatX), borrow=True)


	else:   

		data = load_data_pkl(dataset, division_x, division_y, height, width)
		train_set_x, train_set_y, train_set_y1 = data[0]
		valid_set_x, valid_set_y, valid_set_y1 = data[1]
		test_set_x, test_set_y, test_set_y1 = data[2]

	# compute number of online samples for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
























		######################
		# BUILD ACTUAL MODEL #
		######################

	start_time = time.clock()    
		# start-snippet-1
	x = T.matrix('x')   # the data is presented as rasterized images
	y = T.ivector('y')  # the labels are presented as 1D vector of                      # [int] labels
	y1 = T.matrix('y1')

	index = T.iscalar()  # index to a [mini]batch
	epoch = T.scalar()

	eta = theano.shared(numpy.asarray(initial_learning_rate,dtype=theano.config.floatX))

	# symbolic Hinton's momentum.
	mom = ifelse(epoch < mom_epoch_interval, mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval), mom_end)

	print '... Constructing the Networks ' 
	subnets_fwd = []
	subnets_bck = []
	for i in xrange(division_y * division_x):
		if verbose is True:
 			print "... Initializing Subnet " + str(i + 1) 
 			print "... 						Forward Network"
		subnets_fwd.append( 
						sub_net(
							x = x,	
							y = y,
							y1 = y1,
							net_id = i,
							height = height  / division_x , 
							width = width / division_y , 
							batch_size = batch_size, 				
							filter_size = filter_size, 
							pooling_size = pooling_size, 
							nkerns = nkerns,
							outs = outs,
							dropout_rates = dropout_rates,
							rng = rng,
							activations = activations,
							cnn_activations = cnn_activations,
							use_bias = use_bias,
							svm_flag = svm_flag,
							num_nodes = num_nodes,
							dropout = dropout,
							eta = eta, 
							mom = mom,
							squared_filter_length_limit = squared_filter_length_limit,	
							ada_grad_flag = ada_grad_flag,						
							verbose = verbose 
							) 
    					)
		if verbose is True:
			print "           \n\n"
			print "... 						Backward Network"
		subnets_bck.append( 
						sub_net(
							x = x,	
							y = y,
							y1 = y1,
							net_id = i,
							height = height  / division_x , 
							width = width / division_y , 
							batch_size = 1, 					# This is a different batch_size. This is the one for the stochastic gradient descent.
							filter_size = filter_size, 
							pooling_size = pooling_size, 
							nkerns = nkerns,
							outs = outs,
							dropout_rates = dropout_rates,
							rng = rng,
							activations = activations,
							cnn_activations = cnn_activations,
							use_bias = use_bias,
							svm_flag = svm_flag,
							num_nodes = num_nodes,
							dropout = dropout,
							eta = eta, 
							mom = mom,
							squared_filter_length_limit = squared_filter_length_limit,	
							ada_grad_flag = ada_grad_flag,						
							verbose = verbose 
							) 
    					)
		if verbose is True:
			print "           \n\n\n\n"

	preds =[]
	probs = []
	for i in xrange(division_y*division_x):
		preds.append(subnets_fwd[i].y_pred)						# each one is of length = batch_size 
		probs.append(subnets_fwd[i].probabilities)				# each is of size batch_size X 2  Log proabbility 

	y_pred = T.max ( [preds[i] for j in xrange(division_x*division_y)] , axis = 0 ) 		
	prediction_errors = T.sum(T.neq(y_pred, y))			# Count wrongs in a batch basically or possibly even individually. In this case only individually for online learning.

	# Compile theano function for testing.
	print '... Building the test models '
	test_model = theano.function(
			inputs = [index],	        
	        outputs= prediction_errors,					# Outputs number of wrongs.
	        givens={
	            x: test_set_x[index * batch_size:(index + 1) * batch_size],
	            y: test_set_y[index * batch_size:(index + 1) * batch_size]
	            })

	prediction_test = theano.function(					# Outputs predictions so that you could make a confusion matrix.
		inputs = [index],
	    outputs = y_pred,
	    givens={
	            x: test_set_x[index * batch_size:(index + 1) * batch_size]
	            })


	probs_test =  theano.function(inputs= [ index ],   				# Outputs log probability of each network while training. 
						outputs = [ probs[i] for i in xrange(division_x * division_y) ] ,	
						givens={
							x: test_set_x[index * batch_size:(index + 1) * batch_size]}
						)

	# Compile theano function for validation.
	print '... Building the validation model '
	validate_model = theano.function(
			inputs =  [index],
	        outputs = prediction_errors,				# Output the validation errors of predictions. 
	        givens={
	            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
	            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
	            })


	print '... Building the training prediction models '		
	predict_by_net_train = theano.function(	# Output predictions of each net during training. 
							inputs =  [index],
	       					outputs= [ preds[i] for i in xrange(division_x * division_y) ] , 
	       	 				givens={
	       	    					x: train_set_x[index * batch_size:(index + 1) * batch_size]}) 

	print '... Building the forward propogation models '
	fwd_prop =  theano.function(inputs= [ index ],   				# Outputs log probability of each network while training. 
						outputs = [ probs[i] for i in xrange(division_x * division_y) ] ,	
						givens={
							x: train_set_x[index * batch_size:(index + 1) * batch_size]}
						)

	print '... Building the back propagation models'
	back_prop = []
	for i in xrange(division_x*division_y):
		if verbose is True:
			print " 						Building model for net " + str(i)

		if svm_flag is True:
			back_prop.append( 
				theano.function(inputs= [ index, epoch],
						updates = subnets_bck[i].updates,
						outputs = subnets_bck[i].output,							# Outputs negative log likelihood of each network while training. 
						givens={
						x: train_set_x[index:index + 1],
						y1: train_set_y1[index:index + 1]}												
						))
		else:
			back_prop.append( 
				theano.function(inputs= [index, epoch],
						updates=subnets_bck[i].updates,	
						outputs = subnets_bck[i].output,							# Outputs negative log likelihood of each network while training. 
						givens={
						x: train_set_x[index:index + 1],
						y: train_set_y[index:index + 1]}					
						))

	decay_learning_rate = theano.function(
	       inputs=[], 
	       outputs=eta,												# Just updates the learning rates. 
	       updates={eta: eta * learning_rate_decay}
	        )

	end_time = time.clock()
	print "Building complete, took " + str((end_time - start_time)) + " seconds" 
	print "           \n\n\n\n"















	




	###################
	# PRE-TRAIN MODEL #
	###################
	print "... pre-training"
	start_time = time.clock()
	eta.set_value ( initial_learning_rate )    # reset learning rate after pre-training.
	best_validation_loss = numpy.inf
	this_validation_loss = []
	best_iter = 0
	epoch_counter = 0
	cost_saved = []
	best_params = None
	net_updates_positive_pretrain = numpy.zeros(division_x*division_y , dtype = 'int32')
	net_updates_negative_pretrain = numpy.zeros(division_x*division_y, dtype = 'int32' )
	good_params = subnets_fwd[0].params # initializations ..
	validate_after_epochs = 1
	correct_net = 0
	positive_labels = 0
	positive_per_bag = []

	while (epoch_counter < pre_train_epochs):    				# outside loop on epochs.
		epoch_counter = epoch_counter + 1 						# counts epoch number 
		epoch_time_start = time.clock()		

		print " ... Initiated Pre-Train Epoch " + str(epoch_counter) 

		for batch in xrange (batches2pre_train):			

			if verbose is True: 
				print "... Pre-Train Epoch: " + str(epoch_counter) + " Forward Prop Batch: " + str(batch+1) + " out of " + str(batches2pre_train) + " batches" 

			if mat_flag is True:  # load a batch of data and run on that batch
				train_data_x, train_data_y, train_data_y1, train_data_z = load_data_mat(dataset, batch = batch + 1 , type_set = 'train' , load_z = True)   # Load dataset for first epoch.
				train_data_x = setup_data ( train_data_x ,division_x, division_y , height, width )
				train_set_x.set_value(train_data_x, borrow = True)
				train_set_y.set_value(train_data_y, borrow=True)
				train_set_y1.set_value(train_data_y1, borrow=True)
				train_set_z.set_value(train_data_z, borrow = True)
				n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size		

				for minibatch_index in xrange(n_train_batches):  # this is basically image by image right now for backward prop. For forward its batch wise		
					count = 0
					if verbose is True:
						print "...                      Forward Propagation for minibatch " + str(minibatch_index) + " out of "    + str(n_train_batches)      
					fwd_probability = numpy.squeeze( fwd_prop (minibatch_index) ) # This is probability we want high										
					predict_of_net = numpy.squeeze( predict_by_net_train ( minibatch_index) ) # this is the prediction of each subnet
					inds = [] 
					labels = train_set_y.get_value(borrow = True)[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
					for i in xrange(batch_size):
						inds.append ( numpy.squeeze(fwd_probability[:,i,labels[i]]).argmax() )

					# Back Proppin in here.
					for sample in xrange(batch_size):
						cost_saved.append ( back_prop[inds[sample]](minibatch_index*batch_size + sample, epoch_counter) ) 

						for i in xrange(len(subnets_bck[inds[sample]].params)):
							good_params[i].set_value ( subnets_bck[inds[sample]].params[i].get_value(borrow = True) )

						for i in xrange(division_x*division_y):
							for j in xrange(len(subnets_fwd[i].params)):
								subnets_fwd[i].params[j].set_value ( good_params[j].get_value ( borrow = True ) )
								subnets_bck[i].params[j].set_value ( good_params[j].get_value ( borrow = True ) )

						if int(labels[sample]) == 1:     # If  the true label is 1
							net_updates_positive_pretrain[inds[sample]] = net_updates_positive_pretrain[inds[sample]] + 1	
							positive_labels = positive_labels + 1 						
							if int(inds[sample]) == int(train_set_z.get_value(borrow = True)[minibatch_index*batch_size + sample]):
								correct_net = correct_net + 1 
							if verbose is True:					
								print "...                      True Label 1. Back propogated network "  + str(inds[sample]) + " for label " + str(labels[sample])


						else:  # if true label is 0 		
							net_updates_negative_pretrain[inds[sample]] = net_updates_negative_pretrain[inds[sample]] + 1	
							if verbose is True:					
								print "...                      True Label 0. Back propogated network "  + str(inds[sample]) + " for label " + str(labels[sample])



				if verbose is True:
					print " Negative update histogram "  + str(net_updates_negative_pretrain)
					print " Positive update histogram "  + str(net_updates_positive_pretrain)

			else:       # This whole section needs to mimic the one above ...   
				print " Hold ."

		print " Last trained NLL: " + str(numpy.mean(cost_saved[-1*batch_size:]))
		print " Negative update histogram "  + str(net_updates_negative_pretrain)
		print " Positive update histogram "  + str(net_updates_positive_pretrain)
		print " Total correct net prediction: " + str(correct_net*100./positive_labels) + "%."

		epoch_time_end = time.clock()
		#if verbose is True:
		print " Pre-Train Epoch " + str(epoch_counter) + " complete. It took " +  str(epoch_time_end - epoch_time_start)  +  " seconds to finish."


		# Run validation after an epoch of pre-training
		if  epoch_counter % validate_after_epochs == 0:   			
			# Load Validation Dataset here.
			validation_losses = 0. 






















		if mat_flag is True: 
			count = 0 
			for batch in xrange (batches2validate):
				valid_data_x, valid_data_y, valid_data_y1, valid_data_z = load_data_mat(dataset, batch = batch + 1 , type_set = 'valid' , load_z = True)   # Load dataset for first epoch.
				valid_data_x = setup_data ( valid_data_x ,division_x, division_y , height, width )
				valid_set_x.set_value (valid_data_x, borrow = True)
				valid_set_y.set_value(valid_data_y, borrow=True)
				valid_set_y1.set_value(valid_data_y1, borrow=True)
				valid_set_z.set_value(valid_data_z, borrow = True)
				n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

				validation_losses = validation_losses + numpy.sum([validate_model(i) for i in xrange(n_valid_batches)])
			this_validation_loss = this_validation_loss + [validation_losses]
			print " Validaiton accuracy :" + str(float( n_valid_batches * batches2validate * batch_size- this_validation_loss[-1])*100/(n_valid_batches*batches2validate * batch_size)) + "%, learning_rate={}{}".format(
				eta.get_value(borrow=True), "**" if this_validation_loss[-1] < best_validation_loss else "")
			best_validation_loss = min(best_validation_loss, this_validation_loss[-1])
			#if this_validation_loss < best_validation_loss[-1]:   use this loop to set up the best W and b parameters for testing..
			#	best_validation_loss = this_validation_loss[-1]

			print "           \n\n\n\n"

		else:
			print " Hold. "


	# Reset good parameters 
	for i in xrange(division_x*division_y):
		for j in xrange(len(subnets_fwd[i].params)):
			subnets_fwd[i].params[j].set_value ( good_params[j].get_value ( borrow = True ) )
			subnets_bck[i].params[j].set_value ( good_params[j].get_value ( borrow = True ) )

	print "           \n\n\n\n"




	

















	###############
	# TRAIN MODEL #
	###############
	#print " Reminder: View the Ws and bs to see how they look like."
	#pdb.set_trace()
	print "... training"
	start_time = time.clock()
	eta.set_value ( initial_learning_rate )    # reset learning rate after pre-training.
	best_validation_loss = numpy.inf
	this_validation_loss = []
	best_iter = 0
	epoch_counter = 0
	cost_saved = []
	best_params = None
	net_updates_positive_train = numpy.zeros(division_x*division_y , dtype = 'int32')
	net_updates_negative_train = numpy.zeros(division_x*division_y, dtype = 'int32' )
	validate_after_epochs = 1
	correct_net = 0
	positive_labels = 0

	while (epoch_counter < n_epochs):    				# outside loop on epochs.
		epoch_counter = epoch_counter + 1 						# counts epoch number 
		epoch_time_start = time.clock()		

		print " ... Initiated Train Epoch " + str(epoch_counter) 

		for batch in xrange (batches2train):			

			if verbose is True: 
				print "... Train Epoch: " + str(epoch_counter) + " Forward Prop Batch: " + str(batch+1) + " out of " + str(batches2train) + " batches" 

			if mat_flag is True:  # load a batch of data and run on that batch
				train_data_x, train_data_y, train_data_y1, train_data_z = load_data_mat(dataset, batch = batch + 1 , type_set = 'train' , load_z = True)   # Load dataset for first epoch.
				train_data_x = setup_data ( train_data_x ,division_x, division_y , height, width )
				train_set_x.set_value(train_data_x, borrow = True)
				train_set_y.set_value(train_data_y, borrow=True)
				train_set_y1.set_value(train_data_y1, borrow=True)
				train_set_z.set_value(train_data_z, borrow = True)
				n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size		

				for minibatch_index in xrange(n_train_batches):  # this is basically image by image right now for backward prop. For forward its batch wise		
					count = 0
					if verbose is True:
						print "...                      Forward Propagation for minibatch " + str(minibatch_index) + " out of "    + str(n_train_batches)      

					fwd_probability = numpy.squeeze( fwd_prop (minibatch_index) ) # This is probability we want high										
					predict_of_net = numpy.squeeze( predict_by_net_train ( minibatch_index) ) # this is the prediction of each subnet
					inds = [] 
					labels = train_set_y.get_value(borrow = True)[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
					for i in xrange(batch_size):
						inds.append ( numpy.squeeze(fwd_probability[:,i,labels[i]]).argmax() )
					# Back Proppin in here.
					for sample in xrange(batch_size):
						cost_saved.append ( back_prop[inds[sample]](minibatch_index*batch_size + sample, epoch_counter) ) 
						for i in xrange(len(subnets_bck[inds[sample]].params)):
							subnets_fwd[inds[sample]].params[i].set_value ( subnets_bck[inds[sample]].params[i].get_value(borrow = True) )

						if int(labels[sample]) == 1:     # If  the true label is 1
							net_updates_positive_train[inds[sample]] = net_updates_positive_train[inds[sample]] + 1	
							positive_labels = positive_labels + 1 						
							if int(inds[sample]) == int(train_set_z.get_value(borrow = True)[minibatch_index*batch_size + sample]):
								correct_net = correct_net + 1 
							if verbose is True:					
								print "...                      True Label 1. Back propogated network "  + str(inds[sample]) + " for label " + str(labels[sample])


						else:  # if true label is 0 		
							net_updates_negative_train[inds[sample]] = net_updates_negative_train[inds[sample]] + 1	
							if verbose is True:					
								print "...                      True Label 0. Back propogated network "  + str(inds[sample]) + " for label " + str(labels[sample])



				if verbose is True:
					print " Negative update histogram "  + str(net_updates_negative_train)
					print " Positive update histogram "  + str(net_updates_positive_train)

			else:       # This whole section needs to mimic the one above ...   
				print " Hold ."

		print " Last trained NLL: " + str(numpy.mean(cost_saved[-1*batch_size:]))
		print " Negative update histogram "  + str(net_updates_negative_train)
		print " Positive update histogram "  + str(net_updates_positive_train)
		print " Total correct net prediction: " + str(correct_net*100./positive_labels) + "%."

		epoch_time_end = time.clock()
		#if verbose is True:
		print " Train Epoch " + str(epoch_counter) + " complete. It took " +  str(epoch_time_end - epoch_time_start)  +  " seconds to finish."


		# Run validation after an epoch of pre-training
		if  epoch_counter % validate_after_epochs == 0:   			
			# Load Validation Dataset here.
			validation_losses = 0. 






















		if mat_flag is True: 
			count = 0 
			for batch in xrange (batches2validate):
				valid_data_x, valid_data_y, valid_data_y1, valid_data_z = load_data_mat(dataset, batch = batch + 1 , type_set = 'valid' , load_z = True)   # Load dataset for first epoch.
				valid_data_x = setup_data ( valid_data_x ,division_x, division_y , height, width )
				valid_set_x.set_value (valid_data_x, borrow = True)
				valid_set_y.set_value(valid_data_y, borrow=True)
				valid_set_y1.set_value(valid_data_y1, borrow=True)
				valid_set_z.set_value(valid_data_z, borrow = True)
				n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

				validation_losses = validation_losses + numpy.sum([validate_model(i) for i in xrange(n_valid_batches)])
			this_validation_loss = this_validation_loss + [validation_losses]
			print " Validaiton accuracy :" + str(float( n_valid_batches * batches2validate * batch_size - this_validation_loss[-1])*100/(n_valid_batches*batches2validate * batch_size)) + "%, learning_rate={}{}".format(
				eta.get_value(borrow=True), "**" if this_validation_loss[-1] < best_validation_loss else "")
			best_validation_loss = min(best_validation_loss, this_validation_loss[-1])
			#if this_validation_loss < best_validation_loss[-1]:   use this loop to set up the best W and b parameters for testing..
			#	best_validation_loss = this_validation_loss[-1]
			print "           \n\n\n\n"


		else:
			print " Hold. "

		decay_learning_rate()




















	###############
	# TEST MODEL #
	###############
	def differences(a, b):
		if len(a) != len(b):
			raise ValueError("Lists of different length.")
		return sum(1 for i, j in zip(a, b) if i != j)


	print "... Testing"
	wrong = 0
	predictions = []
	class_prob = []
	labels = []

	start_time = time.clock()
	if mat_flag is False:
		print " Hold. "
	else:
		for batch in xrange(batches2test):
			if verbose is True:
				print ".. Testing batch " + str(batch)
			test_data_x, test_data_y, test_data_y1, test_data_z = load_data_mat(dataset, batch = batch + 1 , type_set = 'test', load_z = True)  
			test_data_x = setup_data ( test_data_x ,division_x, division_y , height, width )
			test_set_x.set_value( test_data_x, borrow = True )
			test_set_y.set_value(test_data_y, borrow=True)
			test_set_y1.set_value(test_data_y1, borrow=True)
			test_set_z.set_value ( test_data_z, borrow = True)

			n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
			labels = labels + test_set_y.get_value(borrow = True).tolist()
			for mini_batch in xrange(n_test_batches):
				#print " Testing Range : " + str(batch * batch_size) + " to " + str((batch + 1) * batch_size)
				wrong = wrong + int(test_model(mini_batch))   
				predictions.extend ( prediction_test(mini_batch) )
				#curr_probs = probs_test (mini_batch)

				#class_prob = class_prob  + 	[curr_probs[max(enumerate(curr_probs[:,int(test_set_y.get_value(borrow = True)[mini_batch])]),key=lambda v: v[1])[0]].tolist()]
			
		print "Total test accuracy : " + str(float((n_test_batches*batches2test*batch_size)-wrong )*100/(n_test_batches*batches2test*batch_size)) + " % out of " + str(n_test_batches*batches2test*batch_size) + " samples."

	correct = 0
	confusion = numpy.zeros((outs,outs), dtype = int)
	for index in xrange(len(predictions)):
		if labels[index] == predictions[index]:
			correct = correct + 1
		confusion[int(predictions[index]),int(labels[index])] = confusion[int(predictions[index]),int(labels[index])] + 1


	end_time = time.clock()
	print "Testing complete, took " + str((end_time - start_time)/ 60.) + " minutes"    
	print "           \n\n\n\n"























	###################
	# Save down stuff #
	###################
	print "... Writing down files."
	f = open(results_file_name, 'w')
	for i in xrange(len(predictions)):
		f.write(str(i))
		f.write("\t")
		f.write(str(labels[i]))
		f.write("\t")
		f.write(str(predictions[i]))
		f.write("\t")
		#for j in xrange(outs):
		#	f.write(str(class_prob[i][j]))
		#	f.write("\t")
		#f.write('\n')

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
	print "Confusion Matrix with accuracy : " + str(float(correct)/len(predictions)*100)
	print confusion
	print "Done"

	pdb.set_trace()


















## Boiler Plate ## 
if __name__ == '__main__':
    
    import sys
    
  
    # dropout rate for each layer
    dropout_rates = [ 0.5, 0.5 ]
     
    #### the params for momentum
    mom_start = 0.5
    mom_end = 0.99
    # for epoch in [0, mom_epoch_interval] the momentum increases linearly
    # from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
    mom_epoch_interval = 50
    mom_params = {"start": mom_start,
                  "end": mom_end,
                  "interval": mom_epoch_interval}

    if len(sys.argv) < 2:
        print "Usage: {0} [dropout|backprop]".format(sys.argv[0])
        exit(1)

    elif sys.argv[1] == "dropout":
        dropout = True
        results_file_name = "mil_results_dropout.txt"
        error_file_name = "mil_error_dropout.txt"
        cost_file_name = "mil_cost_dropout.txt"

    elif sys.argv[1] == "backprop":
        dropout = False
        results_file_name = "results_backprop.txt"
        error_file_name = "error_dropout.txt"
        cost_file_name = "cost_dropout.txt"

    else:
        print "I don't know how to '{0}'".format(sys.argv[1])
        exit(1)














    mi_lenet_dropout(
                
             initial_learning_rate = 1,                   # Learning rate begining             
             learning_rate_decay= 1,                   # Decay of Learninig rate after each iteration of SGD
             squared_filter_length_limit=15,
             n_epochs = 2,                                # Number of Epochs to run
         	 validate_after_epochs = 1,
             batch_size = 250,                              # Only online training works as of now. This is not batch_Size of SGD but how many samples before a back prop happens.
             mom_params=mom_params, 
             activations= [  ReLU , ReLU ],
             cnn_activations = [ ReLU, ReLU, ReLU],
             dropout=dropout, 
             dropout_rates = [ 0.5 , 0.5 ],             # Dropout rates for input, followed by all hidden layers
             dataset='../dataset/simple_mnist/dataset/',
             batches2pre_train = 1,             
             batches2train = 1,
             batches2test = 1,
             batches2validate = 1,
             results_file_name=results_file_name,
             error_file_name=error_file_name,
             cost_file_name=cost_file_name,
             use_bias = True,
             random_seed = 23455,
             nkerns = [ 6, 10 ], 
             outs = 2,                                   # Number of output nodes
             filter_size =  [ 7, 5  ],                      # Receptive field of each CNN layer
             pooling_size = [ 2, 1  ],                     # Pooling field of each CNN layer
             num_nodes =    [ 490 ], 							# Number of nodes in each MLP layer
             height = 84,
             width = 84,                   
             svm_flag = False,                            # True makes the last layer a SVM
             division_x = 3,					# number of instanes division in rows
             division_y = 3, 					# number of instances division in columns
             mat_flag = True,
             ada_grad_flag = True,
             pre_train_epochs = 1,
             verbose = True 

                )



