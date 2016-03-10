#!/usr/bin/python
import os
import sys
import numpy
import scipy.io
import cPickle
import cv2
from random import randint
import skdata
from skdata import caltech
from scipy.misc import imread
import time
from scipy import linalg
from sklearn.base import TransformerMixin, BaseEstimator

# for loading matlab based data.
def load_data_mat(n_classes, height, width, channels, dataset = '../dataset/waldo/' ,batch = 1, type_set = 'train', load_z = False):

	# Use this code if the data was created in matlab in the right format and needed to be loaded
	# print "... Loading " + type_set + " batch number " + str(batch)
	#- ----------  Load Dataet ------- -#
	mat = scipy.io.loadmat(dataset  +  type_set + '/batch_' + str(batch) + '.mat')
	data_x = numpy.asarray(mat['x'], dtype = 'float32')
	if data_x.max() > 1:
		data_x = data_x/data_x.max() # this is not normalize. This just scales. 
	for i in xrange(data_x.shape[0]):
		temp = data_x[i,];
		if channels > 1:
			temp = numpy.reshape (temp,[ height, width, channels ] , order = 'F')
			temp = numpy.reshape (temp,[1, height * width * channels ]) 
		else:
			temp = numpy.reshape (temp,[ height, width ] , order = 'F')
			temp = numpy.reshape (temp,[1, height * width ])
		data_x[i] = temp					 
	data_y = numpy.array(numpy.squeeze(mat['y']), dtype = 'int32')
	if load_z is True:
	    data_z = numpy.array(numpy.squeeze(mat['z']), dtype='float32' )
	y1 = -1 * numpy.ones((data_y.shape[0], n_classes))
	y1[numpy.arange(data_y.shape[0]), data_y] = 1
	if load_z is False:
	    return (data_x,data_y,y1.astype( dtype = 'float32' ))
	else:
	    return (data_x,data_y,y1.astype( dtype = 'float32' ),data_z)

# for MNIST of skdata
def load_skdata_mnist ():
	from skdata import mnist
	mn = mnist.dataset.MNIST()
	mn.fetch(True)
	meta = mn.build_meta()

	train_x = mn.arrays['train_images'][0:50000]
	valid_x = mn.arrays['train_images'][50000:]
	test_x = mn.arrays['test_images']
	train_y = mn.arrays['train_labels'][0:50000]
	valid_y = mn.arrays['train_labels'][50000:]
	test_y = mn.arrays['test_labels']

	# this is a hack. 
	rval = [(train_x, train_y, train_y), (valid_x, valid_y, valid_y), (test_x, test_y, test_y)]
	return rval

def load_skdata_mnist_noise1():
	from skdata import larochelle_etal_2007
	mn = larochelle_etal_2007.MNIST_Noise1()  
	mn.fetch(True)
	meta = mn.build_meta()
	data_x = mn._inputs
	data_y = mn._labels 
	train_x = data_x[0:10000]
	train_y = data_y[0:10000]

	test_x = data_x[10000:12000]
	test_y = data_y[10000:12000]

	valid_x = data_x[12000:]
	valid_y = data_y[12000:]

	rval = [(train_x, train_y, train_y), (valid_x, valid_y, valid_y), (test_x, test_y, test_y)]
	return rval

def load_skdata_mnist_noise2():
	from skdata import larochelle_etal_2007
	mn = larochelle_etal_2007.MNIST_Noise2()  
	mn.fetch(True)
	meta = mn.build_meta()
	data_x = mn._inputs
	data_y = mn._labels 
	train_x = data_x[0:10000]
	train_y = data_y[0:10000]

	test_x = data_x[10000:12000]
	test_y = data_y[10000:12000]

	valid_x = data_x[12000:]
	valid_y = data_y[12000:]
	
	rval = [(train_x, train_y, train_y), (valid_x, valid_y, valid_y), (test_x, test_y, test_y)]
	return rval



def load_skdata_mnist_noise3():
	from skdata import larochelle_etal_2007
	mn = larochelle_etal_2007.MNIST_Noise3()  
	mn.fetch(True)
	meta = mn.build_meta()
	data_x = mn._inputs
	data_y = mn._labels 
	train_x = data_x[0:10000]
	train_y = data_y[0:10000]

	test_x = data_x[10000:12000]
	test_y = data_y[10000:12000]

	valid_x = data_x[12000:]
	valid_y = data_y[12000:]

	rval = [(train_x, train_y, train_y), (valid_x, valid_y, valid_y), (test_x, test_y, test_y)]
	return rval



def load_skdata_mnist_noise4():
	from skdata import larochelle_etal_2007
	mn = larochelle_etal_2007.MNIST_Noise4()  
	mn.fetch(True)
	meta = mn.build_meta()
	data_x = mn._inputs
	data_y = mn._labels 
	train_x = data_x[0:10000]
	train_y = data_y[0:10000]

	test_x = data_x[10000:12000]
	test_y = data_y[10000:12000]

	valid_x = data_x[12000:]
	valid_y = data_y[12000:]

	rval = [(train_x, train_y, train_y), (valid_x, valid_y, valid_y), (test_x, test_y, test_y)]
	return rval



def load_skdata_mnist_noise5():
	from skdata import larochelle_etal_2007
	mn = larochelle_etal_2007.MNIST_Noise5()  
	mn.fetch(True)
	meta = mn.build_meta()
	data_x = mn._inputs
	data_y = mn._labels 

	train_x = data_x[0:10000]
	train_y = data_y[0:10000]

	test_x = data_x[10000:12000]
	test_y = data_y[10000:12000]

	valid_x = data_x[12000:]
	valid_y = data_y[12000:]

	rval = [(train_x, train_y, train_y), (valid_x, valid_y, valid_y), (test_x, test_y, test_y)]
	return rval


def load_skdata_mnist_noise6():
	from skdata import larochelle_etal_2007
	mn = larochelle_etal_2007.MNIST_Noise6()  
	mn.fetch(True)
	meta = mn.build_meta()
	data_x = mn._inputs
	data_y = mn._labels 

	train_x = data_x[0:10000]
	train_y = data_y[0:10000]

	test_x = data_x[10000:12000]
	test_y = data_y[10000:12000]

	valid_x = data_x[12000:]
	valid_y = data_y[12000:]

	rval = [(train_x, train_y, train_y), (valid_x, valid_y, valid_y), (test_x, test_y, test_y)]
	return rval


def load_skdata_mnist_bg_images():
	from skdata import larochelle_etal_2007
	mn = larochelle_etal_2007.MNIST_BackgroundImages()  
	mn.fetch(True)
	meta = mn.build_meta()
	data_x = mn._inputs
	data_y = mn._labels 

	train_x = data_x[0:40000]
	train_y = data_y[0:40000]

	test_x = data_x[50000:]
	test_y = data_y[50000:]

	valid_x = data_x[40000:50000]
	valid_y = data_y[40000:50000]

	rval = [(train_x, train_y, train_y), (valid_x, valid_y, valid_y), (test_x, test_y, test_y)]
	return rval


def load_skdata_mnist_bg_rand():
	from skdata import larochelle_etal_2007
	mn = larochelle_etal_2007.MNIST_BackgroundRandom()  
	mn.fetch(True)
	meta = mn.build_meta()
	data_x = mn._inputs
	data_y = mn._labels 

	train_x = data_x[0:40000]
	train_y = data_y[0:40000]

	test_x = data_x[50000:]
	test_y = data_y[50000:]

	valid_x = data_x[40000:50000]
	valid_y = data_y[40000:50000]

	rval = [(train_x, train_y, train_y), (valid_x, valid_y, valid_y), (test_x, test_y, test_y)]
	return rval


def load_skdata_mnist_rotated():
	from skdata import larochelle_etal_2007
	mn = larochelle_etal_2007.MNIST_Rotated()  
	mn.fetch(True)
	meta = mn.build_meta()
	data_x = mn._inputs
	data_y = mn._labels 

	train_x = data_x[0:40000]
	train_y = data_y[0:40000]

	test_x = data_x[50000:]
	test_y = data_y[50000:]

	valid_x = data_x[40000:50000]
	valid_y = data_y[40000:50000]

	rval = [(train_x, train_y, train_y), (valid_x, valid_y, valid_y), (test_x, test_y, test_y)]
	return rval


def load_skdata_mnist_rotated_bg():
	from skdata import larochelle_etal_2007
	mn = larochelle_etal_2007.MNIST_RotatedBackgroundImages()  
	mn.fetch(True)
	meta = mn.build_meta()
	data_x = mn._inputs
	data_y = mn._labels 

	train_x = data_x[0:40000]
	train_y = data_y[0:40000]

	test_x = data_x[50000:]
	test_y = data_y[50000:]

	valid_x = data_x[40000:50000]
	valid_y = data_y[40000:50000]

	rval = [(train_x, train_y, train_y), (valid_x, valid_y, valid_y), (test_x, test_y, test_y)]
	return rval


# for cifar10 of skdata
def load_skdata_cifar10():
	from skdata import cifar10
	ci = cifar10.dataset.CIFAR10()
	ci.fetch(True)
	meta = ci.build_meta()
	#ci.clean_up() # if you wish to erase the dataset from your comp. 

	data_x = numpy.reshape(ci._pixels,[60000,3072])/255.
	data_y = ci._labels

	# shuffle the data
	rand_perm = numpy.random.permutation(data_y.shape[0])
	data_x = data_x[rand_perm]
	data_y = data_y[rand_perm]

	train_x = data_x[0:40000]		
	train_y = data_y[0:40000]
	test_x = data_x[40000:50000]
	test_y = data_y[40000:50000]
	valid_x = data_x[50000:]
	valid_y = data_y[50000:]

	rval = [(train_x, train_y, train_y), (valid_x, valid_y, valid_y), (test_x, test_y, test_y)]
	return rval
"""
# Pascal 2007
def load_pascal_2007 (batch_size, n_train_images, n_test_images, n_valid_images, 
						   rand_perm, batch = 1, type_set = 'train', height = 256, width = 256, verbose = False ):
	from skdata import pascal
    pa = pascal.VOC2007()
    pa.fetch()
"""    

# caltech 101 of skdata 
def load_skdata_caltech101(batch_size, n_train_images, n_test_images, n_valid_images,
						   rand_perm, batch = 1, type_set = 'train', height = 256, width = 256, verbose = False ):
	# load_batches * batch_size is supplied into batch_size 
	from skdata import caltech
	from scipy.misc import imread
	cal = caltech.Caltech101()
	cal.fetch()
	meta = cal._get_meta()
	img,data_y = cal.img_classification_task()
	img = numpy.asarray(img.objs[0])
	img = img[rand_perm]								# Shuffle so that the ordering of classes is changed, but use the same shuffle so that loading works consistently.
	data_y = data_y[rand_perm] - 1
	data_x = numpy.asarray(numpy.zeros((batch_size,height*width *3)), dtype = 'float32' )
	
	if type_set == 'train':
		push = 0 + batch * batch_size 	
	elif type_set == 'test':
		push = n_train_images + batch * batch_size 
	elif type_set == 'valid':
		push = n_train_images + n_test_images + batch * batch_size
		
	if verbose is True:
		print "Processing image:  " + str(push)
	data_y = numpy.asarray(data_y[push : push + batch_size ] , dtype = 'int32' )	
	
	for i in range(batch_size):
					
		temp_img = imread(img[push + i])
		temp_img = temp_img.astype('float32')
		
		if temp_img.ndim != 3:
	    	# This is a temporary solution. I am allocating to all channels the grayscale values... 
			temp_img = temp_img.astype('float32')
			temp_img = cv2.resize(temp_img,(height,width))
			temp_img1 = numpy.zeros((height,width,3))
			temp_img1 [:,:,0] = temp_img
			temp_img1 [:,:,1] = temp_img
			temp_img1 [:,:,2] = temp_img
			data_x[i] = numpy.reshape(temp_img1,[1,height*width*3] )
		else:
			data_x[i] = numpy.reshape(cv2.resize(temp_img,(height,width)),[1,height*width*3] )
	
	return (data_x,data_y)

# caltech 256 of skdata 
def load_skdata_caltech256(batch_size, n_train_images, n_test_images, n_valid_images,
						   rand_perm, batch = 1, type_set = 'train', height = 256, width = 256, verbose = False ):

	# load_batches * batch_size is supplied into batch_size 
	import skdata
	from skdata import caltech
	from scipy.misc import imread
	cal = caltech.Caltech256()
	cal.fetch()
	meta = cal._get_meta()
	img,data_y = cal.img_classification_task()
	img = numpy.asarray(img.objs[0])
	img = img[rand_perm]								# Shuffle so that the ordering of classes is changed, but use the same shuffle so that loading works consistently.
	data_y = data_y[rand_perm] - 1
	data_x = numpy.asarray(numpy.zeros((batch_size,height*width *3)), dtype = 'float32' )
	
	if type_set == 'train':
		push = 0 + batch * batch_size 	
	elif type_set == 'test':
		push = n_train_images + batch * batch_size 
	elif type_set == 'valid':
		push = n_train_images + n_test_images + batch * batch_size
		
	if verbose is True:
		print "Processing image:  " + str(push)
	data_y = numpy.asarray(data_y[push : push + batch_size ] , dtype = 'int32' )	
	
	for i in range(batch_size):
					
		temp_img = imread(img[push + i])
		temp_img = temp_img.astype('float32')
		
		if temp_img.ndim != 3:
			# This is a temporary solution. I am allocating to all channels the grayscale values... 
			temp_img = temp_img.astype('float32')
			temp_img = cv2.resize(temp_img,(height,width))
			temp_img1 = numpy.zeros((height,width,3))
			temp_img1 [:,:,0] = temp_img
			temp_img1 [:,:,1] = temp_img
			temp_img1 [:,:,2] = temp_img
			data_x[i] = numpy.reshape(temp_img1,[1,height*width*3] )
		else:
			data_x[i] = numpy.reshape(cv2.resize(temp_img,(height,width)),[1,height*width*3] )
	
	return (data_x,data_y)

		
def rgb2gray(rgb):

    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
			
def preprocessing( data, height, width, channels, params): 
	
	normalize 		= params [ "normalize" ]
	GCN 			= params [ "GCN" ]      
	ZCA 			= params [ "ZCA" ]	 
	gray 			= params [ "gray" ]                     
	# Assume that the data is already resized on height and widht and all ... 
	if len(data.shape) == 2 and channels > 1: 	
		data = numpy.reshape ( data, (data.shape[0], height, width, channels)) 
	elif len(data.shape) == 2:
		data = numpy.reshape ( data, (data.shape[0], height, width)) 
	shp = data.shape
	
	out_shp_flag = False
	if gray is True and len(shp) == 4 and out_shp_flag is False: 	
		temp_data = numpy.zeros(shp)
		data = rgb2gray(data)
		out_shp = (shp[0], shp[1]*shp[2])
		
	if len(shp) == 2 and out_shp_flag is False:
		out_shp = shp
		
	if gray is False and len(shp) == 4 and out_shp_flag is False:
		out_shp = (shp[0], shp[1]*shp[2]*shp[3])
		
	if len(shp) == 3 and out_shp_flag is False:
		out_shp = (shp[0], shp[1]*shp[2])
	
	# from here on data is processed as a 2D matrix
	data = numpy.reshape(data,out_shp)
	if normalize is True or ZCA is True: 
		data = data / (data.max(axis = 0) + 1e-7)
		data = data - data.mean(axis = 0)		# do this normalization thing in batch mode.
	
	if ZCA is True:		

		sigma = numpy.dot(data.T,data) / data.shape[1]
		U, S, V = linalg.svd(sigma)		
		# data_rotated = numpy.dot(U.T, data) , full_matrices = True
		temp = numpy.dot(U, numpy.diag(1/numpy.sqrt(S + 1e-7)))
		temp = numpy.dot(temp, U.T)
		data = numpy.dot(data, temp)	
			
	# if GCN is True :
	return data
		
# Load initial data         
class setup_dataset (object):
	def __init__(self, data_params, preprocess_params, outs):
		thismodule = sys.modules[__name__]
		print "... setting up dataset "
		# Load Batch 1.
		self.data_struct         = data_params # This command makes it possible to save down
		self.dataset             = data_params [ "loc" ]
		self.data_type           = data_params [ "type" ]
		self.height              = data_params [ "height" ]
		self.width               = data_params [ "width" ]
		self.batch_size          = data_params [ "batch_size" ]    
		self.load_batches        = data_params [ "load_batches"  ] * self.batch_size
		if self.load_batches < self.batch_size and (self.dataset == "caltech101" or self.dataset == "caltech256"):
			AssertionError("load_batches is improper for this dataset " + self.dataset)
		self.batches2train       = data_params [ "batches2train" ]
		self.batches2test        = data_params [ "batches2test" ]
		self.batches2validate    = data_params [ "batches2validate" ] 
		self.channels            = data_params [ "channels" ]
		
		start_time = time.clock()
		
		temp_dir = './_datasets/_dataset_' + str(randint(11111,99999))			
		os.mkdir(temp_dir)
		os.mkdir(temp_dir + "/train" )
		os.mkdir(temp_dir + "/test"  )
		os.mkdir(temp_dir + "/valid" )
		print "... creating dataset " + temp_dir 			
		# load matlab files as self.dataset.
		if self.data_type == 'mat':
			
			print "... 		--> training data "
			for i in xrange(self.batches2train):		# for each batch_i file.... 
				data_x, data_y, data_y1 = load_data_mat(dataset = self.dataset, batch = i + 1, type_set = 'train' , n_classes = outs, height = self.height, width = self.width, channels = self.channels)
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )
				
				# compute number of minibatches for training, validation and testing
				self.n_train_batches = data_x.shape[0] / self.batch_size			
				f = open(temp_dir + "/train/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()
				 
			print "... 		--> testing data "	
			for i in xrange(self.batches2test):		# for each batch_i file.... 
				data_x, data_y, data_y1 = load_data_mat(dataset = self.dataset, batch = i + 1, type_set = 'test' , n_classes = outs, height = self.height, width = self.width, channels = self.channels)
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )
				
				# compute number of minibatches for training, validation and testing
				self.n_test_batches = data_x.shape[0] / self.batch_size			
				f = open(temp_dir + "/test/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()
				
			print "... 		--> validation data "	
			for i in xrange(self.batches2validate):		# for each batch_i file.... 
				data_x, data_y, data_y1 = load_data_mat(dataset = self.dataset, batch = i + 1, type_set = 'valid' , n_classes = outs, height = self.height, width = self.width, channels = self.channels)
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )
				
				# compute number of minibatches for training, validation and testing
				self.n_valid_batches = data_x.shape[0] / self.batch_size			
				f = open(temp_dir + "/valid/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()	
					
			self.multi_load = True 			
			new_data_params = {
					"type"               : 'base',                                   
					"loc"                : temp_dir,                                          
					"batch_size"         : self.batch_size,                                    
					"load_batches"       : -1,
					"batches2train"      : self.batches2train,                                      
					"batches2test"       : self.batches2test,                                     
					"batches2validate"   : self.batches2validate,                                       
					"height"             : self.height,                                      
					"width"              : self.width,                                       
					"channels"           : self.channels,
					"multi_load"		 : self.multi_load,
					"n_train_batches"	 : self.n_train_batches,
					"n_test_batches"	 : self.n_test_batches,
					"n_valid_batches"	 : self.n_valid_batches                                        
					}
				
		# load pkl data as is shown in theano tutorials
		elif self.data_type == 'pkl':   
		
			data = load_data_pkl(self.dataset)            						
			print "... setting up dataset "
			print "... 		--> training data "			
			data_x, data_y, data_y1 = data[0]
			data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )			
			n_train_images = data_x.shape[0]
			n_train_batches_all = n_train_images / self.batch_size 
			self.n_train_batches = data_x.shape[0] / self.batch_size			
			f = open(temp_dir + "/train/" + 'batch_' + str(0) + '.pkl', 'wb')
			obj = (data_x, data_y )
			cPickle.dump(obj, f, protocol=2)
			f.close()		
			
			print "... 		--> validation data "			
			data_x, data_y, data_y1 = data[1]
			data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )			
			n_valid_images = data_x.shape[0]
			n_valid_batches_all = n_valid_images / self.batch_size 
			self.n_valid_batches = data_x.shape[0] / self.batch_size			
			f = open(temp_dir + "/valid/" + 'batch_' + str(0) + '.pkl', 'wb')
			obj = (data_x, data_y )
			cPickle.dump(obj, f, protocol=2)
			f.close()				
			
			print "... 		--> testing data "			
			data_x, data_y, data_y1 = data[2]
			data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )			
			n_test_images = data_x.shape[0]
			n_test_batches_all = n_test_images / self.batch_size 
			self.n_test_batches = data_x.shape[0] / self.batch_size			
			f = open(temp_dir + "/test/" + 'batch_' + str(0) + '.pkl', 'wb')
			obj = (data_x, data_y )
			cPickle.dump(obj, f, protocol=2)
			f.close()	
							
			if ( (n_train_batches_all < self.batches2train) or 
				(n_test_batches_all < self.batches2test) or 
					(n_valid_batches_all < self.batches2validate) ):   
				# You can't have so many batches.
				print "...  !! dataset doens't have so many batches. "
				raise AssertionError()			
				
			self.multi_load = False
			
			new_data_params = {
					"type"               : 'base',                                   
					"loc"                : temp_dir,                                          
					"batch_size"         : self.batch_size,                                    
					"load_batches"       : -1,
					"batches2train"      : self.batches2train,                                      
					"batches2test"       : self.batches2test,                                     
					"batches2validate"   : self.batches2validate,                                       
					"height"             : self.height,                                      
					"width"              : self.width,                                       
					"channels"           : self.channels,
					"multi_load"		 : self.multi_load,
					"n_train_batches"	 : self.n_train_batches,
					"n_test_batches"	 : self.n_test_batches,
					"n_valid_batches"	 : self.n_valid_batches  					                                        
					}	
							
		# load skdata ( its a good library that has a lot of self.datasets)
		elif self.data_type == 'skdata':
		
			if (self.dataset == 'mnist' or 
				self.dataset == 'mnist_noise1' or 
				self.dataset == 'mnist_noise2' or
				self.dataset == 'mnist_noise3' or
				self.dataset == 'mnist_noise4' or
				self.dataset == 'mnist_noise5' or
				self.dataset == 'mnist_noise6' or
				self.dataset == 'mnist_bg_images' or
				self.dataset == 'mnist_bg_rand' or
				self.dataset == 'mnist_rotated' or
				self.dataset == 'mnist_rotated_bg') :
		
					print "... importing " + self.dataset + " from skdata"
					data = getattr(thismodule, 'load_skdata_' + self.dataset)()                
					print "... setting up dataset "
					print "... 		--> training data "			
					data_x, data_y, data_y1 = data[0]
					data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )			
					n_train_images = data_x.shape[0]
					n_train_batches_all = n_train_images / self.batch_size 
					self.n_train_batches = data_x.shape[0] / self.batch_size			
					f = open(temp_dir + "/train/" + 'batch_' + str(0) + '.pkl', 'wb')
						
					obj = (data_x, data_y )
					cPickle.dump(obj, f, protocol=2)
					f.close()							
					
					print "... 		--> validation data "			
					data_x, data_y, data_y1 = data[1]
					data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )			
					n_valid_images = data_x.shape[0]
					n_valid_batches_all = n_valid_images / self.batch_size 
					self.n_valid_batches = data_x.shape[0] / self.batch_size			
					f = open(temp_dir + "/valid/" + 'batch_' + str(0) + '.pkl', 'wb')
					obj = (data_x, data_y )					
					cPickle.dump(obj, f, protocol=2)
					f.close()				
					
					print "... 		--> testing data "			
					data_x, data_y, data_y1 = data[2]
					data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )			
					n_test_images = data_x.shape[0]
					n_test_batches_all = n_test_images / self.batch_size 
					self.n_test_batches = data_x.shape[0] / self.batch_size			
					f = open(temp_dir + "/test/" + 'batch_' + str(0) + '.pkl', 'wb')
					obj = (data_x, data_y )					
					cPickle.dump(obj, f, protocol=2)
					f.close()	
									
					if ( (n_train_batches_all < self.batches2train) or 
						(n_test_batches_all < self.batches2test) or 
							(n_valid_batches_all < self.batches2validate) ):   
						# You can't have so many batches.
						print "...  !! dataset doens't have so many batches. "
						raise AssertionError()			
					
					self.multi_load = False
					
					new_data_params = {
						"type"               : 'base',                                   
						"loc"                : temp_dir,                                          
						"batch_size"         : self.batch_size,                                    
						"load_batches"       : -1,
						"batches2train"      : self.batches2train,                                      
						"batches2test"       : self.batches2test,                                     
						"batches2validate"   : self.batches2validate,                                       
						"height"             : self.height,                                      
						"width"              : self.width,                                       
						"channels"           : self.channels,
						"multi_load"		 : self.multi_load, 
						"n_train_batches"	 : self.n_train_batches,
						"n_test_batches"	 : self.n_test_batches,
						"n_valid_batches"	 : self.n_valid_batches  						                                       
						}
					
			elif self.dataset == 'cifar10':
				print "... importing cifar 10 from skdata"
				data = load_skdata_cifar10()
				print "... setting up dataset "
				print "... 		--> training data "			
				data_x, data_y, data_y1 = data[0]
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )			
				n_train_images = data_x.shape[0]
				n_train_batches_all = n_train_images / self.batch_size 
				self.n_train_batches = data_x.shape[0] / self.batch_size			
				f = open(temp_dir + "/train/" + 'batch_' + str(0) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()		
				
				print "... 		--> validation data "			
				data_x, data_y, data_y1 = data[1]
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )			
				n_valid_images = data_x.shape[0]
				n_valid_batches_all = n_valid_images / self.batch_size 
				self.n_valid_batches = data_x.shape[0] / self.batch_size			
				f = open(temp_dir + "/valid/" + 'batch_' + str(0) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()				
				
				print "... 		--> testing data "			
				data_x, data_y, data_y1 = data[2]
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )			
				n_test_images = data_x.shape[0]
				n_test_batches_all = n_test_images / self.batch_size 
				self.n_test_batches = data_x.shape[0] / self.batch_size			
				f = open(temp_dir + "/test/" + 'batch_' + str(0) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()	
								
				if ( (n_train_batches_all < self.batches2train) or 
					(n_test_batches_all < self.batches2test) or 
						(n_valid_batches_all < self.batches2validate) ):   
				# You can't have so many batches.
					print "...  !! dataset doens't have so many batches. "
					raise AssertionError()			
				
				self.multi_load = False
								
				new_data_params = {
					"type"               : 'base',                                   
					"loc"                : temp_dir,                                          
					"batch_size"         : self.batch_size,                                    
					"load_batches"       : -1,
					"batches2train"      : self.batches2train,                                      
					"batches2test"       : self.batches2test,                                     
					"batches2validate"   : self.batches2validate,                                       
					"height"             : self.height,                                      
					"width"              : self.width,                                       
					"channels"           : self.channels,
					"multi_load"		 : self.multi_load,
					"n_train_batches"	 : self.n_train_batches,
					"n_test_batches"	 : self.n_test_batches,
					"n_valid_batches"	 : self.n_valid_batches  					                                        
					}
					
			elif self.dataset == 'caltech101':
				verbose = False
				print "... importing caltech 101 from skdata"
				
				# shuffle the data
				total_images_in_dataset = 9144 
				self.rand_perm = numpy.random.permutation(total_images_in_dataset)  
				# create a constant shuffle, so that data can be loaded in batchmode with the same random shuffle
				
				n_train_images = self.batch_size * self.batches2train
				n_test_images = self.batch_size * self.batches2test
				n_valid_images = self.batch_size * self.batches2validate
				
				assert n_valid_images + n_train_images + n_test_images == total_images_in_dataset  
				
				print ".... setting up dataset"				
				print "... 		--> training data "						
				looper = n_train_images / self.load_batches						
				for i in xrange(looper):		# for each batch_i file.... 
					data_x, data_y  = load_skdata_caltech101(
													n_train_images = n_train_images,
													n_test_images = n_test_images,
													n_valid_images = n_valid_images,
													batch_size = self.load_batches, 
													rand_perm = self.rand_perm, 
													batch = i , 
													type_set = 'train' ,
													height = self.height,
													width = self.width,
													verbose = verbose )  													
					data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )				
					# compute number of minibatches for training, validation and testing
					self.n_train_batches = data_x.shape[0] / self.batch_size			
					f = open(temp_dir + "/train/" + 'batch_' + str(i) + '.pkl', 'wb')
					obj = (data_x, data_y )
					cPickle.dump(obj, f, protocol=2)
					f.close()
			
					
				self.n_train_batches = data_x.shape[0] / self.batch_size											
				print "... 		--> testing data "				
				looper = n_test_images / self.load_batches		
				for i in xrange(looper):		# for each batch_i file.... 
					data_x, data_y  = load_skdata_caltech101(
													n_train_images = n_train_images,
													n_test_images = n_test_images,
													n_valid_images = n_valid_images,
													batch_size = self.load_batches, 
													rand_perm = self.rand_perm, 
													batch = i , 
													type_set = 'test' ,
													height = self.height,
													width = self.width,
													verbose = verbose )  
					data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )				
					# compute number of minibatches for training, validation and testing
					self.n_train_batches = data_x.shape[0] / self.batch_size			
					f = open(temp_dir + "/test/" + 'batch_' + str(i) + '.pkl', 'wb')
					obj = (data_x, data_y )
					cPickle.dump(obj, f, protocol=2)
					f.close()							  
				self.n_test_batches = data_x.shape[0] / self.batch_size																
				
				print "... 		--> validation data "	
				looper = n_valid_images / self.load_batches									
				for i in xrange(looper):		# for each batch_i file.... 
					data_x, data_y  = load_skdata_caltech101(
													n_train_images = n_train_images,
													n_test_images = n_test_images,
													n_valid_images = n_valid_images,
													batch_size = self.load_batches, 
													rand_perm = self.rand_perm, 
													batch = i , 
													type_set = 'valid' ,
													height = self.height,
													width = self.width,
													verbose = verbose  )  
					data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )				
					# compute number of minibatches for training, validation and testing
					self.n_train_batches = data_x.shape[0] / self.batch_size			
					f = open(temp_dir + "/valid/" + 'batch_' + str(i) + '.pkl', 'wb')
					obj = (data_x, data_y )
					cPickle.dump(obj, f, protocol=2)
					f.close()							  
				self.n_valid_batches = data_x.shape[0] / self.batch_size																	
				self.multi_load = True
				
				new_data_params = {
					"type"               : 'base',                                   
					"loc"                : temp_dir,                                          
					"batch_size"         : self.batch_size,                                    
					"load_batches"       : self.load_batches / self.batch_size,
					"batches2train"      : self.batches2train / (self.load_batches / self.batch_size) ,                                      
					"batches2test"       : self.batches2test / (self.load_batches / self.batch_size),                                     
					"batches2validate"   : self.batches2validate / (self.load_batches / self.batch_size),                                       
					"height"             : self.height,                                      
					"width"              : self.width,                                       
					"channels"           : self.channels,
					"multi_load"		 : self.multi_load,
					"n_train_batches"	 : self.n_train_batches,
					"n_test_batches"	 : self.n_test_batches,
					"n_valid_batches"	 : self.n_valid_batches  					                                        
					}
					
			elif self.dataset == 'caltech256':
				print "... importing caltech 256 from skdata"
				
					# shuffle the data
				total_images_in_dataset = 30607 
				self.rand_perm = numpy.random.permutation(total_images_in_dataset)  
				# create a constant shuffle, so that data can be loaded in batchmode with the same random shuffle
				
				n_train_images = self.batch_size * self.batches2train
				n_test_images = self.batch_size * self.batches2test
				n_valid_images = self.batch_size * self.batches2validate
				
				
				assert n_valid_images + n_train_images + n_test_images == total_images_in_dataset  
				
				print ".... setting up dataset"				
				print "... 		--> training data "						
				looper = n_train_images / self.load_batches						
				for i in xrange(looper):		# for each batch_i file.... 
					data_x, data_y  = load_skdata_caltech101(
													n_train_images = n_train_images,
													n_test_images = n_test_images,
													n_valid_images = n_valid_images,
													batch_size = self.load_batches, 
													rand_perm = self.rand_perm, 
													batch = i , 
													type_set = 'train' ,
													height = self.height,
													width = self.width,
													verbose = verbose )  													
					data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )				
					# compute number of minibatches for training, validation and testing
					self.n_train_batches = data_x.shape[0] / self.batch_size			
					f = open(temp_dir + "/train/" + 'batch_' + str(i) + '.pkl', 'wb')
					obj = (data_x, data_y )
					cPickle.dump(obj, f, protocol=2)
					f.close()
				
					
				self.n_train_batches = data_x.shape[0] / self.batch_size											
				print "... 		--> testing data "				
				looper = n_test_images / self.load_batches		
				for i in xrange(looper):		# for each batch_i file.... 
					data_x, data_y  = load_skdata_caltech101(
													n_train_images = n_train_images,
													n_test_images = n_test_images,
													n_valid_images = n_valid_images,
													batch_size = self.load_batches, 
													rand_perm = self.rand_perm, 
													batch = i , 
													type_set = 'test' ,
													height = self.height,
													width = self.width,
													verbose = verbose )  
					data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )				
					# compute number of minibatches for training, validation and testing
					self.n_train_batches = data_x.shape[0] / self.batch_size			
					f = open(temp_dir + "/test/" + 'batch_' + str(i) + '.pkl', 'wb')
					obj = (data_x, data_y )
					cPickle.dump(obj, f, protocol=2)
					f.close()							  
				self.n_test_batches = data_x.shape[0] / self.batch_size																
				
				print "... 		--> validation data "	
				looper = n_valid_images / self.load_batches									
				for i in xrange(looper):		# for each batch_i file.... 
					data_x, data_y  = load_skdata_caltech101(
													n_train_images = n_train_images,
													n_test_images = n_test_images,
													n_valid_images = n_valid_images,
													batch_size = self.load_batches, 
													rand_perm = self.rand_perm, 
													batch = i , 
													type_set = 'valid' ,
													height = self.height,
													width = self.width,
													verbose = verbose  )  
					data_x = preprocessing ( data_x, self.height, self.width, self.channels, preprocess_params )				
					# compute number of minibatches for training, validation and testing
					self.n_train_batches = data_x.shape[0] / self.batch_size			
					f = open(temp_dir + "/valid/" + 'batch_' + str(i) + '.pkl', 'wb')
					obj = (data_x, data_y )
					cPickle.dump(obj, f, protocol=2)
					f.close()							  
				self.n_valid_batches = data_x.shape[0] / self.batch_size																	
				self.multi_load = True
				
				
				new_data_params = {
					"type"               : 'base',                                   
					"loc"                : temp_dir,                                          
					"batch_size"         : self.batch_size,                                    
					"load_batches"       : self.load_batches / self.batch_size,
					"batches2train"      : self.batches2train / (self.load_batches / self.batch_size),                                      
					"batches2test"       : self.batches2test / (self.load_batches / self.batch_size),                                     
					"batches2validate"   : self.batches2validate / (self.load_batches / self.batch_size),                                       
					"height"             : self.height,                                      
					"width"              : self.width,                                       
					"channels"           : self.channels,
					"multi_load"		 : self.multi_load,
					"n_train_batches"	 : self.n_train_batches,
					"n_test_batches"	 : self.n_test_batches,
					"n_valid_batches"	 : self.n_valid_batches  					                                        
					}
									
		if preprocess_params["gray"] is True:
			new_data_params ["channels"] = 1 
			self.channels = 1 
		
		assert ( self.height * self.width * self.channels == numpy.prod(data_x.shape[1:]) )
		f = open(temp_dir +  '/data_params.pkl', 'wb')
		cPickle.dump(new_data_params, f, protocol=2)
		f.close()				  	
		end_time = time.clock()
		print "...         time taken is " +str(end_time - start_time) + " seconds"
		
