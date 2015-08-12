#!/usr/bin/python


import os
import sys
import numpy
import pdb
import scipy.io
import gzip
import cPickle
import theano
import cv2





##################################
## Data Loading Functions        ##
##################################

# From the Theano Tutorials
def shared_dataset(data_xy, borrow=True, svm_flag = True):

	data_x, data_y = data_xy

	shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
	shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX),  borrow=borrow)
	                                 
	if svm_flag is True:
		# one-hot encoded labels as {-1, 1}
		n_classes = len(numpy.unique(data_y))  # dangerous?
		y1 = -1 * numpy.ones((data_y.shape[0], n_classes))
		y1[numpy.arange(data_y.shape[0]), data_y] = 1
		shared_y1 = theano.shared(numpy.asarray(y1,dtype=theano.config.floatX), borrow=borrow)

		return shared_x, theano.tensor.cast(shared_y, 'int32'), shared_y1
	else:
		return shared_x, theano.tensor.cast(shared_y, 'int32') 

# from theano tutorials for loading pkl files like what is used in theano tutorials.
def load_data_pkl(dataset):
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

	# Load the dataset	
	f = gzip.open(dataset, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()
	#train_set, valid_set, test_set format: tuple(input, target)
	#input is an numpy.ndarray of 2 dimensions (a matrix)
	#witch row's correspond to an example. target is a
	#numpy.ndarray of 1 dimensions (vector)) that have the same length as
	#the number of rows in the input. It should give the target
	#target to the example with the same index in the input.

	test_set_x, test_set_y, test_set_y1 = shared_dataset(test_set)
	valid_set_x, valid_set_y, valid_set_y1 = shared_dataset(valid_set)
	train_set_x, train_set_y, train_set_y1 = shared_dataset(train_set)

	rval = [(train_set_x, train_set_y, train_set_y1), (valid_set_x, valid_set_y, valid_set_y1), (test_set_x, test_set_y, test_set_y1)]
	return rval

# for loading matlab based data.
def load_data_mat(dataset = '../dataset/waldo/' ,batch = 1, type_set = 'train', load_z = False):

	# Use this code if the data was created in matlab in the right format and needed to be loaded
	# print "... Loading " + type_set + " batch number " + str(batch)
	#- ----------  Load Dataet ------- -#
	mat = scipy.io.loadmat(dataset  +  type_set + '/batch_' + str(batch) + '.mat')
	data_x = numpy.asarray(mat['x'], dtype = theano.config.floatX)
	data_y = numpy.array(numpy.squeeze(mat['y']), dtype = 'int32')
	if load_z is True:
	    data_z = numpy.array(numpy.squeeze(mat['z']), dtype=theano.config.floatX )
	n_classes = len(numpy.unique(data_y))  # dangerous?
	y1 = -1 * numpy.ones((data_y.shape[0], n_classes))
	y1[numpy.arange(data_y.shape[0]), data_y] = 1
	if load_z is False:
	    return (data_x,data_y,y1.astype( dtype = theano.config.floatX ))
	else:
	    return (data_x,data_y,y1.astype( dtype = theano.config.floatX ),data_z)

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
	#mn.clean_up() # if you wish to erase the dataset from your comp. 

	test_set_x, test_set_y, test_set_y1 = shared_dataset((numpy.reshape(test_x/255.,[10000,784]),test_y))
	valid_set_x, valid_set_y, valid_set_y1 = shared_dataset((numpy.reshape(valid_x/255.,[10000,784]),valid_y))
	train_set_x, train_set_y, train_set_y1 = shared_dataset((numpy.reshape(train_x/255.,[50000,784]),train_y))

	rval = [(train_set_x, train_set_y, train_set_y1), (valid_set_x, valid_set_y, valid_set_y1), (test_set_x, test_set_y, test_set_y1)]
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

	test_set_x, test_set_y, test_set_y1 = shared_dataset((numpy.reshape(test_x,[2000,784]),test_y))
	valid_set_x, valid_set_y, valid_set_y1 = shared_dataset((numpy.reshape(valid_x,[2000,784]),valid_y))
	train_set_x, train_set_y, train_set_y1 = shared_dataset((numpy.reshape(train_x,[10000,784]),train_y))

	rval = [(train_set_x, train_set_y, train_set_y1), (valid_set_x, valid_set_y, valid_set_y1), (test_set_x, test_set_y, test_set_y1)]
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

	test_set_x, test_set_y, test_set_y1 = shared_dataset((numpy.reshape(test_x,[2000,784]),test_y))
	valid_set_x, valid_set_y, valid_set_y1 = shared_dataset((numpy.reshape(valid_x,[2000,784]),valid_y))
	train_set_x, train_set_y, train_set_y1 = shared_dataset((numpy.reshape(train_x,[10000,784]),train_y))

	rval = [(train_set_x, train_set_y, train_set_y1), (valid_set_x, valid_set_y, valid_set_y1), (test_set_x, test_set_y, test_set_y1)]
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

	test_set_x, test_set_y, test_set_y1 = shared_dataset((numpy.reshape(test_x,[2000,784]),test_y))
	valid_set_x, valid_set_y, valid_set_y1 = shared_dataset((numpy.reshape(valid_x,[2000,784]),valid_y))
	train_set_x, train_set_y, train_set_y1 = shared_dataset((numpy.reshape(train_x,[10000,784]),train_y))

	rval = [(train_set_x, train_set_y, train_set_y1), (valid_set_x, valid_set_y, valid_set_y1), (test_set_x, test_set_y, test_set_y1)]
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

	test_set_x, test_set_y, test_set_y1 = shared_dataset((numpy.reshape(test_x,[2000,784]),test_y))
	valid_set_x, valid_set_y, valid_set_y1 = shared_dataset((numpy.reshape(valid_x,[2000,784]),valid_y))
	train_set_x, train_set_y, train_set_y1 = shared_dataset((numpy.reshape(train_x,[10000,784]),train_y))

	rval = [(train_set_x, train_set_y, train_set_y1), (valid_set_x, valid_set_y, valid_set_y1), (test_set_x, test_set_y, test_set_y1)]
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

	test_set_x, test_set_y, test_set_y1 = shared_dataset((numpy.reshape(test_x,[2000,784]),test_y))
	valid_set_x, valid_set_y, valid_set_y1 = shared_dataset((numpy.reshape(valid_x,[2000,784]),valid_y))
	train_set_x, train_set_y, train_set_y1 = shared_dataset((numpy.reshape(train_x,[10000,784]),train_y))

	rval = [(train_set_x, train_set_y, train_set_y1), (valid_set_x, valid_set_y, valid_set_y1), (test_set_x, test_set_y, test_set_y1)]
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

	test_set_x, test_set_y, test_set_y1 = shared_dataset((numpy.reshape(test_x,[2000,784]),test_y))
	valid_set_x, valid_set_y, valid_set_y1 = shared_dataset((numpy.reshape(valid_x,[2000,784]),valid_y))
	train_set_x, train_set_y, train_set_y1 = shared_dataset((numpy.reshape(train_x,[10000,784]),train_y))

	rval = [(train_set_x, train_set_y, train_set_y1), (valid_set_x, valid_set_y, valid_set_y1), (test_set_x, test_set_y, test_set_y1)]
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

	test_set_x, test_set_y, test_set_y1 = shared_dataset((numpy.reshape(test_x,[12000,784]),test_y))
	valid_set_x, valid_set_y, valid_set_y1 = shared_dataset((numpy.reshape(valid_x,[10000,784]),valid_y))
	train_set_x, train_set_y, train_set_y1 = shared_dataset((numpy.reshape(train_x,[40000,784]),train_y))

	rval = [(train_set_x, train_set_y, train_set_y1), (valid_set_x, valid_set_y, valid_set_y1), (test_set_x, test_set_y, test_set_y1)]
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

	test_set_x, test_set_y, test_set_y1 = shared_dataset((numpy.reshape(test_x,[12000,784]),test_y))
	valid_set_x, valid_set_y, valid_set_y1 = shared_dataset((numpy.reshape(valid_x,[10000,784]),valid_y))
	train_set_x, train_set_y, train_set_y1 = shared_dataset((numpy.reshape(train_x,[40000,784]),train_y))

	rval = [(train_set_x, train_set_y, train_set_y1), (valid_set_x, valid_set_y, valid_set_y1), (test_set_x, test_set_y, test_set_y1)]
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

	test_set_x, test_set_y, test_set_y1 = shared_dataset((numpy.reshape(test_x,[12000,784]),test_y))
	valid_set_x, valid_set_y, valid_set_y1 = shared_dataset((numpy.reshape(valid_x,[10000,784]),valid_y))
	train_set_x, train_set_y, train_set_y1 = shared_dataset((numpy.reshape(train_x,[40000,784]),train_y))

	rval = [(train_set_x, train_set_y, train_set_y1), (valid_set_x, valid_set_y, valid_set_y1), (test_set_x, test_set_y, test_set_y1)]
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

	test_set_x, test_set_y, test_set_y1 = shared_dataset((numpy.reshape(test_x,[12000,784]),test_y))
	valid_set_x, valid_set_y, valid_set_y1 = shared_dataset((numpy.reshape(valid_x,[10000,784]),valid_y))
	train_set_x, train_set_y, train_set_y1 = shared_dataset((numpy.reshape(train_x,[40000,784]),train_y))

	rval = [(train_set_x, train_set_y, train_set_y1), (valid_set_x, valid_set_y, valid_set_y1), (test_set_x, test_set_y, test_set_y1)]
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

	test_set_x, test_set_y, test_set_y1 = shared_dataset((test_x,test_y))
	valid_set_x, valid_set_y, valid_set_y1 = shared_dataset((valid_x,valid_y))
	train_set_x, train_set_y, train_set_y1 = shared_dataset((train_x,train_y))

	rval = [(train_set_x, train_set_y, train_set_y1), (valid_set_x, valid_set_y, valid_set_y1), (test_set_x, test_set_y, test_set_y1)]
	return rval

# caltech 101 of skdata 
def load_skdata_caltech101(batch_size, rand_perm, batch = 1, type_set = 'train', height = 256, width = 256 ):
	import skdata
	from skdata import caltech
	from scipy.misc import imread
	cal = caltech.Caltech101()
	cal.fetch()
	meta = cal._get_meta()
	img,data_y = cal.img_classification_task()
	img = numpy.asarray(img.objs[0])
	img = img[rand_perm]								# Shuffle so that the ordering of classes is changed, but use the same shuffle so that loading works consistently.
	data_y = data_y[rand_perm]
	data_x = numpy.asarray(numpy.zeros((3*batch_size,height*width*3)), dtype = theano.config.floatX )
	data_y = numpy.asarray(data_y[0:3*batch_size] , dtype = 'int32' )
	for i in range(batch_size):
		temp_img = imread(img[3*batch_size*batch + i])
		temp_img = cv2.normalize(temp_img.astype(theano.config.floatX), None, 0.0, 1.0, cv2.NORM_MINMAX)
		if temp_img.ndim != 3:
	    	# This is a temporary solution. I am allocating to all channels the grayscale values... 
			temp_img = cv2.normalize(temp_img.astype(theano.config.floatX), None, 0.0, 1.0, cv2.NORM_MINMAX)
			temp_img = cv2.resize(temp_img,(height,width))
			temp_img1 = numpy.zeros((height,width,3))
			temp_img1 [:,:,0] = temp_img
			temp_img1 [:,:,1] = temp_img
			temp_img1 [:,:,2] = temp_img
			data_x[i] = numpy.reshape(temp_img1,[1,height*width*3] )
		else:
			data_x[i] = numpy.reshape(cv2.resize(temp_img,(height,width)),[1,height*width*3] )

	train_x = data_x[0:batch_size]		
	train_y = data_y[0:batch_size]
	test_x = data_x[batch_size:2*batch_size]
	test_y = data_y[batch_size:2*batch_size]
	valid_x = data_x[2*batch_size:]
	valid_y = data_y[2*batch_size:]

	
	if type_set == 'train':
		return (train_x,train_y)
	elif type_set == 'test':
		return (test_x,test_y)
	else:
		return (valid_x,valid_y)

# caltech 256 of skdata 
def load_skdata_caltech256(batch_size, rand_perm, batch = 1, type_set = 'train', height = 256, width = 256 ):
	import skdata
	from skdata import caltech
	from scipy.misc import imread
	cal = caltech.Caltech256()
	cal.fetch()
	meta = cal._get_meta()
	img,data_y = cal.img_classification_task()
	img = numpy.asarray(img.objs[0])
	img = img[rand_perm]		
	data_y = data_y[rand_perm]
	data_x = numpy.asarray(numpy.zeros((3*batch_size, height*width*3)), dtype = theano.config.floatX )
	data_y = numpy.asarray(data_y[0:3*batch_size] , dtype = 'int32' )
	for i in range(batch_size):
		temp_img = imread(img[3*batch_size*batch + i])
		temp_img = cv2.normalize(temp_img.astype(theano.config.floatX), None, 0.0, 1.0, cv2.NORM_MINMAX)
		if temp_img.ndim != 3:
	    	# This is a temporary solution. I am allocating to all channels the grayscale values... 
			temp_img = cv2.normalize(temp_img.astype(theano.config.floatX), None, 0.0, 1.0, cv2.NORM_MINMAX)
			temp_img = cv2.resize(temp_img,(height,width))
			temp_img1 = numpy.zeros((height,width,3))
			temp_img1 [:,:,0] = temp_img
			temp_img1 [:,:,1] = temp_img
			temp_img1 [:,:,2] = temp_img
			data_x[i] = numpy.reshape(temp_img1,[1,height*width*3] )
		else:
			data_x[i] = numpy.reshape(cv2.resize(temp_img,(height,width)),[1,height*width*3] )

	train_x = data_x[0:batch_size]		
	train_y = data_y[0:batch_size]
	test_x = data_x[batch_size:2*batch_size]
	test_y = data_y[batch_size:2*batch_size]
	valid_x = data_x[2*batch_size:]
	valid_y = data_y[2*batch_size:]

	
	if type_set == 'train':
		return (train_x,train_y)
	elif type_set == 'test':
		return (test_x,test_y)
	else:
		return (valid_x,valid_y)

