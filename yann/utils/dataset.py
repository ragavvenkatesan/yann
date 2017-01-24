"""
Todo:

	* None of the PASCAL dataset retrievers from ``skdata`` is working. This need to be coded
      in.
	* Need a method to create dataset from a directory of images. - prepare for imagenet and 
	  coco.
	* See if support can be made for fuel.
"""

import os
import sys
import time

import numpy
import scipy.io
import cPickle
import imp

try:
	imp.find_module('scipy')
	scipy_installed = True
except ImportError:
	scipy_installed = False

if scipy_installed is True:
	from scipy import linalg

from random import randint

try:
	imp.find_module('skdata')
	skdata_installed = True
except ImportError:
	skdata_installed = False

if skdata_installed is True:
	import skdata

import theano
import theano.tensor as T
from theano import shared 

thismodule = sys.modules[__name__]

# from sklearn.base import TransformerMixin, BaseEstimator

def load_data_mat(classes, 
                  height,
				  width,
			      channels, location = '../dataset/waldo/',
				  batch = 1, 
				  type_set = 'train', 
				  load_z = False):
	"""
	Use this code if the data was created in matlab in the right format and needed to be 
	loaded. The way to create is to have variables ``x, y, z`` with ``z`` being an optional 
	data to load. ``x`` is assumed to be the data in matrix ``double`` format with rows being 
	each image in vectorized fashion and ``y`` is assumed to be lables in ``int`` or 
	``double``.
	
	The files are stored in the following format: ``loc/type/batch_0.mat``. This code needs 
	scipy to run. 
	
	Args: 
			classes: Number of unique classes in the dataset. You can use ``unique(y)`` in
					 MATLAB to get this value.
			height: The height of each image in the dataset.
			width: The width of each image in the dataset.
			channels: ``3`` if RGB, ``1`` if grayscale and so on.
			location: Location of the dataset.
			batch: if multi batch, then how many batches of data is present if not use ``1``

	Returns:
		float32 tuple: Tuple `(data_x, data_y)` if requested, also `(data_x,data_y,data_z)`

	Todo:
		Need to add preprocessing in this.			
		
	"""
	print "... Loading " + type_set + " batch number " + str(batch)
	if scipy_installed is False:
		raise Exception("Scipy needed for cooking this dataset. Please install")
	mat = scipy.io.loadmat(location  +  type_set + '/batch_' + str(batch) + '.mat')
	data_x = numpy.asarray(mat['x'], dtype = 'float32')
	if data_x.max() > 1:
		data_x = data_x/data_x.max() # this is not normalize. This just scales.

	for i in xrange(data_x.shape[0]):
		temp = data_x[i,]
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
	"""
	Function that downloads the dataset from skdata and returns the dataset in full

	Returns:
		list: ``[(train_x, train_y, train_y),(valid_x, valid_y, valid_y), (test_x, test_y, test_y)]``
	"""
	if skdata_installed is False:
		raise Exception("This dataset cooks from skdata. Please install skdata")
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
	"""
	Function that downloads the dataset from skdata and returns the dataset in full

	Returns:
		list: ``[(train_x, train_y, train_y),(valid_x, valid_y, valid_y), (test_x, test_y, test_y)]``
	"""    	
	if skdata_installed is False:
		raise Exception("This dataset cooks from skdata. Please install skdata")	
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
	"""
	Function that downloads the dataset from skdata and returns the dataset in full

	Returns:
		list: ``[(train_x, train_y, train_y),(valid_x, valid_y, valid_y), (test_x, test_y, test_y)]``
	"""    	
	if skdata_installed is False:
		raise Exception("This dataset cooks from skdata. Please install skdata")	
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
	"""
	Function that downloads the dataset from skdata and returns the dataset 
	in full

	Returns:
		list: ``[(train_x, train_y, train_y),(valid_x, valid_y, valid_y), (test_x, test_y, test_y)]``
	"""    	
	if skdata_installed is False:
		raise Exception("This dataset cooks from skdata. Please install skdata")	
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
	"""
	Function that downloads the dataset from skdata and returns the dataset 
	in full

	Returns:
		list: ``[(train_x, train_y, train_y),(valid_x, valid_y, valid_y), (test_x, test_y, test_y)]``
	"""    	
	if skdata_installed is False:
		raise Exception("This dataset cooks from skdata. Please install skdata")	
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
	"""
	Function that downloads the dataset from skdata and returns the dataset in full

	Returns:
		list: ``[(train_x, train_y, train_y),(valid_x, valid_y, valid_y), (test_x, test_y, test_y)]``
	"""    	
	if skdata_installed is False:
		raise Exception("This dataset cooks from skdata. Please install skdata")	
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
	"""
	Function that downloads the dataset from skdata and returns the dataset in full

	Returns:
		list: ``[(train_x, train_y, train_y),(valid_x, valid_y, valid_y),(test_x, test_y, test_y)]``
	"""    	
	if skdata_installed is False:
		raise Exception("This dataset cooks from skdata. Please install skdata")	
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
	"""
	Function that downloads the dataset from skdata and returns the dataset in full

	Returns:
		list: ``[(train_x, train_y, train_y),(valid_x, valid_y, valid_y), (test_x, test_y, test_y)]``
	"""    	
	if skdata_installed is False:
		raise Exception("This dataset cooks from skdata. Please install skdata")	
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
	"""
	Function that downloads the dataset from skdata and returns the dataset in full

	Returns:
		list: ``[(train_x, train_y, train_y),(valid_x, valid_y, valid_y), (test_x, test_y, test_y)]``
	"""    	
	if skdata_installed is False:
		raise Exception("This dataset cooks from skdata. Please install skdata")	
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
	"""
	Function that downloads the dataset from skdata and returns the dataset in full

	Returns:
		list: ``[(train_x, train_y, train_y),(valid_x, valid_y, valid_y), (test_x, test_y, test_y)]``
	"""    	
	if skdata_installed is False:
		raise Exception("This dataset cooks from skdata. Please install skdata")	
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
	"""
	Function that downloads the dataset from skdata and returns the dataset in full

	Returns:
		list: ``[(train_x, train_y, train_y),(valid_x, valid_y, valid_y), (test_x, test_y, test_y)]``
	"""    	
	if skdata_installed is False:
		raise Exception("This dataset cooks from skdata. Please install skdata")	
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
	"""
	Function that downloads the dataset from skdata and returns the dataset in full

	Returns:
		list: ``[(train_x, train_y, train_y),(valid_x, valid_y, valid_y), (test_x, test_y, test_y)]``
	"""    	
	if skdata_installed is False:
		raise Exception("This dataset cooks from skdata. Please install skdata")	
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


# caltech 101 of skdata 
def load_skdata_caltech101(mini_batch_size, 
						   n_train_images, 
						   n_test_images, 
						   n_valid_images,
						   rand_perm, batch = 1, 
						   type_set = 'train', 
						   height = 256, 
						   width = 256, 
						   verbose = False ):
	"""
	Function that downloads the dataset from skdata and returns the dataset in part

	Args:
		mini_batch_size: What is the size of the batch.
		n_train_images: number of training images.
		n_test_images: number of testing images.
		n_valid_images: number of validating images.
		rand_perm: Create a random permutation list of images to be sampled to batches.
		type_set: What dataset you need, test, train or valid.
		height: Height of the image
		width: Width of the image.
		verbose: similar to dataset.

	Todo:
		This is not a finished function.

	Returns:
		list: ``[(train_x, train_y, train_y),(valid_x, valid_y, valid_y), (test_x, test_y, test_y)]``
	"""    
	# load_batches * mini_batch_size is supplied into mini_batch_size 
	if skdata_installed is False:
		raise Exception("This dataset cooks from skdata. Please install skdata")	
	from skdata import caltech
	if scipy_installed is False:
		raise Exception("Scipy needed for cooking this dataset. Please install")
	from scipy.misc import imread
	cal = caltech.Caltech101()
	cal.fetch()
	meta = cal._get_meta()
	img,data_y = cal.img_classification_task()
	img = numpy.asarray(img.objs[0])
	# Shuffle so that the ordering of classes is changed, 
	# but use the same shuffle so that loading works consistently.	
	img = img[rand_perm] 
	data_y = data_y[rand_perm] 
	data_x = numpy.asarray(numpy.zeros((mini_batch_size,height*width *3)), dtype = 'float32' )

	if type_set == 'train':
		push = 0 + batch * mini_batch_size 	
	elif type_set == 'test':
		push = n_train_images + batch * mini_batch_size 
	elif type_set == 'valid':
		push = n_train_images + n_test_images + batch * mini_batch_size
		
	if verbose is True:
		print "Processing image:  " + str(push)
	data_y = numpy.asarray(data_y[push : push + mini_batch_size ] , dtype = 'int32' )	

	for i in range(mini_batch_size):
					
		temp_img = imread(img[push + i])
		temp_img = temp_img.astype('float32')
		
		if temp_img.ndim != 3:
			# This is a temporary solution. 
			# I am allocating to all channels the grayscale values... 
			temp_img = temp_img.astype('float32')
			temp_img = cv2.resize(temp_img,(height,width))
			temp_img1 = numpy.zeros((height,width,3))
			temp_img1 [:,:,0] = temp_img
			temp_img1 [:,:,1] = temp_img
			temp_img1 [:,:,2] = temp_img
			data_x[i] = numpy.reshape(temp_img1,[1,height*width*3] )
		else:
			data_x[i] = numpy.reshape(cv2.resize(temp_img,(height,width)), [1,height*width*3])

	return (data_x,data_y)

# caltech 256 of skdata 
def load_skdata_caltech256(mini_batch_size, 
				           n_train_images, 
						   n_test_images, 
						   n_valid_images,
						   rand_perm, 
						   batch = 1, 
						   type_set = 'train', 
						   height = 256, 
						   width = 256, 
						   verbose = False ):
	"""
	Function that downloads the dataset from skdata and returns the dataset in part

	Args:
		mini_batch_size: What is the size of the batch.
		n_train_images: number of training images.
		n_test_images: number of testing images.
		n_valid_images: number of validating images.
		rand_perm: Create a random permutation list of images to be sampled to batches.
		type_set: What dataset you need, test, train or valid.
		height: Height of the image
		width: Width of the image.
		verbose: similar to dataset.

	Todo:
		This is not a finished function.

	Returns:
		list: ``[(train_x, train_y, train_y),(valid_x, valid_y, valid_y), (test_x, test_y, test_y)]``
	"""    
	# load_batches * mini_batch_size is supplied into mini_batch_size 
	if skdata_installed is False:
		raise Exception("This dataset cooks from skdata. Please install skdata")	
	import skdata	
	from skdata import caltech
	if scipy_installed is False:
		raise Exception("Scipy needed for cooking this dataset. Please install")	
	from scipy.misc import imread
	cal = caltech.Caltech256()
	cal.fetch()
	meta = cal._get_meta()
	img,data_y = cal.img_classification_task()
	img = numpy.asarray(img.objs[0])
	# Shuffle so that the ordering of classes is changed, 
	# but use the same shuffle so that loading works consistently.	
	img = img[rand_perm]								
	data_y = data_y[rand_perm] 
	data_x = numpy.asarray(numpy.zeros((mini_batch_size,height*width *3)), dtype = 'float32' )
	
	if type_set == 'train':
		push = 0 + batch * mini_batch_size 	
	elif type_set == 'test':
		push = n_train_images + batch * mini_batch_size 
	elif type_set == 'valid':
		push = n_train_images + n_test_images + batch * mini_batch_size
		
	if verbose is True:
		print "Processing image:  " + str(push)
	data_y = numpy.asarray(data_y[push : push + mini_batch_size ] , dtype = 'int32' )	
	
	for i in range(mini_batch_size):
					
		temp_img = imread(img[push + i])
		temp_img = temp_img.astype('float32')
		
		if temp_img.ndim != 3:
			# This is a temporary solution. 
			# I am allocating to all channels the grayscale values... 
			temp_img = temp_img.astype('float32')
			temp_img = cv2.resize(temp_img,(height,width))
			temp_img1 = numpy.zeros((height,width,3))
			temp_img1 [:,:,0] = temp_img
			temp_img1 [:,:,1] = temp_img
			temp_img1 [:,:,2] = temp_img
			data_x[i] = numpy.reshape(temp_img1,[1,height*width*3] )
		else:
			data_x[i] = numpy.reshape(cv2.resize(temp_img,(height,width)), [1,height*width*3])
	
	return (data_x,data_y)

		
def rgb2gray(rgb):
	"""
	Function that takes as input one rgb image array and returns a grayscale image. It applies 
	the following transform:

	.. math::
	
		I_{gray} = 0.2989I_r + 0.5870I_g + 0.1140I_b

	Args:
		rgb: ``numpy ndarray`` of a four-dimensional image batch of the form 
												<number of images, height, width, channels>

	Returns:
		numpy ndarray: gray
	"""
	if len(rgb.shape) == 4:
		r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
	elif len(rgb.shape) == 3:
		r, g, b = rgb[:,:,0], rgb [:,:,1], rgb[:,:,2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return gray
			
# Concatenate three (height, width)s into one (height, width, 3).
def gray2rgb(r, g, b, channels_dim = 3):
	""" 
	Concatenates ``r, g ,b``, three two dimensional arrays into one rgb image.   
				
	Args:
		r:  Red channel pixels
		g:  Green channel pixels
		b:  Blue channel pixels 
		channels_dim: Which channel to concatenate to.
						All inputs must have the same shape.
				
	Returns:
		rgb: Concatenated image.            
	"""

	assert r.ndim == 2 and g.ndim == 2 and b.ndim == 2
	rgb = (r[..., numpy.newaxis], g[..., numpy.newaxis], b[..., numpy.newaxis])
	out = numpy.concatenate(rgb, axis=-1)
	if channels_dim == 3:
		return out	
	elif channels_dim == 1:
		return out.transpose((0,3,1,2))
    		

def preprocessing( data, height, width, channels, args): 
	"""
	This function does some basic image processing and preprocessing work on loaded data. Its 
	better to do these preprocessing once before hand and during setup than to keep doing them
	over and over again. The following operations are available now:

	Normalization: 

	Args: 
		data: provide an array of floating point numbers. If data is a two dimensional array, 
			  assume <image vector, number of images> If data is four dimensional assume 
			  <number of images, height, width, channels>	
		height: integer
		width: integer
		channels: integer (1 for grayscale and 3 for rgb)
		args: provide preprocessing arguments. This is a dictionary of the form:

			.. code-block:: python

				args =  {

					"normalize" : <bool> True for normalize across batches
					"GCN"	: True for global contrast normalization
					"ZCA"	: True, kind of like a PCA representation (not fully tested)
					"grayscale"  : Convert the image to grayscale
					"mean_subtract" : Subtracts the mean of the image. 

						} 

	Returns:
		numpy ndarray: data 
	
	"""
	normalize 		= args [ "normalize" ]
	GCN 			= args [ "GCN" ]      
	ZCA 			= args [ "ZCA" ]	 
	gray 			= args [ "grayscale" ]
	mean_subtract   = args [ "mean_subtract" ]

	# Assume that the data is already resized on height and width and all ... 
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
	if mean_subtract is True:
		if normalize is True or ZCA is True: 
			data = data  / (data.max() + 1e-7)
			data = data - data.mean()
			# do this normalization thing in batch mode.	
	else:
    		if normalize is True or ZCA is True:
			data = data / (data.max() + 1e-7)
	
	if ZCA is True:		

		sigma = numpy.dot(data.T,data) / data.shape[1]
		if scipy_installed is False:
			raise Exception("Scipy needs to be installed for performing ZCA")		
		U, S, V = linalg.svd(sigma)		
		# data_rotated = numpy.dot(U.T, data) , full_matrices = True
		temp = numpy.dot(U, numpy.diag(1/numpy.sqrt(S + 1e-7)))
		temp = numpy.dot(temp, U.T)
		data = numpy.dot(data, temp)	
			
	# if GCN is True :
	return data
	
def check_type(data, type):
	"""
	This checks and sets data as whatever the type is.

	Args:
		data: Whatever is the data. Numpy format usually.
		type: Whichever type to test and set.
	"""
	if not data.dtype == type:
		data = numpy.asarray(data, dtype = type)
		return data
	else:
		return data

def pickle_dataset(loc,batch,data):
	""" 
	Function that stores down an object as a pickle file given its filename and obj

	Args:
		loc: Provide location to save as a string
		batch: provide a batch number to save the file as
		data: Pass the data that needs to be picked down. Could also be a tuple

	"""
	f = open(loc + 'batch_' + str(batch) +  '.pkl' , 'wb')				
	cPickle.dump(data, f, protocol=2)
	f.close()	

# From the Theano Tutorials
def create_shared_memory_dataset(data_xy, 
								 borrow=True,
								 verbose = 1,
								 **kwargs):
	"""
	This function creates a shared theano memory to be used for dataset purposes.
	
	Args: 
		data_xy: ``[data_x, data_y]`` that will be assigned to ``shared_x`` and ``shared_y`` 
					on output.
		borrow: default value is ``True``. This is a theano shared memory type variabe.
		verbose: Similar to verbose everywhere else.
		svm: default is ``False``. If ``True``, we also return a ``shared_svm_y`` for 
				 max-margin type last layer.
				 
	Returns:
		theano.shared: ``shared_x, shared_y`` is ``svm`` is ``False``. If not, ``shared_x, 
		                shared_y, shared_svm_y`` 
	"""
	if 'svm' in kwargs.keys():
		svm = kwargs["svm"]
	else:
		svm = False

	if svm is True:
		data_x, data_y, data_y1 = data_xy
		data_y1 = check_type(data_y1, theano.config.floatX)
		shared_y1 = shared(data_y1, borrow=borrow)
	else:
		data_x, data_y = data_xy
	# Theano recommends storing on gpus only as floatX and casts them to ints during use.
    # I don't know why, but I am following their recommendations blindly.
	data_x = check_type(data_x, theano.config.floatX)
	data_y = check_type(data_y, theano.config.floatX)
	shared_x = shared(data_x, borrow=borrow)
	shared_y = shared(data_y, borrow=borrow)

	if svm is True:
		return shared_x, shared_y, shared_y1
	else:
		return shared_x, shared_y 


# Load initial data         
class setup_dataset (object):
	"""
	The setup_dataset class is used to create and assemble datasets that are friendly to the 
	Yann toolbox.

	Todo:
		``images`` option for the ``source``.
		``skdata pascal`` isn't working
		``imagenet`` dataset and ``coco`` needs to be setup.

	Args: 
		dataset_init_args: is a dictonary of the form:

			.. code-block:: none

				data_init_args = {

					"source" : <where to get the dataset from>
						        'pkl' : A theano tutorial style 'pkl' file.
						        'skdata' : Download and setup from skdata
						        'matlab' : Data is created and is being used from Matlab																				
					"name" : necessary only for skdata
						      supports 'mnist','mnist_noise1', 'mnist_noise2', 'mnist_noise3',
						    'mnist_noise4', 'mnist_noise5', 'mnist_noise6', 'mnist_bg_images',
						     'mnist_bg_rand', 'mnist_rotated', 'mnist_rotated_bg'. Refer to 
						     original paper by Hugo Larochelle [1] for these dataset details.
					"location"           		: #necessary for 'pkl' and 'matlab'                  
					"mini_batch_size"         		: 500,                                     
					"mini_batches_per_batch"    : (100, 20, 20), # trianing, testing, validation 
					"batches2train"      		: 1,                                      
					"batches2test"       		: 1,                                      
					"batches2validate"   		: 1,                                        
					"height"             		: 28,                                       
					"width"              		: 28,                                       
					"channels"           		: 1 ,

						}

		preprocess_init_args: provide preprocessing arguments. This is a dictionary:

			.. code-block:: none

				args =  {
					"normalize" : <bool> True for normalize across batches
					"GCN"	    : True for global contrast normalization
					"ZCA"	    : True, kind of like a PCA representation (not fully tested)
					"grayscale" : Convert the image to grayscale
						}  

		save_directory: <string> a location where the dataset is going to be saved.
		
	.. [#] Larochelle H, Erhan D, Courville A, Bergstra J, Bengio Y. An empirical evaluation 
			of deep architectures on problems with many factors of variation. InProceedings 
			of the 24th international conference on Machine learning 2007 Jun 20 
			(pp. 473-480). ACM.

	Notes:

		Yann toolbox takes datasets in a ``.pkl`` format. The dataset requires a directory 
		structure such as the following:

		.. code-block:: python

			location/_dataset_XXXXX  
			|_ data_params.pkl
			|_ train
				|_ batch_0.pkl
				|_ batch_1.pkl
				.
				.
				.
			|_ valid
				|_ batch_0.pkl
				|_ batch_1.pkl
				.
				.
				.
			|_ test
				|_ batch_0.pkl
				|_ batch_1.pkl
				.
				.
				.				 

		The location id (``XXXXX``) is generated by this class file. The five digits that are 
		produced is the unique id of the dataset.

		The file ``data_params.pkl`` contains one variable ``dataset_args`` of the form:

		.. code-block:: python 

			dataset_args = {
					"location"           		: <location>,                                          
					"mini_batch_size"         		: <int>,                                    
					"mini_batches_per_batch"    : (<int>,<int>,<int>),
					"batches2train"      		: <int>,                                      
					"batches2test"       		: <int>,                                     
					"batches2validate"   		: <int>,                                       
					"height"             		: <int>,                                      
					"width"              		: <int>,                                       
					"channels"           		: <int>,
				}

		This variable is used to load up the dataset.  
	
	"""
	def __init__(self, 
				 dataset_init_args, 
				 save_directory = '_datasets', 
				 verbose = 1,
				 **kwargs):
		"""
		Look at the class definition
		"""
		if verbose >= 1:
			print ". Setting up dataset "

		self.source              = dataset_init_args [ "source" ]
		if self.source == 'skdata':
			self.name = dataset_init_args ["name"]
		else:
			self.location        = dataset_init_args [ "location" ]
		
		if "height" in dataset_init_args.keys():			
			self.height              = dataset_init_args [ "height" ]
		else:
			self.height = 28

		if "width" in dataset_init_args.keys():								
			self.width               = dataset_init_args [ "width" ]
		else:
			self.width = 28
		
		if "channels" in dataset_init_args.keys():
			self.channels            = dataset_init_args [ "channels" ]
		else:
			self.channels = 1
		
		if "mini_batch_size" in dataset_init_args.keys():
			self.mini_batch_size          = dataset_init_args [ "mini_batch_size" ]
		else:
			self.mini_batch_size = 20
		
		if "mini_batches_per_batch" in dataset_init_args.keys():		
			self.mini_batches_per_batch       = dataset_init_args [ "mini_batches_per_batch" ]
		else:
			self.mini_batches_per_batch = (100, 20, 20)
		self.cache_images        = (self.mini_batches_per_batch[0] * self.mini_batch_size,
									self.mini_batches_per_batch[1] * self.mini_batch_size,
									self.mini_batches_per_batch[2] * self.mini_batch_size)

		if "batches2train" in dataset_init_args.keys():			
			self.batches2train       = dataset_init_args [ "batches2train"]
		else:
			self.batches2train = 1

		if "batches2test" in dataset_init_args.keys():		
			self.batches2test        = dataset_init_args [ "batches2test" ]
		else:
			self.batches2test = 1

		if "batches2validate" in dataset_init_args.keys():			
			self.batches2validate    = dataset_init_args [ "batches2validate" ]
		else:
			self.batches2validate = 1

		self.cache =  not( self.batches2train == 1 and 
		                   self.batches2test == 1 and
						   self.batches2validate == 1 )
		
		# create some directory for storing all this data
		self.id = str(randint(11111,99999))
		self.key_root = '/_dataset_'
		self.root = save_directory + self.key_root + self.id 	
		if not os.path.exists(save_directory):
			os.mkdir(save_directory)

		os.mkdir(self.root)
		os.mkdir(self.root + "/train" )												
		os.mkdir(self.root + "/test"  )
		os.mkdir(self.root + "/valid" )

		if "preprocess_init_args" in kwargs.keys():
			self.preprocessor = kwargs['preprocess_init_args']
		else: 
			self.preprocessor =  { 
                            "normalize"     : True,
                            "GCN"           : False,
                            "ZCA"           : False,
                            "grayscale"     : True,
                        	}
		start_time = time.clock()		
		if self.source == 'skdata':
			self._create_skdata(verbose = verbose)
		end_time = time.clock()
		if verbose >=1: 
			print ". Dataset " + self.id + " is created."
			print ". Time taken is " +str(end_time - start_time) + " seconds"			

	def dataset_location (self):
		"""
		Use this function that return the location of dataset.
		"""
		return self.root

	def _create_skdata(self,verbose=1):
		"""
		This is an internal function, create any skdata function.
		"""
		if verbose >=3: 
			print ".. setting up skdata"
		# if hugo larochelle dataset... 
		if (self.name == 'mnist' or 
			self.name == 'mnist_noise1' or 
			self.name == 'mnist_noise2' or
			self.name == 'mnist_noise3' or
			self.name == 'mnist_noise4' or
			self.name == 'mnist_noise5' or
			self.name == 'mnist_noise6' or
			self.name == 'mnist_bg_images' or
			self.name == 'mnist_bg_rand' or
			self.name == 'mnist_rotated' or
			self.name == 'mnist_rotated_bg' or
			self.name == 'cifar10' ) :
	
			self._create_skdata_mnist(verbose = verbose)			
	
	def _create_skdata_mnist(self, verbose = 1): 
		"""
		Interal function. Use this to create mnist and cifar image datasets
		"""
		if verbose >=3:												
			print "... Importing " + self.name + " from skdata"
		data = getattr(thismodule, 'load_skdata_' + self.name)() 

		if verbose >=2:               
			print ".. setting up dataset"
			print ".. training data"		

		data_x, data_y, data_y1  = data[0]		
		data_x = preprocessing ( data = data_x,
								 height = self.height,
								 width = self.width,
								 channels = self.channels,
								 args = self.preprocessor )			
		training_sample_size = data_x.shape[0]
		training_batches_available  = training_sample_size / self.mini_batch_size

		if not self.batches2train * self.mini_batches_per_batch[0] == training_batches_available:
			if training_batches_available < self.batches2train * self.mini_batches_per_batch[0]:
				raise Exception("Not as many training batches available")
			else: 
				data_x = data_x[:self.batches2train * self.cache_images[0]]
				data_y = data_y[:self.batches2train * self.cache_images[0]] 
		loc = self.root + "/train/"	
		data_x = check_type(data_x, theano.config.floatX)
		data_y = check_type(data_y, theano.config.floatX)	
		

		for batch in xrange(self.batches2train):
			start_index = batch * self.cache_images[0]
			end_index = start_index + self.cache_images[0]
			data2save = (data_x [start_index:end_index,], data_y[start_index:end_index,] )
			pickle_dataset(loc = loc, data = data2save, batch=batch)	
		
		if verbose >=2: 			
			print ".. validation data "

		data_x, data_y, data_y1  = data[1]
		data_x = preprocessing ( data = data_x,
								 height = self.height,
								 width = self.width,
								 channels = self.channels,
								 args = self.preprocessor )	
		validation_sample_size = data_x.shape[0]
		validation_batches_available = validation_sample_size / self.mini_batch_size 

		if not self.batches2validate * self.mini_batches_per_batch[1] == \
		                                                 validation_batches_available:
			if validation_batches_available < self.batches2validate * \
			                                           self.mini_batches_per_batch[1]: 
				raise Exception("Not as many validation batches available")
			else: 
				data_x = data_x[:self.batches2validate * self.cache_images[1]]
				data_y = data_y[:self.batches2validate * self.cache_images[1]] 
		loc = self.root + "/valid/"		
		data_x = check_type(data_x, theano.config.floatX)
		data_y = check_type(data_y, theano.config.floatX)				

		for batch in xrange(self.batches2validate):
			start_index = batch * self.cache_images[1]
			end_index = start_index + self.cache_images[1]
			data2save = (data_x [start_index:end_index,], data_y[start_index:end_index,] )
			pickle_dataset(loc = loc, data = data2save, batch=batch)
		
		if verbose >=2: 						
			print ".. testing data "			
		data_x, data_y, data_y1 = data[2]
		data_x = preprocessing ( data = data_x,												
								 height = self.height,
								 width = self.width,
								 channels = self.channels,
								 args = self.preprocessor )	
		testing_sample_size = data_x.shape[0]
		testing_batches_available = testing_sample_size / self.mini_batch_size 
		
		if not self.batches2test * self.mini_batches_per_batch[2] == testing_batches_available:		
			if testing_batches_available < self.batches2test * self.mini_batches_per_batch[2]: 
				raise Exception("Not as many testing batches available")
			else: 
				data_x = data_x[:self.batches2test * self.cache_images[2]]
				data_y = data_y[:self.batches2test * self.cache_images[2]] 
		loc = self.root + "/test/"	
		data_x = check_type(data_x, theano.config.floatX)
		data_y = check_type(data_y, theano.config.floatX)	

		for batch in xrange(self.batches2test):
    			start_index = batch * self.cache_images[2]
			end_index = start_index + self.cache_images[2]
			data2save = (data_x [start_index:end_index,], data_y[start_index:end_index,] )
			pickle_dataset(loc = loc, data = data2save, batch=batch)
										
		dataset_args = {
				"location"          		: self.root,                                          
				"mini_batch_size"           : self.mini_batch_size,                                    
				"cache_batches"    			: self.mini_batches_per_batch,
				"batches2train"     		: self.batches2train,                                      
				"batches2test"      		: self.batches2test,                                     
				"batches2validate"  		: self.batches2validate,                                       
				"height"            		: self.height,                                      
				"width"             		: self.width,                                       
				"channels"          	: 1 if self.preprocessor ["grayscale"] else self.channels,
				"cache"             	: self.cache, 						                                       
				}
			
		assert ( self.height * self.width * self.channels == numpy.prod(data_x.shape[1:]) )
		f = open(self.root +  '/data_params.pkl', 'wb')
		cPickle.dump(dataset_args, f, protocol=2)
		f.close()	


if __name__ == '__main__':
	pass

"""
elif self.name == 'cifar10':
    			print "... importing cifar 10 from skdata"
			data = load_skdata_cifar10()
			print "... setting up dataset "
			print "... 		--> training data "			
			data_x, data_y, data_y1 = data[0]
			data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )			
			n_train_images = data_x.shape[0]
			n_train_batches_all = n_train_images / self.mini_batch_size 
			self.n_train_batches = data_x.shape[0] / self.mini_batch_size			
			f = open(temp_dir + "/train/" + 'batch_' + str(0) + '.pkl', 'wb')
			obj = (data_x, data_y )
			cPickle.dump(obj, f, protocol=2)
			f.close()		
			
			print "... 		--> validation data "			
			data_x, data_y, data_y1 = data[1]
			data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )			
			n_valid_images = data_x.shape[0]
			n_valid_batches_all = n_valid_images / self.mini_batch_size 
			self.n_valid_batches = data_x.shape[0] / self.mini_batch_size			
			f = open(temp_dir + "/valid/" + 'batch_' + str(0) + '.pkl', 'wb')
			obj = (data_x, data_y )
			cPickle.dump(obj, f, protocol=2)
			f.close()				
			
			print "... 		--> testing data "			
			data_x, data_y, data_y1 = data[2]
			data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )			
			n_test_images = data_x.shape[0]
			n_test_batches_all = n_test_images / self.mini_batch_size 
			self.n_test_batches = data_x.shape[0] / self.mini_batch_size			
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
				"mini_batch_size"         : self.mini_batch_size,                                    
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
				
		elif self.name == 'caltech101':
			verbose = False
			print "... importing caltech 101 from skdata"
			
			# shuffle the data
			total_images_in_dataset = 9144 
			self.rand_perm = numpy.random.permutation(total_images_in_dataset)  
			# create a constant shuffle, so that data can be loaded in batchmode with the same random shuffle
			
			n_train_images = self.mini_batch_size * self.batches2train
			n_test_images = self.mini_batch_size * self.batches2test
			n_valid_images = self.mini_batch_size * self.batches2validate
			
			assert n_valid_images + n_train_images + n_test_images == total_images_in_dataset  
			
			print ".... setting up dataset"				
			print "... 		--> training data "						
			looper = n_train_images / self.load_batches						
			for i in xrange(looper):		# for each batch_i file.... 
				data_x, data_y  = load_skdata_caltech101(
												n_train_images = n_train_images,
												n_test_images = n_test_images,
												n_valid_images = n_valid_images,
												mini_batch_size = self.load_batches, 
												rand_perm = self.rand_perm, 
												batch = i , 
												type_set = 'train' ,
												height = self.height,
												width = self.width,
												verbose = verbose )  													
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )				
				# compute number of minibatches for training, validation and testing
				self.n_train_batches = data_x.shape[0] / self.mini_batch_size			
				f = open(temp_dir + "/train/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()
		
				
			self.n_train_batches = data_x.shape[0] / self.mini_batch_size											
			print "... 		--> testing data "				
			looper = n_test_images / self.load_batches		
			for i in xrange(looper):		# for each batch_i file.... 
				data_x, data_y  = load_skdata_caltech101(
												n_train_images = n_train_images,
												n_test_images = n_test_images,
												n_valid_images = n_valid_images,
												mini_batch_size = self.load_batches, 
												rand_perm = self.rand_perm, 
												batch = i , 
												type_set = 'test' ,
												height = self.height,
												width = self.width,
												verbose = verbose )  
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )				
				# compute number of minibatches for training, validation and testing
				self.n_train_batches = data_x.shape[0] / self.mini_batch_size			
				f = open(temp_dir + "/test/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()							  
			self.n_test_batches = data_x.shape[0] / self.mini_batch_size																
			
			print "... 		--> validation data "	
			looper = n_valid_images / self.load_batches									
			for i in xrange(looper):		# for each batch_i file.... 
				data_x, data_y  = load_skdata_caltech101(
												n_train_images = n_train_images,
												n_test_images = n_test_images,
												n_valid_images = n_valid_images,
												mini_batch_size = self.load_batches, 
												rand_perm = self.rand_perm, 
												batch = i , 
												type_set = 'valid' ,
												height = self.height,
												width = self.width,
												verbose = verbose  )  
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )				
				# compute number of minibatches for training, validation and testing
				self.n_train_batches = data_x.shape[0] / self.mini_batch_size			
				f = open(temp_dir + "/valid/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()							  
			self.n_valid_batches = data_x.shape[0] / self.mini_batch_size																	
			self.multi_load = True
			
			new_data_params = {
				"type"               : 'base',                                   
				"loc"                : temp_dir,                                          
				"mini_batch_size"         : self.mini_batch_size,                                    
				"load_batches"       : self.load_batches / self.mini_batch_size,
				"batches2train"      : self.batches2train / (self.load_batches / self.mini_batch_size) ,                                      
				"batches2test"       : self.batches2test / (self.load_batches / self.mini_batch_size),                                     
				"batches2validate"   : self.batches2validate / (self.load_batches / self.mini_batch_size),                                       
				"height"             : self.height,                                      
				"width"              : self.width,                                       
				"channels"           : self.channels,
				"multi_load"		 : self.multi_load,
				"n_train_batches"	 : self.n_train_batches,
				"n_test_batches"	 : self.n_test_batches,
				"n_valid_batches"	 : self.n_valid_batches  					                                        
				}
				
		elif self.name == 'caltech256':
			print "... importing caltech 256 from skdata"
			
				# shuffle the data
			total_images_in_dataset = 30607 
			self.rand_perm = numpy.random.permutation(total_images_in_dataset)  
			# create a constant shuffle, so that data can be loaded in batchmode with the same random shuffle
			
			n_train_images = self.mini_batch_size * self.batches2train
			n_test_images = self.mini_batch_size * self.batches2test
			n_valid_images = self.mini_batch_size * self.batches2validate
			
			
			assert n_valid_images + n_train_images + n_test_images == total_images_in_dataset  
		elif self.name == 'cifar10':
    			print "... importing cifar 10 from skdata"
			data = load_skdata_cifar10()
			print "... setting up dataset "
			print "... 		--> training data "			
			data_x, data_y, data_y1 = data[0]
			data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )			
			n_train_images = data_x.shape[0]
			n_train_batches_all = n_train_images / self.mini_batch_size 
			self.n_train_batches = data_x.shape[0] / self.mini_batch_size			
			f = open(temp_dir + "/train/" + 'batch_' + str(0) + '.pkl', 'wb')
			obj = (data_x, data_y )
			cPickle.dump(obj, f, protocol=2)
			f.close()		
			
			print "... 		--> validation data "			
			data_x, data_y, data_y1 = data[1]
			data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )			
			n_valid_images = data_x.shape[0]
			n_valid_batches_all = n_valid_images / self.mini_batch_size 
			self.n_valid_batches = data_x.shape[0] / self.mini_batch_size			
			f = open(temp_dir + "/valid/" + 'batch_' + str(0) + '.pkl', 'wb')
			obj = (data_x, data_y )
			cPickle.dump(obj, f, protocol=2)
			f.close()				
			
			print "... 		--> testing data "			
			data_x, data_y, data_y1 = data[2]
			data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )			
			n_test_images = data_x.shape[0]
			n_test_batches_all = n_test_images / self.mini_batch_size 
			self.n_test_batches = data_x.shape[0] / self.mini_batch_size			
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
				"mini_batch_size"         : self.mini_batch_size,                                    
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
				
		elif self.name == 'caltech101':
			verbose = False
			print "... importing caltech 101 from skdata"
			
			# shuffle the data
			total_images_in_dataset = 9144 
			self.rand_perm = numpy.random.permutation(total_images_in_dataset)  
			# create a constant shuffle, so that data can be loaded in batchmode with the same random shuffle
			
			n_train_images = self.mini_batch_size * self.batches2train
			n_test_images = self.mini_batch_size * self.batches2test
			n_valid_images = self.mini_batch_size * self.batches2validate
			
			assert n_valid_images + n_train_images + n_test_images == total_images_in_dataset  
			
			print ".... setting up dataset"				
			print "... 		--> training data "						
			looper = n_train_images / self.load_batches						
			for i in xrange(looper):		# for each batch_i file.... 
				data_x, data_y  = load_skdata_caltech101(
												n_train_images = n_train_images,
												n_test_images = n_test_images,
												n_valid_images = n_valid_images,
												mini_batch_size = self.load_batches, 
												rand_perm = self.rand_perm, 
												batch = i , 
												type_set = 'train' ,
												height = self.height,
												width = self.width,
												verbose = verbose )  													
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )				
				# compute number of minibatches for training, validation and testing
				self.n_train_batches = data_x.shape[0] / self.mini_batch_size			
				f = open(temp_dir + "/train/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()
		
				
			self.n_train_batches = data_x.shape[0] / self.mini_batch_size											
			print "... 		--> testing data "				
			looper = n_test_images / self.load_batches		
			for i in xrange(looper):		# for each batch_i file.... 
				data_x, data_y  = load_skdata_caltech101(
												n_train_images = n_train_images,
												n_test_images = n_test_images,
												n_valid_images = n_valid_images,
												mini_batch_size = self.load_batches, 
												rand_perm = self.rand_perm, 
												batch = i , 
												type_set = 'test' ,
												height = self.height,
												width = self.width,
												verbose = verbose )  
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )				
				# compute number of minibatches for training, validation and testing
				self.n_train_batches = data_x.shape[0] / self.mini_batch_size			
				f = open(temp_dir + "/test/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()							  
			self.n_test_batches = data_x.shape[0] / self.mini_batch_size																
			
			print "... 		--> validation data "	
			looper = n_valid_images / self.load_batches									
			for i in xrange(looper):		# for each batch_i file.... 
				data_x, data_y  = load_skdata_caltech101(
												n_train_images = n_train_images,
												n_test_images = n_test_images,
												n_valid_images = n_valid_images,
												mini_batch_size = self.load_batches, 
												rand_perm = self.rand_perm, 
												batch = i , 
												type_set = 'valid' ,
												height = self.height,
												width = self.width,
												verbose = verbose  )  
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )				
				# compute number of minibatches for training, validation and testing
				self.n_train_batches = data_x.shape[0] / self.mini_batch_size			
				f = open(temp_dir + "/valid/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()							  
			self.n_valid_batches = data_x.shape[0] / self.mini_batch_size																	
			self.multi_load = True
			
			new_data_params = {
				"type"               : 'base',                                   
				"loc"                : temp_dir,                                          
				"mini_batch_size"         : self.mini_batch_size,                                    
				"load_batches"       : self.load_batches / self.mini_batch_size,
				"batches2train"      : self.batches2train / (self.load_batches / self.mini_batch_size) ,                                      
				"batches2test"       : self.batches2test / (self.load_batches / self.mini_batch_size),                                     
				"batches2validate"   : self.batches2validate / (self.load_batches / self.mini_batch_size),                                       
				"height"             : self.height,                                      
				"width"              : self.width,                                       
				"channels"           : self.channels,
				"multi_load"		 : self.multi_load,
				"n_train_batches"	 : self.n_train_batches,
				"n_test_batches"	 : self.n_test_batches,
				"n_valid_batches"	 : self.n_valid_batches  					                                        
				}
				
		elif self.name == 'caltech256':
			print "... importing caltech 256 from skdata"
			
				# shuffle the data
			total_images_in_dataset = 30607 
			self.rand_perm = numpy.random.permutation(total_images_in_dataset)  
			# create a constant shuffle, so that data can be loaded in batchmode with the same random shuffle
			
			n_train_images = self.mini_batch_size * self.batches2train
			n_test_images = self.mini_batch_size * self.batches2test
			n_valid_images = self.mini_batch_size * self.batches2validate
			
			
			assert n_valid_images + n_train_images + n_test_images == total_images_in_dataset  
			
			print ".... setting up dataset"				
			print "... 		--> training data "						
			looper = n_train_images / self.load_batches						
			for i in xrange(looper):		# for each batch_i file.... 
				data_x, data_y  = load_skdata_caltech101(
												n_train_images = n_train_images,
												n_test_images = n_test_images,
												n_valid_images = n_valid_images,
												mini_batch_size = self.load_batches, 
												rand_perm = self.rand_perm, 
												batch = i , 
												type_set = 'train' ,
												height = self.height,
												width = self.width,
												verbose = verbose )  													
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )				
				# compute number of minibatches for training, validation and testing
				self.n_train_batches = data_x.shape[0] / self.mini_batch_size			
				f = open(temp_dir + "/train/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()
			
				
			self.n_train_batches = data_x.shape[0] / self.mini_batch_size											
			print "... 		--> testing data "				
			looper = n_test_images / self.load_batches		
			for i in xrange(looper):		# for each batch_i file.... 
				data_x, data_y  = load_skdata_caltech101(
												n_train_images = n_train_images,
												n_test_images = n_test_images,
												n_valid_images = n_valid_images,
												mini_batch_size = self.load_batches, 
												rand_perm = self.rand_perm, 
												batch = i , 
												type_set = 'test' ,
												height = self.height,
												width = self.width,
												verbose = verbose )  
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )				
				# compute number of minibatches for training, validation and testing
				self.n_train_batches = data_x.shape[0] / self.mini_batch_size			
				f = open(temp_dir + "/test/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()							  
			self.n_test_batches = data_x.shape[0] / self.mini_batch_size																
			
			print "... 		--> validation data "	
			looper = n_valid_images / self.load_batches									
			for i in xrange(looper):		# for each batch_i file.... 
				data_x, data_y  = load_skdata_caltech101(
												n_train_images = n_train_images,
												n_test_images = n_test_images,
												n_valid_images = n_valid_images,
												mini_batch_size = self.load_batches, 
												rand_perm = self.rand_perm, 
												batch = i , 
												type_set = 'valid' ,
												height = self.height,
												width = self.width,
												verbose = verbose  )  
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )				
				# compute number of minibatches for training, validation and testing
				self.n_train_batches = data_x.shape[0] / self.mini_batch_size			
				f = open(temp_dir + "/valid/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()							  
			self.n_valid_batches = data_x.shape[0] / self.mini_batch_size																	
			self.multi_load = True
			
			
			new_data_params = {
				"type"               : 'base',                                   
				"loc"                : temp_dir,                                          
				"mini_batch_size"         : self.mini_batch_size,                                    
				"load_batches"       : self.load_batches / self.mini_batch_size,
				"batches2train"      : self.batches2train / (self.load_batches / self.mini_batch_size),                                      
				"batches2test"       : self.batches2test / (self.load_batches / self.mini_batch_size),                                     
				"batches2validate"   : self.batches2validate / (self.load_batches / self.mini_batch_size),                                       
				"height"             : self.height,                                      
				"width"              : self.width,                                       
				"channels"           : self.channels,
				"multi_load"		 : self.multi_load,
				"n_train_batches"	 : self.n_train_batches,
				"n_test_batches"	 : self.n_test_batches,
				"n_valid_batches"	 : self.n_valid_batches  					                                        
				}
								
	if self.preprocessor["gray"] is True:
		new_data_params ["channels"] = 1 
		self.channels = 1 
	
	assert ( self.height * self.width * self.channels == numpy.prod(data_x.shape[1:]) )
	f = open(temp_dir +  '/data_params.pkl', 'wb')
	cPickle.dump(new_data_params, f, protocol=2)
	f.close()				  	
	end_time = time.clock()
	print "...         time taken is " +str(end_time - start_time) + " seconds"


	
			print ".... setting up dataset"				
			print "... 		--> training data "						
			looper = n_train_images / self.load_batches						
			for i in xrange(looper):		# for each batch_i file.... 
				data_x, data_y  = load_skdata_caltech101(
												n_train_images = n_train_images,
												n_test_images = n_test_images,
												n_valid_images = n_valid_images,
												mini_batch_size = self.load_batches, 
												rand_perm = self.rand_perm, 
												batch = i , 
												type_set = 'train' ,
												height = self.height,
												width = self.width,
												verbose = verbose )  													
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )				
				# compute number of minibatches for training, validation and testing
				self.n_train_batches = data_x.shape[0] / self.mini_batch_size			
				f = open(temp_dir + "/train/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()
			
				
			self.n_train_batches = data_x.shape[0] / self.mini_batch_size											
			print "... 		--> testing data "				
			looper = n_test_images / self.load_batches		
			for i in xrange(looper):		# for each batch_i file.... 
				data_x, data_y  = load_skdata_caltech101(
												n_train_images = n_train_images,
												n_test_images = n_test_images,
												n_valid_images = n_valid_images,
												mini_batch_size = self.load_batches, 
												rand_perm = self.rand_perm, 
												batch = i , 
												type_set = 'test' ,
												height = self.height,
												width = self.width,
												verbose = verbose )  
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )				
				# compute number of minibatches for training, validation and testing
				self.n_train_batches = data_x.shape[0] / self.mini_batch_size			
				f = open(temp_dir + "/test/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()							  
			self.n_test_batches = data_x.shape[0] / self.mini_batch_size																
			
			print "... 		--> validation data "	
			looper = n_valid_images / self.load_batches									
			for i in xrange(looper):		# for each batch_i file.... 
				data_x, data_y  = load_skdata_caltech101(
												n_train_images = n_train_images,
												n_test_images = n_test_images,
												n_valid_images = n_valid_images,
												mini_batch_size = self.load_batches, 
												rand_perm = self.rand_perm, 
												batch = i , 
												type_set = 'valid' ,
												height = self.height,
												width = self.width,
												verbose = verbose  )  
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )				
				# compute number of minibatches for training, validation and testing
				self.n_train_batches = data_x.shape[0] / self.mini_batch_size			
				f = open(temp_dir + "/valid/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()							  
			self.n_valid_batches = data_x.shape[0] / self.mini_batch_size																	
			self.multi_load = True
			
			
			new_data_params = {
				"type"               : 'base',                                   
				"loc"                : temp_dir,                                          
				"mini_batch_size"         : self.mini_batch_size,                                    
				"load_batches"       : self.load_batches / self.mini_batch_size,
				"batches2train"      : self.batches2train / (self.load_batches / self.mini_batch_size),                                      
				"batches2test"       : self.batches2test / (self.load_batches / self.mini_batch_size),                                     
				"batches2validate"   : self.batches2validate / (self.load_batches / self.mini_batch_size),                                       
				"height"             : self.height,                                      
				"width"              : self.width,                                       
				"channels"           : self.channels,
				"multi_load"		 : self.multi_load,
				"n_train_batches"	 : self.n_train_batches,
				"n_test_batches"	 : self.n_test_batches,
				"n_valid_batches"	 : self.n_valid_batches  					                                        
				}
								
	if self.preprocessor["gray"] is True:
		new_data_params ["channels"] = 1 
		self.channels = 1 
	
	assert ( self.height * self.width * self.channels == numpy.prod(data_x.shape[1:]) )
	f = open(temp_dir +  '/data_params.pkl', 'wb')
	cPickle.dump(new_data_params, f, protocol=2)
	f.close()				  	
	end_time = time.clock()
	print "...         time taken is " +str(end_time - start_time) + " seconds"



		if self.data_type == 'mat':
			
			print "... 		--> training data "
			for i in xrange(self.batches2train):		# for each batch_i file.... 
				data_x, data_y, data_y1 = load_data_mat(dataset = self.name, batch = i + 1, type_set = 'train' , n_classes = outs, height = self.height, width = self.width, channels = self.channels)
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )
				
				# compute number of minibatches for training, validation and testing
				self.n_train_batches = data_x.shape[0] / self.mini_batch_size			
				f = open(temp_dir + "/train/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()
				 
			print "... 		--> testing data "	
			for i in xrange(self.batches2test):		# for each batch_i file.... 
				data_x, data_y, data_y1 = load_data_mat(dataset = self.name, batch = i + 1, type_set = 'test' , n_classes = outs, height = self.height, width = self.width, channels = self.channels)
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )
				
				# compute number of minibatches for training, validation and testing
				self.n_test_batches = data_x.shape[0] / self.mini_batch_size			
				f = open(temp_dir + "/test/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()
				
			print "... 		--> validation data "	
			for i in xrange(self.batches2validate):		# for each batch_i file.... 
				data_x, data_y, data_y1 = load_data_mat(dataset = self.name, batch = i + 1, type_set = 'valid' , n_classes = outs, height = self.height, width = self.width, channels = self.channels)
				data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )
				
				# compute number of minibatches for training, validation and testing
				self.n_valid_batches = data_x.shape[0] / self.mini_batch_size			
				f = open(temp_dir + "/valid/" + 'batch_' + str(i) + '.pkl', 'wb')
				obj = (data_x, data_y )
				cPickle.dump(obj, f, protocol=2)
				f.close()	
					
			self.multi_load = True 			
			new_data_params = {
					"type"               : 'base',                                   
					"loc"                : temp_dir,                                          
					"mini_batch_size"         : self.mini_batch_size,                                    
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
		
			data = load_data_pkl(self.name)            						
			print "... setting up dataset "
			print "... 		--> training data "			
			data_x, data_y, data_y1 = data[0]
			data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )			
			n_train_images = data_x.shape[0]
			n_train_batches_all = n_train_images / self.mini_batch_size 
			self.n_train_batches = data_x.shape[0] / self.mini_batch_size			
			f = open(temp_dir + "/train/" + 'batch_' + str(0) + '.pkl', 'wb')
			obj = (data_x, data_y )
			cPickle.dump(obj, f, protocol=2)
			f.close()		
			
			print "... 		--> validation data "			
			data_x, data_y, data_y1 = data[1]
			data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )			
			n_valid_images = data_x.shape[0]
			n_valid_batches_all = n_valid_images / self.mini_batch_size 
			self.n_valid_batches = data_x.shape[0] / self.mini_batch_size			
			f = open(temp_dir + "/valid/" + 'batch_' + str(0) + '.pkl', 'wb')
			obj = (data_x, data_y )
			cPickle.dump(obj, f, protocol=2)
			f.close()				
			
			print "... 		--> testing data "			
			data_x, data_y, data_y1 = data[2]
			data_x = preprocessing ( data_x, self.height, self.width, self.channels, self.preprocessor )			
			n_test_images = data_x.shape[0]
			n_test_batches_all = n_test_images / self.mini_batch_size 
			self.n_test_batches = data_x.shape[0] / self.mini_batch_size			
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
					"mini_batch_size"         : self.mini_batch_size,                                    
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
							
		# load skdata ( its a good library that has a lot of self.names)
		elif self.data_type == 'skdata':
		

		



# this loads up the data_params from a folder and sets up the initial databatch.         
def reset ( dataset, data_params ):
    import pdb
    pdb.set_trace()
    os.remove(dataset + '/data_params.pkl')
    f = open(dataset +  '/data_params.pkl', 'wb')
    cPickle.dump(new_data_params, f, protocol=2)
    f.close()			
"""
