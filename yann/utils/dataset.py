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
#import cPickle
#for python3 compatability
import pickle as cPickle
import imp

from image import *
from yann.utils.image import preprocessing
from yann.utils.image import check_type

# for xrange python2 and 3 compatability
try:
    xrange
except NameError:
    xrange = range

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

from yann.utils.image import preprocessing
from scipy.misc import imresize as imresize

thismodule = sys.modules[__name__]

def download_data (url, location):
    """
    """
    import urllib2
    file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    f = open(location + file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print("Downloading: %s Bytes: %s" % (file_name, file_size))
    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print(status),
    f.close()

def load_cifar100 ():
    """
    Function that downloads the cifar 100 dataset and returns the dataset in full

    TODO: Need to implement this.
    """
    print("Not implemented yet.")

def load_data_mat(height,
                  width,
                  channels, 
                  location = '../dataset/waldo/',
                  batch = 0,
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
    print("... Loading " + type_set + " batch number " + str(batch))
    if scipy_installed is False:
        raise Exception("Scipy needed for cooking this dataset. Please install")
    mat = scipy.io.loadmat(location  + '/' +  type_set + '/batch_' + str(batch) + '.mat')
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

    # y1 = -1 * numpy.ones((data_y.shape[0], max(data_y)))  # is this max dangerous ?
    # y1[numpy.arange(data_y.shape[0]), data_y] = 1

    if load_z is False:
        # return (data_x,data_y,y1.astype( dtype = 'float32' ))
        return (data_x,data_y)        
    else:
        return (data_x,data_y,data_z)


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
def load_skdata_caltech101(batch_size,
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
        batch_size: What is the size of the batch.
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
    # load_batches * batch_size is supplied into batch_size
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

    data_x = numpy.asarray(numpy.zeros((batch_size,height*width *3)), dtype = 'float32' )

    if type_set == 'train':
        push = 0 + batch * batch_size
    elif type_set == 'test':
        push = n_train_images + batch * batch_size
    elif type_set == 'valid':
        push = n_train_images + n_test_images + batch * batch_size

    if verbose is True:
        print("Processing image:  " + str(push))
    data_y = numpy.asarray(data_y[push : push + batch_size ] , dtype = 'int32' )

    for i in range(batch_size):

        temp_img = imread(img[push + i])
        temp_img = temp_img.astype('float32')

        if temp_img.ndim != 3:
            # This is a temporary solution.
            # I am allocating to all channels the grayscale values...
            temp_img = temp_img.astype('float32')
            temp_img = imresize(temp_img,(height,width))
            temp_img1 = numpy.zeros((height,width,3))
            temp_img1 [:,:,0] = temp_img
            temp_img1 [:,:,1] = temp_img
            temp_img1 [:,:,2] = temp_img
            data_x[i] = numpy.reshape(temp_img1,[1,height*width*3] )
        else:
            data_x[i] = numpy.reshape(imresize(temp_img,(height,width)), [1,height*width*3])

    return (data_x,data_y)

# caltech 256 of skdata
def load_skdata_caltech256(batch_size,
                           n_train_images,
                           n_test_images,
                           n_valid_images,
                           rand_perm, batch = 1,
                           type_set = 'train',
                           height = 256,
                           width = 256,
                           verbose = False):
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
    data_x = numpy.asarray(numpy.zeros((batch_size,height*width *3)), dtype = 'float32' )

    if type_set == 'train':
        push = 0 + batch * batch_size
    elif type_set == 'test':
        push = n_train_images + batch * batch_size
    elif type_set == 'valid':
        push = n_train_images + n_test_images + batch * batch_size

    if verbose is True:
        print("Processing image:  " + str(push))
    data_y = numpy.asarray(data_y[push : push + batch_size ] , dtype = 'int32' )

    for i in range(batch_size):

        temp_img = imread(img[push + i])
        temp_img = temp_img.astype('float32')

        if temp_img.ndim != 3:
            # This is a temporary solution.
            # I am allocating to all channels the grayscale values...
            temp_img = temp_img.astype('float32')
            temp_img = imresize(temp_img,(height,width))
            temp_img1 = numpy.zeros((height,width,3))
            temp_img1 [:,:,0] = temp_img
            temp_img1 [:,:,1] = temp_img
            temp_img1 [:,:,2] = temp_img
            data_x[i] = numpy.reshape(temp_img1,[1,height*width*3] )
        else:
            data_x[i] = numpy.reshape(imresize(temp_img,(height,width)), [1,height*width*3])

    return (data_x,data_y)

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
                              supports
                                * ``'mnist'``
                                * ``'mnist_noise1'``
                                * ``'mnist_noise2'``
                                * ``'mnist_noise3'``
                                * ``'mnist_noise4'``
                                * ``'mnist_noise5'``
                                * ``'mnist_noise6'``
                                * ``'mnist_bg_images'``
                                * ``'mnist_bg_rand'``
                                * ``'mnist_rotated'``
                                * ``'mnist_rotated_bg'``.
                                * ``'cifar10'``
                                * ``'caltech101'``
                                * ``'caltech256'``
                        Refer to original paper by Hugo Larochelle [1] for these dataset details.

                    "location"                  : #necessary for 'pkl' and 'matlab'
                    "mini_batch_size"           : 500,
                    "mini_batches_per_batch"    : (100, 20, 20), # trianing, testing, validation
                    "batches2train"             : 1,
                    "batches2test"              : 1,
                    "batches2validate"          : 1,
                    "height"                    : 28,
                    "width"                     : 28,
                    "channels"                  : 1 ,

                        }

        preprocess_init_args: provide preprocessing arguments. This is a dictionary:

            .. code-block:: none

                args =  {
                    "normalize" : <bool> True for normalize across batches
                    "GCN"       : True for global contrast normalization
                    "ZCA"       : True, kind of like a PCA representation (not fully tested)
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

        The file ``data_params.pkl`` contains one variable ``dataset_args`` used by datastream.
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
            print(". Setting up dataset ")

        self.source              = dataset_init_args [ "source" ]

        if self.source == 'skdata':
            self.name = dataset_init_args ["name"]

        elif self.source == 'matlab':
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
                            "ZCA"           : False,
                            "grayscale"     : True,
                            "zero_mean"		: False,
                            }
        start_time = time.clock()
        if self.source == 'skdata':
            self._create_skdata(verbose = verbose)

        if self.source == 'matlab':
            self._mat2yann( verbose = verbose )

        end_time = time.clock()
        if verbose >=1:
            print(". Dataset " + self.id + " is created.")
            print(". Time taken is " +str(end_time - start_time) + " seconds")

    def _mat2yann (self, verbose = 2):
        """
        This method will convert a matlab dataset into yann.        
        Refer to the svhn example in tutorials as to how to create a matlab dataset for yann.

        Notes:

            This will convert all the mat files into pkl files. If you need lesser batches or 
            mini batch sizes, alter the mat files accordingly. 
        """
        for type in ['train', 'test', 'valid']:
            if verbose >= 2:
                print ( ".. creating data " + type )
            if type == 'train': 
                batches = self.batches2train
            elif type == 'test':
                batches = self.batches2test
            else:
                batches = self.batches2validate
            for batch in xrange(batches):		# for each batch_i file....
                if verbose >= 3:
                    print ( "... batch " +str(batch) )
                data_x, data_y = load_data_mat(location = self.location, 
                                                        batch = batch, 
                                                        type_set = type,
                                                        height = self.height,
                                                        width = self.width,
                                                        channels = self.channels)
                                                        
                data_x = preprocessing ( data_x, 
                                        self.height,
                                        self.width,
                                        self.channels,
                                        self.preprocessor )

                if verbose >=3: 
                    print ("... Dumping batch " + str(batch))
                # compute number of minibatches for training, validation and testing
                f = open(self.root + "/" + type + "/" + 'batch_' + str(batch) + '.pkl', 'wb')
                obj = (data_x, data_y )
                cPickle.dump(obj, f, protocol=2)
                f.close()
                if type == 'train':
                    mppb_train = data_x.shape[0] / self.mini_batch_size
                elif type == 'test': 
                    mppb_test = data_x.shape[0] / self.mini_batch_size
                else: 
                    mppb_valid = data_x.shape[0] / self.mini_batch_size
            
        dataset_args = {
                "location"                  : self.root,
                "mini_batch_size"           : self.mini_batch_size,
                "cache_batches"             : (mppb_train, mppb_test, mppb_valid),
                "batches2train"             : self.batches2train,
                "batches2test"              : self.batches2test,
                "batches2validate"          : self.batches2validate,
                "height"                    : self.height,
                "width"                     : self.width,
                "channels"              : 1 if self.preprocessor ["grayscale"] else self.channels,
                "cache"                 : self.cache,
                }
        
        assert ( self.height * self.width * self.channels == numpy.prod(data_x.shape[1:]) )
        f = open(self.root +  '/data_params.pkl', 'wb')
        cPickle.dump(dataset_args, f, protocol=2)
        f.close()

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
            print(".. setting up skdata")
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

        elif self.name == 'caltech101':
            self._create_skdata_caltech101(verbose = verbose)

        elif self.name == 'caltech256':
            self._create_skdata_caltech256(verbose = verbose)

    def _create_skdata_mnist(self, verbose = 1):
        """
        Interal function. Use this to create mnist and cifar image datasets
        """
        if verbose >=3:
            print("... Importing " + self.name + " from skdata")
        data = getattr(thismodule, 'load_skdata_' + self.name)()

        if verbose >=2:
            print(".. setting up dataset")
            print(".. training data")

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
            print(".. validation data ")

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
            print(".. testing data ")
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
                "location"                  : self.root,
                "mini_batch_size"           : self.mini_batch_size,
                "cache_batches"             : self.mini_batches_per_batch,
                "batches2train"             : self.batches2train,
                "batches2test"              : self.batches2test,
                "batches2validate"          : self.batches2validate,
                "height"                    : self.height,
                "width"                     : self.width,
                "channels"              : 1 if self.preprocessor ["grayscale"] else self.channels,
                "cache"                 : self.cache,
                }

        assert ( self.height * self.width * self.channels == numpy.prod(data_x.shape[1:]) )
        f = open(self.root +  '/data_params.pkl', 'wb')
        cPickle.dump(dataset_args, f, protocol=2)
        f.close()

    def _create_skdata_caltech101(self, verbose = 2):
        """
        Interal function. Use this to create mnist and caltech101 image datasets
        """
        # shuffle the data
        total_images_in_dataset = 9144
        self.rand_perm = numpy.random.permutation(total_images_in_dataset)
        # create a constant shuffle, so that data can be loaded in batchmode with the same
        # random shuffle

        n_train_images = self.mini_batches_per_batch[0]*self.mini_batch_size * self.batches2train
        n_test_images = self.mini_batches_per_batch[2]*self.mini_batch_size * self.batches2test
        n_valid_images = self.mini_batches_per_batch[1]*self.mini_batch_size * self.batches2validate

        assert n_valid_images + n_train_images + n_test_images <= total_images_in_dataset

        if verbose >=2:
            print(".. Setting up dataset")
            print(".. Training data")

        looper = n_train_images / ( self.mini_batches_per_batch[0] * self.mini_batch_size )

        for i in xrange(looper):		# for each batch_i file....
            if verbose >= 3:
                print("... Training batch " + str(i))
            data_x, data_y  = load_skdata_caltech101(
                                    n_train_images = n_train_images,
                                    n_test_images = n_test_images,
                                    n_valid_images = n_valid_images,
                                    batch_size = self.mini_batches_per_batch[0] * \
                                                        self.mini_batch_size,
                                    rand_perm = self.rand_perm,
                                    batch = i ,
                                    type_set = 'train',
                                    height = self.height,
                                    width = self.width,
                                    verbose = verbose )
            data_x = preprocessing ( data_x,
                                    self.height,
                                    self.width,
                                    self.channels,
                                    self.preprocessor )
            f = open(self.root + "/train/" + 'batch_' + str(i) + '.pkl', 'wb')
            obj = (data_x, data_y )
            cPickle.dump(obj, f, protocol=2)
            f.close()

        if verbose >=2:
            print(".. Testing data")
        looper = n_test_images / ( self.mini_batches_per_batch[1] * self.mini_batch_size )
        for i in xrange(looper):		# for each batch_i file....
            if verbose >= 3:
                print("... Testing batch " + str(i))
            data_x, data_y  = load_skdata_caltech101(
                                    n_train_images = n_train_images,
                                    n_test_images = n_test_images,
                                    n_valid_images = n_valid_images,
                                    batch_size = self.mini_batches_per_batch[1] * \
                                                    self.mini_batch_size,
                                    rand_perm = self.rand_perm,
                                    batch = i ,
                                    type_set = 'test' ,
                                    height = self.height,
                                    width = self.width,
                                    verbose = verbose )
            data_x = preprocessing ( data_x,
                                self.height,
                                self.width,
                                self.channels,
                                self.preprocessor )
            f = open(self.root + "/test/" + 'batch_' + str(i) + '.pkl', 'wb')
            obj = (data_x, data_y )
            cPickle.dump(obj, f, protocol=2)
            f.close()

        if verbose >=2:
                print(".. Validation data")
        looper = n_valid_images / ( self.mini_batches_per_batch[2] * self.mini_batch_size )
        for i in xrange(looper):		# for each batch_i file....
            if verbose >= 3:
                    print("... Validation batch " + str(i))
            data_x, data_y  = load_skdata_caltech101(
                                            n_train_images = n_train_images,
                                            n_test_images = n_test_images,
                                            n_valid_images = n_valid_images,
                                            batch_size = self.mini_batches_per_batch[2] * \
                                                          self.mini_batch_size,
                                            rand_perm = self.rand_perm,
                                            batch = i ,
                                            type_set = 'valid' ,
                                            height = self.height,
                                            width = self.width,
                                            verbose = verbose  )
            data_x = preprocessing ( data_x,
                                     self.height,
                                     self.width,
                                     self.channels,
                                     self.preprocessor )
            f = open(self.root + "/valid/" + 'batch_' + str(i) + '.pkl', 'wb')
            obj = (data_x, data_y )
            cPickle.dump(obj, f, protocol=2)
            f.close()

        assert ( self.height * self.width * self.channels == numpy.prod(data_x.shape[1:]) )
        data_args = {
            "location"                  : self.root,
            "mini_batch_size"           : self.mini_batch_size,
            "cache_batches"             : self.mini_batches_per_batch,
            "batches2train"             : self.batches2train,
            "batches2test"              : self.batches2test,
            "batches2validate"          : self.batches2validate,
            "height"                    : self.height,
            "width"                     : self.width,
            "channels"              : 1 if self.preprocessor ["grayscale"] else self.channels,
            "cache"                 : self.cache,
            }

        f = open(self.root +  '/data_params.pkl', 'wb')
        cPickle.dump(data_args, f, protocol=2)
        f.close()

    def _create_skdata_caltech256(self, verbose = 2):
        """
        Interal function. Use this to create mnist and caltech101 image datasets
        """
        # shuffle the data
        total_images_in_dataset = 30607
        self.rand_perm = numpy.random.permutation(total_images_in_dataset)
        # create a constant shuffle, so that data can be loaded in batchmode with the same
        # random shuffle

        n_train_images = self.mini_batches_per_batch[0]*self.mini_batch_size * self.batches2train
        n_test_images = self.mini_batches_per_batch[2]*self.mini_batch_size * self.batches2test
        n_valid_images = self.mini_batches_per_batch[1]*self.mini_batch_size * self.batches2validate

        assert n_valid_images + n_train_images + n_test_images <= total_images_in_dataset

        if verbose >=2:
            print(".. Setting up dataset")
            print(".. Training data")

        looper = n_train_images / ( self.mini_batches_per_batch[0] * self.mini_batch_size )

        for i in xrange(looper):		# for each batch_i file....
            if verbose >= 3:
                print("... Training batch " + str(i))
            data_x, data_y  = load_skdata_caltech256(
                                    n_train_images = n_train_images,
                                    n_test_images = n_test_images,
                                    n_valid_images = n_valid_images,
                                    batch_size = self.mini_batches_per_batch[0] * \
                                                        self.mini_batch_size,
                                    rand_perm = self.rand_perm,
                                    batch = i ,
                                    type_set = 'train',
                                    height = self.height,
                                    width = self.width,
                                    verbose = verbose )
            data_x = preprocessing ( data_x,
                                    self.height,
                                    self.width,
                                    self.channels,
                                    self.preprocessor )
            f = open(self.root + "/train/" + 'batch_' + str(i) + '.pkl', 'wb')
            obj = (data_x, data_y )
            cPickle.dump(obj, f, protocol=2)
            f.close()

        if verbose >=2:
            print(".. Testing data")
        looper = n_test_images / ( self.mini_batches_per_batch[1] * self.mini_batch_size )
        for i in xrange(looper):		# for each batch_i file....
            if verbose >= 3:
                print("... Testing batch " + str(i))
            data_x, data_y  = load_skdata_caltech256(
                                    n_train_images = n_train_images,
                                    n_test_images = n_test_images,
                                    n_valid_images = n_valid_images,
                                    batch_size = self.mini_batches_per_batch[1] * \
                                                    self.mini_batch_size,
                                    rand_perm = self.rand_perm,
                                    batch = i ,
                                    type_set = 'test' ,
                                    height = self.height,
                                    width = self.width,
                                    verbose = verbose )
            data_x = preprocessing ( data_x,
                                self.height,
                                self.width,
                                self.channels,
                                self.preprocessor )
            f = open(self.root + "/test/" + 'batch_' + str(i) + '.pkl', 'wb')
            obj = (data_x, data_y )
            cPickle.dump(obj, f, protocol=2)
            f.close()

        if verbose >=2:
                print(".. Validation data")
        looper = n_valid_images / ( self.mini_batches_per_batch[2] * self.mini_batch_size )
        for i in xrange(looper):		# for each batch_i file....
            if verbose >= 3:
                    print("... Validation batch " + str(i))
            data_x, data_y  = load_skdata_caltech256(
                                            n_train_images = n_train_images,
                                            n_test_images = n_test_images,
                                            n_valid_images = n_valid_images,
                                            batch_size = self.mini_batches_per_batch[2] * \
                                                          self.mini_batch_size,
                                            rand_perm = self.rand_perm,
                                            batch = i ,
                                            type_set = 'valid' ,
                                            height = self.height,
                                            width = self.width,
                                            verbose = verbose  )
            data_x = preprocessing ( data_x,
                                     self.height,
                                     self.width,
                                     self.channels,
                                     self.preprocessor )
            f = open(self.root + "/valid/" + 'batch_' + str(i) + '.pkl', 'wb')
            obj = (data_x, data_y )
            cPickle.dump(obj, f, protocol=2)
            f.close()

        assert ( self.height * self.width * self.channels == numpy.prod(data_x.shape[1:]) )
        data_args = {
            "location"                  : self.root,
            "mini_batch_size"           : self.mini_batch_size,
            "cache_batches"             : self.mini_batches_per_batch,
            "batches2train"             : self.batches2train,
            "batches2test"              : self.batches2test,
            "batches2validate"          : self.batches2validate,
            "height"                    : self.height,
            "width"                     : self.width,
            "channels"              : 1 if self.preprocessor ["grayscale"] else self.channels,
            "cache"                 : self.cache,
            }

        f = open(self.root +  '/data_params.pkl', 'wb')
        cPickle.dump(data_args, f, protocol=2)
        f.close()

if __name__ == '__main__':
    pass
