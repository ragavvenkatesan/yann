#!/usr/bin/python

# General Packages
import os
import sys
import time
import numpy
import pdb
from collections import OrderedDict

# Math Packages
import math
import scipy.io
import gzip
import cPickle
import cv2, cv
from random import randint

# Theano Packages
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.ifelse import ifelse


##################################
## Various activation functions ##
##################################

#### rectified linear unit
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
#### sigmoid
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
#### tanh
def Tanh(x):
    y = T.tanh(x)
    return(y)
    
def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    #This code is from theao utilities 
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array

# takes a numpy array of shape (n_imgs, height, width)
def visualize(imgs, tile_shape = None, tile_spacing = (2,2), 
                loc = '../results/', filename = "activities.jpg" , show_img = True ):
    img_shape = (imgs.shape[1], imgs.shape[2])
    if tile_shape is None:
        tile_shape = (numpy.asarray(numpy.ceil(numpy.sqrt(imgs.shape[0])), dtype='int32'), numpy.asarray(imgs.shape[0]/numpy.ceil(numpy.sqrt(imgs.shape[0])), dtype='int32') )
    flattened_imgs = numpy.reshape(imgs,(imgs.shape[0],numpy.prod(imgs.shape[1:])))
    filters_as_image = tile_raster_images(X =flattened_imgs, img_shape = img_shape, tile_shape = tile_shape, tile_spacing = (2,2))
    if show_img is True:
        cv2.imshow(loc + filename + str(randint(0,9)), filters_as_image)
    cv2.imwrite(loc + filename, filters_as_image)

# SVM layer from the discussions in this group
# https://groups.google.com/forum/#!msg/theano-users/on4D16jqRX8/IWGa-Gl07g0J
class SVMLayer(object):

    def __init__(self, input, n_in, n_out, W=None, b=None):


        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        else:
            self.W = W
        # initialize the baises b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)
        else:
            self.b =b

        # parameters of the model
        self.params = [self.W, self.b]
        self.output = T.dot(input, self.W) + self.b
        self.y_pred = T.argmax(self.output, axis=1)

    def hinge(self, u):
            return T.maximum(0, 1 - u)

    def ova_svm_cost(self, y1):
        """ return the one-vs-all svm cost
        given ground-truth y in one-hot {-1, 1} form """
        y1_printed = theano.printing.Print('this is important')(T.max(y1))
        margin = y1 * self.output
        cost = self.hinge(margin).mean(axis=0).sum()
        return cost


    def errors(self, y):
        """ compute zero-one loss
        note, y is in integer form, not one-hot
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.sum(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


# Modified From https://github.com/mdenil/dropout/blob/master/mlp.py
class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None ):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                    value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                    name='W')
        else:
            self.W = W

        # initialize the baises b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(
                    value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                    name='b')
        else:
            self.b = b

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]
        self.probabilities = T.log(self.p_y_given_x)

    def negative_log_likelihood(self, y, regularize_l1 = False, regularize_l2 = False):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        """
 
        rval = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) 
        if regularize_l1 is True: 
            rval = rval + T.sum(abs(self.params[0]))
        if regularize_l2 is True:
            rval = rval + T.sum(self.params[0]**2)
        return rval

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch ;
        zero one loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.sum(T.neq(self.y_pred, y))      # L1 norm of the error. 
        else:
            raise NotImplementedError()

    
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None,
                 use_bias=False):

        self.input = input
        self.activation = activation

        if W is None:
            W_values = numpy.asarray(0.01 * rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')
        
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))
    
        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]


def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, use_bias=use_bias)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


class MLP(object):
    """A multilayer perceptron with all the trappings required to do dropout
    training.

    """
    def __init__(self,
            rng,
            input,
            layer_sizes,
            dropout_rates,
            activations,
            use_bias=True,
            svm_flag = True,
            verbose = True):


        #rectified_linear_activation = lambda x: T.maximum(0.0, x)
        # Set up all the hidden layers
        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        next_layer_input = input
        #first_layer = True
        # dropout the input
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[0])
        layer_counter = 0        
        for n_in, n_out in weight_matrix_sizes[:-1]:
            if verbose is True:
                print "           -->        Initializing MLP Layer with " + str(n_out) + " hidden units taking in input size " + str(n_in)

            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                    input=next_dropout_layer_input,
                    activation=activations[layer_counter],
                    n_in=n_in, n_out=n_out, use_bias=use_bias,
                    dropout_rate=dropout_rates[layer_counter + 1])
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            # Reuse the paramters from the dropout layer here, in a different
            # path through the graph.
            next_layer = HiddenLayer(rng=rng,
                    input=next_layer_input,
                    activation=activations[layer_counter],
                    # scale the weight matrix W with (1-p)
                    W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
                    b=next_dropout_layer.b,
                    n_in=n_in, n_out=n_out,
                    use_bias=use_bias)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            #first_layer = False
            layer_counter += 1
        
        # Set up the output layer
        n_in, n_out = weight_matrix_sizes[-1]
        

        # Again, reuse paramters in the dropout output.
    
        if svm_flag is False:
            if verbose is True:
                print "           -->        Initializing regression layer with " + str(n_out) + " output units"
            dropout_output_layer = LogisticRegression(
                input=next_dropout_layer_input,
                n_in=n_in, n_out=n_out)

            output_layer = LogisticRegression(
                input=next_layer_input,
                # scale the weight matrix W with (1-p)
                W=dropout_output_layer.W * (1 - dropout_rates[-1]),
                b=dropout_output_layer.b,
                n_in=n_in, n_out=n_out)

            self.layers.append(output_layer)
            self.dropout_layers.append(dropout_output_layer)

            self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
            self.negative_log_likelihood = self.layers[-1].negative_log_likelihood

        else:
            if verbose is True:
                print "           -->        Initializing SVM layer with " + str(n_out) + " class predictors"
            dropout_output_layer = SVMLayer(
                input=next_dropout_layer_input,
                n_in=n_in, n_out=n_out)

            output_layer = SVMLayer(input = next_layer_input,
                                    W=dropout_output_layer.W * (1 - dropout_rates[-1]),
                                    b=dropout_output_layer.b,
                                    n_in = n_in,
                                    n_out = n_out)

            self.layers.append(output_layer)
            self.dropout_layers.append(dropout_output_layer)

            self.dropout_negative_log_likelihood = self.dropout_layers[-1].ova_svm_cost
            self.negative_log_likelihood = self.layers[-1].ova_svm_cost

        
        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        
        self.dropout_errors = self.dropout_layers[-1].errors
        self.errors = self.layers[-1].errors

        self.predicts_dropouts = self.layers[-1].y_pred
        self.predicts = self.layers[-1].y_pred
        self.params = [ param for layer in self.dropout_layers for param in layer.params ]

        if svm_flag is True:
            self.probabilities = self.layers[-1].output
        else:
            self.probabilities = self.layers[-1].probabilities            

        # Grab all the parameters together.
        #self.filter_images = T.reshape(self.W, (filter_shape[0], filter_shape[1], numpy.prod(filter_shape[2:])))

       
       
# From theano tutorials
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network .. taken from the theano tutorials"""

    def __init__(self, rng, input, filter_shape, image_shape, poolsize, activation, verbose = True):
       
        assert image_shape[1] == filter_shape[1]
        if verbose is True:
            print "           -->        Initializing Convolutional Layer with " + str(filter_shape[0])  + " kernels"
            print "                                  ....... Kernel size [" + str(filter_shape[2]) + " X " + str(filter_shape[3]) +"]"
            print "                                  ....... Pooling size [" + str(poolsize[0]) + " X " + str(poolsize[1]) + "]"
            print "                                  ....... Input size ["  + str(image_shape[2]) + " " + str(image_shape[3]) + "]"
            #print "                                  ....... Input number of feature maps is " +str(image_shape[1]) 
         
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size =filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width     & height

        self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        # store parameters of this layer
        self.params = [self.W, self.b]

        self.img_shape = (filter_shape[2], filter_shape[3])
        self.tile_shape = (numpy.asarray(numpy.ceil(numpy.sqrt(filter_shape[0]*filter_shape[1])), dtype='int32'), 
                            numpy.asarray(filter_shape[0]*filter_shape[1]/numpy.ceil(filter_shape[0]*filter_shape[1]), dtype='int32') )
        self.filter_img = self.W.reshape((filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]))

    def get_reconstructed_input(self, hidden):
        """ Computes the reconstructed input given the values of the hidden layer """
        repeated_conv = conv.conv2d(input = hidden, filters = self.W_prime, border_mode='full')
        multiple_conv_out = [repeated_conv.flatten()] * numpy.prod(self.poolsize)
        stacked_conv_neibs = T.stack(*multiple_conv_out).T
        stretch_unpooling_out = neibs2images(stacked_conv_neibs, self.pl, self.x.shape) 
        return ReLU(stretch_unpooling_out + self.b_prime.dimshuffle('x', 0, 'x', 'x'))
            



