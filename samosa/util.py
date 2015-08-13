#!/usr/bin/python

# General Packages
import numpy

# Math Packages
import gzip
import cPickle
import cv2, cv
from random import randint

# Theano Packages
import theano
import theano.tensor as T


## Visualizing utilities for CNN. 
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
        cv2.imshow(filename + str(randint(0,9)), filters_as_image)
    cv2.imwrite(loc + filename, filters_as_image)

# in case of filters that are color in the first layer, visuailize them as color images. 
def visualize_color_filters(imgs, tile_shape = None, tile_spacing = (2,2), 
                loc = '../results/', filename = "activities.jpg" , show_img = True ):
    img_shape = (imgs.shape[2], imgs.shape[3])
    if tile_shape is None:
        tile_shape = (numpy.asarray(numpy.ceil(numpy.sqrt(imgs.shape[0])), dtype='int32'), numpy.asarray(imgs.shape[0]/numpy.ceil(numpy.sqrt(imgs.shape[0])), dtype='int32') )
    #flattened_imgs = []
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)]
    filters_as_image = numpy.zeros((out_shape[0],out_shape[1],imgs.shape[1]))
    for i in xrange(imgs.shape[1]):     # only allow  channel color images .
        flattened_imgs  = numpy.reshape(imgs[:,i,:,:],(imgs.shape[0],numpy.prod(imgs.shape[2:])))
        filters_as_image[:,:,i] = tile_raster_images(X =flattened_imgs, img_shape = img_shape, tile_shape = tile_shape, tile_spacing = (2,2))        
    if show_img is True:
        cv2.imshow(filename + str(randint(0,9)), filters_as_image)
    cv2.imwrite(loc + filename, filters_as_image)
    
    

## Pickling utilities to pickle down and sterilize the entire network.
# function to load saved down data previously. Assumes that the first two elements are definitely params and arch_params.
def load_network(filename, data_params = False, optimization_params = False):
    
    f = gzip.open(filename, 'rb')
    params = cPickle.load(f)
    arch_params = cPickle.load(f)
    
    rval = (params, arch_params)
    if data_params is True:
        data_params_out = cPickle.load(f)
        rval = rval + (data_params_out,)
    if optimization_params is True:
        optimization_params = cPickle.load(f)
        rval = rval + (optimization_params,)
    return rval 
