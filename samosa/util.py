#!/usr/bin/python
import numpy
import gzip
import cPickle
import cv2


# Pylear pylearn_visualization
from pylearn2.gui.patch_viewer import make_viewer
    
    
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
    
# Concatenate three (height, width)s into one (height, width, 3).
def gray2rgb(r, g, b):
    assert r.ndim == 2 and g.ndim == 2 and b.ndim == 2
    rgb = (r[..., numpy.newaxis], g[..., numpy.newaxis], b[..., numpy.newaxis])
    return numpy.concatenate(rgb, axis=-1)
    
def visualize (imgs, prefix , is_color = False ):          
    raster = []
    count = 0 
    if is_color is True and imgs.shape[3] % 3 != 0:
        filts = numpy.floor( imgs.shape[3] / 3)
        imgs = imgs[:,:,:,0:filts]
    
    for i in xrange (imgs.shape[3]):
        curr_image = imgs[:,:,:,i]
        if is_color is True:
            raster.append(rgb2gray(numpy.array(make_viewer( curr_image.reshape((curr_image.shape[0],curr_image.shape[1] * curr_image.shape[2])), is_color = False ).get_img())))
            if count == 2:          
                cv2.imwrite(prefix + str(i) + ".jpg", gray2rgb(raster[i-2],raster[i-1],raster[i]) )
                count = -1                            
        else:   
            raster.append(numpy.array(make_viewer( curr_image.reshape((curr_image.shape[0],curr_image.shape[1] * curr_image.shape[2])), is_color = False ).get_img()))             
            cv2.imwrite(prefix + str(i) + ".jpg",raster[i])
            
        count = count + 1
    return raster
        
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
