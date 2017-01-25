import numpy 

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

                    "normalize" : <bool> True for normalize across batches makes image go from 0 - 1
                    "ZCA"	: True, kind of like a PCA representation (not fully tested)
                    "grayscale"  : Convert the image to grayscale
                    "zero_mean" : Subtracts the mean of the image. 

                        } 

    Returns:
        numpy ndarray: data 

    """
    normalize 		= args [ "normalize" ]
    ZCA 			= args [ "ZCA" ]	 
    gray 			= args [ "grayscale" ]
    zero_mean	    = args [ "zero_mean" ]

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

    if normalize is True or ZCA is True:
        data = data / (data.max() + 1e-7)

    if normalize is True and zero_mean is True:
        data = ( data - 0.5 ) * 2
    elif normalize is True and zero_mean is False:
        data = (data - data.mean())

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