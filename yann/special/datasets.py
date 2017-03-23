from yann.utils.dataset import *
import numpy

class combine_split_datasets (object):

    def __init__(self, loc, verbose = 1, **kwargs):
    """
    This will combine two split datasets into one.

    Todo:
        Extend it for non-split datasets also.

    Args:
        loc: A tuple of a list of locations of two dataset to be blended.
        verbose: As always
    """        
        self.loc1 = loc[0]
        self.loc2 - loc[1]

        if verbose >= 3:
            print("... Initializing dataset 1")

        f = open(loc1 + '/data_params.pkl', 'rb')
        data_params_1 = cPickle.load(f)
        f.close()    

        data_splits_1 = data_params_1['splits']
        if data_splits_1['p'] == 0:
            self.n_classes_1 = len (data_splits_1['base'])
        else:
            self.n_classes_1 = len (data_splits_1['base']) + len(data_splits_1['shot'])

        self.mini_batches_per_batch_1  = data_params_1 [ "cache_batches" ]
        self.batches2train_1       = data_params_1 [ "batches2train" ]
        self.batches2test_1      = data_params_1 [ "batches2test" ]
        self.batches2validate_1    = data_params_1 [ "batches2validate" ]
        self.cache_1               = data_params_1 [ "cache" ]

        if verbose >= 3:
            print("... Initializing dataset 2")

        f = open(loc1 + '/data_params.pkl', 'rb')
        data_params_2 = cPickle.load(f)
        f.close()  

        data_splits_2 = data_params_2['splits']
        if data_splits_2['p'] == 0:
            self.n_classes_2 = len (data_splits_2['base'])
        else:
            self.n_classes_2 = len (data_splits_2['base']) + len(data_splits_2['shot'])

        self.mini_batches_per_batch_2  = data_params_2 [ "cache_batches" ]
        self.batches2train_2       = data_params_2 [ "batches2train" ]
        self.batches2test_2      = data_params_2 [ "batches2test" ]
        self.batches2validate_2    = data_params_2 [ "batches2validate" ]
        self.cache_2               = data_params_2 [ "cache" ]

        # Why is this necessary ?
        assert data_params_1 [ "mini_batch_size" ] == data_params_2 [ "mini_batch_size" ]
        self.mini_batch_size = data_params_1['mini_batch_size']

        assert data_params_1 [ "height" ] == data_params_2 [ "height" ]      
        self.height              = data_params_1 [ "height" ]

        assert data_params_1 [ "width" ] == data_params_2 [ "width" ]        
        self.width               = data_params_1 [ "width" ]

        assert data_params_1 [ "channels" ] == data_params_2 [ "channels" ]        
        self.channels            = data_params_1 [ "channels" ]

def cook_mnist_normalized(  verbose = 1, **kwargs):
    """
    Wrapper to cook mnist dataset. Will take as input,

    Args:

        save_directory: which directory to save the cooked dataset onto.
        dataset_parms: default is the dictionary. Refer to :mod:`setup_dataset`
        preprocess_params: default is the dictionary. Refer to :mod:`setup_dataset`

    Notes:
        By default, this will create a dataset that is not mean-subtracted.
    """

    if not 'data_params' in kwargs.keys():

        data_params = {
                   "source"             : 'skdata',
                   "name"               : 'mnist',
                   "location"			: '',
                   "mini_batch_size"    : 500,
                   "mini_batches_per_batch" : (100, 20, 20),
                   "batches2train"      : 1,
                   "batches2test"       : 1,
                   "batches2validate"   : 1,
                   "height"             : 28,
                   "width"              : 28,
                   "channels"           : 1  }

    else:
        data_params = kwargs['data_params']

    if not 'preprocess_params' in kwargs.keys():

    # parameters relating to preprocessing.
        preprocess_params = {
                            "normalize"     : True,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"     : False,
                        }
    else:
        preprocess_params = kwargs['preprocess_params']

    if not 'save_directory' in kwargs.keys():
        save_directory = '_datasets'
    else:
        save_directory = kwargs ['save_directory']

    dataset = setup_dataset(dataset_init_args = data_params,
                            save_directory = save_directory,
                            preprocess_init_args = preprocess_params,
                            verbose = 3)
    return dataset

def cook_mnist_normalized_zero_mean(  verbose = 1,	**kwargs):
    """
    Wrapper to cook mnist dataset. Will take as input,

    Args:

        save_directory: which directory to save the cooked dataset onto.
        dataset_parms: default is the dictionary. Refer to :mod:`setup_dataset`
        preprocess_params: default is the dictionary. Refer to :mod:`setup_dataset`
    """

    if not 'data_params' in kwargs.keys():

        data_params = {
                   "source"             : 'skdata',
                   "name"               : 'mnist',
                   "location"			: '',
                   "mini_batch_size"    : 500,
                   "mini_batches_per_batch" : (100, 20, 20),
                   "batches2train"      : 1,
                   "batches2test"       : 1,
                   "batches2validate"   : 1,
                   "height"             : 28,
                   "width"              : 28,
                   "channels"           : 1  }

    else:
        data_params = kwargs['data_params']

    if not 'preprocess_params' in kwargs.keys():

    # parameters relating to preprocessing.
        preprocess_params = {
                            "normalize"     : True,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"     : True,
                        }
    else:
        preprocess_params = kwargs['preprocess_params']

    if not 'save_directory' in kwargs.keys():
        save_directory = '_datasets'
    else:
        save_directory = kwargs ['save_directory']

    dataset = setup_dataset(dataset_init_args = data_params,
                            save_directory = save_directory,
                            preprocess_init_args = preprocess_params,
                            verbose = 3)
    return dataset

def cook_mnist_multi_load(  verbose = 1, **kwargs):
    """
    Testing code, mainly.
    Wrapper to cook mnist dataset. Will take as input,

    Args:

        save_directory: which directory to save the cooked dataset onto.
        dataset_parms: default is the dictionary. Refer to :mod:`setup_dataset`
        preprocess_params: default is the dictionary. Refer to :mod:`setup_dataset`

    Notes:
        This just creates a ``data_params`` that loads multiple batches without cache. I use this
        to test the cahcing working on datastream module.
    """

    if not 'data_params' in kwargs.keys():

        data_params = {
                   "source"             : 'skdata',
                   "name"               : 'mnist',
                   "location"			: '',
                   "mini_batch_size"    : 500,
                   "mini_batches_per_batch" : (20, 5, 5),
                   "batches2train"      : 5,
                   "batches2test"       : 4,
                   "batches2validate"   : 4,
                   "height"             : 28,
                   "width"              : 28,
                   "channels"           : 1 }

    else:
        data_params = kwargs['data_params']

    if not 'preprocess_params' in kwargs.keys():
        # parameters relating to preprocessing.
        preprocess_params = {
                            "normalize"     : True,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"     : False,
                        }
    else:
        preprocess_params = kwargs['preprocess_params']

    if not 'save_directory' in kwargs.keys():
        save_directory = '_datasets'
    else:
        save_directory = kwargs ['save_directory']

    dataset = setup_dataset(dataset_init_args = data_params,
                            save_directory = save_directory,
                            preprocess_init_args = preprocess_params,
                            verbose = 3)
    return dataset

def cook_cifar10_normalized(verbose = 1, **kwargs):
    """
    Wrapper to cook cifar10 dataset. Will take as input,

    Args:

        save_directory: which directory to save the cooked dataset onto.
        dataset_parms: default is the dictionary. Refer to :mod:`setup_dataset`
        preprocess_params: default is the dictionary. Refer to :mod:`setup_dataset`
    """

    if not 'data_params' in kwargs.keys():

        data_params = {
                   "source"             : 'skdata',
                   "name"               : 'cifar10',
                   "location"			: '',
                   "mini_batch_size"    : 500,
                   "mini_batches_per_batch" : (80, 20, 20),
                   "batches2train"      : 1,
                   "batches2test"       : 1,
                   "batches2validate"   : 1,
                   "height"             : 32,
                   "width"              : 32,
                   "channels"           : 3  }

    else:
        data_params = kwargs['data_params']

    if not 'preprocess_params' in kwargs.keys():

    # parameters relating to preprocessing.
        preprocess_params = {
                            "normalize"     : True,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"     : False,
                        }
    else:
        preprocess_params = kwargs['preprocess_params']

    if not 'save_directory' in kwargs.keys():
        save_directory = '_datasets'
    else:
        save_directory = kwargs ['save_directory']

    dataset = setup_dataset(dataset_init_args = data_params,
                            save_directory = save_directory,
                            preprocess_init_args = preprocess_params,
                            verbose = 3)
    return dataset

def cook_caltech101(verbose = 1, **kwargs):
    """
    Wrapper to cook cifar10 dataset. Will take as input,

    Args:

        save_directory: which directory to save the cooked dataset onto.
        dataset_parms: default is the dictionary. Refer to :mod:`setup_dataset`
        preprocess_params: default is the dictionary. Refer to :mod:`setup_dataset`
    """

    if not 'data_params' in kwargs.keys():

        data_params = {
                    "source"             : 'skdata',
                    "name"               : 'caltech101',
                    "location"           : '',
                    "mini_batch_size"    : 72,
                    "mini_batches_per_batch" : (1, 1, 1),
                    "batches2train"      : 60,
                    "batches2test"       : 37,
                    "batches2validate"   : 30,
                    "height"             : 224,
                    "width"              : 224,
                    "channels"           : 3  }

    else:
        data_params = kwargs['data_params']

    if not 'preprocess_params' in kwargs.keys():

        # parameters relating to preprocessing.
        preprocess_params = {
                            "normalize"     : False,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"     : False,
                        }
    else:
        preprocess_params = kwargs['preprocess_params']

    if not 'save_directory' in kwargs.keys():
        save_directory = '_datasets'
    else:
        save_directory = kwargs ['save_directory']

    dataset = setup_dataset(dataset_init_args = data_params,
                        save_directory = save_directory,
                        preprocess_init_args = preprocess_params,
                        verbose = 3)
    return dataset

def cook_caltech256(verbose = 1, **kwargs):
    """
    Wrapper to cook cifar10 dataset. Will take as input,

    Args:

        save_directory: which directory to save the cooked dataset onto.
        dataset_parms: default is the dictionary. Refer to :mod:`setup_dataset`
        preprocess_params: default is the dictionary. Refer to :mod:`setup_dataset`
    """

    if not 'data_params' in kwargs.keys():

        data_params = {
                    "source"             : 'skdata',
                    "name"               : 'caltech256',
                    "location"           : '',
                    "mini_batch_size"    : 36,
                    "mini_batches_per_batch" : (1, 1, 1),
                    "batches2train"      : 424,
                    "batches2test"       : 226,
                    "batches2validate"   : 200,
                    "height"             : 224,
                    "width"              : 224,
                    "channels"           : 3  }

    else:
        data_params = kwargs['data_params']

    if not 'preprocess_params' in kwargs.keys():

        # parameters relating to preprocessing.
        preprocess_params = {
                            "normalize"     : False,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"     : False,
                        }
    else:
        preprocess_params = kwargs['preprocess_params']

    if not 'save_directory' in kwargs.keys():
        save_directory = '_datasets'
    else:
        save_directory = kwargs ['save_directory']

    dataset = setup_dataset(dataset_init_args = data_params,
                        save_directory = save_directory,
                        preprocess_init_args = preprocess_params,
                        verbose = 3)
    return dataset

def cook_cifar10_normalized_zero_mean(verbose = 1, **kwargs):
    """
    Wrapper to cook cifar10 dataset. Will take as input,

    Args:

        save_directory: which directory to save the cooked dataset onto.
        dataset_parms: default is the dictionary. Refer to :mod:`setup_dataset`
        preprocess_params: default is the dictionary. Refer to :mod:`setup_dataset`
    """

    if not 'data_params' in kwargs.keys():

        data_params = {
                   "source"             : 'skdata',
                   "name"               : 'cifar10',
                   "location"			: '',
                   "mini_batch_size"    : 500,
                   "mini_batches_per_batch" : (80, 20, 20),
                   "batches2train"      : 1,
                   "batches2test"       : 1,
                   "batches2validate"   : 1,
                   "height"             : 32,
                   "width"              : 32,
                   "channels"           : 3  }

    else:
        data_params = kwargs['data_params']

    if not 'preprocess_params' in kwargs.keys():

    # parameters relating to preprocessing.
        preprocess_params = {
                            "normalize"     : True,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"     : True,
                        }
    else:
        preprocess_params = kwargs['preprocess_params']

    if not 'save_directory' in kwargs.keys():
        save_directory = '_datasets'
    else:
        save_directory = kwargs ['save_directory']

    dataset = setup_dataset(dataset_init_args = data_params,
                            save_directory = save_directory,
                            preprocess_init_args = preprocess_params,
                            verbose = 3)
    return dataset

cook_mnist = cook_mnist_normalized
cook_cifar10 = cook_cifar10_normalized

class split_all(setup_dataset):
    """
    Inheriting from the setup dataset. The new methods added will include the split. 
    """    
    def __init__(self,
                 dataset_init_args,
                 save_directory = '_datasets',
                 verbose = 0,
                 **kwargs): 
        """
        This is just a re-use of the setup_dataset. This will construct the split dataset. 

        Args:
            split_args: Is a dictionary of the form,
                    split_args =  { 
                            "base"              : [0,1,2,4,6,8,9],
                            "shot"              : [3,5,7],
                            "p"                 : 0
                            }   
        Notes:
            Arguments are the same as in the case of setup_dataset. With the addtion of one extra
            argument  - split_args.
        """

        if "preprocess_init_args" in kwargs.keys():
            self.preprocessor = kwargs['preprocess_init_args']
        else:
            self.preprocessor =  {
                            "normalize"     : True,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"		: True,
                            }        

        if "split_args" in kwargs.keys():
            self.splits = kwargs['split_args']
        else:
            self.splits = { 
                            "base"              : [0,1,2,3,4,5],
                            "shot"              : [6,7,8,9],
                            "p"                 : 0
                        } 

        self.n_classes = len(self.splits['base']) + len(self.splits['shot'])
        super(split_all,self).__init__(     dataset_init_args = dataset_init_args,
                                            save_directory = save_directory,
                                            preprocess_init_args = self.preprocessor,
                                            verbose = 1)

    def _mat2yann (self, verbose = 1):
        """
        Interal function. Use this to create matlab image datasets
        This is modfied for the split dataset from the original ``setup_dataset`` class.
        """
        if verbose >=2:
            print (".. Creating a split dataset")

        type = 'train'
        if verbose >= 2:
            print ( ".. creating data " + type )
        
        batches = self.batches2train
        new = True
        for batch in xrange(batches):		# for each batch_i file....
            if verbose >= 3:
                print ( "... batch " +str(batch) )

            data_x_batch, data_y_batch = load_data_mat(location = self.location, 
                                                batch = batch, 
                                                type_set = type,
                                                height = self.height,
                                                width = self.width,
                                                channels = self.channels)
            if new is True:
                data_x = data_x_batch
                data_y = data_y_batch
                new = False
            else:
                data_x = numpy.concatenate( (data_x, data_x_batch) , axis = 0)
                data_y = numpy.concatenate( (data_y, data_y_batch) , axis = 0)

        data_x, data_y  = self._split_data (( data_x, data_y ), y1 = False )
        data_x = preprocessing ( data = data_x,
                                height = self.height,
                                width = self.width,
                                channels = self.channels,
                                args = self.preprocessor )

        training_sample_size = data_x.shape[0]
        training_mini_batches_available  = int(numpy.floor(training_sample_size / \
                                                                             self.mini_batch_size))

        if training_mini_batches_available < self.batches2train * self.mini_batches_per_batch[0]:
            #self.mini_batches_per_batch = ( training_batches_available/self.batches2train,
            #                                self.mini_batches_per_batch [1],
            #                                self.mini_batches_per_batch [2] )
            self.batches2train = int(numpy.floor(training_mini_batches_available / \
                                                                    self.mini_batches_per_batch[0]))         

        loc = self.root + "/train/"
        data_x = check_type(data_x, theano.config.floatX)
        data_y = check_type(data_y, theano.config.floatX)

        for batch in xrange(self.batches2train):
            start_index = batch * self.cache_images[0]
            end_index = start_index + self.cache_images[0]
            data2save = (data_x [start_index:end_index,], data_y[start_index:end_index,] )
            pickle_dataset(loc = loc, data = data2save, batch=batch)

            
        type = 'valid'
        if verbose >= 2:
            print ( ".. creating data " + type )
        batches = self.batches2validate
        new = True
        del(data_x)
        del(data_y)

        for batch in xrange(batches):		# for each batch_i file....
            if verbose >= 3:
                print ( "... batch " +str(batch) )

            data_x_batch, data_y_batch = load_data_mat(location = self.location, 
                                                batch = batch, 
                                                type_set = type,
                                                height = self.height,
                                                width = self.width,
                                                channels = self.channels)

            if new is True:
                data_x = data_x_batch
                data_y = data_y_batch
                new = False
            else:
                data_x = numpy.concatenate( (data_x, data_x_batch) , axis = 0)
                data_y = numpy.concatenate( (data_y, data_y_batch) , axis = 0)

        data_x, data_y  = self._split_data (( data_x, data_y ), y1 = False )
        data_x = preprocessing ( data = data_x,
                                height = self.height,
                                width = self.width,
                                channels = self.channels,
                                args = self.preprocessor )


        validation_sample_size = data_x.shape[0]
        validation_mini_batches_available = int(numpy.floor(
                                                validation_sample_size / self.mini_batch_size))

        if validation_mini_batches_available < self.batches2validate * self.mini_batches_per_batch[1]:
            self.batches2validate = int(numpy.floor(validation_mini_batches_available \
                                    / self.mini_batches_per_batch[1]))

        loc = self.root + "/valid/"
        data_x = check_type(data_x, theano.config.floatX)
        data_y = check_type(data_y, theano.config.floatX)

        for batch in xrange(self.batches2validate):
            start_index = batch * self.cache_images[1]
            end_index = start_index + self.cache_images[1]
            data2save = (data_x [start_index:end_index,], data_y[start_index:end_index,] )
            pickle_dataset(loc = loc, data = data2save, batch=batch)

        type = 'train'
        if verbose >= 2:
            print ( ".. creating data " + type )
        batches = self.batches2test
        new = True
        del(data_x)
        del(data_y)

        for batch in xrange(batches):		# for each batch_i file....
            if verbose >= 3:
                print ( "... batch " +str(batch) )

            data_x_batch, data_y_batch = load_data_mat(location = self.location, 
                                                batch = batch, 
                                                type_set = type,
                                                height = self.height,
                                                width = self.width,
                                                channels = self.channels)
            if new is True:
                data_x = data_x_batch
                data_y = data_y_batch
                new = False
            else:
                data_x = numpy.concatenate( (data_x, data_x_batch) , axis = 0)
                data_y = numpy.concatenate( (data_y, data_y_batch) , axis = 0)

        data_x, data_y  = self._split_data (( data_x, data_y ), y1 = False )
        data_x = preprocessing ( data = data_x,
                                height = self.height,
                                width = self.width,
                                channels = self.channels,
                                args = self.preprocessor )

        testing_sample_size = data_x.shape[0]
        testing_mini_batches_available = int(numpy.floor(testing_sample_size / self.mini_batch_size))

        if testing_mini_batches_available < self.batches2test * self.mini_batches_per_batch[2]:
            self.batches2test = int(numpy.floor(testing_mini_batches_available \
                                    / self.mini_batches_per_batch[2]))

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
                "cache"                     : self.cache,
                "splits"                    : self.splits
                }

        assert ( self.height * self.width * self.channels == numpy.prod(data_x.shape[1:]) )
        f = open(self.root +  '/data_params.pkl', 'wb')
        cPickle.dump(dataset_args, f, protocol=2)
        f.close()

    def _create_skdata_mnist(self, verbose = 1):
        """
        Interal function. Use this to create mnist and cifar image datasets
        This is modfied for the split dataset from the original ``setup_dataset`` class.
        """
        if verbose >=2:
            print (".. Creating a split dataset")
        if verbose >=3:
            print("... Importing " + self.name + " from skdata")
        data = getattr(thismodule, 'load_skdata_' + self.name)()

        if verbose >=2:
            print(".. setting up dataset")
            print(".. training data")
        # Assuming this is num classes. Dangerous ?
        self.n_classes = data[0][1].max()

        data_x, data_y, data_y1 = self._split_data (data[0])
        data_x = preprocessing ( data = data_x,
                                 height = self.height,
                                 width = self.width,
                                 channels = self.channels,
                                 args = self.preprocessor )

        training_sample_size = data_x.shape[0]
        training_batches_available  = int(numpy.floor(training_sample_size / self.mini_batch_size))

        if training_batches_available < self.batches2train * self.mini_batches_per_batch[0]:
            self.mini_batches_per_batch = ( training_batches_available/self.batches2train,
                                            self.mini_batches_per_batch [1],
                                            self.mini_batches_per_batch [2] )

        if self.batches2train * self.mini_batches_per_batch[0] < self.cache_images[0]:
            self.cache_images = (self.mini_batches_per_batch[0] * self.mini_batch_size, \
                                        self.cache_images[1],  self.cache_images[2])

        data_x = data_x[:self.cache_images[0]]
        data_y = data_y[:self.cache_images[0]]                

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

        data_x, data_y, data_y1 = self._split_data (data[1])
        data_x = preprocessing ( data = data_x,
                                 height = self.height,
                                 width = self.width,
                                 channels = self.channels,
                                 args = self.preprocessor )

        validation_sample_size = data_x.shape[0]
        validation_batches_available = int(numpy.floor(
                                                validation_sample_size / self.mini_batch_size))

        if validation_batches_available < self.batches2validate * self.mini_batches_per_batch[1]:
            self.mini_batches_per_batch = ( self.mini_batches_per_batch [0],
                                            validation_batches_available/self.batches2validate,
                                            self.mini_batches_per_batch [2] )

        if self.batches2validate * self.mini_batches_per_batch[1] < self.cache_images[1]:
            self.cache_images = (   self.cache_images[0],\
                                    self.mini_batches_per_batch[1] * self.mini_batch_size, \
                                    self.cache_images[2])

        data_x = data_x[:self.cache_images[1]]
        data_y = data_y[:self.cache_images[1]]

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

        data_x, data_y, data_y1 = self._split_data(data[2])
        data_x = preprocessing ( data = data_x,
                                 height = self.height,
                                 width = self.width,
                                 channels = self.channels,
                                 args = self.preprocessor )
        testing_sample_size = data_x.shape[0]
        testing_batches_available = int(numpy.floor(testing_sample_size / self.mini_batch_size))

        if testing_batches_available < self.batches2test * self.mini_batches_per_batch[2]:
            self.mini_batches_per_batch = ( self.mini_batches_per_batch [0],
                                            self.mini_batches_per_batch [1],
                                            testing_batches_available/self.batches2test )

        if self.batches2test * self.mini_batches_per_batch[2] < self.cache_images[2]:
            self.cache_images = (   self.cache_images[0],\
                                    self.cache_images[1], \
                                    self.mini_batches_per_batch[2] * self.mini_batch_size )

        data_x = data_x[:self.cache_images[2]]
        data_y = data_y[:self.cache_images[2]]

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
                "cache"                     : self.cache,
                "splits"                    : self.splits
                }

        assert ( self.height * self.width * self.channels == numpy.prod(data_x.shape[1:]) )
        f = open(self.root +  '/data_params.pkl', 'wb')
        cPickle.dump(dataset_args, f, protocol=2)
        f.close()


    def _split_data (self, data, y1 = True):
        """
        This is an internal method that will split the datasets.

        Args:
            data: train, test and valid batches in a tuple.
        
        Returns:
            tuple: split data in the same format as data.
        """
        n_shots = self.splits["p"]
        if y1 is True:
            data_x, data_y, data_y1  = data
        else:
            data_x, data_y = data
        locs = numpy.zeros(len(data_y), dtype = bool)
        for label in xrange(self.n_classes + 1):
            temp = numpy.zeros(len(data_y), dtype = bool)                                                
            temp[data_y==label] = True
            if label in self.splits["shot"]:
                count = 0        
                for element in xrange(len(temp)):
                    if temp[element] == True:    # numpy needs == rather than 'is'               
                        count = count + 1
                    if count > n_shots:	                     	            
                        temp[element] = False	
                        		     
            locs[temp] = True
        data_x = data_x[locs]
        data_y = data_y[locs]
        if y1 is True:
            data_y1 = data_y1[locs]
            return (data_x, data_y, data_y1)  
        else:
            return (data_x, data_y)

class split_only_train(setup_dataset):
    
    """
    Inheriting from the split dataset. The new methods added will include the split. 
    """    
    def __init__(self,
                 dataset_init_args,
                 save_directory = '_datasets',
                 verbose = 0,
                 **kwargs): 
        """
        This is just a re-use of the setup_dataset. This will construct the split dataset. 

        Args:
            split_args: Is a dictionary of the form,
                splits = { 
                        "base"              : [6,7,8,9],
                        "shot"              : [0,1,2,3,4,5],
                        "p"                 : 0
                    }   
        Notes:
            Arguments are the same as in the case of setup_dataset. With the addtion of one extra
            argument  - split_args.
        """
        if "preprocess_init_args" in kwargs.keys():
            self.preprocessor = kwargs['preprocess_init_args']
        else:
            self.preprocessor =  {
                            "normalize"     : True,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"		: True,
                            }        

        if "split_args" in kwargs.keys():
            self.splits = kwargs['split_args']
        else:
            self.splits =  { 
                        "base"              : [6,7,8,9],
                        "shot"              : [0,1,2,3,4,5],
                        "p"                 : 0
                            }   

        self.n_classes = len(self.splits['base']) + len(self.splits['shot'])
        super(split_only_train,self).__init__( 
                                        dataset_init_args = dataset_init_args,
                                        save_directory = save_directory,
                                        preprocess_init_args = self.preprocessor,
                                        verbose = 1)

    def _create_skdata_mnist(self, verbose = 1):
        """
        Interal function. Use this to create mnist and cifar image datasets
        This is modfied for the split dataset from the original ``setup_dataset`` class.
        """
        if verbose >=2:
            print (".. Creating a split dataset")
        if verbose >=3:
            print("... Importing " + self.name + " from skdata")
        data = getattr(thismodule, 'load_skdata_' + self.name)()

        if verbose >=2:
            print(".. setting up dataset")
            print(".. training data")

        data_x, data_y, data_y1 = self._split_data (data[0])

        data_x = preprocessing ( data = data_x,
                                 height = self.height,
                                 width = self.width,
                                 channels = self.channels,
                                 args = self.preprocessor )

        training_sample_size = data_x.shape[0]
        training_batches_available  = int(numpy.floor(training_sample_size / self.mini_batch_size))

        if training_batches_available < self.batches2train * self.mini_batches_per_batch[0]:
            self.mini_batches_per_batch = ( training_batches_available/self.batches2train,
                                            self.mini_batches_per_batch [1],
                                            self.mini_batches_per_batch [2] )

        if self.batches2train * self.mini_batches_per_batch[0] < self.cache_images[0]:
            self.cache_images = (self.mini_batches_per_batch[0] * self.mini_batch_size, \
                                        self.cache_images[1],  self.cache_images[2])

        data_x = data_x[:self.cache_images[0]]
        data_y = data_y[:self.cache_images[0]]                

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

        data_x, data_y, data_y1 = data[1]
        data_x = preprocessing ( data = data_x,
                                 height = self.height,
                                 width = self.width,
                                 channels = self.channels,
                                 args = self.preprocessor )

        validation_sample_size = data_x.shape[0]
        validation_batches_available = int(numpy.floor(
                                                validation_sample_size / self.mini_batch_size))

        if validation_batches_available < self.batches2validate * self.mini_batches_per_batch[1]:
            self.mini_batches_per_batch = ( self.mini_batches_per_batch [0],
                                            validation_batches_available/self.batches2validate,
                                            self.mini_batches_per_batch [2] )

        if self.batches2validate * self.mini_batches_per_batch[1] < self.cache_images[1]:
            self.cache_images = (   self.cache_images[0],\
                                    self.mini_batches_per_batch[1] * self.mini_batch_size, \
                                    self.cache_images[2])

        data_x = data_x[:self.cache_images[1]]
        data_y = data_y[:self.cache_images[1]]

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
        testing_batches_available = int(numpy.floor(testing_sample_size / self.mini_batch_size))

        if testing_batches_available < self.batches2test * self.mini_batches_per_batch[2]:
            self.mini_batches_per_batch = ( self.mini_batches_per_batch [0],
                                            self.mini_batches_per_batch [1],
                                            testing_batches_available/self.batches2test )

        if self.batches2test * self.mini_batches_per_batch[2] < self.cache_images[2]:
            self.cache_images = (   self.cache_images[0],\
                                    self.cache_images[1], \
                                    self.mini_batches_per_batch[2] * self.mini_batch_size )

        data_x = data_x[:self.cache_images[2]]
        data_y = data_y[:self.cache_images[2]]

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
                "cache"                     : self.cache,
                "splits"                    : self.splits
                }

        assert ( self.height * self.width * self.channels == numpy.prod(data_x.shape[1:]) )
        f = open(self.root +  '/data_params.pkl', 'wb')
        cPickle.dump(dataset_args, f, protocol=2)
        f.close()        

    def _mat2yann (self, verbose = 1):
        """
        Interal function. Use this to create matlab image datasets
        This is modfied for the split dataset from the original ``setup_dataset`` class.
        """
        if verbose >=2:
            print (".. Creating a split dataset")

        type = 'train'
        if verbose >= 2:
            print ( ".. creating data " + type )
        
        batches = self.batches2train
        new = True
        for batch in xrange(batches):       # for each batch_i file....
            if verbose >= 3:
                print ( "... batch " +str(batch) )

            data_x_batch, data_y_batch = load_data_mat(location = self.location, 
                                                batch = batch, 
                                                type_set = type,
                                                height = self.height,
                                                width = self.width,
                                                channels = self.channels)
            if new is True:
                data_x = data_x_batch
                data_y = data_y_batch
                new = False
            else:
                data_x = numpy.concatenate( (data_x, data_x_batch) , axis = 0)
                data_y = numpy.concatenate( (data_y, data_y_batch) , axis = 0)

        data_x, data_y  = self._split_data (( data_x, data_y ), y1 = False )
        data_x = preprocessing ( data = data_x,
                                height = self.height,
                                width = self.width,
                                channels = self.channels,
                                args = self.preprocessor )

        training_sample_size = data_x.shape[0]
        training_mini_batches_available  = int(numpy.floor(training_sample_size / self.mini_batch_size))

        if training_mini_batches_available < self.batches2train * self.mini_batches_per_batch[0]:
            #self.mini_batches_per_batch = ( training_batches_available/self.batches2train,
            #                                self.mini_batches_per_batch [1],
            #                                self.mini_batches_per_batch [2] )
            self.batches2train = int(numpy.floor(training_mini_batches_available / self.mini_batches_per_batch[0]))         

        loc = self.root + "/train/"
        data_x = check_type(data_x, theano.config.floatX)
        data_y = check_type(data_y, theano.config.floatX)

        for batch in xrange(self.batches2train):
            start_index = batch * self.cache_images[0]
            end_index = start_index + self.cache_images[0]
            data2save = (data_x [start_index:end_index,], data_y[start_index:end_index,] )
            pickle_dataset(loc = loc, data = data2save, batch=batch)

            
        type = 'valid'
        if verbose >= 2:
            print ( ".. creating data " + type )
        batches = self.batches2validate
        new = True
        del(data_x)
        del(data_y)

        for batch in xrange(batches):       # for each batch_i file....
            if verbose >= 3:
                print ( "... batch " +str(batch) )

            data_x_batch, data_y_batch = load_data_mat(location = self.location, 
                                                batch = batch, 
                                                type_set = type,
                                                height = self.height,
                                                width = self.width,
                                                channels = self.channels)

            if new is True:
                data_x = data_x_batch
                data_y = data_y_batch
                new = False
            else:
                data_x = numpy.concatenate( (data_x, data_x_batch) , axis = 0)
                data_y = numpy.concatenate( (data_y, data_y_batch) , axis = 0)

        # data_x, data_y  = self._split_data (( data_x, data_y ), y1 = False )
        data_x = preprocessing ( data = data_x,
                                height = self.height,
                                width = self.width,
                                channels = self.channels,
                                args = self.preprocessor )


        validation_sample_size = data_x.shape[0]
        validation_mini_batches_available = int(numpy.floor(
                                                validation_sample_size / self.mini_batch_size))

        if validation_mini_batches_available < self.batches2validate * self.mini_batches_per_batch[1]:
            self.batches2validate = int(numpy.floor(validation_mini_batches_available \
                                    / self.mini_batches_per_batch[1]))

        loc = self.root + "/valid/"
        data_x = check_type(data_x, theano.config.floatX)
        data_y = check_type(data_y, theano.config.floatX)

        for batch in xrange(self.batches2validate):
            start_index = batch * self.cache_images[1]
            end_index = start_index + self.cache_images[1]
            data2save = (data_x [start_index:end_index,], data_y[start_index:end_index,] )
            pickle_dataset(loc = loc, data = data2save, batch=batch)

        type = 'train'
        if verbose >= 2:
            print ( ".. creating data " + type )
        batches = self.batches2test
        new = True
        del(data_x)
        del(data_y)

        for batch in xrange(batches):       # for each batch_i file....
            if verbose >= 3:
                print ( "... batch " +str(batch) )

            data_x_batch, data_y_batch = load_data_mat(location = self.location, 
                                                batch = batch, 
                                                type_set = type,
                                                height = self.height,
                                                width = self.width,
                                                channels = self.channels)
            if new is True:
                data_x = data_x_batch
                data_y = data_y_batch
                new = False
            else:
                data_x = numpy.concatenate( (data_x, data_x_batch) , axis = 0)
                data_y = numpy.concatenate( (data_y, data_y_batch) , axis = 0)

        # data_x, data_y  = self._split_data (( data_x, data_y ), y1 = False )
        data_x = preprocessing ( data = data_x,
                                height = self.height,
                                width = self.width,
                                channels = self.channels,
                                args = self.preprocessor )

        testing_sample_size = data_x.shape[0]
        testing_mini_batches_available = int(numpy.floor(testing_sample_size / self.mini_batch_size))

        if testing_mini_batches_available < self.batches2test * self.mini_batches_per_batch[2]:
            self.batches2test = int(numpy.floor(testing_mini_batches_available \
                                    / self.mini_batches_per_batch[2]))

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
                "cache"                     : self.cache,
                "splits"                    : self.splits
                }

        assert ( self.height * self.width * self.channels == numpy.prod(data_x.shape[1:]) )
        f = open(self.root +  '/data_params.pkl', 'wb')
        cPickle.dump(dataset_args, f, protocol=2)
        f.close()

    def _split_data (self, data, y1 = True):
        """
        This is an internal method that will split the datasets.

        Args:
            data: train, test and valid batches in a tuple.
        
        Returns:
            tuple: split data in the same format as data.
        """
        n_shots = self.splits["p"]
        if y1 is True:
            data_x, data_y, data_y1  = data
        else:
            data_x, data_y = data
        locs = numpy.zeros(len(data_y), dtype = bool)
        for label in xrange(self.n_classes + 1):
            temp = numpy.zeros(len(data_y), dtype = bool)                                                
            temp[data_y==label] = True
            if label in self.splits["shot"]:
                count = 0        
                for element in xrange(len(temp)):
                    if temp[element] == True:    # numpy needs == rather than 'is'               
                        count = count + 1
                    if count > n_shots:                                     
                        temp[element] = False   
                                     
            locs[temp] = True
        data_x = data_x[locs]
        data_y = data_y[locs]
        if y1 is True:
            data_y1 = data_y1[locs]
            return (data_x, data_y, data_y1)  
        else:
            return (data_x, data_y)


if __name__ == '__main__':
    pass
