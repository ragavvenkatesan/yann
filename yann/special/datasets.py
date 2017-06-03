from yann.utils.dataset import *
import numpy
import os

def download_celebA ( data_dir = 'celebA'):
    """ 
    This method downloads celebA dataset into directory _data/``data_dir``.

    Args:
        data_dir: Location to save the data.
    """ 

    def _download_file(id, destination):
        """
        Helper function to download the dataset from Google Drive.
        """
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()

        response = session.get(URL, params={ 'id': id }, stream=True)
        token = _get_token(response)

        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params=params, stream=True)

        _save_content(response, destination)

    def _get_token(response):
        """
        Helper function for token confirmation.
        """
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def _save_content(response, destination, chunk_size=32*1024):
        total_size = int(response.headers.get('content-length', 0))
        print("... Downloading the dataset")
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)

    dirpath = '_data/'
    if os.path.exists(os.path.join(dirpath, data_dir)):
        print('Celeb-A already downloaded')

    else:
        filename, drive_id  = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
        save_path = os.path.join(dirpath, filename)

        if os.path.exists(save_path):
            print('[*] {} already exists'.format(save_path))
        else:
            os.mkdir(dirpath)
            _download_file(drive_id, save_path)

        print("... Dataset downloaded")
        print("... Extracting images")
        zip_dir = ''
        with zipfile.ZipFile(save_path) as zf:
            zip_dir = zf.namelist()[0]
            zf.extractall(dirpath)
        os.remove(save_path)
        os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))
        print("... Celeb-A extracted successfully")

    filepath = os.path.join(os.path.join(dirpath, data_dir), '*.jpg')
    filelist = glob.glob(filepath)

class combine_split_datasets_train_only (object):
    """
    This will combine two split datasets into one.

    Todo:
        Extend it for non-split datasets also.

    Args:
        loc: A tuple of a list of locations of two dataset to be blended.
        verbose: As always
    
    Notes:
        At this moment, mini_batches_per_batch and mini_batch_size of both datasets must be the 
        same.
        This only splits the train data with shot. The test and valid hold both. 
        This is designed for the incremental learning.
    """     
    def __init__(self, loc, verbose = 1, **kwargs):
        """
        Initializer method.
        """
        if verbose >= 2:
            print (".. Initializing combining datasets")

        self.loc1 = loc[0]
        self.loc2 = loc[1]

        if verbose >= 3:
            print("... Initializing dataset 1")

        f = open(self.loc1 + '/data_params.pkl', 'rb')
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

        f = open(self.loc2 + '/data_params.pkl', 'rb')
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

        save_directory = '_datasets'
        self.id = str(randint(11111,99999))
        self.key_root = '/_dataset_'
        self.root = save_directory + self.key_root + self.id
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)        
        os.mkdir(self.root)
        os.mkdir(self.root + "/train" )
        os.mkdir(self.root + "/test"  )
        os.mkdir(self.root + "/valid" )          

        if not 'splits' in kwargs.keys():
            self.splits = { "base"              :  [ x + \
                                                self.n_classes_1 for x in \
                                                list(range(self.n_classes_2)) ] ,
                        "shot"              : list(range(self.n_classes_1)),
                        "p"                 : 0
                    }              
        else:
            self.splits = kwargs ['splits']
        
        assert self.mini_batches_per_batch_1 == self.mini_batches_per_batch_2
        self.mini_batches_per_batch = (self.mini_batches_per_batch_2[0],
                                       self.mini_batches_per_batch_2[1] * 2,
                                       self.mini_batches_per_batch_2[2] * 2)
        self.combine ( verbose = verbose )  
        print ("Dataset " + self.id + " is combined and makde")
        
    def combine(self, verbose = 1):
        """
        Thie method runs the combine.

        Args:
            verbose: As Always
        """
        if verbose >= 2:
            print (".. Begining combining")
        self._combine_and_save ( type = 'train', verbose = 2 )
        self._combine_and_save ( type = 'test', verbose = 2 )
        self._combine_and_save ( type = 'valid', verbose = 2 )
        self._combine_parameters ( verbose = 2 )

    def _combine_parameters (self, verbose = 1 ):
        """
        This method prints the last parameter file
        """
        self.cache =  not( self.batches2train_1 == 1 and
                           self.batches2test_1 == 1 and
                           self.batches2validate_1 == 1 and 
                           self.batches2train_2 == 1 and
                           self.batches2test_2 == 1 and
                           self.batches2validate_2 == 1 )

        dataset_args = {
                "location"                  : self.root,
                "mini_batch_size"           : self.mini_batch_size,
                "cache_batches"             : self.mini_batches_per_batch_1 + \
                                              self.mini_batches_per_batch_2,
                "batches2train"             : max(self.batches2train_1, self.batches2train_2),
                "batches2test"              : max(self.batches2test_1, self.batches2test_2),
                "batches2validate"          : max(self.batches2validate_1, self.batches2validate_2),
                "height"                    : self.height,
                "width"                     : self.width,
                "channels"              : self.channels,
                "cache"                 : self.cache,
                "mini_batches_per_batch": self.mini_batches_per_batch,
                "splits"                : self.splits,
                }

        f = open(self.root +  '/data_params.pkl', 'wb')
        cPickle.dump(dataset_args, f, protocol=2)
        f.close()

    def _combine_and_save ( self, type = 'train', verbose = 1 ):
        """
        This is an internal methof that would combine the training data.
        """
        if type == 'train':
            n_batches_1 = self.batches2train_1
            n_batches_2 = self.batches2train_2
        elif type == 'test':
            n_batches_1 = self.batches2test_1
            n_batches_2 = self.batches2test_2
        else:
            n_batches_1 = self.batches2validate_1
            n_batches_2 = self.batches2validate_2

        for batch in xrange(max(n_batches_1,n_batches_2)):
            data = self.load_data ( type = type,  batch = batch, n_batches_1 = n_batches_1, \
                                            n_batches_2 = n_batches_2, verbose = verbose)
            data_x_1, data_y_1, data_x_2, data_y_2 = data
            data_y_2 = self._convert_labels ( labels = data_y_2, verbose = verbose )

            if not type == 'train':
                data_x = numpy.concatenate( (data_x_1, data_x_2), axis = 0 )
                data_y = numpy.concatenate( (data_y_1, data_y_2), axis = 0 )
            else:
                data_x = data_x_2
                data_y = data_y_2              
            self.save_data(data_x = data_x, data_y = data_y, type = type, batch = batch )
        
        self._combine_parameters ()

    def _convert_labels ( self, labels, verbose = 1):
        """
        This method convert the labels from old to new for the second dataset. Only works for 
        second dataset, not for the first dataset.

        Args:
            verbose: As usual.
        """ 
        return labels + self.n_classes_1

    def save_data (self, data_x, data_y, type = 'train', batch = 0, verbose = 2):
        """
        Saves down a batch of data. 
        """
        if verbose >=3: 
            print ("... Dumping batch " + str(batch))
        # compute number of minibatches for training, validation and testing
        f = open( self.root + "/" + type + "/" + 'batch_' + str(batch) + '.pkl', 'wb')
        obj = (data_x, data_y )
        obj = shuffle (obj)
        cPickle.dump(obj, f, protocol=2)
        f.close() 
    

    def load_data(self, n_batches_1, n_batches_2, type = 'train', batch = 0,verbose = 2):
        """
        Will load the data from the file and will return the data. Will supply two batches one from
        each set respectively.

        Args:
            type: ``train``, ``test`` or ``valid``.
                  default is ``train``
            batch: Supply an integer
            n_batches_1: Number of batches in dataset 1
            n_batches_2: Number of batches in dataset 2
            verbose: Simliar to verbose in toolbox.

        Todo:
            Create and load dataset for type = 'x'

        Returns:
            numpy.ndarray: ``data_x, data_y``
        """
        batch_load = batch % n_batches_1        
        if verbose >= 1:
            print("... loading " + type + " data batch " + str(batch_load))
        
        f = open(self.loc1 + '/' + type + '/batch_' + str(batch_load) +'.pkl', 'rb')

        data_x_1 , data_y_1 = cPickle.load(f)
        f.close()

        data_x_1 = check_type (data_x_1, theano.config.floatX)
        data_y_1 = check_type (data_y_1, theano.config.floatX)
        # Theano recommends storing on gpus only as floatX and casts them to ints during use.
        # I don't know why, but I am following their recommendations blindly.

        if verbose >= 3:
            print("... data from set 1 is loaded")

        batch_load = batch % n_batches_2
        if verbose >= 1:
            print("... loading " + type + " data batch " + str(batch_load))

        f = open(self.loc2+ '/' + type + '/batch_' + str(batch_load) +'.pkl', 'rb')

        data_x_2 , data_y_2 = cPickle.load(f)
        f.close()

        data_x_2 = check_type (data_x_2, theano.config.floatX)
        data_y_2 = check_type (data_y_2, theano.config.floatX)
        # Theano recommends storing on gpus only as floatX and casts them to ints during use.
        # I don't know why, but I am following their recommendations blindly.
        if verbose >= 3:
            print("... data from set 2 is loaded")

        return (data_x_1, data_y_1, data_x_2, data_y_2)

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

def cook_celeba_normalized_zero_mean(verbose = 1, location = '_data/celebA', **kwargs):
    """
    Wrapper to cook Celeb-A dataset in preparation for GANs. Will take as input,

    Args:
        location: Location where celebA was downloaded using 
                    ``yann.specials.datasets.download_celebA``
        save_directory: which directory to save the cooked dataset onto.
        dataset_parms: default is the dictionary. Refer to :mod:`setup_dataset`
        preprocess_params: default is the dictionary. Refer to :mod:`setup_dataset`
    """

    if not 'data_params' in kwargs.keys():

        data_params = {
                    "source"             : 'images-only',
                    "name"               : 'celeba',
                    "location"           : location,
                    "mini_batch_size"    : 500,
                    "mini_batches_per_batch" : (1, 1, 1),
                    "batches2train"      : 403,
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
