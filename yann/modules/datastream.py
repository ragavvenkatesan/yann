import numpy
import cPickle

import theano
import theano.tensor as T

from yann.utils.dataset import create_shared_memory_dataset
from yann.utils.dataset import check_type
from abstract import module

class datastream(module):
    """
    This module initializes the dataset to the network class and provides all dataset related
    functionalities. It also provides for dynamically loading and caching dataset batches.
    :mod: ``add_layer`` will use this to initialize.

    Args:
        dataset_init_args: Is a dictionary of the form:
        borrow: Theano's borrow. Default value is ``True``.

            .. code-block:: python

                dataset_init_args = {
                            "dataset":  <location>
                            "svm"    :  False or True
                                 ``svm`` if ``True``, a one-hot label set will also be setup.
                            "n_classes": <int>
                                ``n_classes`` if ``svm`` is ``True``, we need to know how
                                 many ``n_classes`` are present.
                            "id": id of the datastream
                    }

        verbose: Similar to verbose throughout the toolbox.

    Returns:
        dataset: A dataset module object that has the details of loader and other things.

    Todo:
        * Datastream should work with Fuel perhaps ?
        * Support HDf5 perhaps

    """

    # this loads up the data_params from a folder and sets up the initial databatch.
    def __init__ ( self,
                   dataset_init_args,
                   borrow = True,
                   verbose = 1):

        if "id" in dataset_init_args.keys():
            id = dataset_init_args["id"]
        else:
            id = '-1'
        super(datastream,self).__init__(id = id, type = 'datastream')

        dataset = dataset_init_args ["dataset"]
        self.dataset = dataset
        self.borrow = borrow

        if verbose >= 3:
            print "... Initializing datastream with " + dataset

        f = open(dataset + '/data_params.pkl', 'rb')
        data_params = cPickle.load(f)
        f.close()

        self.dataset_location    = data_params [ "location"]
        self.mini_batch_size     = data_params [ "mini_batch_size" ]
        self.mini_batches_per_batch  = data_params [ "cache_batches" ]
        self.batches2train       = data_params [ "batches2train" ]
        self.batches2test        = data_params [ "batches2test" ]
        self.batches2validate    = data_params [ "batches2validate" ]
        self.height              = data_params [ "height" ]
        self.width               = data_params [ "width" ]
        self.channels            = data_params [ "channels" ]
        self.cache               = data_params [ "cache" ]

        self.current_type = 'train'
        if 'type' in dataset_init_args.keys():
            self.type = dataset_init_args['type']
        else:
            self.type = 'xy'

        if self.type == 'xy':
            if 'svm' in dataset_init_args.keys():
                self.svm = dataset_init_args["svm"]
            else:
                self.svm = False
        elif self.type == 'x':
            self.svm = False


        if self.svm is True:
            if "n_classes" in dataset_init_args.keys():
                self.n_classes = dataset_init_args ["n_classes"]
            else:
                self.n_classes = False

        self.initialize_dataset(verbose = verbose)
        self.batch = 0# initialize the batch to zero. Changing this will produce a new stream.

        self.cached_zeros_x = numpy.zeros((1,),dtype = theano.config.floatX)
        if self.type == 'xy':
            self.cached_zeros_y = numpy.zeros((1,),dtype = theano.config.floatX)

        if verbose >= 3:
            print "... Datastream is initiliazed"

        self.x = T.matrix('x')
        if self.type == 'xy':
            self.y = T.ivector('y')
            self.one_hot_y = T.matrix('one_hot_y')
        elif self.type == 'x':
            self.y = self.x

    def load_data (self, type = 'train', batch = 0, verbose = 2):
        """
        Will load the data from the file and will return the data. The important thing to note
        is that all the datasets in :mod: ``yann`` all require a ``y`` or a variable to
        predict. In case of auto-encoder for instance, the thing to predict is the image
        itself. Setup dataset thusly.

        Args:
            type: ``train``, ``test`` or ``valid``.
                  default is ``train``
            batch: Supply an integer

            verbose: Simliar to verbose in toolbox.

        Todo:
            Create and load dataset for type = 'x'

        Returns:
            numpy.ndarray: ``data_x, data_y``
        """
        if verbose >= 3:
            print "... loading " + type + " data batch " + str(batch)

        f = open(self.dataset + '/' + type + '/batch_' + str(batch) +'.pkl', 'rb')

        data_x, data_y = cPickle.load(f)
        f.close()

        if verbose >= 3:
            print "... data is loaded"

        data_x = check_type (data_x, theano.config.floatX)
        if self.type == 'xy':
            data_y = check_type (data_y, theano.config.floatX)
        # Theano recommends storing on gpus only as floatX and casts them to ints during use.
        # I don't know why, but I am following their recommendations blindly.
        return data_x, data_y

    def set_data (self, type = 'train', batch = 0, verbose = 2):
        """
        This can work only after network is cooked.

        Args:
            batch: which batch of data to load and set
            verbose: as usual
        """
        if verbose >=3 :
            print "... Setting batch " + str(batch) + " of data of type " + type

        data_x, data_y = self.load_data (batch = batch, type = type, verbose = verbose )
        # Doing this just so that I can use set_value instead of set_sub_tensor.
        # Also, I see some elegance in zeroing out stuff.

        if data_x.shape[0] < self.data_cache_size:
            # This will probably used by non-cached datasets heavily.
            data_size_needed = (self.data_cache_size - data_x.shape[0], self.height *
                                                            self.width * self.channels)
            if not self.cached_zeros_x.shape[0] == data_size_needed[0]:
                self.cached_zeros_x = numpy.zeros(data_size_needed,
                                                     dtype = data_x.dtype)
                if verbose >= 3:
                    print "... Cache miss in loading data "

            if self.type == 'xy':
                if not self.cached_zeros_y.shape[0] == data_size_needed[0]:
                    self.cached_zeros_y =  numpy.zeros((data_size_needed[0],), dtype = data_y.dtype)

            data_x = numpy.concatenate((data_x, self.cached_zeros_x), axis=0)
            if self.type == 'xy':  # if its only x, data_y is useless anyway.
                data_y = numpy.concatenate((data_y, self.cached_zeros_y), axis = 0)

        elif data_x.shape[0] > self.data_cache_size:
            # don't know if this case will ever be used.
            data_x = data_x[:self.data_cache_size,]
            data_y = data_y[:self.data_cache_size,]

        self.data_x.set_value (data_x, borrow = self.borrow )
        if self.type == 'xy':
            self.data_y_uncasted.set_value (data_y, borrow = self.borrow )

        if self.svm is True and self.type == 'xy':
            data_one_hot_y = self.one_hot_labels( data_y, verbose = verbose )
            self.data_one_hot_y.set_value ( data_one_hot_y , borrow = self.borrow )

        self.current_type = type

    def one_hot_labels (self, y, verbose = 1):
        """
        Function takes in labels and returns a one-hot encoding. Used for max-margin loss.
        Args:
            y: Labels to be encoded.n_classes
            verbose: Typical as in the rest of the toolbox.

        Notes:
            ``self.n_classes``: Number of unique classes in the labels.

                       This could be found out using the following:
                       .. code-block: python

                           import numpy
                           n_classes = len(numpy.unique(y))

                      This might be potentially dangerous in case of cached dataset. Although
                      this is the default if ``n_classes`` is not provided as input to this
                      module, I discourage anyone from using this.
        Returns:
            numpy ndarray: one-hot encoded label list.
        """

        if self.n_classes is False:
            if verbose >= 3:
                print "... Making a decision to create n_classes variable, not a good idea."
            self.n_classes = len(numpy.unique(y))

        # found this technique online somewhere, forgot where couldn't cite.
        y1 = -1 * numpy.ones((y.shape[0], self.n_classes))
        y1[numpy.arange(y.shape[0]), y] = 1
        y1 = check_type(y1, theano.config.floatX)
        return y1


    def initialize_dataset( self, verbose = 1 ):
        """
        Load the initial training batch of data on to ``data_x`` and ``data_y`` variables
        and create shared memories.

        Todo:
            I am assuming that training has the largest number of data. This is immaterial when
            caching but during set_data routine, I need to be careful.
        Args:
            verbose: Toolbox style verbose.
        """
        if verbose >= 3:
            print ".. Initializing the dataset by loading 0th batch"

        # every dataset will have atleast one batch ..load that.
        # Assumimg that train has more number of data than any other.

        data_x, data_y = self.load_data(type = 'train', batch = 0, verbose = verbose)
        self.data_cache_size = data_x.shape[0]

        if self.svm is False:
            self.data_x, self.data_y_uncasted = create_shared_memory_dataset(
                                                           (data_x, data_y),
                                                            borrow = self.borrow,
                                                            verbose = verbose)
        else:
            data_y1 = self.one_hot_labels (data_y, verbose = verbose)
            self.data_x, self.data_y_uncasted, self.data_one_hot_y = create_shared_memory_dataset(
                                                  (data_x, data_y, data_y1),
                                                            borrow = self.borrow,
                                                            svm = True,
                                                            verbose = verbose)
        if self.type == 'xy':
            self.data_y = T.cast(self.data_y_uncasted, 'int32')

        if verbose >=3:
            print "... dataset is initialized"


if __name__ == '__main__':
    pass
