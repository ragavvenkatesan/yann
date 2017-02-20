from yann.utils.dataset import setup_dataset

def cook_svhn_normalized( location, verbose = 1, **kwargs):
    """
    This method demonstrates how to cook a dataset for yann from matlab. Refer to the 
    ``pantry/matlab/setup_svhn.m`` file first to setup the dataset and make it ready for use with 
    yann.

    Args:

        location: provide the location where the dataset is created and stored.
                  Refer to prepare_svhn.m file to understand how to prepare a dataset.
        save_directory: which directory to save the cooked dataset onto.
        dataset_parms: default is the dictionary. Refer to :mod:`setup_dataset`
        preprocess_params: default is the dictionary. Refer to :mod:`setup_dataset`.

    Notes:
        By default, this will create a dataset that is not mean-subtracted.
    """

    if not 'save_directory' in kwargs.keys():
        save_directory = '_datasets'
    else:
        save_directory = kwargs ['save_directory']
        
    if not 'data_params' in kwargs.keys():

        data_params = {
                   "source"             : 'matlab',
                   # "name"               : 'yann_svhn', # some name.
                   "location"			: location,    # some location to load from.  
                   "height"             : 32,
                   "width"              : 32,
                   "channels"           : 3,
                   "batches2test"       : 42,
                   "batches2train"      : 56,
                   "batches2validate"   : 28,
                   "mini_batch_size"    : 500}

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

    dataset = setup_dataset(dataset_init_args = data_params,
                            save_directory = save_directory,
                            preprocess_init_args = preprocess_params,
                            verbose = 3)
    return dataset

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        raise Exception("Provide Dataset Location")
    else:
        location = sys.argv[1]
    cook_svhn_normalized(location)