"""
This is a tutorial to setup any dataset in matlab format to be used by YANN.
Still working on this. 
"""
def cook_svhn_normalized( location, verbose = 1, **kwargs):
    """
    This method demonstrates how to cook a dataset for yann from matlab.

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
                   "source"             : 'mat',
                   "name"               : 'yann_Svhn', # some name.
                   "location"			: location,    # some location
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

    dataset = setup_dataset(dataset_init_args = data_params,
                            save_directory = save_directory,
                            preprocess_init_args = preprocess_params,
                            verbose = 3)
    return dataset