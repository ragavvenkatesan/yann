.. _mat2yann:

Cooking a matlab dataset for Yann.
==================================

By virture of being here, it is assumed that you have gone through the :ref:`quick_start`.

This tutorial will help you convert a dataset from matlab workspace to yann. To begin let us 
acquire `Google's Street View House Numbers dataset in Matlab`_ [1]. Download from the url three
.mat files: test_32x32.mat, train_32x32.mat and extra_32x32.mat. Once downloaded we need to 
divide this mat dump of data into training, testing and validation minibatches appropriately as 
used by yann. This can be accomplished by the steps outlined in the code 
``yann\pantry\matlab\make_svhn.m``. This will create data with 500 samples per mini batch with 
56 training batches, 42 testing batches and 28 validation batches. 

Once the mat files are setup appropriately, they are ready for yann to load and convert them into 
yann data. In case of data that is not form svhn, you can open one of the 'batch' files in matlab
to understand how the data is spread. Typically, the ``x`` variable is vectorized images, in this 
case 500X3072 (500 images per batch, 32*32*3 pixels per image). ``y`` is an integer vector labels 
going from 0-10 in this case. 

.. rubric:: References

.. [#] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng
        Reading Digits  in Natural Images with Unsupervised Feature Learning NIPS Workshop
        on Deep Learning and Unsupervised Feature Learning 2011. 
        
To convert the code into yann, we can use the ``setup_dataset`` module at ``yann.utils.dataset.py``
file. Simply call the initializer as,

.. code-block:: python

    dataset = setup_dataset(dataset_init_args = data_params,
                            save_directory = save_directory,
                            preprocess_init_args = preprocess_params,
                            verbose = 3 )

where, ``data_params`` contains information about the dataset thusly, 

.. code-block:: python

    data_params = {
                   "source"             : 'mat',
                   # "name"               : 'yann_svhn', # some name.
                   "location"			: location,    # some location to load from.  
                   "height"             : 32,
                   "width"              : 32,
                   "channels"           : 3,
                   "batches2test"       : 42,
                   "batches2train"      : 56,
                   "batches2validate"   : 28,
                   "mini_batch_size"    : 500  }

and the ``preprocess_params`` contains information on how to process the images thusly,

.. code-block:: python 

    preprocess_params = {
                            "normalize"     : True,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"     : False,
                        }

``save_directory`` is simply a location to save the yann dataset. Customarialy, it is 
``save_directory = '_datasets'``

The full code for this tutorial with additional commentary can be found in the file 
``pantry.tutorials.mat2yann.py``. 

If you have toolbox cloned or downloaded or just the tutorials 
downloaded, Run the code using,

.. automodule:: pantry.tutorials.mat2yann
   :members:

.. autoclass:: yann.utils.dataset.setup_dataset

.. _Google's Street View House Numbers dataset in Matlab: http://ufldl.stanford.edu/housenumbers/

