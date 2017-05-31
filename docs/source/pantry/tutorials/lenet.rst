.. _lenet:

Convolutional Neural Network.
=============================

By virture of being here, it is assumed that you have gone through the :ref:`quick_start`.

Building a convolutional neural network is just as similar as an MLNN. The convolutional-pooling
layer or convpool layer could be added using the following statement:

.. code-block :: python

        net.add_layer ( type = "conv_pool",
                        origin = "input",
                        id = "conv_pool_1",
                        num_neurons = 40,
                        filter_size = (5,5),
                        pool_size = (2,2),
                        activation = ('maxout', 'maxout', 2),
                        batch_norm = True,           
                        regularize = True,                             
                        verbose = verbose
                    )

Here the layer has 40 filters each 5X5 followed by batch normalization followed by a maxpooling 
of 2X2 all with stride 1. The activation used is maxout with maxout by 2. A simpler relu layer
could be added thus, 

.. code-block :: python 

        net.add_layer ( type = "conv_pool",
                    origin = "input",
                    id = "conv_pool_1",
                    num_neurons = 40,
                    filter_size = (5,5),
                    pool_size = (2,2),
                    activation = 'relu',
                    verbose = verbose
                    )

Refer to the APIs for more details on the convpool layer.
It is often useful to visualize the filters learnt in a CNN, so we introduce the visualizer module 
here along with the CNN tutorial. The visualizer can be setup using the ``add_module`` method of 
``net`` object. 


.. code-block :: python

        net.add_module ( type = 'visualizer',
                     params = visualizer_params,
                     verbose = verbose 
                    )

where the ``visualizer_params`` is a dictionary of the following format.

.. code-block :: python

        visualizer_params = {
                    "root"       : 'lenet5',
                    "frequency"  : 1,
                    "sample_size": 144,
                    "rgb_filters": True,
                    "debug_functions" : False,
                    "debug_layers": False,  
                    "id"         : 'main'
                        }   

``root`` is the location where the visualizations are saved, ``frequency`` is the number of epochs
for which visualizations are saved down, ``sample_size`` number of images are saved each time.
``rgb_filters`` make the filters save in color. Along with the activities of each layer for the 
exact same images as the data itself, the filters of neural network are also saved down. 
For more options of parameters on visualizer refer to the `visualizer documentation`_ .

.. _visualizer documentation: http://yann.readthedocs.io/en/master/yann/modules/visualizer.html

The full code for this tutorial with additional commentary can be found in the file 
``pantry.tutorials.lenet.py``. This tutorial runs a CNN for the lenet dataset. 
If you have toolbox cloned or downloaded or just the tutorials downloaded, Run the code using,

.. automodule:: pantry.tutorials.lenet
   :members:

