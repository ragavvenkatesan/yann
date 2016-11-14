.. _index: 
.. Yann Toolbox documentation master file, created by sphinx
 
==================================
Yet Another Neural Network Toolbox       
==================================    

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: license.html
   :alt: MIT License

.. image:: https://readthedocs.org/projects/yann/badge/?version=latest
    :target: http://yann.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status   

.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :target: https://github.com/ragavvenkatesan/yann/issues
    :alt: Raise issues
    
.. image:: https://badges.gitter.im/yann-users/Lobby.svg
    :target: https://gitter.im/yann-users/Lobby
    :alt: Gitter Lobby

.. image:: https://requires.io/github/ragavvenkatesan/yann/requirements.svg?branch=master
    :target: https://requires.io/github/ragavvenkatesan/yann/requirements/?branch=master
    :alt: Requirements Status

.. image:: https://codecov.io/gh/ragavvenkatesan/yann/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/ragavvenkatesan/yann
    :alt: Code Coverage


Welcome to the Yann Toolbox. It is a toolbox for convolutional neural networks, built on top 
of `theano`_. To setup the toolbox refer the :ref:`setup` guide. Once setup you may start with the 
:ref:`quick_start` guide or try your hand at the :ref:`tutorial` and the guide to 
:ref:`getting_started`. A user base discussion group is setup on `gitter`_.

.. _gitter: https://gitter.im/yann-users/Lobby
.. _theano: http://theano.readthedocs.io

.. warning ::

    Yann is currently under its early phases and is presently undergoing massive development. 
    Expect a lot of changes.

.. warning ::
    
      As of now, there are no unit-tests. The toolbox will be 
      formalized in the future but at this moment, the authorship, coverage and maintanence of the 
      toolbox is under extremely limited manpower.

.. note ::
    
    While, there are more formal and wholesome toolboxes that are similar and have a much larger 
    userbase such as `Lasagne`_, `Keras`_, `Blocks`_ and `Caffe`_, this toolbox is much more
    simple and versatile. Yann is designed as a supplement to an upcoming book on Convolutional 
    Neural Networks and also the toolbox of choice for a deep learning course. Because of this 
    reason, Yann is specifically designed to be intuitive and easy to use for beginners. That does 
    not compromise Yann of any of its features. 
    It is still a good choice for a toolbox for running pre-trained models and build 
    complicated, non-vannilla CNN architectures that are not easy to build with the other toolboxes.

.. _Lasagne: https://github.com/Lasagne/Lasagne
.. _Keras: http://keras.io/
.. _Caffe: http://caffe.berkeleyvision.org/
.. _Blocks: https://blocks.readthedocs.io/en/latest/
.. _Installation Guide: _setup

.. _getting_started: 

Getting Started
===============

The following will help you get acquinted with Yann. 

.. toctree::
   :maxdepth: 1
   :name: get_start   

   setup       
   tutorial
   organization


.. _quick_start:

Quick Start 
===========    

The easiest way to get going with Yann is to follow this quick start guide. If you are not 
satisfied and want a more detailed introduction to the toolbox, you may refer to the 
:ref:`tutorial` and the :ref:`organization`.  

The start and the end of Yann toolbox is the :mod:`network` module. The :mod:`yann.network`.
``network`` object is where all the magic happens. Start by importing :mod:`network` and creating a
``network`` object. 

.. code-block:: python

    from yann.network import network
    net = network()

Voila! We have thus created a new network. The network doesn't have any layers or modules in it. Tte
proof to can be seen by verifying the ``net.layers`` property of the ``net`` object.

.. code-block:: python

    net.layers
    
This will produce an output which is essentially an empty dictionary ``{}``. Let's add some layers!
The toolbox comes with the `MNIST dataset <http://yann.lecun.com/exdb/mnist/>`_ of handwritten 
characters pre-packaged for this toolbox. The dataset is located at ``_datasets/_dataset_71367``. 
Refer to the :ref:`tutorial` on how to convert your own dataset for yann.

The first layer that we need to add is an ``input`` layer. Every ``input``layer requries a dataset
to be associated with it. Let us create this layer.

.. code-block:: python

    dataset_params  = { "dataset": "_datasets/_dataset_71367", "n_classes" : 10 }
    net.add_layer(type = "input", dataset_init_args = dataset_params)

This piece of code creates and adds a new ``datastream`` module to the ``net`` and wires up the 
newly added ``input`` layer with this ``datastream``. Let us add a ``classifier`` layer. The default
classifier that Yann is setup with is the logistic regression classifier. Refer to the :ref:`yann`
or :ref:`tutorial` for other types of layers. Let us create a this ``classifier`` layer for now.

.. code-block:: python

    net.add_layer(type = "classifier" , num_classes = 10)
    net.add_layer(type = "objective")

The layer ``objective`` creates the loss function from the classifier that can be used as a learning
metric. It also provides a scope for other moduels such as the :mod:`optimizer` module. Refer
:ref:`organization` and :ref:`yann` for more details on modules. Now that our network is created and
constructed we can see that the ``net`` objects have ``layers`` populated. 

.. code-block:: python

    net.layers
    >>{'1': <yann.network.layers.classifier_layer object at 0x7eff9a7d0050>, '0': 
      <yann.network.layers.input_layer object at 0x7effa410d6d0>, '2': 
      <yann.network.layers.objective_layer object at 0x7eff9a71b210>}

The keys of the dictionary such as ``'1'``, ``'0'`` and ``'2'`` are the ``id`` of the layer. We 
could have created a layer with a custom id by supplying an ``id`` argument to the ``add_layer``
method. 

Now our network is finally ready to be trained. Before training we need to build an 
:mod:`optimizer` and other tools, but for now let us use the default ones. Once all of this is done,
before training, yann requires that the network be 'cooked'. For more details on cooking refer
:ref:`organization`. For now let us imagine that cooking a network will finalize the wiring, 
architecture, cache and prepare the first batch of data, prepare the modules and in general 
prepare the network for training. 

.. code-block:: python

    net.cook()

Cooking would take a few seconds and might print what it is doing along the way. Once cooked, we may
notice for instance that the network has a :mod:`optimizer` module. 


.. code-block:: python

    net.optimizer
    >>{'main': <yann.network.modules.optimizer object at 0x7eff9a7c1b10>}

To train the model that we have just cooked, we can use the ``train`` function that becomes 
available to us once the network is cooked.

.. code-block:: python

    net.train()

This will print a progress for each epoch and will show validation accuracy after each epoch on a
validation set that is independent from the training set. By default the training will run for 40
epochs: 20 on a higher learning rate and 20 more on a fine tuning learning rate. 

Every layer also has an ``layer.output`` object. The ``output`` can be probed by using the 
``layer_activity`` method as long as it is directly or in-directly associated with a 
:mod:`datastream` module through an ``input`` layer and the network was cooked.
Let us observe the activity of the input layer for trial. Once trained we can observe this output.
The layer activity will just be a ``numpy`` array of numbers, so let us print its shape instead.

.. code-block:: python

    net.layer_activity(id = '0').shape
    net.layers['0'].output_shape

The second line of code will verify the output we produced in the first line. An interesting layer
output is the output of the ``objective`` layer, which will give us the current 
negative log likelihood of the network, the one that we are trying to minimize. 

.. code-block:: python

    net.layer_activity(id = '2')
    >>array(2.1561384201049805, dtype=float32)

Once we are done training, we can run the network feedforward on the testing set to produce a
generalization performance result. 


.. code-block:: python
    
    net.test()

Congratualations, you now know how to use the yann toolbox successfully. A full-fledge code of the 
logistic regression that we implemented here can be found in 
``pantry/examples/log_reg.py``. That piece of code also has in commentary other options
that could be supplied to some of the function calls we made here that explain the processes better.

Hope you liked this quick start guide to the Yann toolbox and have fun!










.. toctree::
   :maxdepth: 3
   :name: hidden  
   :hidden:   
         
   yann/index
   trailer   
   license   
   genindex


Indices of functions and modules
--------------------------------

* :ref:`genindex`
* :ref:`modindex`
