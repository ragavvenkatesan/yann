.. _organization:

=============================
Structure of the Yann network
=============================

.. figure:: _static/imgs/net_obj.png
   :alt: The structure of the net object.

The core of the yann toolbox and its operations are built around the :mod:`yann.network.network`
class, which is present in the file ``yann/network.py``. The above figure shows the organization of 
the :mod:`yann.network.network` class. The :func:`add_xxxx` methods add either a layer or module as 
nomenclature. The network class can hold various layers and modules in various connections and 
architecture that are added using the ``add_`` methods. 


Verbose
-------

Throughout the toolbox, various methods take an argument called ``verbose`` as input. ``verbose`` is
by default always ``2``. ``verbose = 1`` implies a silent run and therefore the code doesn't print 
anything unless absolutely needed. ``verbose=2`` prints quite the standard amount of information and 
``verbose==3``, which is friendly when being used for debugging prints annoyingly too much 
information.  


Initializing a network class
----------------------------

A :mod:`network` pbject can quite simply be initialized by calling 


.. code-block:: python

    from yann.network import network
    net = network()






Each layer takes in as argument While prepping the network for learning, we 
can (or may) need only certain modules and layers. The process of preparing the network by selecting 
and building the training, testing and validation parts of network is called cooking.  

.. figure:: _static/imgs/cooked_net.png
   :alt: A cooked network. The objects that are in gray and are shaded are uncooked parts of the 
            network.

The above figure shows a cooked network. The objects that are in gray and are shaded are uncooked 
parts of the network. Once cooked, the network is ready for training and testing all by using other 
methods within the network. The network class also has several properties such as layers, which is 
a dictionary of the layers that are added to it and params, which is a dictionary of all the 
parameters. All layers and modules contain a property called `id` through which they are referred.









