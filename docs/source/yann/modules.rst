.. _modules:

:mod:`modules` - Modules that are external to the network but the network can use
==================================================================================

The file ``yann.modules.py`` contains the definition for the network modules. It contains 
    various modules including:

    * :mod:`visualizer` is used to produce network visualizations. It will take the activities, 
      filters and data from a network and produce activations.
    * :mod:`resultor` is used to save down network results.
    * :mod:`optimizer`is the backbone of the SGD and optimization.
    * :mod:`dataset` is the module that creates, loads, caches and feeds data to the network.    

.. automodule:: yann.modules
   :members:
