.. _modules:

:mod:`modules` - Modules that are external to the network which the network uses
================================================================================

The module :mod:`yann.modules` contains the definition for the network modules. It contains 
    various modules including:

    * :mod:`visualizer` is used to produce network visualizations. It will take the activities, 
      filters and data from a network and produce activations.
    * :mod:`resultor` is used to save down network results.
    * :mod:`optimizer` is the backbone of the SGD and optimization.
    * :mod:`dataset` is the module that creates, loads, caches and feeds data to the network.    


.. toctree::
   :maxdepth: 3
   :name: Yann Modules     

   optimizer
   datastream
   visualizer
   resultor
   