.. _pickle:

:mod:`pickle` - provides a way to save the network' parameters as a pickle file.
================================================================================

The file ``yann.utils.pickle.py`` contains the definition for the pickle methods. Use pickle 
method in the file to save the params down as a pickle file. Note that this only saves the 
parameters down and not the architecture or optimizers or other modules. The id of the layers
will also be saved along as dictionary keys so you can use them to create a network. 

The documentation follows:

.. automodule:: yann.utils.pickle
   :members: