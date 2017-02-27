.. _batch_norm:

:mod:`batch_norm` - Batch normalization  layer classes
======================================================

The file ``yann.layers.batch_norm.py`` contains the definition for the batch norm layers. Batch norm
can by default be applied to convolution and fully connected layers by sullying an argument
``batch_norm = True``, in the layer arguments. But this in-built method applies batch norm 
prior to layer activation. Some architectures including ResNet involves batch norms after the 
activations of the layer. Therefore there is a need for an independent batch norm layer that simply
applies batch norm for some outputs. The layers in this module can do that. 

There are four classes in this file. Two for one-dimensions and two for two-dimnensions.

.. automodule:: yann.layers.batch_norm
   :members:
