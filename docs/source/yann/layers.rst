.. _layers:

:mod:`layers` - Contains the definitions of all the types of layers.
====================================================================

The file ``yann.layers.py`` contains the definition for the different types of layers
that are accessible in ``yann``. It contains various layers including:

    * :mod:`input_layer` 
    * :mod:`dropout_dot_product_layer` and :mod:`dot_product_layer`
    * :mod:`dropout_conv_pool_layer_2d` and :mod:`conv_pool_layer_2d`
    * :mod:`classifier_layer`
    * :mod:`objective_layer`

All these are inherited classes from :mod:`layer` class, which is abstract.

.. automodule:: yann.layers
   :members:
