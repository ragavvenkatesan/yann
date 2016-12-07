.. _layers:

:mod:`layers` - Contains the definitions of all the types of layers.
====================================================================

The module :mod:`yann.layers` contains the definition for the different types of layers
that are accessible in ``yann``. It contains various layers including:

    * :mod:`abstract.layer`
    * :mod:`input.input_layer` 
    * :mod:`fully_connected.dropout_dot_product_layer` and 
        :mod:`fully_connected.dot_product_layer`
    * :mod:`conv_pool.dropout_conv_pool_layer_2d` and 
        :mod:`conv_pool.conv_pool_layer_2d`
    * :mod:`ouput.classifier_layer`
    * :mod:`output.objective_layer`

All these are inherited classes from :mod:`layer` class, which is abstract.

Specific layers that can be used are 

.. toctree::
   :maxdepth: 3
   :name: Layers    

   abstract
   input
   fully_connected
   conv_pool
   output


