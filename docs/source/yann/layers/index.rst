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
    * :mod:`merge.merge_layer`
    * :mod:`flatten.flatten_layer`
    * :mod:`flatten.unflatten_layer`
    * :mod:`random.random_layer`
    * :mod:`batch_norm.batch_norm_layer_2d` and
        :mod:`batch_norm.dropout_batch_norm_layer_2d`        
    * :mod:`batch_norm.batch_norm_layer_1d` and
        :mod:`batch_norm.dropout_batch_norm_layer_1d`  

All these are inherited classes from :mod:`layer` class, which is abstract.

Specific layers that can be used are 

.. toctree::
   :maxdepth: 3
   :name: Layers    

   abstract
   input
   fully_connected
   conv_pool
   merge
   flatten
   output   
   random
   transform
   batchnorm


