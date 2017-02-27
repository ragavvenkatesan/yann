.. _batch_norm:

Batch Normalization.
====================

Batch normalization has become one important operation in faster and stable learning of neural
networks. In batch norm we do the following:

.. math::

    x = (\frac{x - \mu_b}{\sigma_b})\gamma + \beta

The :math:`x` is the input (and the output) of this operation, :math:`\mu_b` and :math:`\sigma_b`
are the mean and the variance of the minibatch of :math:`x` supplied. :math:`\gamma` and 
:math:`\beta` are learnt using back propagation. This will also store a running mean and a running 
variance, which is used during inference time. 

By default batch normalization can be performed on convolution and dot product layers using 
the argument ``batch_norm = True`` supplied to the :mod:`yann.network.add_layer` method. This 
will apply the batch normalization before the activation and after the core layer operation. 

While this is the technique that was described in the original batch normalization paper[1]. Some 
modern networks such as the Residual network [2],[3] use a re-orderd version of layer operations 
that require the batch norm to be applied post-activation. This is particularly used when using 
ReLU or Maxout networks[4][5]. Therefore we also provide a layer type ``batch_norm``, that could
create a layer that simply does batch normalization on the input supplied. These layers could be 
used to create a post-activation batch normalization. 

This tutorial demonstrates the use of both these techniques using the same architecutre of networks
used in the :ref:`lenet` tutorial. The codes for these can be found in the following module methods
in :mod:`pantry.tutorials`.

.. rubric:: References

.. [#]   Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network 
         training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015).
.. [#]   He, Kaiming, et al. "Identity mappings in deep residual networks." European Conference on 
         Computer Vision. Springer International Publishing, 2016.
.. [#]   He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the 
         IEEE Conference on Computer Vision and Pattern Recognition. 2016.
.. [#]   Nair, Vinod, and Geoffrey E. Hinton. "Rectified linear units improve restricted boltzmann 
         machines." Proceedings of the 27th International Conference on Machine Learning (ICML-10). 
         2010.
.. [#]   Goodfellow, Ian J., et al. “Maxout networks.” arXiv preprint arXiv:1302.4389 (2013).
         
.. automodule:: pantry.tutorials.lenet
   :members:

