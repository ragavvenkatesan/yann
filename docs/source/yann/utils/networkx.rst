.. networkx:

:mod:`networkx` - provides a nice port to networkx methods related to Yann
============================================================================

The file ``yann.utils.networkx.py`` contains the definition for the networkx ports. If 
:mod:`networkx` was installed, each network class also creates a ``networkx.DiGraph`` within itself
which is accessible through `net = network()`, `net.graph`. Layers will be `net.graph.nodes()`
and its attributes will be layer prorperties such as `type`, `output_shape` and so on. 

:mod:'yann.utils.networkx` has some ports that uses this networkx graph. 

This includes:

        * `draw_network` which draws the network.

.. automodule:: yann.utils.networkx
   :members: