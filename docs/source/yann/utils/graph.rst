.. networkx:

:mod:`graph` - provides a nice port to networkx methods related to Yann
=======================================================================

The file ``yann.utils.graph.py`` contains the definition for the networkx ports. If 
:mod:`networkx` was installed, each network class also creates a ``networkx.DiGraph`` within itself
which is accessible through `net = network()`, `net.graph`. In each layer some representative
nodes (max limited) will be added to this graph and can be seen at `net.graph.nodes()`.
Its attributes will be layer prorperties such as `type`, `output_shape` and so on. 

:mod:`yann.utils.graph` has some ports that uses this networkx graph. 

This includes:

        * `draw_network` which draws the network.

The documentation follows.

.. automodule:: yann.utils.graph
   :members: