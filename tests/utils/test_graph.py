import unittest
import networkx as nx
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch
import yann.utils.graph as util_graph
from os.path import isfile
from os import remove
class TestGraph(unittest.TestCase):

    def setUp(self):
        self.G = nx.Graph()

    def test_draw_network(self):
        util_graph.draw_network(self.G, "test.pdf", verbose=3)
        self.assertTrue(isfile("test.pdf"))
        remove("test.pdf")
