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
    @patch('yann.utils.graph.to_pydot')
    def test_draw_network(self, mock_pydot):
        mock_pydot_obj = mock_pydotplus()
        mock_pydot.return_value = mock_pydot_obj
        util_graph.draw_network(self.G, "test.pdf", verbose=3)
        print(mock_pydot_obj.called)
        print()
        self.assertEqual(mock_pydot_obj.called, 3)
        self.assertEqual(mock_pydot_obj.filename, "test.pdf")

class mock_pydotplus:
    called = 0
    filename =''
    def write_png(self, filename):
        self.filename=filename
        self.called += 1
        return True

    def set_node_defaults(self, style="filled", fillcolor="grey"):
        self.called += 1
        return self
    def set_edge_defaults(self,color="blue", arrowhead="vee", weight="0"):
        self.called += 1
        return self