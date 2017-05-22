import unittest
import numpy as np
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch
import yann.utils.pickle as util_pickle
from yann import network
class TestGraph(unittest.TestCase):

    def setUp(self):
        self.net = network.network()
        self.params = {'1':np.random.rand(5), '2':np.random.rand(5), '3':np.random.rand(5)}

    @patch('yann.network.network.get_params')
    @patch('yann.utils.pickle.cPickle.dump')
    def test_pickle(self, mock_dump,mock_params):
        mock_dump.return_value = Mock(spec=file)
        util_pickle.pickle(self.net, "test_file", verbose=3)
        self.assertTrue(mock_params.called)
        self.assertTrue(mock_dump.called)

    @patch('yann.utils.pickle.cPickle.load')
    @patch('yann.utils.pickle.shared_params')
    @patch('yann.utils.pickle.open')
    def test_load(self,mock_open, mock_shared_params, mock_load):
        util_pickle.load('file', verbose=3)
        self.assertTrue(mock_load.called)
        self.assertTrue(mock_shared_params.called)
        self.assertTrue(mock_open.called)

    def test_shared_params(self):
        shared_params = util_pickle.shared_params(self.params, verbose=3)
        self.assertEqual(len(shared_params), len(self.params))