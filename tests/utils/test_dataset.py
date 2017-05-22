import unittest
import numpy as np
import yann.utils.dataset as util_dataset
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch

from os.path import isfile
from os import remove
class TestDataset(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.test_data = (np.random.rand(10,20), np.random.rand(10,1))
        self.test_data_svm = (np.random.rand(10, 20), np.random.rand(10, 1),np.random.rand(10, 1))
        self.perm = np.random.permutation(self.test_data[0].shape[0])
        self.test_data_perm = (self.test_data[0][self.perm], self.test_data[1][self.perm])
        self.test_file_loc = "./"
        self.test_file_url_working = "https://raw.githubusercontent.com/ragavvenkatesan/yann/master/requirements.txt"
        self.test_file_url_not_working = "https://raw.githubusercontent.com/ragavvenkatesan/yann/master/requirements.txt1"
        self.filename_not_working = self.test_file_url_not_working.split('/')[-1]
        self.data_mat_check = {'x' : np.random.rand(20) * 10, 'y' : np.random.rand(20), 'z' : np.random.rand(20)}
        self.data_mat_check_channels = {'x': np.random.rand(20,2), 'y': np.random.rand(20), 'z': np.random.rand(20)}
        self.dataset_init_args_matlab =  {
               "source"             : 'matlab',
               "location"                   : 'loc',    # some location to load from.
               "height"             : 1,
               "width"              : 1,
               "channels"           : 1,
               "batches2test"       : 1,
               "batches2train"      : 1,
               "batches2validate"   : 1,
               "mini_batches_per_batch": [1,1,1],
               "mini_batch_size"    : 500,
                 }
        self.preprocess_init_args = {
                            "normalize"     : False,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"		: False,
                            }
        self.preprocess_init_args_default = {
            "normalize": True,
            "ZCA": False,
            "grayscale": True,
            "zero_mean"	: False,
        }
        self.dataset_init_args_skdata = {
               "source"             : 'skdata',
               "name"               : 'mnist', # some name.
  }
        # self.fake_minst

    @patch('numpy.random.permutation')
    def test_shuffle(self,mock_permutation):
        mock_permutation.return_value = self.perm;
        result = util_dataset.shuffle(self.test_data)
        self.assertTrue(np.allclose(result[0], self.test_data_perm[0]))
        self.assertTrue(np.allclose(result[1], self.test_data_perm[1]))
    def test_create_shared_memory_dataset_without_args(self):
        result = util_dataset.create_shared_memory_dataset(self.test_data)
        self.assertTrue(np.allclose(result[0].eval(), self.test_data[0]))
        self.assertTrue(np.allclose(result[1].eval(), self.test_data[1]))

    def test_create_shared_memory_dataset_with_args(self):
        result = util_dataset.create_shared_memory_dataset(self.test_data_svm, svm=True)
        self.assertTrue(np.allclose(result[0].eval(), self.test_data_svm[0]))
        self.assertTrue(np.allclose(result[1].eval(), self.test_data_svm[1]))
        self.assertTrue(np.allclose(result[2].eval(), self.test_data_svm[2]))
    @patch('yann.utils.dataset.cPickle.dump')
    @patch('yann.utils.dataset.open')
    def test_pickle_dataset(self, mock_pickle,mock_open):
        util_dataset.pickle_dataset('.', '0', self.test_data)
        self.assertTrue(mock_pickle.called == 1)
        self.assertTrue(mock_open.called == 1 )
    def test_download_data(self):
        print(self.test_file_url_working.split('/')[-1])

        util_dataset.download_data(self.test_file_url_working, self.test_file_loc)
        print('printed')
        self.assertTrue(isfile(self.test_file_url_working.split('/')[-1]))
        remove(self.test_file_url_working.split('/')[-1])
    @patch('scipy.io.loadmat')
    def test_load_mat(self, mock_loadmat):
        mock_loadmat.return_value = self.data_mat_check
        val = util_dataset.load_data_mat(1,1,1,'mock location', 0, '', False)
        valz = util_dataset.load_data_mat(1, 1, 1, 'mock location', 0, '', True)
        self.assertEqual(mock_loadmat.called, 1)
        self.assertEqual(len(val), 2)
        self.assertEqual(len(valz), 3)

    @patch('scipy.io.loadmat')
    def test_load_mat_channels(self, mock_loadmat):
        mock_loadmat.return_value = self.data_mat_check_channels
        val_channel = util_dataset.load_data_mat(1,1,2,'mock location',0,'', False)
        self.assertEqual(val_channel[0].shape[1], 2)

    def test_skdata_mnist(self):
        mnist = util_dataset.load_skdata_mnist()
        # print(mnist[0][0].shape, mnist[0][].shape)
        self.assertEqual(len(mnist[0][0]) + len(mnist[1][0]),60000)
        self.assertEqual(len(mnist[0][1]) + len(mnist[1][1]), 60000)

    @patch('yann.utils.dataset.setup_dataset._mat2yann')
    @patch('os.mkdir')
    def test_setup_dataset_matlab(self,mock_dir, mock_mat2yann ):
        self.setup_dataset = util_dataset.setup_dataset(self.dataset_init_args_matlab, preprocess_init_args = self.preprocess_init_args, save_directory='some/unknown/directory')
        self.assertEqual(mock_mat2yann.called, 1)
        self.assertEqual(mock_dir.call_count, 5)
        self.assertEqual(self.setup_dataset.source, self.dataset_init_args_matlab['source'])
        self.assertEqual(self.setup_dataset.height, self.dataset_init_args_matlab['height'])
        self.assertEqual(self.setup_dataset.width, self.dataset_init_args_matlab['width'])
        self.assertEqual(self.setup_dataset.channels, self.dataset_init_args_matlab['channels'])
        self.assertEqual(self.setup_dataset.mini_batch_size, self.dataset_init_args_matlab['mini_batch_size'])
        self.assertEqual(self.setup_dataset.mini_batches_per_batch, self.dataset_init_args_matlab['mini_batches_per_batch'])
        self.assertEqual(self.setup_dataset.batches2train,
                         self.dataset_init_args_matlab['batches2train'])
        self.assertEqual(self.setup_dataset.batches2test,
                         self.dataset_init_args_matlab['batches2test'])
        self.assertEqual(self.setup_dataset.batches2validate,
                         self.dataset_init_args_matlab['batches2validate'])
        self.assertEqual(self.setup_dataset.preprocessor, self.preprocess_init_args)

    @patch('yann.utils.dataset.setup_dataset._create_skdata')
    @patch('os.mkdir')
    def test_setup_dataset_skdata(self, mock_dir, mock_create_skdata):
        self.setup_dataset = util_dataset.setup_dataset(self.dataset_init_args_skdata)
        self.assertEqual(mock_create_skdata.called, 1)
        self.assertEqual(mock_dir.call_count, 4)
        self.assertEqual(self.setup_dataset.source, self.dataset_init_args_skdata['source'])
        self.assertEqual(self.setup_dataset.height, 28)
        self.assertEqual(self.setup_dataset.width, 28)
        self.assertEqual(self.setup_dataset.name, self.dataset_init_args_skdata['name'])
        self.assertEqual(self.setup_dataset.channels, 1)
        self.assertEqual(self.setup_dataset.mini_batch_size, 20)
        self.assertEqual(self.setup_dataset.mini_batches_per_batch, (100,20,20))
        self.assertEqual(self.setup_dataset.batches2train,
                         1)
        self.assertEqual(self.setup_dataset.batches2test,
                         1)
        self.assertEqual(self.setup_dataset.batches2validate,
                         1)
        self.assertEqual(self.setup_dataset.preprocessor, self.preprocess_init_args_default)
        self.assertEqual(self.setup_dataset.dataset_location().split('/')[0], '_datasets')


    @patch('scipy.io.loadmat')
    @patch('yann.utils.dataset.cPickle.dump')
    @patch('yann.utils.dataset.open')
    @patch('yann.utils.dataset.preprocessing')
    def test_mat2yann(self,mock_preprocessor, mock_open, mock_dump, mock_loadmat):
        mock_open.return_value = Mock(spec=file)
        mock_loadmat.return_value = self.data_mat_check
        val = util_dataset.load_data_mat(1, 1, 1, 'mock location', 0, '', False)
        mock_preprocessor.return_value = val[0]
        self.setup_dataset = util_dataset.setup_dataset(self.dataset_init_args_matlab, verbose=3)
        self.assertEqual(mock_open.call_count, 4)
        self.assertEqual(mock_dump.call_count, 4)
        self.assertEqual(mock_open.call_count, 4)
        self.assertEqual(mock_preprocessor.call_count, 3)

    @patch('yann.utils.dataset.setup_dataset._create_skdata_mnist')
    @patch('yann.utils.dataset.setup_dataset._create_skdata_caltech101')
    @patch('yann.utils.dataset.setup_dataset._create_skdata_caltech256')
    def test_create_skdata(self, mock_caltech256, mock_caltech101, mock_minst):
        self.setup_dataset = util_dataset.setup_dataset(self.dataset_init_args_skdata, verbose=3)
        self.assertEqual(mock_minst.call_count, 1)

        self.dataset_init_args_skdata['name'] = 'caltech101'
        self.setup_dataset = util_dataset.setup_dataset(self.dataset_init_args_skdata, verbose=3)
        self.assertEqual(mock_caltech101.call_count, 1)

        self.dataset_init_args_skdata['name'] = 'caltech256'
        self.setup_dataset = util_dataset.setup_dataset(self.dataset_init_args_skdata, verbose=3)
        self.assertEqual(mock_caltech256.call_count, 1)
