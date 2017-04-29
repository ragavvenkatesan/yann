#location 'root' is always initialized, so exception is not reachable
#update_plot is yet to implement
import unittest
import numpy
from yann.modules.resultor import resultor as r

try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch
from StringIO import StringIO

class TestResultor(unittest.TestCase):

    def setUp(self):
        self.verbose = 3
        self.resultor_init_args = {
                                    'id':'test',
                                    'root':'test',
                                    'results':'test',
                                    'costs':'test',
                                    'confusion':'test',
                                    'learning_rate':'test',
                                    'momentum':'test',
                                    'save_confusion': True
                                  }

        self.wrong_path = 'C:/wrong/path'
        self.exception_msg = 'root variable has not been provided. \
                                            Without a root folder, no save can be performed'
        self.confusion =  numpy.eye(3)
    @patch('yann.modules.resultor.open')
    @patch('os.makedirs')
    def test1_resultor_no_vals(self,mock_makedir,mock_open):
        mock_makedir.return_value = ""
        mock_open.return_value = StringIO('test')
        try:
            r(
                resultor_init_args ={},
                    verbose = self.verbose)
            self.assertEqual(True,True)
        except Exception, c:
            self.assertEqual(True,False)


    @patch('yann.modules.resultor.open')
    @patch('os.makedirs')
    def test2_resultor_vals(self,mock_makedir,mock_open):
        mock_makedir.return_value = ""
        mock_open.return_value = StringIO('test')
        try:
            r(
                resultor_init_args =self.resultor_init_args,
                    verbose = self.verbose)
            self.assertEqual(True,True)
        except Exception, c:
            self.assertEqual(True,False)


    # @patch('yann.modules.resultor.open')
    # @patch('os.makedirs')
    # def test3_resultor_exception(self,mock_makedir,mock_open):
    #     mock_makedir.return_value = ""
    #
    #     mock_open.return_value = StringIO('test')
    #     try:
    #         self.layer = r(
    #             resultor_init_args ={'root':self.wrong_path},
    #                 verbose = self.verbose)
    #         self.assertEqual(True,True)
    #     except Exception, c:
    #         self.assertEqual(True,False)

    @patch('yann.modules.resultor.open')
    @patch('os.makedirs')
    def test4_resultor_process_results(self,mock_makedir,mock_open):
        mock_makedir.return_value = ""
        mock_open.return_value = mock_open(read_data='foo\nbar\nbaz\n')
        try:
            self.res = r(
                resultor_init_args =self.resultor_init_args,
                    verbose = self.verbose)
            self.res.process_results(        cost=10,
                            lr=1,
                            mom=1,verbose = self.verbose    )
            self.assertEqual(True,True)
        except Exception, c:
            self.assertEqual(True,False)


    @patch('yann.modules.resultor.resultor._store_confusion_img')
    @patch('os.path.exists')
    @patch('yann.modules.resultor.open')
    @patch('os.makedirs')
    def test5_resultor_print_confusion(self,mock_makedir,mock_open,mock_path_exists,mock_con_omg):
        mock_con_omg.return_value = ""
        mock_path_exists.return_value = False
        mock_makedir.return_value = ""
        mock_open.return_value = StringIO('test')
        try:
            self.res =r(
                resultor_init_args =self.resultor_init_args,
                    verbose = self.verbose)
            self.res.print_confusion ( epoch=0, train = "train", valid = "valid", test = "test", verbose = self.verbose)
            self.assertEqual(True,True)
        except Exception, c:
            self.assertEqual(True,False)

    @patch('yann.modules.resultor.plt.set_cmap')
    @patch('yann.modules.resultor.plt.xlabel')
    @patch('yann.modules.resultor.plt.ylabel')
    @patch('yann.modules.resultor.plt.colorbar')
    @patch('yann.modules.resultor.plt.title')
    @patch('yann.modules.resultor.plt.text')
    @patch('yann.modules.resultor.plt.close')
    @patch('yann.modules.resultor.plt.savefig')
    @patch('yann.modules.resultor.plt.figure')
    @patch('yann.modules.resultor.plt.matshow')
    @patch('os.path.exists')
    @patch('yann.modules.resultor.open')
    @patch('os.makedirs')
    def test6_resultor_store_confusion_img(self,mock_makedir,mock_open,mock_path_exists,mock_matshow,mock_figure,mock_savefig,mock_close,mock_text,mock_title, mock_colorbar,mock_ylabel,mock_xlabel, mock_cmap):
        mock_savefig.return_value=""
        mock_close.return_value = ""
        mock_matshow.return_value = ""
        mock_figure.return_value = ""
        mock_path_exists.return_value = False
        mock_makedir.return_value = ""
        mock_text.return_value =""
        mock_title.return_value = ""
        mock_colorbar.return_value = ""
        mock_ylabel.return_value = ""
        mock_xlabel.return_value = ""
        mock_cmap.return_value = ""
        mock_open.return_value = StringIO('test')
        try:
            self.res =r(
                resultor_init_args =self.resultor_init_args,
                    verbose = self.verbose)
            self.res._store_confusion_img(confusion=self.confusion, filename="test", verbose = self.verbose)
            self.assertEqual(True,True)
        except Exception as e:
            self.assertEqual(True,False)
