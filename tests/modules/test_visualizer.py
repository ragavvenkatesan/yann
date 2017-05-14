import unittest
import numpy
from yann.modules.visualizer import save_images
from yann.modules.visualizer import visualizer as v
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch

class TestVisualizer(unittest.TestCase):
    def setUp(self):
        self.input_ndarray = numpy.zeros((1,10,10,1))
        self.input_ndarray_multichannel_3 = numpy.zeros((3,10,10,1))
        self.input_ndarray_multichannel_2 = numpy.zeros((2, 10, 10, 3))
        self.verbose = 3
        self.prefix = "."
        self.input_ndarray_dim_2 = numpy.zeros((1, 1))
        self.input_ndarray_dim_2_case_else_case = numpy.zeros((3, 10))
        self.visualizer_init_args = {
                                    'id':'test',
                                    'root':'test',
                                    'frequency':'test',
                                    'sample_size':'test',
                                    'rgb_filters':'test',
                                    'debug_functions':'test',
                                    'debug_layers':'test',
                                    'save_confusion': True
                                  }

        self.wrong_path = 'C:/wrong/path'
        self.input_batch_size = 1
        self.indices_size = (self.input_batch_size,)
    @patch('yann.modules.visualizer.imsave')
    def test1_save_images_channels_1(self,mock_imsave):
        try:
            mock_imsave.return_value = ""
            save_images(imgs= self.input_ndarray, prefix = self.prefix, is_color= True, verbose = self.verbose)
            self.assertEqual(True, True)
        except Exception, c:
            self.assertEqual(True, False)

    @patch('yann.modules.visualizer.imsave')
    def test2_save_images_channels_3(self,mock_imsave):
        try:
            mock_imsave.return_value = ""
            save_images(imgs= self.input_ndarray_multichannel_3, prefix = self.prefix, is_color= True, verbose = self.verbose)
            self.assertEqual(True, True)
        except Exception, c:
            self.assertEqual(True, False)

    @patch('yann.modules.visualizer.imsave')
    def test3_save_images_channels_2(self,mock_imsave):
        try:
            mock_imsave.return_value = ""
            save_images(imgs= self.input_ndarray_multichannel_2, prefix = self.prefix, is_color= True, verbose = self.verbose)
            self.assertEqual(True, True)
        except Exception, c:
            self.assertEqual(True, False)


    @patch('yann.modules.visualizer.imsave')
    def test4_save_images_channels_dim_2(self,mock_imsave):
        try:
            mock_imsave.return_value = ""
            save_images(imgs= self.input_ndarray_dim_2, prefix = self.prefix, is_color= True, verbose = self.verbose)
            self.assertEqual(True, True)
        except Exception, c:
            self.assertEqual(True, False)

    @patch('yann.modules.visualizer.imsave')
    def test5_save_images_channels_dim_2_else_case(self,mock_imsave):
        try:
            mock_imsave.return_value = ""
            save_images(imgs= self.input_ndarray_dim_2_case_else_case, prefix = self.prefix, is_color= True, verbose = self.verbose)
            self.assertEqual(True, True)
        except Exception, c:
            self.assertEqual(True, False)

    @patch('os.getcwd')
    @patch('os.makedirs')
    def test6_visualizer_no_vals(self,mock_makedir,mock_getcwd):
        mock_getcwd.return_value = self.wrong_path
        mock_makedir.return_value = ""
        try:
            v(
                visualizer_init_args ={},
                    verbose = self.verbose)
            self.assertEqual(True,True)
        except Exception:
            self.assertEqual(True,False)

    @patch('os.getcwd')
    @patch('os.makedirs')
    def test7_visualizer_vals(self,mock_makedir,mock_getcwd):
        mock_getcwd.return_value = self.wrong_path
        mock_makedir.return_value = ""
        try:
            v(
                visualizer_init_args =self.visualizer_init_args,
                    verbose = self.verbose)
            self.assertEqual(True,True)
        except Exception:
            self.assertEqual(True,False)

    @patch('os.getcwd')
    @patch('os.makedirs')
    def test8_visualizer_initialize(self,mock_makedir,mock_getcwd):
        mock_getcwd.return_value = self.wrong_path
        mock_makedir.return_value = ""

        self.v = v(
            visualizer_init_args = {},
                verbose = self.verbose)
        self.assertEqual(True,True)
        self.v.sample_size = self.input_batch_size
        self.v.initialize(batch_size=3, verbose= self.verbose)
        self.assertEqual(self.v.indices.shape,self.indices_size)


    @patch('yann.modules.visualizer.static_theano_print')
    @patch('yann.modules.visualizer.static_printer_import')
    @patch('yann.modules.visualizer.dynamic_printer_import')
    @patch('os.getcwd')
    @patch('os.makedirs')
    def test9_theano_function_visualizer_import_true(self,mock_makedir,mock_getcwd,mock_dynamic,mock_static,mock_theano_print):
        mock_dynamic.return_value = True
        mock_static.return_value = True
        mock_getcwd.return_value = self.wrong_path
        mock_makedir.return_value = ""
        mock_theano_print.return_value = ""
        self.v = v(
            visualizer_init_args = {},
                verbose = self.verbose)
        self.assertEqual(True,True)
        self.v.sample_size = self.input_batch_size
        self.v.initialize(batch_size=3, verbose= self.verbose)
        self.v.theano_function_visualizer(function= "",
                                    short_variable_names = False,
                                    format ='pdf',
                                    verbose = self.verbose)
        self.assertEqual(self.v.indices.shape,self.indices_size)

    @patch('theano.printing.pydot_imported')
    @patch('yann.modules.visualizer.static_theano_print')
    @patch('yann.modules.visualizer.static_printer_import')
    @patch('yann.modules.visualizer.dynamic_printer_import')
    @patch('os.getcwd')
    @patch('os.makedirs')
    def test10_theano_function_visualizer_exp_static(self,mock_makedir,mock_getcwd,mock_dynamic,mock_static,mock_theano_print,mock_pydot):
        mock_dynamic.return_value = False
        mock_static.return_value = True
        mock_getcwd.return_value = self.wrong_path
        mock_makedir.return_value = ""
        mock_theano_print.side_effect = OSError('abc')
        mock_pydot.return_value = False
        try:
            self.v = v(
                visualizer_init_args = {},
                    verbose = self.verbose)
            self.v.sample_size = self.input_batch_size
            self.v.initialize(batch_size=3, verbose= self.verbose)
            self.v.theano_function_visualizer(function="",
                                              short_variable_names="abs",
                                              format='pdf1',
                                              verbose=self.verbose)
            self.assertEqual(True,False)
        except Exception:
            self.assertEqual(True,True)


    @patch('theano.printing.pydot_imported')
    @patch('yann.modules.visualizer.static_theano_print')
    @patch('yann.modules.visualizer.static_printer_import')
    @patch('yann.modules.visualizer.dynamic_printer_import')
    @patch('os.getcwd')
    @patch('os.makedirs')
    def test11_theano_function_visualizer_exp_dynamic(self,mock_makedir,mock_getcwd,mock_dynamic,mock_static,mock_theano_print,mock_pydot):
        mock_dynamic.return_value = True
        mock_static.return_value = False
        mock_getcwd.return_value = self.wrong_path
        mock_makedir.return_value = ""
        mock_theano_print.side_effect = OSError('abc')
        mock_pydot.return_value = False
        try:
            self.v = v(
                visualizer_init_args = {},
                    verbose = self.verbose)
            self.v.sample_size = self.input_batch_size
            self.v.initialize(batch_size=3, verbose= self.verbose)
            self.v.theano_function_visualizer(function="",
                                              short_variable_names="abs",
                                              format='pdf1',
                                              verbose=self.verbose)
            self.assertEqual(True,False)
        except Exception:
            self.assertEqual(True,True)

    @patch('yann.modules.visualizer.save_images')
    @patch('yann.modules.visualizer.static_theano_print')
    @patch('yann.modules.visualizer.static_printer_import')
    @patch('yann.modules.visualizer.dynamic_printer_import')
    @patch('os.getcwd')
    @patch('os.makedirs')
    def test12_theano_function_visualizer_visualize_images(self,mock_makedir,mock_getcwd,mock_dynamic,mock_static,mock_theano_print,mock_save_images):
        mock_dynamic.return_value = True
        mock_static.return_value = True
        mock_getcwd.return_value = self.wrong_path
        mock_makedir.return_value = ""
        mock_theano_print.return_value = ""
        mock_save_images.return_value = ""
        try:
            self.v = v(
                visualizer_init_args = {},
                    verbose = self.verbose)
            self.assertEqual(True,True)
            self.v.sample_size = self.input_batch_size
            self.v.initialize(batch_size=3, verbose= self.verbose)
            self.v.visualize_images(imgs= self.input_ndarray, loc = None, verbose = self.verbose)
            self.assertEqual(True,True)
        except Exception:
            self.assertEqual(True,False)

    # @patch('yann.modules.visualizer.save_images')
    # @patch('yann.modules.visualizer.static_theano_print')
    # @patch('yann.modules.visualizer.static_printer_import')
    # @patch('yann.modules.visualizer.dynamic_printer_import')
    # @patch('os.getcwd')
    # @patch('os.makedirs')
    # def test13_theano_function_visualizer_visualize_activities(self,mock_makedir,mock_getcwd,mock_dynamic,mock_static,mock_theano_print,mock_save_images):
    #     mock_dynamic.return_value = True
    #     mock_static.return_value = True
    #     mock_getcwd.return_value = self.wrong_path
    #     mock_makedir.return_value = ""
    #     mock_theano_print.return_value = ""
    #     mock_save_images.return_value = ""
    #     # try:
    #     self.v = v(
    #         visualizer_init_args = {},
    #             verbose = self.verbose)
    #     self.assertEqual(True,True)
    #     self.v.sample_size = self.input_batch_size
    #     self.v.initialize(batch_size=3, verbose= self.verbose)
    #     self.layer_activities ={   'a': ['xyz1', 'xyz2'],
    #                                 'b': ['xyz3', 'xyz4'],
    #                                 'c': ['xyz5'],
    #                                 'd': ['xyz6']}
    #     self.v.visualize_activities(layer_activities=self.layer_activities, epoch =0, index = 0, verbose = self.verbose)
    #     self.assertEqual(True,True)
        # except Exception:
        #     self.assertEqual(True,False)
