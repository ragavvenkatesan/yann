import unittest
import numpy as np
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch
import yann.utils.image as util_image
class TestImage(unittest.TestCase):

    def setUp(self):
        self.rgb3d = np.random.rand(5,5,3)
        self.rgb4d = np .random.rand(5, 5,5, 3)
        self.r = np.random.rand(5,5)
        self.g = np.random.rand(5,5)
        self.b = np.random.rand(5,5)
        self.r3d = np.random.rand(5, 5,5)
        self.g3d = np.random.rand(5, 5,5)
        self.b3d = np.random.rand(5, 5,5)
        self.preprocess_init_args1 = {
            "normalize": False,
            "ZCA": False,
            "grayscale": False,
            "zero_mean"	: False,
        }
        self.preprocess_init_args2 = {
            "normalize": True,
            "ZCA": True,
            "grayscale": True,
            "zero_mean"	: True,
        }
        self.preprocess_init_args3 = {
            "normalize": False,
            "ZCA": True,
            "grayscale": True,
            "zero_mean"	: True,
        }
        self.preprocess_init_args4 = {
            "normalize": False,
            "ZCA": True,
            "grayscale": False,
            "zero_mean"	: True,
        }
        self.data = np.random.rand(6)
    def test_rgb2gray(self):
        gray = util_image.rgb2gray(self.rgb3d)
        self.assertEqual(len(gray.shape), 2)
        gray = util_image.rgb2gray(self.rgb4d)
        self.assertEqual(len(gray.shape), 3)

    def test_gray2rgb(self):
        rgb = util_image.gray2rgb(self.r,self.g,self.b, 3)
        self.assertEqual(rgb.shape[2], 3)
        rgb = util_image.gray2rgb(self.r, self.g, self.b, 1)
        self.assertEqual(rgb.shape[0], 3)
        rgb = util_image.gray2rgb(self.r, self.g, self.b, 1)
        self.assertEqual(rgb.shape[0], 3)

    @patch('yann.utils.image.numpy.mean')
    def test_preprocessing(self, mock_mean):
        preprocessed_data = util_image.preprocessing(self.r, 5,1,1,self.preprocess_init_args1)
        self.assertEqual(preprocessed_data.shape, (5,5))
        preprocessed_data = util_image.preprocessing(np.random.rand(5,9), 3, 1, 3, self.preprocess_init_args2)
        self.assertEqual(preprocessed_data.shape, (5, 3))
        preprocessed_data = util_image.preprocessing(np.random.rand(5, 9), 3, 1, 3, self.preprocess_init_args4)
        self.assertEqual(preprocessed_data.shape, (5, 9))
        preprocessed_data = util_image.preprocessing(np.random.rand(5, 9), 3, 1, 3, self.preprocess_init_args1)
        self.assertEqual(preprocessed_data.shape, (5, 9))
        preprocessed_data = util_image.preprocessing(np.random.rand(5, 9), 9, 1, 1, self.preprocess_init_args3)
        self.assertEqual(preprocessed_data.shape, (5, 9))

    def test_check_type(self):
        data_checked_type = util_image.check_type(self.data, 'float32')
        self.assertEqual(data_checked_type.dtype, 'float32')
        data_checked_type = util_image.check_type(self.data, self.data.dtype)
        self.assertEqual(data_checked_type.dtype, self.data.dtype)
