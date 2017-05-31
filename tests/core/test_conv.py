import unittest
import numpy
import theano
from yann.core.conv import convolver_2d as cv
from yann.core.conv import deconvolver_2d as dv
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch

class TestConv(unittest.TestCase):
    def setUp(self):
        self.verbose = 3
        self.input_shape = (1,1,10,10)
        self.input_ndarray = numpy.random.rand(1,1,10,10)
        self.input_tensor = theano.shared(self.input_ndarray)
        self.default_param_value = [1.]
        self.custom_param_value = [1., 1.,1.]
        self.filter_shape = (1,1,5,5)
        self.filter_shape_exception = (1, 5, 5, 5)
        self.filter_ndarray = numpy.random.rand(1,1,5,5)
        self.filter_tensor = theano.shared(self.filter_ndarray)
        self.subsample = (1,1)
        self.conv_shape = (1,1,1,1)
        self.exception_msg_conv = "input_shape[1] and filter_shape[1] must match"
        self.exception_msg_deconv =" This dimensionality of th output image cannot be achieved."

    @patch('yann.core.conv.conv_shape')
    @patch('yann.core.conv.conv2d')
    def test1_conv_layer_2d(self,mock_conv2d,mock_conv_shape):
        mock_conv2d.return_value = self.input_ndarray
        mock_conv_shape.return_value = self.conv_shape
        self.conv_layer_2d = cv(
                            input = self.input_tensor,
                            filters = self.filter_tensor ,
                            subsample = self.subsample,
                            filter_shape = self.filter_shape,
                            image_shape = self.input_shape,
                            border_mode='valid',
                            verbose = self.verbose
        )
        self.assertEqual(self.conv_layer_2d.out_shp,(self.conv_shape[2],self.conv_shape[3]))
        self.assertTrue(numpy.allclose(self.conv_layer_2d.out,self.input_ndarray))

    @patch('yann.core.conv.conv_shape')
    @patch('yann.core.conv.conv2d')
    def test2_conv_layer_2d_exception(self,mock_conv2d,mock_conv_shape):
        mock_conv2d.return_value = self.input_ndarray
        mock_conv_shape.return_value = self.conv_shape
        try:
            self.conv_layer_2d = cv(
                                input = self.input_tensor,
                                filters = self.filter_tensor ,
                                subsample = self.subsample,
                                filter_shape = self.filter_shape_exception,
                                image_shape = self.input_shape,
                                border_mode='valid',
                                verbose = self.verbose
            )
        except Exception,c:
            self.assertEqual(c.message,self.exception_msg_conv)

    @patch('yann.core.conv.conv_shape')
    @patch('yann.core.conv.deconv2d')
    def test3_deconv_layer_2d(self,mock_deconv2d,mock_conv_shape):
        mock_deconv2d.return_value = self.input_ndarray
        mock_conv_shape.return_value = self.input_shape
        self.deconv_layer_2d = dv(
                            input = self.input_tensor,
                            filters = self.filter_tensor ,
                            subsample = self.subsample,
                            filter_shape = self.filter_shape,
                            image_shape = self.input_shape,
                            output_shape= self.input_shape,
                            border_mode='valid',
                            verbose = self.verbose
        )
        self.assertEqual(self.deconv_layer_2d.out_shp,(self.input_shape[2],self.input_shape[3]))
        self.assertTrue(numpy.allclose(self.deconv_layer_2d.out,self.input_ndarray))

    @patch('yann.core.conv.conv_shape')
    @patch('yann.core.conv.deconv2d')
    def test4_deconv_layer_2d_exception(self,mock_deconv2d,mock_deconv_shape):
        mock_deconv2d.return_value = self.input_ndarray
        mock_deconv_shape.return_value = (2,2,0,10)
        try:
            self.deconv_layer_2d = dv(
                                input = self.input_tensor,
                                filters = self.filter_tensor ,
                                subsample = self.subsample,
                                filter_shape = self.filter_shape,
                                image_shape = self.input_shape,
                                output_shape= self.input_shape,
                                border_mode='valid',
                                verbose = self.verbose
            )
        except Exception,c:
            self.assertEqual(c.message, self.exception_msg_deconv)
