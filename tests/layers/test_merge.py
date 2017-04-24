#dropout_merge_layer code has issues
import unittest
import numpy
import theano
from yann.layers.merge import merge_layer as ml, dropout_merge_layer as dml
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch

class TestMerge(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.verbose = 3
        self.merge_layer_id = "merge"
        self.dropout_merge_layer_id = "dmerge"
        self.input_shape = (1,1,3,3)
        self.input_shape_dim2 = (1,1)
        self.input_ndarray1 = numpy.random.rand(1,1,3,3)
        self.input_tensor1 = theano.shared(self.input_ndarray1)
        self.input_ndarray2 = numpy.random.rand(1,1,3,3)
        self.input_tensor2= theano.shared(self.input_ndarray2)
        self.output_error_shape = (1,)
        self.output_batch_shape = (2,1,3,3)
        self.output_batch_shape_dim2 = (2,1)
        self.output_concat_shape = (1,2,3,3)
        self.output_concat_shape_dim2 = (1,2)
        self.output_error = 1
        self.morethan2_exception =  "Use merge layer for merging only two layers. If you want \
                                    to merge more than one you may use this layer more than once"
        self.notype_exception = " This type is not allowed. "

    @patch('yann.layers.merge.rmse')
    def test1_merge_layer_error_rmse(self,mock_error):
        mock_error.return_value = self.output_error
        self.merge = ml(
            x = (self.input_tensor1,self.input_tensor2),
            input_shape = (self.input_shape,self.input_shape),
            id = self.merge_layer_id,
            type = 'error',
            error = 'rmse',
            verbose= self.verbose
        )
        self.assertEqual(self.merge.output,self.output_error)
        self.assertEqual(self.merge.output_shape,self.output_error_shape)
        self.assertEqual(self.merge.id, self.merge_layer_id)

    @patch('yann.layers.merge.l1')
    def test2_merge_layer_error_l1(self,mock_error):
        mock_error.return_value = self.output_error
        self.merge = ml(
            x = (self.input_tensor1,self.input_tensor2),
            input_shape = (self.input_shape,self.input_shape),
            id = self.merge_layer_id,
            type = 'error',
            error = 'l1',
            verbose= self.verbose
        )
        self.assertEqual(self.merge.output,self.output_error)
        self.assertEqual(self.merge.output_shape,self.output_error_shape)
        self.assertEqual(self.merge.id, self.merge_layer_id)


    @patch('yann.layers.merge.cross_entropy')
    def test3_merge_layer_error_crossentropy(self,mock_error):
        mock_error.return_value = self.output_error
        self.merge = ml(
            x = (self.input_tensor1,self.input_tensor2),
            input_shape = (self.input_shape,self.input_shape),
            id = self.merge_layer_id,
            type = 'error',
            error = 'cross_entropy',
            verbose= self.verbose
        )
        self.assertEqual(self.merge.output,self.output_error)
        self.assertEqual(self.merge.output_shape,self.output_error_shape)
        self.assertEqual(self.merge.id, self.merge_layer_id)


    def test4_merge_layer_exception(self):
        try:
            self.merge = ml(
                x = (self.input_tensor1,self.input_tensor1,self.input_tensor2),
                input_shape = (self.input_shape,self.input_shape,self.input_shape),
                id = self.merge_layer_id,
                type = 'error',
                error = 'cross_entropy',
                verbose= self.verbose
            )
        except Exception,c:
            self.assertEqual(c.message,self.morethan2_exception)


    def test5_merge_layer_sum(self):
        self.merge = ml(
            x = (self.input_tensor1,self.input_tensor2),
            input_shape = (self.input_shape,self.input_shape),
            id = self.merge_layer_id,
            type = 'sum',
            verbose= self.verbose
        )
        self.assertEqual(self.merge.output_shape,self.input_shape)
        self.assertEqual(self.merge.id, self.merge_layer_id)

    def test6_merge_layer_concat(self):
        self.merge = ml(
            x = (self.input_tensor1,self.input_tensor2),
            input_shape = (self.input_shape,self.input_shape),
            id = self.merge_layer_id,
            type = 'concatenate',
            verbose= self.verbose
        )
        self.assertEqual(self.merge.output_shape,self.output_concat_shape)
        self.assertEqual(self.merge.id, self.merge_layer_id)

    def test7_merge_layer_batch(self):
        self.merge = ml(
            x = (self.input_tensor1,self.input_tensor2),
            input_shape = (self.input_shape,self.input_shape),
            id = self.merge_layer_id,
            type = 'batch',
            verbose= self.verbose
        )
        self.assertEqual(self.merge.output_shape,self.output_batch_shape)
        self.assertEqual(self.merge.id, self.merge_layer_id)

    def test6_merge_layer_concat_dim2(self):
        self.merge = ml(
            x = (self.input_tensor1,self.input_tensor2),
            input_shape = (self.input_shape_dim2,self.input_shape_dim2),
            id = self.merge_layer_id,
            type = 'concatenate',
            verbose= self.verbose
        )
        self.assertEqual(self.merge.output_shape,self.output_concat_shape_dim2)
        self.assertEqual(self.merge.id, self.merge_layer_id)

    def test8_merge_layer_batch_dim2(self):
        self.merge = ml(
            x = (self.input_tensor1,self.input_tensor2),
            input_shape = (self.input_shape_dim2,self.input_shape_dim2),
            id = self.merge_layer_id,
            type = 'batch',
            verbose= self.verbose
        )
        self.assertEqual(self.merge.output_shape,self.output_batch_shape_dim2)
        self.assertEqual(self.merge.id, self.merge_layer_id)

    def test9_merge_layer_no_type(self):
        try:
            self.merge = ml(
                x = (self.input_tensor1,self.input_tensor2),
                input_shape = (self.input_shape,self.input_shape),
                id = self.merge_layer_id,
                type = 'WrongType',
                error = 'cross_entropy',
                verbose= self.verbose
            )
        except Exception,c:
            self.assertEqual(c.message,self.notype_exception)

    def test10_merge_layer_batch_dim2(self):
        self.merge = ml(
            x = (self.input_tensor1,self.input_tensor2),
            input_shape = (self.input_shape_dim2,self.input_shape_dim2),
            id = self.merge_layer_id,
            type = 'batch',
            verbose= self.verbose
        )
        self.assertEqual(self.merge.output_shape,self.output_batch_shape_dim2)
        self.assertEqual(self.merge.id, self.merge_layer_id)

    def test11_merge_layer_loss(self):
        self.merge = ml(
            x = (self.input_tensor1,self.input_tensor2),
            input_shape = (self.input_shape_dim2,self.input_shape_dim2),
            id = self.merge_layer_id,
            type = 'sum',
            verbose= self.verbose
        )
        self.merge_l = self.merge.loss()
        self.assertTrue(numpy.allclose(self.merge_l.eval(),self.merge.output.eval()))

    def test12_merge_layer_loss_log(self):
        self.merge = ml(
            x = (self.input_tensor1,self.input_tensor2),
            input_shape = (self.input_shape_dim2,self.input_shape_dim2),
            id = self.merge_layer_id,
            type = 'sum',
            verbose= self.verbose
        )
        self.merge_l = self.merge.loss('log')
        self.assertFalse(numpy.allclose(self.merge_l.eval(),self.merge.output.eval()))


