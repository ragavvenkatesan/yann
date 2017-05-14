import unittest
import numpy
import theano
import theano.tensor as T
import yann.network as n
from yann.network import network as net
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch


class TestErrors(unittest.TestCase):

    def setUp(self):
        self.verbose = 3
        self.input_shape = (1,10,10,10)
        self.input_ndarray = numpy.random.rand(1,1,10,10)
        self.datastream_exception_msg = "Can't setup an input layer without dataset initialized"
        self.conv_pool_orgin_exception_msg = "You can't create a convolutional layer without an" + \
                                    " origin layer."
        self.deconv_pool_origin_exception_msg = "You can't create a deconvolutional layer without an" + \
                                    " origin layer."
        self.output_shape_exception_msg = "output shape not provided for the deconv layer"
        self.flatten_origin_exception_msg = "You can't create a flatten without a layer to flatten."
        self.unflatten_shape_exception_msg = "This type of layer needs a shape variable to unflatten to"
        self.dot_prod_origin_exception_msg = "You can't create a fully connected layer without an" + \
                                    " origin layer."
        self.classifier_origin_exception_msg = "You can't create a softmax layer without an" + \
                                    " origin layer."
        self.classifier_class_exception_msg = "Supply number of classes"
        self.objective_origin_exception_msg = "You can't create an abstract objective layer without a" + \
                                    " classifier layer."
        self.objective_origin_type_exception_msg = "layer-layer loss is not supported by the objective layer, use \
                                  merge layer for it ."

        self.objective_no_lossfn_exception_msg = "Layer input doesn't provide a loss function"
        self.merge_origin_exception_msg = "You can't create an merge layer without atleast two" + \
                                    "layer ids as input."
        self.single_input_merge_layer_exception = "layer-layer loss needs a tuple as origin"
        self.merge_tensor_type_exception = "If using tensor as a type, you need to supply input shapes"
        self.random_origin_type_exception = "You can't create a fully connected layer without an" + \
                                    " origin layer."
        self.tensor_no_input_exception_msg = "Needs an input tensor"
        self.tensor_no_input_shape_exception_msg = "Needs an input shape"
        self.batch_norm_layer_exception_msg = "You can't create a batch norm layer without an" + \
                                    " origin layer."


    def test1_sink(self):
        self.assertEqual(n._sink(),None)

    def test2_init_True(self):
        self.net = net(verbose=self.verbose)
        self.assertEqual(self.net.layer_graph,{})
        self.assertEqual(self.net.borrow,True)

    @patch('yann.network.nx_installed')
    def test3_init_False(self,mock_nx):
        mock_nx.return_value = False
        self.net = net(verbose=self.verbose,**{'borrow':False})
        self.assertEqual(self.net.borrow,False)

    def test4_layer_activity(self):
        self.net = net(verbose=self.verbose)
        x = theano.tensor.dscalar()
        self.func = theano.function([x], x)
        self.index = 0
        self.net.layer_activities = {self.index:self.func}
        layer_activity =self.net.layer_activity(index=self.index,id=0,verbose=self.verbose)
        self.assertEqual(layer_activity,0)


    @patch('yann.modules.resultor.resultor')
    def test5_add_resultor_no_inputs(self, mock_resultor):
        mock_resultor.return_value = "test"
        self.net = net(verbose=self.verbose)
        self.net._add_resultor(resultor_params = None, verbose = self.verbose)
        self.assertEqual(self.net.last_resultor_created,'1')

    @patch('yann.modules.resultor.resultor')
    def test6_add_resultor_inputs(self, mock_resultor):
        mock_resultor.return_value = "test"
        self.net = net(verbose=self.verbose)
        self.input_id = '2'
        self.resultor_params = {'id':self.input_id}
        self.net._add_resultor(resultor_params = self.resultor_params, verbose = self.verbose)
        self.assertEqual(self.net.last_resultor_created,self.input_id)

    @patch('yann.modules.visualizer.visualizer')
    def test7_add_visualizer_no_inputs(self, mock_visualizer):
        mock_visualizer.return_value = "test"
        self.net = net(verbose=self.verbose)
        self.net._add_visualizer(visualizer_params = {}, verbose = self.verbose)
        self.assertEqual(self.net.last_visualizer_created,1)

    @patch('yann.modules.visualizer.visualizer')
    def test8_add_visualizer_inputs(self, mock_visualizer):
        mock_visualizer.return_value = "test"
        self.net = net(verbose=self.verbose)
        self.input_id = 2
        self.visualizer_params = {'id': self.input_id}
        self.net._add_visualizer(visualizer_params = self.visualizer_params, verbose = self.verbose)
        self.assertEqual(self.net.last_visualizer_created,2)

    @patch('yann.modules.optimizer.optimizer')
    def test9_add_optimizer_no_inputs(self, mock_optimizer):
        mock_optimizer.return_value = "test"
        self.net = net(verbose=self.verbose)
        self.net._add_optimizer(optimizer_params={}, verbose=self.verbose)
        self.assertEqual(self.net.last_optimizer_created, '1')

    @patch('yann.modules.optimizer.optimizer')
    def test10_add_optimizer_inputs(self, mock_optimizer):
        mock_optimizer.return_value = "test"
        self.net = net(verbose=self.verbose)
        self.input_id = 2
        self.optimizer_params = {'id': self.input_id}
        self.net._add_optimizer(optimizer_params=self.optimizer_params, verbose=self.verbose)
        self.assertEqual(self.net.last_optimizer_created, 2)

    @patch('yann.modules.datastream.datastream')
    def test11_add_datastream_no_inputs(self, mock_datastream):
        mock_datastream.return_value = "test"
        self.net = net(verbose=self.verbose)
        self.net._add_datastream(dataset_params={}, verbose=self.verbose)
        self.assertEqual(self.net.last_datastream_created, '1')

    @patch('yann.modules.datastream.datastream')
    def test12_add_datastream_inputs(self,mock_datastream):
        mock_datastream.return_value = "test"
        self.net = net(verbose=self.verbose)
        self.input_id = 2
        self.datastream_params = {'id': self.input_id}
        self.net._add_datastream(dataset_params=self.datastream_params, verbose=self.verbose)
        self.assertEqual(self.net.last_datastream_created, 2)

    def test13_add_input_layer_exception(self):
        try:
            self.net = net(verbose=self.verbose)
            self.net._add_input_layer(id="input",options={},verbose=self.verbose)
        except Exception,c:
            self.assertEqual(c.message,self.datastream_exception_msg)

    @patch('yann.layers.input.input_layer')
    @patch('yann.layers.input.dropout_input_layer')
    @patch('yann.network.network._add_datastream')
    def test14_add_input_layer_with_datastream_params_without_init(self,mock_datastream,mock_dl,mock_l):
        mock_l.return_value = MockLayer()
        mock_dl.return_value = MockLayer()
        mock_datastream.return_value = ""
        self.net = net(verbose=self.verbose)
        self.mockds = MockDS()
        self.net.datastream = {'test_dataset':self.mockds}
        self.net.last_datastream_created = 'test_dataset'
        self.dataset_init_args = {'id':'test_dataset'}
        self.options = {'dataset_init_args':self.dataset_init_args}
        self.net._add_input_layer(id="input", options=self.options, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["input"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["input"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["input"].origin)>0)

    @patch('yann.layers.input.input_layer')
    @patch('yann.layers.input.dropout_input_layer')
    @patch('yann.network.network._add_datastream')
    def test15_add_input_layer_with_datastream_params_with_init(self,mock_datastream,mock_dl,mock_l):
        mock_l.return_value = MockLayer()
        mock_dl.return_value = MockLayer()
        mock_datastream.return_value = ""
        self.net = net(verbose=self.verbose)
        self.mockds = MockDS()
        self.net.datastream = {'test_dataset':self.mockds}
        self.net.last_datastream_created = 'test_dataset'
        self.dataset_init_args = {}
        self.options = {'dataset_init_args':self.dataset_init_args,'origin':'test_dataset','mean_subtract':0,'dropout_rate':0}
        self.net._add_input_layer(id="input", options=self.options, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["input"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["input"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["input"].origin)>0)

    def test16_add_conv_layer_no_init(self):
        self.net = net(verbose=self.verbose)
        try:
            self.net._add_conv_layer(id="conv",options={},verbose=self.verbose)
            self.assertEqual(False, True)
        except Exception, c:
            self.assertEqual(c.message,self.conv_pool_orgin_exception_msg)

    @patch('yann.layers.conv_pool.dropout_conv_pool_layer_2d')
    @patch('yann.layers.conv_pool.conv_pool_layer_2d')
    def test17_add_conv_layer_origin(self,mock_cl,mock_dcl):
        mock_cl.return_value = MockLayer()
        mock_dcl.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input':MockInputLayer()}
        self.net.dropout_layers = {'input':MockInputLayer()}
        self.net.inference_layers = {'input':MockInputLayer()}
        self.net.last_layer_created = "input"
        self.net._add_conv_layer(id="conv", options={}, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["conv"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["conv"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["conv"].origin)>0)

    @patch('yann.layers.conv_pool.dropout_conv_pool_layer_2d')
    @patch('yann.layers.conv_pool.conv_pool_layer_2d')
    def test18_add_conv_layer_values(self,mock_cl,mock_dcl):
        mock_cl.return_value = MockLayer()
        mock_dcl.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input':MockInputLayer()}
        self.net.dropout_layers = {'input':MockInputLayer()}
        self.net.inference_layers = {'input':MockInputLayer()}
        self.net.last_layer_created = "input"
        self.options = {
            'origin': 'input',
            'num_neurons': 10,
            'filter_size' : (3,3),
            'activation' : 'relu',
            'border_mode': 'valid',
            'stride' : (1,1),
            'batch_norm': True,
            'pool_size' : (1,1),
            'pool_type': 'max',
            'input_params': None,
            'dropout_rate' : 0.5,
            'regularize' : True

        }
        self.net._add_conv_layer(id="conv", options=self.options, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["conv"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["conv"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["conv"].origin)>0)


    def test19_add_deconv_layer_no_init(self):
        self.net = net(verbose=self.verbose)
        try:
            self.net._add_deconv_layer(id="deconv",options={},verbose=self.verbose)
            self.assertEqual(False, True)
        except Exception, c:
            self.assertEqual(c.message,self.deconv_pool_origin_exception_msg)


    @patch('yann.layers.conv_pool.dropout_deconv_layer_2d')
    @patch('yann.layers.conv_pool.deconv_layer_2d')
    def test20_add_deconv_layer_origin_output_shape_exception(self,mock_cl,mock_dcl):
        mock_cl.return_value = MockLayer()
        mock_dcl.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        try:
            self.net.layers = {'input':MockInputLayer()}
            self.net.dropout_layers = {'input':MockInputLayer()}
            self.net.inference_layers = {'input':MockInputLayer()}
            self.net.last_layer_created = "input"
            self.net._add_deconv_layer(id="deconv", options={}, verbose=self.verbose)
            self.assertEqual(False,True)
        except Exception,c:
            self.assertEqual(c.message,self.output_shape_exception_msg)


    @patch('yann.layers.conv_pool.dropout_deconv_layer_2d')
    @patch('yann.layers.conv_pool.deconv_layer_2d')
    def test21_add_deconv_layer_no_init_except_outshape(self,mock_cl,mock_dcl):
        mock_cl.return_value = MockLayer()
        mock_dcl.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input':MockInputLayer()}
        self.net.dropout_layers = {'input':MockInputLayer()}
        self.net.inference_layers = {'input':MockInputLayer()}
        self.net.last_layer_created = "input"
        self.options = {'output_shape':(1,1,10,10)}
        self.net._add_deconv_layer(id="deconv", options=self.options, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["deconv"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["deconv"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["deconv"].origin)>0)

    @patch('yann.layers.conv_pool.dropout_deconv_layer_2d')
    @patch('yann.layers.conv_pool.deconv_layer_2d')
    def test22_add_deconv_layer_all_vals(self,mock_cl,mock_dcl):
        mock_cl.return_value = MockLayer()
        mock_dcl.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input':MockInputLayer()}
        self.net.dropout_layers = {'input':MockInputLayer()}
        self.net.inference_layers = {'input':MockInputLayer()}
        self.net.last_layer_created = "input"
        self.options = {
            'origin': 'input',
            'num_neurons': 10,
            'filter_size' : (3,3),
            'activation' : 'relu',
            'border_mode': 'valid',
            'stride' : (1,1),
            'batch_norm': True,
            'input_params': None,
            'dropout_rate' : 0.5,
            'regularize' : True,
            'output_shape':(1,1,10,10)
        }
        self.net._add_deconv_layer(id="deconv", options=self.options, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["deconv"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["deconv"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["deconv"].origin)>0)

    def test23_add_flatten_layer_origin_exception(self):
        self.net = net(verbose=self.verbose)
        try:
            self.net._add_flatten_layer(id="flatten",options={},verbose=self.verbose)
            self.assertEqual(False, True)
        except Exception, c:
            self.assertEqual(c.message,self.flatten_origin_exception_msg)


    @patch('yann.layers.flatten.flatten_layer')
    def test24_add_flatten_layer_no_init(self,mock_fl):
        mock_fl.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input':MockInputLayer()}
        self.net.dropout_layers = {'input':MockInputLayer()}
        self.net.inference_layers = {'input':MockInputLayer()}
        self.net.last_layer_created = "input"
        self.options = {}
        self.net._add_flatten_layer(id="flatten", options=self.options, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["flatten"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["flatten"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["flatten"].origin)>0)

    @patch('yann.layers.flatten.flatten_layer')
    def test25_add_flatten_layer_with_origin(self,mock_fl):
        mock_fl.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input':MockInputLayer()}
        self.net.dropout_layers = {'input':MockInputLayer()}
        self.net.inference_layers = {'input':MockInputLayer()}
        self.net.last_layer_created = "input"
        self.options = {'origin':"input"}
        self.net._add_flatten_layer(id="flatten", options=self.options, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["flatten"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["flatten"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["flatten"].origin)>0)

    def test26_add_unflatten_layer_origin_exception(self):
        self.net = net(verbose=self.verbose)
        try:
            self.net._add_unflatten_layer(id="unflatten",options={},verbose=self.verbose)
            self.assertEqual(False, True)
        except Exception, c:
            self.assertEqual(c.message,self.flatten_origin_exception_msg)

    def test27_add_unflatten_layer_origin_exception(self):
        self.net = net(verbose=self.verbose)
        try:
            self.net.last_layer_created = "input"
            self.net._add_unflatten_layer(id="unflatten",options={},verbose=self.verbose)
            self.assertEqual(False, True)
        except Exception, c:
            self.assertEqual(c.message,self.unflatten_shape_exception_msg)

    @patch('yann.layers.flatten.unflatten_layer')
    def test28_add_unflatten_layer(self, mock_ufl):
        mock_ufl.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.last_layer_created = "input"
        self.net.layers = {'input':MockInputLayer()}
        self.net.dropout_layers = {'input':MockInputLayer()}
        self.net.inference_layers = {'input':MockInputLayer()}
        self.options = {'origin':'input','shape':(1,1,10,10)}
        self.net._add_unflatten_layer(id="unflatten",options=self.options,verbose=self.verbose)
        self.assertTrue(len(self.net.layers["unflatten"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["unflatten"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["unflatten"].origin)>0)


    def test29_add_dot_product_layer_origin_exception(self):
        self.net = net(verbose=self.verbose)
        try:
            self.net._add_dot_product_layer(id="dp",options={},verbose=self.verbose)
            self.assertEqual(False, True)
        except Exception, c:
            self.assertEqual(c.message,self.dot_prod_origin_exception_msg)

    @patch('yann.network.network.add_layer')
    @patch('yann.layers.fully_connected.dropout_dot_product_layer')
    @patch('yann.layers.fully_connected.dot_product_layer')
    def test30_add_dot_product_layer_no_init(self,mock_dpl,mock_ddpl,mock_add_layer):
        mock_add_layer.return_value = ""
        mock_dpl.return_value = MockLayer()
        mock_ddpl.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input':MockInputLayer()}
        self.net.dropout_layers = {'input':MockInputLayer()}
        self.net.inference_layers = {'input':MockInputLayer()}
        self.net.last_layer_created = "input"
        self.options = {}
        self.net._add_dot_product_layer(id="dp", options=self.options, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["dp"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["dp"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["dp"].origin)>0)

    @patch('yann.network.network.add_layer')
    @patch('yann.layers.fully_connected.dropout_dot_product_layer')
    @patch('yann.layers.fully_connected.dot_product_layer')
    def test31_add_dot_product_layer_values(self,mock_dpl,mock_ddpl,mock_add_layer):
        mock_add_layer.return_value = ""
        mock_dpl.return_value = MockLayer()
        mock_ddpl.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input':MockInputLayer()}
        self.net.dropout_layers = {'input':MockInputLayer()}
        self.net.inference_layers = {'input':MockInputLayer()}
        self.net.last_layer_created = "input"
        self.options = {
            'origin': 'input',
            'num_neurons': 10,
            'activation': 'relu',
            'batch_norm': True,
            'input_params': None,
            'dropout_rate': 0.5,
            'regularize': True
        }
        self.net._add_dot_product_layer(id="dp", options=self.options, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["dp"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["dp"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["dp"].origin)>0)

    def test32_add_classfier_product_layer_origin_exception(self):
        self.net = net(verbose=self.verbose)
        try:
            self.net._add_classifier_layer(id="cl",options={},verbose=self.verbose)
            self.assertEqual(False, True)
        except Exception, c:
            self.assertEqual(c.message,self.classifier_origin_exception_msg)

    @patch('yann.network.network.add_layer')
    @patch('yann.layers.output.classifier_layer')
    def test33_add_classfier_layer_no_init(self,mock_cl,mock_add_layer):
        mock_add_layer.return_value = ""
        mock_cl.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input':MockInputLayer()}
        self.net.dropout_layers = {'input':MockInputLayer()}
        self.net.inference_layers = {'input':MockInputLayer()}
        self.net.last_layer_created = "input"
        self.options = {'num_classes': 10,}
        self.net._add_classifier_layer(id="cl", options=self.options, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["cl"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["cl"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["cl"].origin)>0)

    @patch('yann.network.network.add_layer')
    @patch('yann.layers.output.classifier_layer')
    def test34_add_classfier_layer_values(self,mock_dpl,mock_add_layer):
        mock_add_layer.return_value = ""
        mock_dpl.return_value = MockLayer()
        mock_add_layer.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input':MockInputLayer()}
        self.net.dropout_layers = {'input':MockInputLayer()}
        self.net.inference_layers = {'input':MockInputLayer()}
        self.net.last_layer_created = "input"
        self.options = {
            'origin': 'input',
            'num_classes': 10,
            'activation': 'relu',
            'input_params': None,
            'regularize': True
        }
        self.net._add_classifier_layer(id="cl", options=self.options, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["cl"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["cl"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["cl"].origin)>0)


    @patch('yann.network.network.add_layer')
    @patch('yann.layers.output.classifier_layer')
    def test35_add_classfier_layer_no_class_exception(self,mock_dpl,mock_add_layer):
        self.net = net(verbose=self.verbose)
        self.net.last_layer_created = "input"
        self.options = {'origin': 'input',}
        mock_add_layer.return_value = ""
        mock_dpl.return_value = MockLayer()
        mock_add_layer.return_value = MockLayer()
        self.net.layers = {'input': MockInputLayer()}
        self.net.dropout_layers = {'input':MockInputLayer()}
        self.net.inference_layers = {'input':MockInputLayer()}
        try:
            self.net._add_classifier_layer(id="cl", options=self.options, verbose=self.verbose)
            self.assertEqual(False, True)
        except Exception, c:
            self.assertEqual(c.message, self.classifier_class_exception_msg)


    def test36_add_objective_layer_origin_exception(self):
        self.net = net(verbose=self.verbose)
        try:
            self.net._add_objective_layer(id="ol",options={},verbose=self.verbose)
            self.assertEqual(False, True)
        except Exception, c:
            self.assertEqual(c.message,self.objective_origin_exception_msg)

    def test37_add_objective_layer_origin_type_exception(self):
        self.net = net(verbose=self.verbose)
        try:
            self.options = {'origin': ['input','batch']}
            self.net._add_objective_layer(id="ol",options=self.options,verbose=self.verbose)
            self.assertEqual(False, True)
        except Exception, c:
            self.assertEqual(c.message,self.objective_origin_type_exception_msg)



    @patch('yann.network.network.add_layer')
    @patch('yann.layers.output.objective_layer')
    def test38_add_objective_layer_no_init(self,mock_ol,mock_add_layer):
        try:
            mock_add_layer.return_value = ""
            mock_ol.return_value = MockLayer()
            self.net = net(verbose=self.verbose)
            self.net.layers = {'input':MockInputLayer()}
            self.net.dropout_layers = {'input':MockInputLayer()}
            self.net.inference_layers = {'input':MockInputLayer()}
            self.net.last_classifier_created = "input"
            self.options = {'type': "classifier",}
            self.last_classifier_created = "cl"
            self.net._add_objective_layer(id="ol", options=self.options, verbose=self.verbose)
            self.assertEqual(False, True)
        except Exception, c:
            self.assertEqual(c.message,self.objective_no_lossfn_exception_msg)

    @patch('yann.network.network.add_layer')
    @patch('yann.layers.output.objective_layer')
    def test39_add_objective_layer_no_no_lossfn_exception(self, mock_ol, mock_add_layer):
        try:
            mock_add_layer.return_value = ""
            mock_ol.return_value = MockLayer()
            self.net = net(verbose=self.verbose)
            self.net.layers = {'input': MockInputLayer()}
            self.net.dropout_layers = {'input': MockInputLayer()}
            self.net.inference_layers = {'input': MockInputLayer()}
            self.net.last_classifier_created = "input"
            self.options = {'type': "classifier", }
            self.last_classifier_created = "cl"
            self.net._add_objective_layer(id="ol", options=self.options, verbose=self.verbose)
            self.assertEqual(False, True)
        except Exception, c:
            self.assertEqual(c.message, self.objective_no_lossfn_exception_msg)

    @patch('yann.network.network.add_layer')
    @patch('yann.layers.output.objective_layer')
    def test40_add_objective_layer_init1(self, mock_ol, mock_add_layer):
        mock_add_layer.return_value = ""
        mock_ol.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input': MockInputLayerLoss()}
        self.net.dropout_layers = {'input': MockInputLayer()}
        self.net.inference_layers = {'input': MockInputLayer()}
        self.net.last_classifier_created = "input"
        self.mockds = MockDS()
        self.net.datastream = {'test_dataset':self.mockds}
        self.net.last_datastream_created = 'test_dataset'
        self.dataset_init_args = {'id':'test_dataset'}
        self.options = {'type': "value",
                        'layer_type' : 'decision',
                        'objective' : 'softmax',
                        'dataset_origin':'ds',
                        'regularizer':[0,0]
                        }
        self.last_classifier_created = "cl"
        self.net._add_objective_layer(id="ol", options=self.options, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["ol"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["ol"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["ol"].origin)>0)

    @patch('yann.network.network.add_layer')
    @patch('yann.layers.output.objective_layer')
    def test41_add_objective_layer_init2(self, mock_ol, mock_add_layer):
        mock_add_layer.return_value = ""
        mock_ol.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input': MockInputLayerLoss()}
        self.net.dropout_layers = {'input': MockInputLayer()}
        self.net.inference_layers = {'input': MockInputLayer()}
        self.net.last_classifier_created = "input"
        self.mockds = MockDS()
        self.net.datastream = {'test_dataset':self.mockds}
        self.net.last_datastream_created = 'test_dataset'
        self.dataset_init_args = {'id':'test_dataset'}
        self.options = {'type': "value",
                        'layer_type' : 'decision',
                        'objective' : 'hinge',
                        'dataset_origin':'ds',
                        'regularizer':[0,0]
                        }
        self.last_classifier_created = "cl"
        self.net._add_objective_layer(id="ol", options=self.options, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["ol"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["ol"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["ol"].origin)>0)


    def test42_add_merge_layer_no_orgin_exception(self):
        self.net = net(verbose=self.verbose)
        try:
            self.net._add_merge_layer(id="ml",options={},verbose=self.verbose)
            self.assertEqual(False, True)
        except Exception, c:
            self.assertEqual(c.message,self.merge_origin_exception_msg)

    def test43_add_merge_layer_tuple_exception(self):
        self.net = net(verbose=self.verbose)
        try:
            self.options = {'origin':"input"}
            self.net._add_merge_layer(id="ml",options=self.options,verbose=self.verbose)
            self.assertEqual(False, True)
        except Exception, c:
            self.assertEqual(c.message,self.single_input_merge_layer_exception)

    @patch('yann.layers.merge.merge_layer')
    def test44_add_merge_layer_no_init(self, mock_merge):
        mock_merge.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.options = {'origin': ("input","input")}
        self.net.layers = {'input': MockInputLayer()}
        self.net.dropout_layers = {'input': MockDropoutLayerMerge()}
        self.net.inference_layers = {'input': MockInputLayer()}
        self.net._add_merge_layer(id="ml", options=self.options, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["ml"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["ml"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["ml"].origin)>0)

    @patch('yann.layers.merge.merge_layer')
    def test45_add_merge_layer_init(self, mock_merge):
        mock_merge.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.options = {'origin': ("input","input"),
                        'error':'rmse',
                        'layer_type':'error',
                        'input_type':'layer'
                        }
        self.net.layers = {'input': MockInputLayer()}
        self.net.dropout_layers = {'input': MockDropoutLayerMerge()}
        self.net.inference_layers = {'input': MockInputLayer()}
        self.net._add_merge_layer(id="ml", options=self.options, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["ml"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["ml"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["ml"].origin)>0)

    @patch('yann.layers.merge.merge_layer')
    def test46_add_merge_layer_tensor_exception(self, mock_merge):
        mock_merge.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.options = {'origin': ("input","input"),
                        'error':'rmse',
                        'layer_type':'error',
                        'input_type': 'tensor'
                        }
        self.net.layers = {'input': MockInputLayer()}
        self.net.dropout_layers = {'input': MockDropoutLayerMerge()}
        self.net.inference_layers = {'input': MockInputLayer()}
        try:
            self.net._add_merge_layer(id="ml", options=self.options, verbose=self.verbose)
        except Exception, c:
            self.assertEqual(c.message,self.merge_tensor_type_exception)

    @patch('yann.layers.merge.merge_layer')
    def test47_add_merge_layer_init2(self, mock_merge):
        mock_merge.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.options = {'origin': ("input", "input"),
                        'error': 'rmse',
                        'layer_type': 'error',
                        'input_type': 'tensor',
                        'input_shape':(1,1,10,10)
                        }
        self.net.layers = {'input': MockInputLayer()}
        self.net.dropout_layers = {'input': MockDropoutLayerMerge()}
        self.net.inference_layers = {'input': MockInputLayer()}
        self.net._add_merge_layer(id="ml", options=self.options, verbose=self.verbose)
        self.assertEqual(self.net.inference_layers["ml"].origin,MockLayer().origin)

    @patch('yann.layers.random.random_layer')
    def test48_add_random_layer_no_init(self,mock_random):
        mock_random.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input': MockInputLayer()}
        self.net.dropout_layers = {'input': MockInputLayer()}
        self.net.inference_layers = {'input': MockInputLayer()}
        self.options = {}
        self.net._add_random_layer(id="rl", options=self.options, verbose=self.verbose)
        self.assertEqual(self.net.layers["rl"].origin,MockLayer().origin)
        self.assertEqual(self.net.dropout_layers["rl"].origin,MockLayer().origin)
        self.assertEqual(self.net.inference_layers["rl"].origin,MockLayer().origin)

    @patch('yann.layers.random.random_layer')
    def test49_add_random_layer_init(self,mock_random):
        mock_random.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input': MockInputLayer()}
        self.net.dropout_layers = {'input': MockInputLayer()}
        self.net.inference_layers = {'input': MockInputLayer()}
        self.options = {
                        'distribution':'binomial',
                        'num_neurons' :100
                        }
        self.net._add_random_layer(id="rl", options=self.options, verbose=self.verbose)
        self.assertEqual(self.net.layers["rl"].origin,MockLayer().origin)
        self.assertEqual(self.net.dropout_layers["rl"].origin,MockLayer().origin)
        self.assertEqual(self.net.inference_layers["rl"].origin,MockLayer().origin)

    def test50_add_rotate_layer_origin_exception(self):
        self.net = net(verbose=self.verbose)
        try:
            self.net._add_rotate_layer(id="rl",options={},verbose=self.verbose)
            self.assertEqual(False, True)
        except Exception, c:
            self.assertEqual(c.message,self.random_origin_type_exception)

    @patch('yann.network.network.add_layer')
    @patch('yann.layers.transform.rotate_layer')
    @patch('yann.layers.transform.dropout_rotate_layer')
    def test51_add_rotate_layer_no_init(self, mock_drl,mock_rl, mock_add_layer):
        mock_add_layer.return_value = ""
        mock_rl.return_value = MockLayer()
        mock_drl.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input': MockInputLayerLoss()}
        self.net.dropout_layers = {'input': MockInputLayer()}
        self.net.inference_layers = {'input': MockInputLayer()}
        self.net.last_classifier_created = "input"
        self.mockds = MockDS()
        self.net.datastream = {'test_dataset':self.mockds}
        self.net.last_layer_created = 'input'
        self.options = {
                        }
        self.net._add_rotate_layer(id="rl", options=self.options, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["rl"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["rl"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["rl"].origin)>0)

    @patch('yann.network.network.add_layer')
    @patch('yann.layers.transform.rotate_layer')
    @patch('yann.layers.transform.dropout_rotate_layer')
    def test52_add_rotate_layer_init(self, mock_drl,mock_rl, mock_add_layer):
        mock_add_layer.return_value = ""
        mock_rl.return_value = MockLayer()
        mock_drl.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input': MockInputLayerLoss()}
        self.net.dropout_layers = {'input': MockInputLayer()}
        self.net.inference_layers = {'input': MockInputLayer()}
        self.net.last_classifier_created = "input"
        self.mockds = MockDS()
        self.net.datastream = {'test_dataset':self.mockds}
        self.net.last_layer_created = 'input'
        self.options = {
                        'origin':'input',
                        'angle' : None
                        }
        self.net._add_rotate_layer(id="rl", options=self.options, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["rl"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["rl"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["rl"].origin)>0)

    def test53_tensor_layer_input_exception(self):
        self.net = net(verbose=self.verbose)
        try:
            self.net._add_tensor_layer(id="rl", options={}, verbose=self.verbose)
        except Exception,c:
            self.assertEqual(c.message,self.tensor_no_input_exception_msg)

    def test54_tensor_layer_input_shape_exception(self):
        self.net = net(verbose=self.verbose)
        try:
            self.net._add_tensor_layer(id="rl", options={'input':self.input_ndarray}, verbose=self.verbose)
        except Exception,c:
            self.assertEqual(c.message,self.tensor_no_input_shape_exception_msg)

    @patch('yann.layers.input.dropout_tensor_layer')
    @patch('yann.layers.input.tensor_layer')
    def test55_tensor_layer_init(self,mock_tensor, mock_dtensor):
        mock_tensor.return_value = MockLayer()
        mock_dtensor.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input': MockInputLayer()}
        self.net.dropout_layers = {'input': MockInputLayer()}
        self.net.inference_layers = {'input': MockInputLayer()}
        self.net._add_tensor_layer(id="tl", options={
                                    'input':self.input_ndarray,
                                    'input_shape':self.input_shape}, verbose=self.verbose)
        self.assertEqual(self.net.layers["tl"].origin,MockLayer().origin)
        self.assertEqual(self.net.dropout_layers["tl"].origin,MockLayer().origin)
        self.assertEqual(self.net.inference_layers["tl"].origin,MockLayer().origin)

    @patch('yann.layers.input.dropout_tensor_layer')
    @patch('yann.layers.input.tensor_layer')
    def test56_tensor_layer_init_dropout(self,mock_tensor, mock_dtensor):
        mock_tensor.return_value = MockLayer()
        mock_dtensor.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input': MockInputLayer()}
        self.net.dropout_layers = {'input': MockInputLayer()}
        self.net.inference_layers = {'input': MockInputLayer()}
        self.net._add_tensor_layer(id="tl", options={
                                    'dropout_rate' : 0,
                                    'input':self.input_ndarray,
                                    'input_shape':self.input_shape}, verbose=self.verbose)
        self.assertEqual(self.net.layers["tl"].origin,MockLayer().origin)
        self.assertEqual(self.net.dropout_layers["tl"].origin,MockLayer().origin)
        self.assertEqual(self.net.inference_layers["tl"].origin,MockLayer().origin)

    def test57_add_batch_norm_layer_origin_exception(self):
        self.net = net(verbose=self.verbose)
        try:
            self.net._add_batch_norm_layer(id="bn",options={},verbose=self.verbose)
            self.assertEqual(False, True)
        except Exception, c:
            self.assertEqual(c.message,self.batch_norm_layer_exception_msg)

    @patch('yann.layers.batch_norm.batch_norm_layer_2d')
    def test58_add_batch_norm_layer_no_init(self,mock_bn):
        mock_bn.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input': MockInputLayer()}
        self.net.dropout_layers = {'input': MockInputLayer()}
        self.net.inference_layers = {'input': MockInputLayer()}
        self.net.last_layer_created = "input"
        self.net._add_batch_norm_layer(id="bn", options={}, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["bn"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["bn"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["bn"].origin)>0)

    @patch('yann.layers.batch_norm.batch_norm_layer_1d')
    def test59_add_batch_norm_layer_init(self,mock_bn):
        mock_bn.return_value = MockLayer()
        self.net = net(verbose=self.verbose)
        self.net.layers = {'input': MockInputLayer1d()}
        self.net.dropout_layers = {'input': MockInputLayer1d()}
        self.net.inference_layers = {'input': MockInputLayer1d()}
        self.net._add_batch_norm_layer(id="bn", options={
                        'origin':"input",
                        'input_params':None,
                        'dropout_rate':0
        }, verbose=self.verbose)
        self.assertTrue(len(self.net.layers["bn"].origin) > 0)
        self.assertTrue(len(self.net.dropout_layers["bn"].origin) > 0)
        self.assertTrue(len(self.net.inference_layers["bn"].origin)>0)



class MockDS():
    def __init__(self):
        self.svm = True
        self.x = 1
        self.mini_batch_size =1
        self.height = 1
        self.width = 1
        self.channels = 1
        self.y = 1
class MockLayer():
    def __init__(self):
        self.origin = []
        self.w = 1
        self.b = 0
        self.gamma = numpy.ones((1,),dtype=theano.config.floatX)
        self.beta = numpy.ones((1,), dtype=theano.config.floatX)
        self.running_mean = numpy.ones((1,), dtype=theano.config.floatX)
        self.running_var = numpy.ones((1,), dtype=theano.config.floatX)
        self.L1 = 0
        self.L2 = 0
        self.params ={}


class MockInputLayer():
    def __init__(self):
        self.output_shape =(1,10,10,10)
        self.output = numpy.random.rand(1,1,10,10)
        self.inference = self.output
        self.destination = []
        self.origin = []

class MockInputLayer1d():
    def __init__(self):
        self.output_shape =(10,10)
        self.output = numpy.random.rand(10,10)
        self.inference = self.output
        self.destination = []
        self.origin = []

class MockInputLayerLoss():
    def __init__(self):
        self.output_shape =(1,10,10,10)
        self.output = numpy.random.rand(1,1,10,10)
        self.inference = self.output
        self.destination = []
        self.origin = []
        self.loss = 'nll'
class MockDropoutLayerMerge():
    def __init__(self):
        self.origin = ["input"]
        self.output_shape =(1,10,10,10)
        self.output = numpy.random.rand(1,1,10,10)
        self.inference = self.output
        self.destination = []

