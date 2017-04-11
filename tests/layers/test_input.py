import unittest
import numpy as np
import theano
import theano.tensor as T
from yann.layers.input import input_layer as il
from yann.network import network


class TestInput(unittest.TestCase):
    def setUp(self):
        self.verbose = 3
        from yann.special.datasets import cook_mnist
        data = cook_mnist (verbose = self.verbose)
        self.dataset = data.dataset_location()
        self.dataset_id = data.id
        dataset_params  = {
                        "dataset"   :  self.dataset,
                        "svm"       :  False, 
                        "n_classes" : 10,
                        "id"        : 'data'
                }

        self.dataset_size = data.mini_batch_size
        self.dataset_height = data.height
        self.dataset_width = data.width
        self.dataset_channels = data.channels
        self.dataset_params  = {
                        "dataset"   :  self.dataset,
                        "svm"       :  False, 
                        "n_classes" : 10,
                        "id"        : 'data'
                }               
        from yann.modules.datastream import datastream
        self.datastream = {}
        self.datastream[0] = datastream ( dataset_init_args = self.dataset_params, verbose = self.verbose)
    def test_dataset(self):
    #    self.layer = il(
    #                     x = self.datastream[0].x,
    #                     mini_batch_size = self.dataset_size,
    #                     id = self.dataset_id,
    #                     height = self.dataset_height,
    #                     width = self.dataset_width,
    #                     channels = self.dataset_channels,
    #                     mean_subtract = False,
    #                     verbose = self.verbose)
           self.layer = il(
                        x = self.datastream[0].x,
                        mini_batch_size = self.datastream[0].mini_batch_size,
                        id = self.datastream[0].id,
                        height = self.datastream[0].height,
                        width = self.datastream[0].width,
                        channels = self.datastream[0].channels,
                        mean_subtract = False,
                        verbose = self.verbose)
        #    print(self.layer.id)
        #    print(self.layer.print_layer("", True, True ))
        #    print(self.layer.output_shape)
           self.assertEqual(self.layer.id,"data")
        #    net = network(   borrow = True,
        #              verbose = self.verbose ) 
        #    net.add_layer(self.layer)
        #    net.pretty_print()
