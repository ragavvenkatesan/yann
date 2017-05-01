import unittest
import networkx as nx
import numpy as np
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch
import yann.utils.raster as util_raster
from os.path import isfile
from os import remove
class TestGraph(unittest.TestCase):

    def setUp(self):
        self.ndar = np.random.rand(10) * 100
        self.ndar_image = np.random.rand(100, 30)
        self.r = np.random.rand(100,30)
        self.g = np.random.rand(100, 30)
        self.b = np.random.rand(100, 30)
        self.t = np.random.rand(100, 30)
        self.random_tuple = (self.r, self.g, self.b, self.t)
        self.random_tuple_none = (self.r, self.g, self.b, None)
    def test_scale_to_unit_interval(self):
        scaled_ndar = util_raster.scale_to_unit_interval(self.ndar)
        self.assertLessEqual(scaled_ndar.max(), 1)

    def test_tile_raster_images(self):
        tiled_image = util_raster.tile_raster_images(self.ndar_image, (10,3), (10,10) )
        self.assertLessEqual(tiled_image.shape, (100, 30))
        tiled_image = util_raster.tile_raster_images((self.r, self.g, self.b, self.t), (10, 3), (10, 10),scale_rows_to_unit_interval=False)
        self.assertLessEqual(tiled_image.shape, (100, 30, 4))
        tiled_image = util_raster.tile_raster_images(self.random_tuple, (10, 3), (10, 10),
                                                     scale_rows_to_unit_interval=False, output_pixel_vals=False)

        self.assertLessEqual(tiled_image.shape, (100, 30, 4))
        tiled_image = util_raster.tile_raster_images(self.random_tuple_none, (10, 3), (10, 10),
                                                     scale_rows_to_unit_interval=False, output_pixel_vals=True)
        self.assertLessEqual(tiled_image.shape, (100, 30, 4))