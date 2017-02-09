static_printer_import = True
dynamic_printer_import = True

import os
from abstract import module
from yann.utils.dataset import rgb2gray, gray2rgb
from matplotlib.image import imsave
import numpy as np

try:
    from theano.printing import pydotprint as static_theano_print
except:
    static_printer_import = False
try:
    from theano.d3viz import d3viz as dynamic_theano_print # refer todo on top.
except:
    dynamic_printer_import = False

def save_images(imgs, prefix, is_color, verbose = 2):
    """
    This functions produces a visualiation of the filters.

    Args:
        imgs: images of shape .. [num_imgs, height, width, channels] Note the change in order.
              it can also be in lots of other shapes and they will be reshaped and saved as images.
        prefix: address to save the image to
        is_color: If the image is color or not. True only if image shape is of color images.
        verbose: As Always
    """
    """
    vis = False
    try:
        # Pylearn2 pylearn_visualization
        from pylearn2.gui.patch_viewer import make_viewer
        vis = True
    except ImportError:
        if verbose >=3:
            print "... Install pylearn2 before using visualize, not visualizing"
    """
    from yann.utils.raster import tile_raster_images

    #if vis is True:
    if verbose >= 3:
        print "... Rasterizing"

    raster = []
    count = 0

    if len(imgs.shape) > 2:

        if is_color is True:
            if imgs.shape[3] == 1:
                is_color = False

            if imgs.shape[3] > 1 and imgs.shape[3] % 3 != 0:
                filts = np.floor( imgs.shape[3] / 3) # consider only the first so an so channels
                imgs = imgs[:,:,:,0:int(filts)*3]

        for i in xrange (imgs.shape[3]):
            curr_image = imgs[:,:,:,i]
            num_imgs = curr_image.shape[0]
            tile_raster = np.floor(np.sqrt(num_imgs))

            if int(tile_raster ** 2) == num_imgs:
                tile_shape = (tile_raster, tile_raster)
            else:
                remain = num_imgs - (tile_raster ** 2)
                remain_rows = np.ceil(tile_raster / remain)
                tile_shape = (tile_raster + remain_rows, tile_raster)

            tile_shape = (int(tile_shape[0]), int(tile_shape[1]))
            if is_color is True:
                I = np.array(tile_raster_images(
                                X = curr_image.reshape((curr_image.shape[0],curr_image.shape[1] * \
                                        curr_image.shape[2])),
                                    img_shape = (curr_image.shape[1], curr_image.shape[2]),
                                    tile_shape = tile_shape ))
                if len(I.shape) == 3:
                    raster.append(rgb2gray(I))
                else:
                    raster.append(I)
                if count == 2:
                    imsave(prefix + str(i) + ".jpg", gray2rgb(raster[i-2],raster[i-1],raster[i]) )
                    count = -1
            else:
                raster.append(np.array(tile_raster_images(
                                X = curr_image.reshape((curr_image.shape[0],curr_image.shape[1] * \
                                        curr_image.shape[2])),
                                    img_shape = (curr_image.shape[1], curr_image.shape[2]),
                                    tile_shape = tile_shape )))
                assert len(raster[i].shape) == 2
                imsave(prefix + str(i) + ".jpg",raster[i], cmap = 'gray')
            count = count + 1

    else:
        num_imgs = imgs.shape[0]
        tile_raster = np.floor(np.sqrt(num_imgs))

        lt = int(np.floor(np.sqrt(imgs.shape[1])))
        imgs = imgs[:,:lt*lt]

        if int(tile_raster ** 2) == num_imgs:
            tile_shape = (tile_raster, tile_raster)
        else:
            remain = num_imgs - (tile_raster ** 2)
            remain_rows = np.ceil(tile_raster / remain)
            tile_shape = (tile_raster + remain_rows, tile_raster)
        tile_shape = (int(tile_shape[0]), int(tile_shape[1]))
        raster.append(np.array(tile_raster_images(X = imgs, img_shape = (lt,lt),
                                                                    tile_shape = tile_shape)))
        is_color = False
        imsave(prefix + "0.jpg",raster[0], cmap ='gray')
    return raster
    #else:
    #    return None

class visualizer(module):
    """
    Visualizer saves down images to visualize. The initilizer only initializes the directories
    for storing visuals. Three types of visualizations are saved down:

        * filters of each layer
        * activations of each layer
        * raw images to check the activations against

    Args:
        verbose               : Similar to any 3-level verbose in the toolbox.
        visualizer_init_args  : ``visualer_params`` is a dictionary of the form:

            .. code-block:: none

                visualizer_init_args = {
                    "root"       : <location to save the visualizations at>,
                    "frequency"  : <integer>, after how many epochs do you need to
                                    visualize. Default value is 1
                    "sample_size": <integer, prefer squares>, simply save down random
                                    images from the datasets saves down activations for the
                                    same images also. Default value is 16
                    "rgb_filters": <bool> flag. if True a 3D-RGB rendition of the CNN
                                    filters is rendered. Default value is False.
                    "debug_functions" : <bool> visualize train and test and other theano functions.
                                        default is False. Needs pydot and dv2viz to be installed.
                    "debug_layers" : <bool> Will print layer activities from input to that layer
                                     output. ( this is almost always useless because test debug
                                     function will combine all these layers and print directly.)
                    "id"         : id of the visualizer
                                }

    Returns:
        yann.modules.visualizer: A visualizer object.
    """
    def __init__( self, visualizer_init_args, verbose = 2 ):
        if "id" in visualizer_init_args.keys():
            id = visualizer_init_args["id"]
        else:
            id = 'main'
        super(visualizer,self).__init__(id = id, type = 'visualizer')

        if verbose >= 3:
            print "... Creating visualizer directories"

        if "root" in visualizer_init_args.keys():
            self.root         = visualizer_init_args ["root"] + "/visualizer"
        else:
            self.root   = os.getcwd() + '/visualizer'

        if "frequency" in visualizer_init_args.keys():
            self.frequency    = visualizer_init_args ["frequency" ]
        else:
            self.frequency    = 1

        if "sample_size" in visualizer_init_args.keys():
            self.sample_size  = visualizer_init_args ["sample_size" ]
        else:
            self.sample_size  = 16

        if "rgb_filters" in visualizer_init_args.keys():
            self.rgb_filters  = visualizer_init_args ["rgb_filters" ]
        else:
            self.rgb_filters  = False

        if "debug_functions" in visualizer_init_args.keys():
            self.debug_functions = visualizer_init_args ["debug_functions"]
        else:
            self.debug_functions = False

        if "debug_layers" in visualizer_init_args.keys():
            self.debug_layers = visualizer_init_args ["debug_layers"]
        else:
            self.debug_layers = False

        """ Needs to be done after mini_batch_size is setup.
            self.shuffle_batch_ind = np.arange(self.mini_batch_size)
            np.random.shuffle(self.shuffle_batch_ind)
            self.visualize_ind = self.shuffle_batch_ind[0:self.n_visual_images]

            assert self.mini_batch_size >= self.n_visual_images

            # loop around and make folders for kernels and layers for visualizer
            for i in xrange(len(self.nkerns)):
                os.makedirs('../visuals/filters/layer_'+str(i))
        """

        # create all directories required for saving visuals
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if not os.path.exists(self.root + '/activities'):
            os.makedirs(self.root + '/activities')

        if not os.path.exists(self.root + '/filters'):
            os.makedirs(self.root + '/filters')

        if not os.path.exists(self.root + '/data'):
            os.makedirs(self.root + '/data')

        if not os.path.exists(self.root + '/computational_graphs'):
            os.makedirs(self.root + '/computational_graphs')
            os.makedirs(self.root + '/computational_graphs/static')
            os.makedirs(self.root + '/computational_graphs/dynamic') # refer the todo on top.

        if verbose >= 3:
            print "... Visualizer is initiliazed"

    def initialize (self, batch_size, verbose = 2):
        """
        Function that cooks the visualizer for some dataset.

        Args:
            batch_size: form dataset
            verbose: as always
        """
        self.batch_size = batch_size
        shuffle_batch_ind = np.arange(self.batch_size)
        np.random.shuffle(shuffle_batch_ind)
        self.indices = shuffle_batch_ind[0:self.sample_size]

    def theano_function_visualizer( self,
                                    function,
                                    short_variable_names = False,
                                    format ='pdf',
                                    verbose = 2):
        """
        This basically prints a visualization of any theano function using the in-built theano
        visualizer. It will save both a interactive html file and a plain old png file. This is
        just a wrapper to theano's visualization tools.

        Args:
            function: theano function to print
            short_variable_names: If True will print variables in short.
            format: Any pydot supported format. Default is 'pdf'
            verbose: As usual.
        """
        if verbose >=3:
            print "... creating visualizations of computational graph"
        # this try and except is bad coding, but this seems to be OS dependent and I don't want to
        # bother with this.

        if static_printer_import is True:
            filename = self.root + '/computational_graphs/static/' + function.name
            try:
                static_theano_print(fct = function, outfile = filename + '.' + format,
                                                                print_output_file = False,
                                                                format = format,
                                                        var_with_name_simple = short_variable_names)
            except:
                if verbose >= 3:
                    print "... Something is wrong with the setup of installers for pydot"

        if dynamic_printer_import is True:
            filename = self.root + '/computational_graphs/dynamic/' + function.name
            try:
                dynamic_theano_print(fct = function, outfile = filename + '.html')
                                                # this is not working for something is
                                                # wrong with path. Refer todo on top of the code.
            except:
                if verbose >= 3:
                    print "... Something is wrong with the setup of installers for dv3viz"



    def visualize_images(self, imgs, loc = None, verbose = 2):
        """
        Visualize the images in the dataset. Assumes that the data in the tensor variable imgs is
        in shape (batch_size, height, width, channels). Assumes that batchsize does not change.

        Args:
            imgs: tensor of data
            verbose: as usual.
        """
        if verbose >=3 :
            print "... saving down images"
        if imgs.shape[0] == self.batch_size:
            imgs = imgs[self.indices]

        if loc is None:
            loc = self.root + '/data/image_'
        else:
            loc = loc + '/image_'
        imgs = save_images(   imgs = imgs,
                            prefix = loc,
                            is_color = self.rgb_filters)

    def visualize_activities(self, layer_activities, epoch, index = 0, verbose = 2):
        """
        This method saves down all activities.

        Args:
            layer_activities: network's layer_activities as created
            epoch: what epoch are we running currently.
            verbose: as always
        """
        if verbose >= 3:
            print "... Visualizing Activities"

        loc = self.root + '/activities/epoch_' + str(epoch)
        if not os.path.exists(loc):
            os.makedirs(loc)
        for id, activity in layer_activities.iteritems():
            imgs = activity(index)
            if verbose >= 3:
                print "... Visualizing Activities :: id = %s" % id
            if len(imgs.shape) == 2:
                if not os.path.exists(loc + '/layer_' + id):
                    os.makedirs(loc + '/layer_' + id)
                if not os.path.exists(loc + '/layer_' + id + '/straight'):
                    os.makedirs(loc + '/layer_' + id + '/straight')
                self.visualize_images(  imgs = imgs,
                                        loc = loc + '/layer_' + id + '/straight',
                                        verbose = verbose )
                if not os.path.exists(loc + '/layer_' + id + '/transposed'):
                    os.makedirs(loc + '/layer_' + id + '/transposed')
                self.visualize_images(  imgs = imgs.transpose(),
                                        loc = loc + '/layer_' + id + '/transposed',
                                        verbose = verbose )
            elif len(imgs.shape) == 4:
                imgs = imgs.transpose(0,2,3,1)
                if not os.path.exists(loc + '/layer_' + id):
                    os.makedirs(loc + '/layer_' + id)
                self.visualize_images(imgs, loc = loc + '/layer_' + id ,verbose = verbose)

    def visualize_filters(self, layers, epoch, index = 0, verbose = 2):
        """
        This method saves down all activities.

        Args:
            layers: network's layer dictionary
            epoch: what epoch are we running currently.
            verbose: as always
        """
        if verbose >= 3:
            print "... Visualizing Layers"

        loc = self.root + '/filters/epoch_' + str(epoch)
        if not os.path.exists(loc):
            os.makedirs(loc)
        for id, layer in layers.iteritems():
            if layer.params is not None:
                if verbose >= 3:
                    print "... saving down visualization of layer " + id

                imgs = layer.get_params()[0]
                if len(imgs.shape) == 4:
                    if not os.path.exists(loc + '/layer_' + id):
                        os.makedirs(loc + '/layer_' + id)
                    imgs = imgs.transpose(0,2,3,1)
                    self.visualize_images(   imgs = imgs,
                                             loc = loc + '/layer_' + id ,
                                             verbose = verbose )
                elif len(imgs.shape) == 2:
                    if not os.path.exists(loc + '/layer_' + id):
                        os.makedirs(loc + '/layer_' + id)
                    if not os.path.exists(loc + '/layer_' + id + '/straight'):
                        os.makedirs(loc + '/layer_' + id + '/straight')
                    self.visualize_images(   imgs = imgs,
                                             loc = loc + '/layer_' + id + '/straight',
                                             verbose = verbose )
                    if not os.path.exists(loc + '/layer_' + id + '/transposed'):
                        os.makedirs(loc + '/layer_' + id + '/transposed')
                    self.visualize_images(   imgs = imgs.transpose(),
                                             loc = loc + '/layer_' + id + '/transposed',
                                             verbose = verbose )


if __name__ == '__main__':
    pass
