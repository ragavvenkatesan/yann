import imp
try:
    imp.find_module('progressbar')
    progressbar_installed = True
except ImportError:
    progressbar_installed = False
if progressbar_installed is True:
    import progressbar

try:
    imp.find_module('networkx')
    nx_installed = True
except ImportError:
    nx_installed = False
if nx_installed is True:
    import networkx as nx

import time
from collections import OrderedDict

import numpy
import theano
import theano.tensor as T

import yann
from yann.core.operators import copy_params

max_neurons_to_display = 7

class network(object):
    """
    Todo:

        * Origin to be taken from another network outside of this one.
        * Need to expand beyond just ``classifier`` type networks.

    Class definition for the class :mod:`network`:

    All network properties, network variables and functionalities are initialized using this
    class and are contained by the network object. The ``network.__init__`` method initializes
    the network class. The ``network.__init___`` function has many purposes depending on the
    arguments supplied.

    Provide any or all of the following arguments. Appropriate errors will be thrown if the
    parameters are not supplied correctly.

    Args:
        verbose             : Similar to any 3-level verbose in the toolbox.
        type                : option takes only 'classifier' for now. Will add 'encoders'
                              and others later
        borrow: Check ``theano's`` borrow. Default is ``True``.

    Returns:
        ``yann.network.network``: network object with parameters setup.
    """

    def __init__(  self, verbose = 2, **kwargs ):
        """ Refer to the definition in the class docstrings for details """

        if verbose >= 1:
            print(". Initializing the network")

        if nx_installed is True:
            self.graph = nx.DiGraph()
            self.layer_graph = {}
        else:
            self.graph = None

        self.layers = {} # create an empty dictionary that we can populate later.
        self.active_params = []
        self.params = []
        self.dropout_layers = {} # These are just weights dropped out. Contains references
        self.inference_layers = {} # This is used for when doing only forward prop.
        self.num_layers = 0  # just maintain for the sake of it.

        # All these are just bookkeeping variables.
        self.last_layer_created = None
        self.last_resultor_created = None
        self.last_datastream_created = None
        self.last_visualizer_created = None
        self.last_optimizer_created = None
        self.last_objective_layer_created = None
        self.last_classifier_created = None
        self.layer_activities_created = True

        # create empty dictionary of modules.
        self.visualizer = {}
        self.optimizer  = {}
        self.resultor   = {}
        self.datastream = {}
        self.num_layers = 0
        self.rng = numpy.random    # each network will have its own unique randomizer
        self.borrow = True
        self.L1 = 0
        self.L2 = 0
        self.layer_activities = {}

        # for each argument supplied by kwargs, intialize something.
        if 'borrow' in kwargs.keys():
            self.borrow = kwargs['borrow']
        else:
            self.borrow = True

    def layer_activity(self, id, index=0, verbose = 2):
        """
        Use this function to visualize or print out the outputs of each layer.
        I don't know why this might be useful, but its fun to check this out I guess. This will only
        work after the dataset is initialized.

        Args:
            id: id of the layer that you want to visualize the output for.
            index: Which batch of data should I use for producing the outputs.
                    Default is ``0``
        """
        return self.layer_activities[id](index)

    def add_layer(self, type, verbose = 2, **kwargs):
        """

        Todo:
            Need to add the following:
            * Merge Layer: Concatenate, Embed, Add, max, mean
            * Inception Layer.
            * LSTM layer.
            * MaskedConvPool Layer.
            * ...
            * when ``type`` is ``'objective'``, I need to allow taking a loss between two layers to
            be propagated. Right now ``origin`` has to be a classifier layer only. This needs to
            change to be able to implement generality and mentor networks.
            * basically objective has to be a flag for an error function if origin is a tuple.


        Args:
            type: <string> options include
                  'input' or 'data' - which indicates an input layer.
                  'conv_pool' or 'fully_connected' - indicates a convolutional - pooling layer
                  'dot_product' or 'hidden' or 'mlp' or 'fully_connected' - indicates a hidden fully
                  connected layer
                  'classifier' or 'softmax' or 'output' or 'label' - indicates a classifier layer
                  'objective' or 'loss' or 'energy' - a layer that creates a loss function
                  'merge' or 'join' - a layer that merges two layers.
                  'flatten' - a layer that produces a flattened output of a block data.
                  'random' - a layer that produces random numbers.
                  'rotate' - a layer that rotate the input images.
                  From now on everything is optional args..
            id: <string> how to identify the layer by.
                Default is just layer number that starts with ``0``.
            origin: ``id`` will use the output of that layer as input to the new layer.
                     Default is the last layer created. This variable for ``input`` type of layers
                     is not a layer, but a datastream id. For ``merge`` layer, this is a
                     tuple of two layer ids.
            verbose: similar to the rest of the toolbox.
            mean_subtract: if ``True`` we will subtract the mean from each image, else not.
            num_neurons: number of neurons in the layer
            dataset: <string> Location to the dataset.
                     used when layer ``type`` is ``input``.
            activation: String, takes options that are listed in :mod:`activations` Needed for
                        layers that use activations.
                        Some activations also take support parameters, for instance ``maxout``
                        takes maxout type and size, ``softmax`` takes an option temperature.
                        Refer to the module :mod:`activations` to know more.
            stride: tuple ``(int , int)``. Used as convolution stride. Default ``(1,1)``
            batch_norm: If provided will be used, default is ``False``.
            border_mode: Refer to ``border_mode`` variable in ``yann.core.conv``, module
                         :mod:`conv`
            pool_size: Subsample size, default is ``(1,1)``.
            pool_type: Refer to :mod:`pool` for details. {'max', 'sum', 'mean', 'max_same_size'}
            learnable: Default is ``True``, if ``True`` we backprop on that layer. If ``False``
                       Layer is obstinate.
            shape: tuple of shape to unflatten to ( height, width, channels ) in case layer was an
                    unflatten layer
            input_params: Supply params or initializations from a pre-trained system.
            dropout_rate: If you want to dropout this layer's output provide the output.
            regularize: ``True`` is you want to apply regularization, ``False`` if not.
            num_classes: ``int`` number of classes to classify.
            objective:  objective provided by classifier
                        ``nll``-negative log likelihood,
                        ``cce``-categorical cross entropy,
                        ``bce``-binary cross entropy,
                        ``hinge``-hinge loss . For classifier layer.
            dataset_init_args: same as for the dataset module. In fact this argument is needed
                                only when dataset module is not setup.
            datastream_id: When using input layer or during objective layer, use this to identify
                           which datastream to take data from.
            regularizer: Default is ``(0.001, 0.001)`` coeffients for L1, L2 regulaizer
                            coefficients.
            error: ``merge`` layers take an option called ``'error'`` which can be None or others
                    which are methods in ``yann.core.errors``.
            angle:  Takes value between [0,1] to capture the angle between [0,180] degrees
                    Default is None. If None is specified, random number is generated from a uniform
                    distriibution between 0 and 1.
            layer_type: If ``value`` supply, else it is default ``'discriminator'``

        """
        if not 'id' in kwargs.keys():
            id = str(self.num_layers)
        else:
            id = kwargs["id"]

        # increase the layer counter
        self.num_layers = self.num_layers + 1

        if verbose >= 2 :
            print(".. Adding "+ type + " layer " + id)

        if not 'learnable' in kwargs.keys():
            if type == 'input' or \
               type == 'data' or \
               type == 'objective' or  \
               type == 'merge' or \
               type == 'flatten' or \
               type == 'unflatten' or \
               type == 'random' or \
               type == 'rotate' or \
               type == 'loss' or \
               type == 'energy' or \
               type == 'connect' or \
               type == 'join':
                if verbose >= 3:
                    print("... Making learnable False as it is not provided")
                learnable = False
            else:
                if verbose >= 3:
                    print("... Making learnable True as it is not provided")
                learnable = True
        else:
            learnable = kwargs['learnable']

        if type == 'input' or \
           type == 'data':
            if not learnable == False:
                raise Exception (" You cannot learn this type of a layer")
            self._add_input_layer(id =id, options = kwargs, verbose = verbose)

        elif type == 'conv_pool' or \
             type == 'convolution':
            self._add_conv_layer(id = id, options = kwargs, verbose = verbose)
            self.params = self.params + self.dropout_layers[id].params

        elif type == 'dot_product' or \
             type == 'hidden' or  \
             type == 'mlp' or  \
             type == 'fully_connected':
            self._add_dot_product_layer(id =id, options = kwargs, verbose = verbose)
            self.params = self.params + self.dropout_layers[id].params

        elif type == 'classifier' or  \
             type ==  'softmax' or \
             type ==  'output' or \
             type == 'label':
            self._add_classifier_layer(id =id, options = kwargs, verbose = verbose)
            self.last_classifier_created = id
            self.params = self.params + self.dropout_layers[id].params

        elif type == 'objective' or  \
             type == 'loss' or \
             type == 'energy':
            self._add_objective_layer(id =id, options = kwargs, verbose = verbose)
            self.last_objective_layer_created = id

        elif type == 'merge' or \
             type == 'join' or \
             type == 'connect':
            self._add_merge_layer(id = id, options = kwargs, verbose = verbose)

        elif type == 'flatten':
            self._add_flatten_layer( id = id, options = kwargs, verbose = verbose)

        elif type == 'unflatten':
            self._add_unflatten_layer( id = id, options = kwargs, verbose = verbose)

        elif type == 'random':
            self._add_random_layer (id = id, options = kwargs, verbose = verbose)

        elif type == 'rotate':
            self._add_rotate_layer (id = id, options = kwargs, verbose = verbose)

        else:
            raise Exception('No layer called ' + type + ' exists in yann')

        if learnable is True:
            self.active_params = self.active_params + self.dropout_layers[id].active_params
            # .. Note :: using append is troublesome here.

        self.last_layer_created = id
        self.layers[id].active = learnable
        self.dropout_layers[id].active = learnable
        # addlayer to self.graph if it exists.
        if not self.graph is None: # graph is being created
            attributes = self.layers[id]._graph_attributes()  # collect all attributes of layer

            if len(attributes["output_shape"]) == 4:  # output is an image
                node_shape = 'square'
                num_neurons = attributes["output_shape"][1]  # number of kernels output
                node_size = attributes["output_shape"][2] * attributes["output_shape"][3]
            else:
                node_shape = 'circle'
                num_neurons = attributes['output_shape'][-1]
                node_size = 300   # why ? default.

            if num_neurons > max_neurons_to_display:
                neurons = range(0,max_neurons_to_display - 2) + range(num_neurons - 2,num_neurons)
            else:
                neurons = range(num_neurons)

            self.layer_graph[id] = nx.DiGraph()
            self.layer_graph[id].name = id
            for i in neurons:
                self.layer_graph[id].add_node(id + "-" + str(i) ,attributes, shape = node_shape,
                                                                            size = node_size)
            self.graph.subgraph(self.layer_graph[id])
            for origin in attributes["origin"]:
                if origin in self.layers.keys():
                    origin_attr = self.layers[origin]._graph_attributes()
                    origin_num_neurons = origin_attr["output_shape"]
                    if len(origin_num_neurons) == 1:
                        origin_num_neurons = origin_num_neurons[0] # for layers like merge and obj
                    else:
                        origin_num_neurons = origin_num_neurons[1]

                    if origin_num_neurons > max_neurons_to_display:
                        origin_neurons = range(0,max_neurons_to_display - 2) + \
                                                 range(origin_num_neurons - 2, origin_num_neurons)
                    else:
                        origin_neurons = range(origin_num_neurons)

                    for edge_from in origin_neurons:
                        for edge_to in neurons:
                            self.graph.add_edge(origin + "-" + str(edge_from),
                                                               id + "-" + str(edge_to))
                else:
                    for edge_to in neurons:
                        self.graph.add_edge(origin , id + "-" + str(edge_to))
        if verbose >= 3:
            print("... Layer " + id + " is created and it learnablity is " + \
                                                                    str(self.layers[id].active))
    def add_module (self, type, params = None, verbose = 2):
        """
        Use this function to add a module to the net.

        Args:

            type: which module to add. Options are ``'resultor'``, ``'visualizer'``, ``'optimizer'``
                  ``'datastream'``
            params:
                If the ``type`` was ``'resultor'`` params is a dictionary of the form:

                .. code-block:: none

                    params    =    {
                            "root"     : "<root directory to save stuff inside>"
                            "results"  : "<results_file_name>.txt",
                            "errors"   : "<error_file_name>.txt",
                            "costs"    : "<cost_file_name>.txt",
                            "confusion": "<confusion_file_name>.txt",
                            "network"  : "<network_save_file_name>.pkl"
                            "id"       : id of the resultor
                                    }

                While the filenames are optional, ``root`` must be provided. If a particular file
                is not provided, that value will not be saved. This value is supplied to setup the
                resultor module of :mod: `network`.

                If the ``type`` was ``'visualizer'`` params is a dictionary of the form:

                .. code-block:: none

                    parmas = {
                            "root"        : location to save the visualizations
                            "frequency"   : <integer>, after how many epochs do you need to
                                            visualize. Default value is 1import os

                            "sample_size" : <integer, prefer squares>, simply save down random
                                            images from the datasets also saves down activations
                                            for the same images also. Default value is 16
                            "rgb_filters" : <bool> flag. if True 3D-RGB CNN filters are rendered.
                                            Default value is False
                            "id"          : id of the visualizer
                                    }

                If the  ``type`` was ``'optimizer'`` params is a dictionary of the form:

                .. code-block:: none

                    params =  {
                            "momentum_type"       : <option> takes 'false' <no momentum>, 'polyak'
                                                    and 'nesterov'. Default value is 'polyak'
                            "momentum_params"   : (<value in [0,1]>, <value in [0,1]>, <int>),
                                                    (momentum coeffient at start, at end, at what
                                                    epoch to end momentum increase). Default is
                                                    the tuple (0.5, 0.95,50)
                            "learning_rate"   : (initial_learning_rate, fine_tuning_learning_rate,
                                                    annealing_decay_rate). Default is the tuple
                                                    (0.1,0.001,0.005)
                            "regularization"    : (l1_coeff, l2_coeff). Default is (0.001, 0.001)
                            "optimizer_type": <option>, takes 'sgd', 'adagrad', 'rmsprop', 'adam'.
                                                    Default is 'rmsprop'
                            "objective_function": <option>,  takes
                                                    'nll'-negative log likelihood,
                                                    'cce'-categorical cross entropy,
                                                    'bce'-binary cross entropy.
                                                    Default is 'nll'
                            "id"                : id of the optimizer
                                }

                If the ``type was ``'datastream'`` params is a dictionary of the form:

                .. code-block:: python

                    params = {
                                "dataset":  <location>
                                "svm"    :  False or True
                                    ``svm`` if ``True``, a one-hot label set will also be setup.
                                "n_classes": <int>
                                    ``n_classes`` if ``svm`` is ``True``, we need to know how
                                    many ``n_classes`` are present.
                                "id": id of the datastream
                        }

            verbose: Similar to rest of the toolbox.
        """
        if verbose >= 2:
            print(".. Setting up the " + type)

        # input parameter `viualizer` is used
        if type == 'visualizer':
            self._add_visualizer(visualizer_params = params, verbose = verbose)

        # input parameter `optimizer` is used
        elif type == 'optimizer':
            self._add_optimizer(optimizer_params = params, verbose = verbose)

        elif type == 'datastream':
            self._add_datastream(dataset_params = params, verbose = verbose)

        elif type == 'resultor':
            self._add_resultor(resultor_params = params, verbose = verbose)

        else:
            raise Exception ('No module called ' + type)

    def _add_resultor(self, resultor_params = None, verbose = 2):
        """
        This function is used to add a resultor to the network.

        Args:
            resultor_params: parameters for resultor_init_args for resultor modele.
                             Refer to the network or resultor class for details.
            verbose: Similar to what is found in the rest of the toolbox.
        """
        if resultor_params is None:
            resultor_params = {
                    "root"      : "resultor",
                    "results"   : "results.txt",
                    "errors"    : "errors.txt",
                    "costs"     : "costs.txt",
                    "confusion" : "confusion.txt",
                    "network"   : "network.pkl",
                    "learning_rate" : "learning_rate.txt",
                    "momentum"  : "momentum.txt",
                    "visualize" : False,
            }

        if not "id" in resultor_params.keys():
            id = len(self.resultor) + 1
            resultor_params["id"] = id
        else:
            id = resultor_params['id']

        from yann.modules.resultor import resultor
        self.resultor[id] = resultor ( resultor_init_args = resultor_params, verbose = verbose )
        self.last_resultor_created = id

    def _add_visualizer(self, visualizer_params, verbose = 2):
        """
        This function is used to add a visualizer to the network.

        Args:
            visualizer_params: parameters for resultor_init_args for visualizer.
                               Refer to the network or visualizer class for details.
            verbose: Similar to what is found in the rest of the toolbox.
        """
        if not "id" in visualizer_params.keys():
            id = len(self.visualizer) + 1
            visualizer_params["id"] = id
        else:
            id = visualizer_params['id']
        from yann.modules.visualizer import visualizer
        self.visualizer[id] = visualizer( visualizer_init_args = visualizer_params,
                                                                                 verbose = verbose )
        self.last_visualizer_created = id

    def _add_optimizer(self, optimizer_params, verbose = 2):
        """
        This function is used to add a optimizer to the network.

        Args:
            optimizer_params: parameters for optimizer_init_args for visualizer.
                               Refer to the network or optimizer class for details.
            verbose: Similar to what is found in the rest of the toolbox.
        """
        if not "id" in optimizer_params.keys():
            id = len(self.optimizer) + 1
            optimizer_params["id"] = id
        else:
            id = optimizer_params['id']
        from yann.modules.optimizer import optimizer
        self.optimizer[id] = optimizer ( optimizer_init_args = optimizer_params, verbose = verbose )
        self.last_optimizer_created = id

    def _add_datastream(self, dataset_params, verbose = 2):
        """
        This function is used to add a datastream to the network.

        Args:
            visualizer_params: parameters for dataset_init_args for datastream.
                               Refer to the network or datastream class for details.
            verbose: Similar to what is found in the rest of the toolbox.
        """
        if not "id" in dataset_params.keys():
            id = len(self.datastream) + 1
            dataset_params["id"] = id
        else:
            id = dataset_params['id']

        from yann.modules.datastream import datastream
        self.datastream[id] = datastream ( dataset_init_args = dataset_params, verbose = verbose)
        self.last_datastream_created = id

        if not self.graph is None: # graph is being created
            self.graph.add_node(id,shape = 'square', size = 1000, style='filled',fillcolor='red')

    def _add_input_layer(self, id, options, verbose = 2):
        """
        This is an internal function. Use ``add_layer`` instead of this from outside the class.

        Args:
            options: Basically kwargs supplied to the add_layer function.
            verbose: simiar to everywhere on the toolbox.
        """
        if verbose >=3:
            print("... Adding an input layer")

        if 'dataset_init_args' in options.keys():
            dataset_params = options["dataset_init_args"]
            if 'id' in dataset_params.keys():
                datastream_id = dataset_params['id']
            else:
                datastream_id = '-1'  # this is temp datastream will initialize a new id.
            if not datastream_id in self.datastream.keys():
                self._add_datastream(dataset_params = dataset_params, verbose = 2)
                if verbose >= 3:
                    print("... Created a new datastream module also")
            else:
                if verbose >= 3:
                    print("... Datastream already created, will use it straight away")

        if 'origin' in options.keys():
            datastream_id = options['origin']

        elif len(self.datastream) == 0:
            raise Exception("Can't setup an input layer without dataset initialized")

        else:
            datastream_id = self.last_datastream_created

        self.svm = self.datastream[datastream_id].svm

        if not 'mean_subtract' in options.keys():
            if verbose >=3:
                print("... mean_subtract not provided. Assuming False")
            mean_subtract = False
        else:
            mean_subtract = options["mean_subtract"]

        if not 'dropout_rate' in options.keys():
            if verbose >= 3:
                print("... dropout_rate not provided. Assuming 0")
            dropout_rate = 0
        else:
            dropout_rate = options ["dropout_rate"]

        from yann.layers.input import dropout_input_layer as dil
        from yann.layers.input import input_layer as il

        self.dropout_layers[id] = dil (
                            dropout_rate = dropout_rate,
                            x = self.datastream[datastream_id].x,
                            rng = self.rng,
                            id = id,
                            mini_batch_size = self.datastream[datastream_id].mini_batch_size,
                            height = self.datastream[datastream_id].height,
                            width = self.datastream[datastream_id].width,
                            channels = self.datastream[datastream_id].channels,
                            mean_subtract = mean_subtract,
                            verbose =verbose)

        self.layers[id] = il(
                            x = self.datastream[datastream_id].x,
                            mini_batch_size = self.datastream[datastream_id].mini_batch_size,
                            id = id,
                            height = self.datastream[datastream_id].height,
                            width = self.datastream[datastream_id].width,
                            channels = self.datastream[datastream_id].channels,
                            mean_subtract = mean_subtract,
                            verbose =verbose)

        self.inference_layers[id] = il(
                            x = self.datastream[datastream_id].x,
                            mini_batch_size = self.datastream[datastream_id].mini_batch_size,
                            id = id,
                            height = self.datastream[datastream_id].height,
                            width = self.datastream[datastream_id].width,
                            channels = self.datastream[datastream_id].channels,
                            mean_subtract = mean_subtract,
                            verbose =verbose)

        # create a whole new stream, whether used or not.
        # users who do not need dropout need not know about this. muahhahaha
        self.layers[id].origin.append(datastream_id)
        self.dropout_layers[id].origin.append(datastream_id)
        self.inference_layers[id].origin.append(datastream_id)

    def _add_conv_layer(self, id, options, verbose = 2):
        """
        This is an internal function. Use ``add_layer`` instead of this from outside the class.

        Args:
            options: Basically kwargs supplied to the add_layer function.
            verbose: same as everywhere else on the toolbox
        """
        if verbose >=3:
            print("... Adding a convolution layer")
        if not 'origin' in options.keys():
            if self.last_layer_created is None:
                raise Exception("You can't create a convolutional layer without an" + \
                                    " origin layer.")
            else:
                if verbose >=3:
                    print ("... origin layer not provided, assuming the last layer created.")
                origin = self.last_layer_created
        else:
            origin = options["origin"]

        input_shape = self.layers[origin].output_shape

        if not 'num_neurons' in options.keys():
            if verbose >=3:
                print("... num_neurons not provided for layer " + id + ". Asumming 20")
            num_neurons = 20
        else:
            nkerns = options ["num_neurons"]

        if not 'filter_size' in options.keys():
            if verbose >=3:
                print("... filter_size not provided for layer " + id + ". Asumming (3,3)")
            filter_size = (3,3)
        else:
            filter_size = options ["filter_size"]

        if not 'activation' in options.keys():
            if verbose >=3:
                print("... Activations not provided for layer " + id + ". Using ReLU")
            activation = 'relu'
        else:
            activation = options ["activation"]

        if not 'border_mode' in options.keys():
            if verbose >=3:
                print("... no border_mode setup, going with default")
            border_mode = 'valid'
        else:
            border_mode = options ["border_mode"]

        if not 'stride' in options.keys():
            if verbose >=3:
                print("... No stride provided for layer " + id + ". Using (1,1)")
            stride = (1,1)
        else:
            stride = options ["stride"]

        if not 'batch_norm' in options.keys():
            if verbose >=3:
                print("... No batch norm provided for layer " + id + ". Batch norm is off")
            batch_norm = False
        else:
            batch_norm = options["batch_norm"]

        if not 'pool_size' in options.keys():
            if verbose >=3:
                print("... No pool size provided for layer " + id + " assume (1,1)")
            pool_size = (1,1)
        else:
            pool_size = options ["pool_size"]

        if not 'pool_type' in options.keys():
            if verbose >=3:
                print("... No pool type provided for layer " + id + " assume max")
            pool_type = 'max'
        else:
            pool_type = options ["pool_type"]

        if not 'input_params' in options.keys():
            if verbose >=3:
                print("... No initial params for layer " + id + " assume None")
            input_params = None
        else:
            input_params = options ["input_params"]

        if not 'dropout_rate' in options.keys():
            if verbose >=3:
                print("... No dropout_rate set for layer " + id + " assume 0")
            dropout_rate = 0
        else:
            dropout_rate = options ["dropout_rate"]

        if not 'regularize' in options.keys():
            if verbose >=3:
                print ("... No regularize set for layer " + id + " assume False")
            regularize = False
        else:
            regularize = options ["regularize"]

        if verbose >=3:
            print("... creating the dropout stream")
        # Just create a dropout layer no matter what.

        from yann.layers.conv_pool import dropout_conv_pool_layer_2d as dcpl2d
        from yann.layers.conv_pool import conv_pool_layer_2d as cpl2d

        self.dropout_layers[id] = dcpl2d (
                                        input = self.dropout_layers[origin].output,
                                        dropout_rate = dropout_rate,
                                        nkerns = nkerns,
                                        id = id,
                                        input_shape = self.dropout_layers[origin].output_shape,
                                        filter_shape = filter_size,
                                        poolsize = pool_size,
                                        pooltype = pool_type,
                                        batch_norm = batch_norm,
                                        border_mode = border_mode,
                                        stride = stride,
                                        rng = self.rng,
                                        borrow = self.borrow,
                                        activation = activation,
                                        input_params = input_params,
                                        verbose = verbose,
                                        )
        # If dropout_rate is 0, this is just a wasted multiplication by 1, but who cares.
        if dropout_rate >0:
            w = self.dropout_layers[id].w * (1 - dropout_rate)
        else:
            w = self.dropout_layers[id].w
        b = self.dropout_layers[id].b

        layer_params = [w,b]
        if batch_norm is True:
            # should I halve the gamma for batch norm ?
            gamma = self.dropout_layers[id].gamma
            beta = self.dropout_layers[id].beta
            mean = self.dropout_layers[id].running_mean
            var = self.dropout_layers[id].running_var
            layer_params.append(gamma)
            layer_params.append(beta)
            layer_params.append(mean)
            layer_params.append(var)
        if verbose >=3:
            print("... creating the stable stream")

        self.layers[id] = cpl2d (
                            input = self.layers[origin].output,
                            nkerns = nkerns,
                            id = id,
                            input_shape = self.layers[origin].output_shape,
                            filter_shape = filter_size,
                            poolsize = pool_size,
                            pooltype = pool_type,
                            batch_norm = batch_norm,
                            border_mode = border_mode,
                            stride = stride,
                            rng = self.rng,
                            borrow = self.borrow,
                            activation = activation,
                            input_params = layer_params,
                            verbose = verbose,
                                )

        self.inference_layers[id] = cpl2d (
                            input = self.inference_layers[origin].inference,
                            nkerns = nkerns,
                            id = id,
                            input_shape = self.layers[origin].output_shape,
                            filter_shape = filter_size,
                            poolsize = pool_size,
                            pooltype = pool_type,
                            batch_norm = batch_norm,
                            border_mode = border_mode,
                            stride = stride,
                            rng = self.rng,
                            borrow = self.borrow,
                            activation = activation,
                            input_params = layer_params,
                            verbose = verbose,
                                )

        if regularize is True:
            self.L1 = self.L1 + self.layers[id].L1
            self.L2 = self.L2 + self.layers[id].L2

        self.dropout_layers[id].origin.append(origin)
        self.dropout_layers[origin].destination.append(id)
        self.layers[id].origin.append(origin)
        self.layers[origin].destination.append(id)
        self.inference_layers[id].origin.append(origin)
        self.inference_layers[origin].destination.append(id)

    def _add_flatten_layer( self, id, options, verbose = 2):
        """
        Internal function that adds a flattening layer.

        Args:
            id: id of the layer
            options: basically kwargs supplied to add_layer
            verbose: as usual
        """
        if verbose >= 3:
            print("... Adding a flatten layer")
        if not 'origin' in options.keys():
            if self.last_layer_created is None:
                raise Exception("You can't create a flatten without a layer to flatten.")
            if verbose >=3:
                print("... origin layer is not supplied, assuming the last layer created is.")
            origin = self.last_layer_created
        else:
            origin = options ["origin"]

        input = self.layers[origin].output
        dropout_input = self.dropout_layers[origin].output
        inference_input = self.inference_layers[origin].inference
        input_shape = self.layers[origin].output_shape

        from yann.layers.flatten import flatten_layer as flt
        self.dropout_layers[id] = flt(input = dropout_input, id = id, input_shape = input_shape)
        self.layers[id] = flt(input = input, id = id, input_shape = input_shape)
        self.inference_layers[id] = flt(input = inference_input, id = id, input_shape = input_shape)
        

        self.dropout_layers[id].origin.append(origin)
        self.dropout_layers[origin].destination.append(id)
        self.layers[id].origin.append(origin)
        self.layers[origin].destination.append(id)

    def _add_unflatten_layer( self, id, options, verbose = 2):
        """
        Internal function that adds an unflattening layer.

        Args:
            id: id of the layer
            options: basically kwargs supplied to add_layer
            verbose: as usual
        """
        if verbose >= 3:
            print("... Adding a flatten layer")
        if not 'origin' in options.keys():
            if self.last_layer_created is None:
                raise Exception("You can't create a flatten without a layer to flatten.")
            if verbose >=3:
                print("... origin layer is not supplied, assuming the last layer created is.")
            origin = self.last_layer_created
        else:
            origin = options ["origin"]

        if not 'shape' in options.keys():
            raise Exception ('This type of layer needs a shape variable to unflatten to')
        else:
            shape = options["shape"]

        input = self.layers[origin].output
        dropout_input = self.dropout_layers[origin].output
        inference_input = self.inference_layers[origin].inference
        input_shape = self.layers[origin].output_shape

        if len(shape) == 2:
            shape = (shape[0], shape[1], 1) # If not provided assume 1 channel

        from yann.layers.flatten import unflatten_layer as flt
        self.dropout_layers[id] = flt(input = dropout_input, id = id, shape = shape,
                                                                    input_shape = input_shape)
        self.layers[id] = flt(input = input, id = id, shape = shape, input_shape = input_shape)
        self.inference_layers[id] = flt(input = inference_input, id = id, shape = shape,
                                                                    input_shape = input_shape)
        

        self.dropout_layers[id].origin.append(origin)
        self.dropout_layers[origin].destination.append(id)
        self.layers[id].origin.append(origin)
        self.layers[origin].destination.append(id)
        self.inference_layers[id].origin.append(origin)
        self.inference_layers[origin].destination.append(id)


    def _add_dot_product_layer(self, id, options, verbose = 2):
        """
        This is an internal function. Use ``add_layer`` instead of this from outside the class.

        Args:
            options: Basically kwargs supplied to the add_layer function.
            verbose: simiar to everywhere on the toolbox.
        """
        if verbose >= 3:
            print("... Adding a dot product layer")
        if not 'origin' in options.keys():
            if self.last_layer_created is None:
                raise Exception("You can't create a fully connected layer without an" + \
                                    " origin layer.")
            if verbose >=3:
                print("... origin layer is not supplied, assuming the last layer created is.")
            origin = self.last_layer_created
        else:
            origin = options ["origin"]

        # If the last layer was not a MLP layer, flatten the output signal from there.
        if not len(self.layers[origin].output_shape) == 2:
            if verbose >= 3:
                print("... Can't add a fully connected layer to a 2D image, flattening the" + \
                                                                                  " layer output.")
            self.add_layer(type = 'flatten', origin = origin, verbose = verbose)
            origin = self.last_layer_created

        input = self.layers[origin].output
        dropout_input = self.dropout_layers[origin].output
        inference_input = self.inference_layers[origin].inference
        input_shape = self.layers[origin].output_shape

        if not 'num_neurons' in options.keys():
            if verbose >=3:
                print("... num_neurons not provided, Assuming 100")
            num_neurons = 100
        else:
            num_neurons = options ["num_neurons"]

        if not 'activation' in options.keys():
            if verbose >=3:
                print("... Activations not provided for layer " + id + ". Using ReLU")
            activation = 'relu'
        else:
            activation = options ["activation"]

        if not 'batch_norm' in options.keys():
            if verbose >=3:
                print("... No batch norm provided for layer " + id + ". Batch norm is off")
            batch_norm = False
        else:
            batch_norm = options["batch_norm"]

        if not 'input_params' in options.keys():
            if verbose >=3:
                print("... No initial params for layer " + id + " assume None")
            input_params = None
        else:
            input_params = options ["input_params"]

        if not 'dropout_rate' in options.keys():
            if verbose >=3:
                print("... No dropout_rate set for layer " + id + " assume 0")
            dropout_rate = 0
        else:
            dropout_rate = options ["dropout_rate"]

        if not 'regularize' in options.keys():
            if verbose >=3:
                print("... No regularize set for layer " + id + " assume False")
            regularize = False
        else:
            regularize = options ["regularize"]
        if verbose >=3:
            print("... creating the dropout stream")
        # Just create a dropout layer no matter what.

        from yann.layers.fully_connected import dropout_dot_product_layer as ddpl
        from yann.layers.fully_connected import dot_product_layer as dpl

        self.dropout_layers[id] = ddpl (
                                input = dropout_input,
                                dropout_rate = dropout_rate,
                                num_neurons = num_neurons,
                                id = id,
                                input_shape = input_shape,
                                rng = self.rng,
                                input_params = input_params,
                                borrow = self.borrow,
                                activation = activation,
                                batch_norm = batch_norm,
                                verbose = verbose
                                )
        # If dropout_rate is 0, this is just a wasted multiplication by 1, but who cares.
        w = self.dropout_layers[id].w * (1 - dropout_rate)
        b = self.dropout_layers[id].b 
        layer_params = [w,b]

        if batch_norm is True:
            # Again, should I halve gamma because of dropout ???
            gamma = self.dropout_layers[id].gamma 
            beta = self.dropout_layers[id].beta
            mean = self.dropout_layers[id].running_mean
            var = self.dropout_layers[id].running_var            
            layer_params.append(gamma)
            layer_params.append(beta)
            layer_params.append(mean)
            layer_params.append(var)
        if verbose >=3:
            print("... creating the stable stream")
        self.layers[id] = dpl (
                            input = input,
                            num_neurons = num_neurons,
                            input_shape = input_shape,
                            id = id,
                            rng = self.rng,
                            input_params = layer_params,
                            borrow = self.borrow,
                            activation = activation,
                            batch_norm = batch_norm,
                            verbose = verbose
                                )
        self.inference_layers[id] = dpl (
                            input = inference_input,
                            num_neurons = num_neurons,
                            input_shape = input_shape,
                            id = id,
                            rng = self.rng,
                            input_params = layer_params,
                            borrow = self.borrow,
                            activation = activation,
                            batch_norm = batch_norm,
                            verbose = verbose
                                )                                
        if regularize is True:
            self.L1 = self.L1 + self.layers[id].L1
            self.L2 = self.L2 + self.layers[id].L2

        self.dropout_layers[id].origin.append(origin)
        self.dropout_layers[origin].destination.append(id)
        self.layers[id].origin.append(origin)
        self.layers[origin].destination.append(id)

    def _add_classifier_layer(self, id, options, verbose = 2):
        """
        This is an internal function. Use ``add_layer`` instead of this from outside the class.

        Args:
            options: Basically kwargs supplied to the add_layer function.
            verbose: simiar to everywhere on the toolbox.
        """
        if verbose >=3:
            print("... Adding a classifier layer")

        if not 'origin' in options.keys():
            if self.last_layer_created is None:
                raise Exception("You can't create a softmax layer without an" + \
                                    " origin layer.")
            if verbose >=3:
                print("... origin layer is not supplied, assuming the last layer created is.")
            origin = self.last_layer_created
        else:
            origin = options ["origin"]

        # If the last layer was not a MLP layer, flatten the output signal from there.
        if not len(self.layers[origin].output_shape) == 2:
            if verbose >= 3:
                print "... Can't add a classifier layer to a 2D image, flattening the" + \
                                                                                  " layer output."
            self.add_layer(type = 'flatten', origin = origin, verbose = verbose)
            origin = self.last_layer_created

        input = self.layers[origin].output
        dropout_input = self.dropout_layers[origin].output
        inference_input = self.inference_layers[origin].inference
        input_shape = self.layers[origin].output_shape

        if not 'num_classes' in options.keys():
            raise Exception("Supply number of classes")
        else:
            num_classes = options ["num_classes"]

        if not 'activation' in options.keys():
            if verbose >=3:
                print("... Activations not provided for layer " + id + ". Using ReLU")
            activation = 'softmax'
        else:
            activation = options ["activation"]

        if not 'input_params' in options.keys():
            if verbose >=3:
                print("... No initial params for layer " + id + " assume None")
            input_params = None
        else:
            input_params = options ["input_params"]

        if not 'regularize' in options.keys():
            if verbose >=3:
                print("... No regularize set for layer " + id + " assume False")
            regularize = False
        else:
            regularize = options ["regularize"]

        if verbose >=3:
            print("... creating the dropout stream")
        # Just create a dropout layer no matter what.

        from yann.layers.output import classifier_layer as classifier

        self.dropout_layers[id] = classifier (
                                    input = dropout_input,
                                    id = id,
                                    input_shape = input_shape,
                                    num_classes = num_classes,
                                    rng = self.rng,
                                    input_params = input_params,
                                    borrow = self.borrow,
                                    activation = activation,
                                    verbose = verbose
                                )
        if verbose >=3:
            print("... creating the stable stream")
        params = self.dropout_layers[id].params
        self.layers[id] = classifier (
                                    input = input,
                                    id = id,
                                    input_shape = input_shape,
                                    num_classes = num_classes,
                                    rng = self.rng,
                                    input_params = params,
                                    borrow = self.borrow,
                                    activation = activation,
                                    verbose = verbose
                                )
        self.inference_layers[id] = classifier (
                                    input = inference_input,
                                    id = id,
                                    input_shape = input_shape,
                                    num_classes = num_classes,
                                    rng = self.rng,
                                    input_params = params,
                                    borrow = self.borrow,
                                    activation = activation,
                                    verbose = verbose
                                )                                

        if regularize is True:
            self.L1 = self.L1 + self.layers[id].L1
            self.L2 = self.L2 + self.layers[id].L2

        self.dropout_layers[id].origin.append(origin)
        self.dropout_layers[origin].destination.append(id)
        self.layers[id].origin.append(origin)
        self.layers[origin].destination.append(id)
        self.inference_layers[id].origin.append(origin)
        self.inference_layers[origin].destination.append(id)

    def _add_objective_layer (self, id, options, verbose = 2):
        """
        This is an internal function. Use ``add_layer`` instead of this from outside the class.

        Args:
            options: Basically kwargs supplied to the add_layer function.
            verbose: simiar to everywhere on the toolbox.

        """

        if verbose >=3:
            print("... Adding an objective layer")

        if not 'layer_type' in options.keys():
            if verbose >= 3:
                print("... type is not provided, assuming nll")
            type = 'nll'
        else:
            type = options['layer_type']

        if not 'origin' in options.keys():
            if self.last_classifier_created is None:
                raise Exception("You can't create an abstract objective layer without a" + \
                                    " classifier layer.")
            if verbose >=3:
                print("... origin layer is not supplied, assuming the last classifier layer" + \
                                   " created is the origin.")
            if not type == 'value':
                origin = self.last_classifier_created
            else:
                origin = None
        else:
            origin = options["origin"]
            import types
            if isinstance(origin, types.ListType):
                raise Exception ( "layer-layer loss is not supported by the objective layer, use \
                                  merge layer for it .")
            else:
                origin = options ["origin"]

        if not 'objective' in options.keys():
            if not type == 'value':
                if verbose >= 3:
                    print("... objective not provided, assuming nll")
                objective = type
                # check if the origin layer is a classifier error.
                loss = getattr(self.layers[origin], "loss", None)
                dropout_loss = getattr(self.dropout_layers[origin], "loss", None)

            if loss is None:
                raise Exception ("Layer " + origin + " doesn't provide a loss function")
        else:
            if type =='value':
                loss = options['objective']
                dropout_loss = options['objective']
                objective = 'value'
            else:
                objective = options['objective']
                loss = getattr(self.layers[origin], "loss", None)
                dropout_loss = getattr(self.dropout_layers[origin], "loss", None)

        if 'dataset_origin' in options.keys():
            if 'dataset_origin' in self.datastream.keys():
                datastream_id = options["datset_origin"]
            else:
                if verbose >= 3:
                    print("... Invalid datastream id, switching to last created datastream")
                datastream_id = self.last_datastream_created
        else:
            datastream_id = self.last_datastream_created

        if self.datastream[datastream_id].svm is True and not objective == 'hinge':
            if verbose >=2:
                print(".. Objective should only be hinge if datastream is setup for svm. Beware")
            self.datastream[datastream_id].svm = False

        if objective == 'hinge':
            if not hasattr(self.datastream[datastream_id], 'one_hot_y') is True:
                if verbose >=1:
                    print(". Datastream is not setup for hinge loss " + \
                                                       "switching to negative log likelihood")
                objective = 'nll'
                data_y = self.datastream[datastream_id].y
            else:
                data_y = data_y = self.datastream[datastream_id].one_hot_y
        else:
            data_y = self.datastream[datastream_id].y


        # Just create a dropout layer no matter what.
        if not 'regularizer' in options.keys():
            l1_regularizer_coeff = 0
            l2_regularizer_coeff = 0
        else:
            l1_regularizer_coeff, l2_regularizer_coeff = options['regularizer']

        if verbose >=3:
            print("... creating the dropout stream")

        from yann.layers.output import objective_layer as obj

        self.dropout_layers[id] = obj(
                                    objective = objective,
                                    labels = data_y,
                                    id = id,
                                    loss = dropout_loss,
                                    L1 = self.L1,
                                    L2 = self.L2,
                                    l1_coeff = l1_regularizer_coeff,
                                    l2_coeff = l2_regularizer_coeff,
                                    verbose = verbose )
        if verbose >=3:
            print("... creating the stable stream")

        self.layers[id] = obj(
                            objective = objective,
                            labels = data_y,
                            id = id,
                            loss = loss,
                            L1 = self.L1,
                            L2 = self.L2,
                            l1_coeff = l1_regularizer_coeff,
                            l2_coeff = l2_regularizer_coeff,
                            verbose = verbose )

        self.inference_layers[id] = obj(
                            objective = objective,
                            labels = data_y,
                            id = id,
                            loss = loss,
                            L1 = self.L1,
                            L2 = self.L2,
                            l1_coeff = l1_regularizer_coeff,
                            l2_coeff = l2_regularizer_coeff,
                            verbose = verbose )

        if not origin == None:
            self.dropout_layers[id].origin.append(origin)
            self.dropout_layers[origin].destination.append(id)
            self.layers[id].origin.append(origin)
            self.layers[origin].destination.append(id)
            self.inference_layers[id].origin.append(origin)
            self.inference_layers[origin].destination.append(id)            

    def _add_merge_layer(self, id, options, verbose = 2):
        """
        This is an internal function. Use ``add_layer`` instead of this from outside the class.

        Args:
            options: Basically kwargs supplied to the add_layer function.
            verbose: simiar to everywhere on the toolbox.

        """
        if verbose >=3:
            print("... Adding a merge layer")

        if not 'origin' in options.keys():
            raise Exception("You can't create an merge layer without atleast two" + \
                                    "layer ids as input.")
        else:
            origin = options["origin"]
            if not type(origin) is tuple:
                raise Exception ( "layer-layer loss needs a tuple as origin")
            else:
                origin = options ["origin"]

        if not 'error' in options.keys():
            error = 'rmse'
        else:
            error = options ["error"]

        if not 'layer_type' in options.keys():
            layer_type = 'error'
        else:
            layer_type = options['layer_type']

        if verbose >=3:
            print("... creating the dropout stream")

        from yann.layers.merge import merge_layer as mrg
        inputs = []
        input_shape = []

        for lyr in origin:
            inputs.append ( self.dropout_layers[lyr].output )
            input_shape.append (self.dropout_layers[lyr].output_shape)

        self.dropout_layers[id] = mrg(
                                    id = id,
                                    x = inputs,
                                    type = layer_type,
                                    error = error,
                                    input_shape = input_shape,
                                    verbose = verbose )
        if verbose >=3:
            print("... creating the stable stream")

        inputs = [] # input_shape is going to remain the same from dropout to this.
        for lyr in origin:
            inputs.append(self.layers[lyr].output)

        self.layers[id] = mrg( id = id,
                               x = inputs,
                               error = error,
                               type = layer_type,
                               input_shape = input_shape,
                               verbose = verbose )

        inputs = [] # input_shape is going to remain the same from dropout to this.
        for lyr in origin:
            inputs.append(self.inference_layers[lyr].inference)

        self.inference_layers[id] = mrg( id = id,
                               x = inputs,
                               error = error,
                               type = layer_type,
                               input_shape = input_shape,
                               verbose = verbose )

        for lyr in origin:
            self.dropout_layers[id].origin.append(lyr)
            self.layers[id].origin.append(lyr)
            self.inference_layers[id].origin.append(lyr)
            self.dropout_layers[lyr].destination.append(id)
            self.layers[lyr].destination.append(id)
            self.inference_layers[lyr].destination.append(id)
            

    def _add_random_layer(self, id, options, verbose = 2):
        """
        This is an internal function. Use ``add_layer`` instead of this from outside the class.

        Args:
            options: Basically kwargs supplied to the add_layer function.
            verbose: simiar to everywhere on the toolbox.
        """
        if verbose >=3:
            print("... Adding a random generator layer")

        from yann.layers.random import random_layer as rl

        if 'distribution' in options.keys():
            distribution = options['distribution']
        else:
            distribution = 'binomial'

        if not 'num_neurons' in options.keys():
            if verbose >=3:
                print("... num_neurons not provided, Assuming 100")
            num_neurons = 100
        else:
            num_neurons = options ["num_neurons"]


        self.dropout_layers[id] = rl (
                            id = id,
                            num_neurons = num_neurons,
                            distribution = distribution,
                            options = options,
                            verbose = verbose)

        self.layers[id] = self.dropout_layers[id]
        self.inference_layers[id] = self.dropout_layers[id]

    def _add_rotate_layer(self, id, options, verbose = 2):
        """
        This is an internal function. Use ``add_layer`` instead of this from outside the class.

        Args:
            options: Basically kwargs supplied to the add_layer function.
            angle:  Value between [0,1] to capture the rotation between [0,90] degrees
                    If None is specified, angle is generated randomly from a uniform dist
            verbose: simiar to everywhere on the toolbox.
        """
        if verbose >=3:
            print("... Adding a rotate layer")

        if not 'origin' in options.keys():
            if self.last_layer_created is None:
                raise Exception("You can't create a fully connected layer without an" + \
                                    " origin layer.")
            if verbose >=3:
                print("... origin layer is not supplied, assuming the last layer created is.")
            origin = self.last_layer_created
        else:
            origin = options ["origin"]

        from yann.layers.transform import rotate_layer as rl
        from yann.layers.transform import dropout_rotate_layer as drl

        input = self.layers[origin].output
        dropout_input = self.dropout_layers[origin].output
        inference_input = self.inference_layers[origin].inference
        
        input_shape = self.layers[origin].output_shape

        if 'angle' in options.keys():
            angle = options['angle']
        else:
            angle = None

        self.layers[id] = rl(
                            input = input,
                            input_shape = input_shape,
                            id = id,
                            angle = angle,
                            verbose =verbose)

        self.dropout_layers[id] = drl (
                            input = dropout_input,
                            input_shape = input_shape,
                            id = id,
                            angle = angle,
                            verbose = verbose)

        self.inference_layers[id] = drl (
                            input = inference_input,
                            input_shape = input_shape,
                            id = id,
                            angle = angle,
                            verbose = verbose)

    def _initialize_test_classifier(self, errors, verbose):
        """
        Internal function that creates a test method for a classifier network

        Args:
            errors: a function that returns the error when supplied with y data.
        """
        if verbose >=3:
            print("... creating the classifier testing theano function ")

        index = T.lscalar('index')

        self.mini_batch_test = theano.function(
            inputs = [index],
            outputs = errors(self.y),
            name = 'test',
            givens={
            self.x: self.data_x[ index * self.mini_batch_size:(index + 1) * self.mini_batch_size],
            self.y: self.data_y[ index * self.mini_batch_size:(index + 1) * self.mini_batch_size]})



    def _initialize_test_value(self, errors, verbose):
        """
        Internal function that creates a test method for a classifier network

        Args:
            errors: a function that returns the errors
        """
        if verbose >=3:
            print("... creating the generator testing theano function ")

        index = T.lscalar('index')

        self.mini_batch_test = theano.function(
            inputs = [index],
            outputs = errors,
            name = 'test',
            givens={
            self.x: self.data_x[ index * self.mini_batch_size:(index + 1) * self.mini_batch_size]})

    def _initialize_test (self, verbose = 2, **kwargs):
        """
        Internal function to create the ``self.test_batch``  theano function. ``net.cook`` will use
        this function.

        Args:
            datastream: as always
            classifier: the classifier layer to test out of.
            generator: if generator network, generator layer to test out of.
            value: if value-based objective network, the value layer to test out of.
            verbose: as always

        """
        if verbose>=3 :
            print("... initializing test function")

        if self.cooked_datastream is None:
               raise Exception ("This needs to be run only after datastream is cooked")

        if 'classifier' in kwargs.keys():
            _errors = self.inference_layers[kwargs['classifier']].errors
            self._initialize_test_classifier(errors = _errors, verbose = verbose)

        elif 'value' in kwargs.keys():
            _errors = self.inference_layers[kwargs['value']].inference
            self._initialize_test_value(errors = _errors, verbose = verbose)

        else:
            raise Exception('Supply a type of layer to cook test')


    def _initialize_predict (self, classifier = None, verbose = 2):
        """
        Internal function to create the ``self.predict_batch``  theano function.
        ``net.cook`` will use this function.

        Args:
            datastream: as always
            classifier: the classifier layer whose predictions are needed.
            verbose: as always

        """
        if self.cooked_datastream is None:
               raise Exception ("This needs to be run only after network is cooked")

        if verbose>=3 :
            print("... initializing predict function")

        _predictions = self.inference_layers[classifier].predictions

        index = T.lscalar('index')

        self.mini_batch_predictions = theano.function(
                inputs = [index],
                outputs = _predictions,
                name = 'predict',
                givens={
            self.x: self.data_x[ index * self.mini_batch_size:(index + 1) * self.mini_batch_size]})

    def _initialize_posterior (self, classifier = None, generator = None, verbose = 2):
        """
        Internal function to create the ``self.probabilities_batch``  theano function.
        ``net.cook`` will use this function.

        Args:
            datastream: as always
            classifier: the classifier layer whose predictions are needed.
            verbose: as always

        """
        if not classifier is None:

            if self.cooked_datastream is None:
                raise Exception ("This needs to be run only after network is cooked")

            if verbose>=3 :
                print("... initializing probability output functions")

            _probabilities = self.inference_layers[classifier].probabilities

            index = T.lscalar('index')
            self.mini_batch_posterior = theano.function(
                    inputs = [index],
                    outputs = _probabilities,
                    name = 'posterior',
                    givens={
                self.x: self.data_x[ index * self.mini_batch_size:(index + 1) * \
                                                                             self.mini_batch_size]})
        else:
            if verbose >=3:
                print("... This network does not need a posterior")

    def _initialize_train_classifier(self, objective = None, verbose = 2):
        """
        Internal function that creates a train method for a classifier network

        Args:
            objective: a function that returns the objective when supplied with y data.
        """
        if verbose >=3:
            print("... creating the classifier training theano function ")

        index = T.lscalar('index')
        if self.cooked_datastream.svm is False:
            self.mini_batch_train = theano.function(
                    inputs = [index, self.cooked_optimizer.epoch],
                    outputs = objective,
                    name = 'train',
                    givens={
            self.x: self.data_x[index * self.mini_batch_size:(index + 1) * self.mini_batch_size],
            self.y: self.data_y[index * self.mini_batch_size:(index + 1) * self.mini_batch_size]},
                    updates = self.cooked_optimizer.updates, on_unused_input = 'ignore')
        else:
            self.mini_batch_train = theano.function(
                    inputs = [index, self.cooked_optimizer.epoch],
                    outputs = objective,
                    name = 'train',
                    # profile = True, # uncommenting this line will enable profiling
                    givens={
            self.x: self.data_x[ index * self.mini_batch_size:(index + 1) * self.mini_batch_size],
            self.one_hot_y: self.data_one_hot_y[index * self.mini_batch_size:(index + 1) *
                                                                    self.mini_batch_size]},
                    updates = self.cooked_optimizer.updates, on_unused_input = 'ignore')

    def _initialize_train_value(self, objective = None, verbose = 2):
        """
        Internal function that creates a train method for a generator network

        Args:
            objective: a function that returns the objective when supplied with y data.
        """
        if verbose >=3:
            print("... creating the generator training theano function ")

        index = T.lscalar('index')
        if self.cooked_datastream.svm is False:
            self.mini_batch_train = theano.function(
                    inputs = [index, self.cooked_optimizer.epoch],
                    outputs = objective,
                    name = 'train',
                    givens={
            self.x: self.data_x[index * self.mini_batch_size:(index + 1) * self.mini_batch_size]},
                    updates = self.cooked_optimizer.updates, on_unused_input = 'ignore')
        else:
            self.mini_batch_train = theano.function(
                    inputs = [index, self.cooked_optimizer.epoch],
                    outputs = objective,
                    name = 'train',
                    # profile = True, # uncommenting this line will enable profiling
                    givens={
            self.x: self.data_x[ index * self.mini_batch_size:(index + 1) * self.mini_batch_size]},
                    updates = self.cooked_optimizer.updates, on_unused_input = 'ignore')

    def _initialize_train (self,objective = None, verbose = 2):
        """
        Internal function to create the ``self.train_batch``  theano function.
        ``net.cook`` will use this function.

        Args:
            datastream: an id
            optimizer: an id
            objective: a graph that connects to loss or None, use self.dropout_cost
            verbose: as always
        """

        if self.cooked_datastream is None:
               raise Exception ("This needs to be run only after network is cooked")

        if verbose >= 3:
            print("... initializing trainer functions")

        if objective == None:
            objective = self.dropout_cost

        if self.network_type == 'classifier':
            self._initialize_train_classifier(objective = objective, verbose = verbose)

        else:
            self._initialize_train_value(objective = objective, verbose = verbose)

    def _cook_optimizer (self, params = None, objective = None, optimizer = None, verbose = 2):
        """
        Internal function to create the ``self.decay_learning_rate`` and
        ``self.momentum_value`` and ``self.learning_rate``   theano function.
        ``net.cook`` will use this function.

        Args:
            optimizer: an id
            verbose: as always
            params: provide the parameters that you'd want to update. if ``None``, will use
                   ``active_params``
            objective: provide the objective that yo'd need to cook optimizer with. if ``None``,
                        will use ``dropout_cost``.
        """
        if params is None:
            params = self.active_params

        if objective is None:
            objective = self.dropout_cost

        if verbose >=3:
            print("... Cooking Optimizer")

        if optimizer is None:
            optimizer = self.cooked_optimizer

        self.learning_rate = optimizer.learning_rate
        anneal_rate = T.scalar('annealing_rate')
        self.decay_learning_rate = theano.function(
                        inputs=[anneal_rate],          # Just updates the learning rates.
                        name = 'annealing',
                        updates={self.learning_rate: self.learning_rate - self.learning_rate *
                                                                            anneal_rate })
        self.current_momentum = theano.function ( inputs =[optimizer.epoch],
                                                         outputs = optimizer.momentum,
                                                         name = 'momentum' )

        optimizer.calculate_gradients(params = params,
                                       objective = objective,
                                       verbose = verbose)
        optimizer.create_updates (params = params, verbose = verbose)

        # add layer specific updates on to the main optimizer updates.
        for lyr in self.dropout_layers:
            optimizer.updates.update(self.dropout_layers[lyr].updates)


    def _create_layer_activity(self, id, activity, verbose = 2):
        """
        This is an internal method to create activity of one generator layer.

        Args:
            id : id of the layer to calcualte activity of.
            activity: Supply the activity to output to produce activity of.
            verbose: as always

        """
        index = T.lscalar('index')
        if self.network_type == 'classifier':
            self.layer_activities[id] = theano.function(
                    name = 'layer_activity_' + id,
                    inputs = [index],
                    outputs = activity,
                    givens={
                    self.x: self.cooked_datastream.data_x[index *
                                    self.cooked_datastream.mini_batch_size:(index + 1) *
                                    self.cooked_datastream.mini_batch_size],
                    self.y: self.cooked_datastream.data_y[index *
                                    self.cooked_datastream.mini_batch_size:(index + 1) *
                                    self.cooked_datastream.mini_batch_size]},
                                    on_unused_input = 'ignore')
        else:
            self.layer_activities[id] = theano.function(
                    name = 'layer_activity_' + id,
                    inputs = [index],
                    outputs = activity,
                    givens={
                    self.x: self.cooked_datastream.data_x[index *
                                    self.cooked_datastream.mini_batch_size:(index + 1) *
                                    self.cooked_datastream.mini_batch_size]},
                                    on_unused_input = 'ignore')

    def _create_layer_activities(self, datastream = None, verbose = 2):
        """
        Use this function to create activities for  each layer.
        I don't know why this might be useful, but its fun to check this out I guess. This will only
        work after the dataset is initialized.

        Used internally by ``cook`` method. Use the layer_activity

        Args:
            datastream: id of the datastream, Default is latest.
            verbose: as usual

        """
        if verbose >=3:
            print("... creating the activities of all layers ")

        if self.cooked_datastream is None:
           raise Exception ("This needs to be run only after network is cooked")

        self.layer_activities_created = True
        for id, _layer in self.inference_layers.iteritems():
            if verbose >=3 :

                print ("... collecting the activities of layer " + id)
            activity = _layer.inference
            self._create_layer_activity(id = id, activity = activity, verbose = verbose)

    def _new_era (self, new_learning_rate = 0.01, verbose = 2):
        """
        This re-initializes the learning rate to the learning rate variable. This also reinitializes
        the parameters of the network to best_params.

        Args:
            new_learning_rate: rate at which you want fine tuning to begin.
            verbose: Just as the rest of the toolbox.
        """
        if verbose >= 3:
            print("... setting up new era")
        self.learning_rate.set_value(numpy.asarray(new_learning_rate,dtype = theano.config.floatX))
        # copying and removing only active_params. Is that a porblem ?
        copy_params ( source = self.best_params, destination = self.active_params ,
                                                                            borrow = self.borrow)

    def _cook_datastream (self, verbose = 2):
        """
        Drag details from datastream to the network.add_layer

        Args:
            verbose: Just as always
        """
        if verbose >= 3:
            print("... Cooking datastream")

        self.mini_batch_size = self.cooked_datastream.mini_batch_size
        self.height = self.cooked_datastream.height
        self.width = self.cooked_datastream.width
        self.channels = self.cooked_datastream.channels
        self.batches2train = self.cooked_datastream.batches2train
        self.batches2test = self.cooked_datastream.batches2test
        self.batches2validate = self.cooked_datastream.batches2validate
        self.set_data = self.cooked_datastream.set_data
        self.cache = self.cooked_datastream.cache
        self.mini_batches_per_batch = self.cooked_datastream.mini_batches_per_batch
        self.data_x = self.cooked_datastream.data_x
        if self.network_type == 'classifier':
            self.y = self.cooked_datastream.y
            self.data_y = self.cooked_datastream.data_y
            if self.cooked_datastream.svm is True:
                self.data_one_hot_y = self.cooked_datastream.data_one_hot_y
                self.one_hot_y = self.cooked_datastream.one_hot_y
        self.x = self.cooked_datastream.x
        self.current_data_type = self.cooked_datastream.current_type

    def _cache_data (self, type = 'train', batch = 0, verbose = 2):
        """
        This just calls the datastream's ``set_data`` method and sets the appropriate variables.

        Args:
            type: ``'train'``, ``'test'``, ``'valid'``
            batch: Batch number
            verbose: As always.
        """

        if self.cooked_datastream is None:
            raise Exception ("This needs to be run only after network is cooked")

        if verbose >= 3:
            print("... Loading batch " + str(batch) + " of type " + type)
        self.set_data ( batch = batch , type = type, verbose = verbose )
        self.current_data_type = type

    def _cook_visualizer(self, verbose = 2):
        """
        This is an internal function that cooks a visualizer

        Args:
            cook_all: <bool> True will print all layer activities and stuff. False only prints
                     test and train.
            verbose: as always
        """
        if verbose >= 3:
            print("... Cooking visualizer")

        self._create_layer_activities ( verbose = verbose )

        if self.cooked_visualizer.debug_functions is True:
            if hasattr(self,'cooked_optimizer'):
                if verbose >= 3:
                    print("... Saving down visualizations of optimizer")

                self.cooked_visualizer.theano_function_visualizer(function = self.mini_batch_test,
                                                                                verbose = verbose)
                self.cooked_visualizer.theano_function_visualizer(function = self.mini_batch_train,
                                                                                verbose = verbose)
                self.cooked_visualizer.theano_function_visualizer(
                                                            function = self.mini_batch_posterior,
                                                            verbose = verbose)
                self.cooked_visualizer.theano_function_visualizer(
                                                            function = self.mini_batch_predictions,
                                                            verbose = verbose)
        if self.cooked_visualizer.debug_layers and self.layer_activities_created is True:
            for lyr in self.layer_activities.keys():
                self.cooked_visualizer.theano_function_visualizer(
                                                        function = self.layer_activities[lyr],
                                                        verbose = verbose)
        self.cooked_visualizer.initialize(batch_size = self.mini_batch_size, verbose = verbose)
        # datastream needs to be reshaped basically. Why I don't use unflatten or inputs I don't
        # know .
        imgs = self.data_x[:self.mini_batch_size,].reshape((self.mini_batch_size,
                                                self.height, self.width, self.channels)).eval()
        self.cooked_visualizer.visualize_images(imgs = imgs, verbose = verbose)
        self.visualize_after_epochs = 1

    def _cook_resultor (self, resultor = None, verbose = 2):
        """
        This is an internal function that cooks a resultor

        Args:
            verbose: as always
        """

        if verbose > 3:
            print("... Resultor is cooked")

    def visualize_activities( self, epoch = 0, verbose = 2):
        """
        This method will save down all layer activities for the correct epoch.

        Args:
            epoch: What epoch is being running now.
            verbose: As always.
        """
        self.cooked_visualizer.visualize_activities(layer_activities = self.layer_activities,
                                                 epoch = epoch,
                                                 index = 0,
                                                 verbose = verbose)

    def visualize_filters( self, epoch = 0, verbose = 2):
        """
        This method will save down all layer filters for the correct epoch.

        Args:
            epoch: What epoch is being running now.
            verbose: As always.
        """
        self.cooked_visualizer.visualize_filters(layers = self.dropout_layers,
                                                 epoch = epoch,
                                                 verbose = verbose)

    def visualize(self, epoch = 0, verbose =2 ):
        """
        This method will use the cooked visualizer to save down the visualizations

        Args:
            epoch: supply the epoch number ( used to create directories to save
        """
        if (epoch % self.visualize_after_epochs == 0):
            self.visualize_activities(epoch = epoch, verbose = verbose)
            self.visualize_filters(epoch = epoch, verbose = verbose)

    def write_results(self, epoch = 0, verbose =2 ):

        """
        This method will use the cooked visualizer to save down the visualizations

        Args:
            epoch: supply the epoch number ( used to create directories to save
        """
        if (epoch % self.write_results_after_epochs == 0):
            self.write_results(epoch = epoch, verbose = verbose)
            self.visualize_filters(epoch = epoch, verbose = verbose)

    def cook(self, verbose = 2, **kwargs):
        """
        This function builds the backprop network, and makes the trainer, tester and validator
        theano functions. The trainer builds the trainers for a particular objective layer and
        optimizer.

        Args:
            optimizer: Supply which optimizer to use.
                          Default is last optimizer created.
            datastream: Supply which datastream to use.
                          Default is the last datastream created.
            visualizer: Supply a visualizer to cook with.
                          Default is the last visualizer created.
            classifier_layer: supply the layer of classifier.
                          Default is the last classifier layer created.
            objective_layer: Supply the layer id of layer that has the objective function.
                          Default is last objective layer created if no classifier is provided.
            active_layers: Supply a list of active layers. If this parameter is supplied all
                           ``'learnabile'`` of all layers will be ignored and only these layers
                           will be trained. By default, all the learnable layers are used.
            verbose: Similar to the rest of the toolbox.


        """
        if verbose >= 2:
            print(".. Cooking the network")
        if verbose >= 3:
            print("... Building the network's objectives, gradients and backprop network")

        if not 'optimizer' in kwargs.keys():
            optimizer = None
        else:
            optimizer = kwargs['optimizer']

        if self.last_visualizer_created is None:
            visualizer_init_args = { }
            self.add_module(type = 'visualizer', params=visualizer_init_args, verbose = verbose)

        if not 'visualizer' in kwargs.keys():
            visualizer = self.last_visualizer_created
        else:
            visualizer = kwargs['visualizer']
        self.cooked_visualizer = self.visualizer[visualizer]

        if not 'datastream' in kwargs.keys():
            datastream = None
        else:
            datastream = kwargs['datastream']

        self.network_type = None # This is the first palce where the network type is created.
                                 # This is to differentiate between a classifier and other
                                 # types of networks.
        if 'classifier_layer' in kwargs.keys():
            classifier_layer = kwargs['classifier_layer']
            self.network_type = 'classifier'

        else:
            if not self.last_classifier_created is None:
                classifier_layer = self.last_classifier_created
                self.network_type = 'classifier'
            else:
                classifier_layer = None
                self.network_type = 'value'

        if not 'objective_layer' in kwargs.keys():
            objective_layer = self.last_objective_layer_created
        else:
            objective_layer = kwargs['objective_layer']

        if not 'active_layers' in kwargs.keys():
            params = self.active_params
        else:
            params = []
            for lyr in kwargs['active_layers']:
                params.append(lyr.params)

        if not 'resultor' in kwargs.keys():
            resultor = None
        else:
            resultor =  kwargs['resultor']

        if resultor is None:
            if self.last_resultor_created is None:
                if verbose >= 3:
                    print('... No resultor setup, creating a defualt one.')
                self.add_module( type = 'resultor', verbose =verbose )
            else:
                if verbose >= 3:
                    print("... resultor not provided, assuming " + self.last_resultor_created)
            resultor = self.last_resultor_created
        else:
            if not resultor in self.resultor.keys():
                raise Exception ("Resultor " + resultor + " not found.")
        self.cooked_resultor = self.resultor[resultor]

        if optimizer is None:
            if self.last_optimizer_created is None:

                optimizer_params =  {
                            "momentum_type"       : 'false',
                            "momentum_params"     : (0.9, 0.95, 30),
                            "regularization"      : (0, 0),
                            "optimizer_type"      : 'sgd',
                            "id"                  : "main"
                                }
                if verbose >= 3:
                    print('... No optimzier setup, creating a defualt one.')
                self.add_module( type = 'optimizer', params = optimizer_params, verbose =verbose )
            else:
                if verbose >= 3:
                    print("... optimizer not provided, assuming " + self.last_optimizer_created)
            optimizer = self.last_optimizer_created
        else:
            if not optimizer in self.optimizer.keys():
                raise Exception ("Optimzer " + optimizer + " not found.")
        self.cooked_optimizer = self.optimizer[optimizer]

        if datastream is None:
            if self.last_datastream_created is None:
                raise Exception("Cannot build trainer without having an datastream initialized")

            if verbose >= 3:
                print("... datastream not provided, assuming " + self.last_datastream_created)
            datastream = self.last_datastream_created
        else:
            if not datastream in self.datastream.keys():
                raise Exception ("Datastream " + datastream + " not found.")
        self.cooked_datastream = self.datastream[datastream]

        if objective_layer is None:
            if self.last_objective_layer_created is None:
                raise Exception ("Cannot build trainer without having an objective layer created")
            else:
                objective_layer = self.last_objective_layer_created

        if verbose >=2 :
            print (".. All checks complete, cooking continues" )

        self.cost = self.layers[objective_layer].output
        self.dropout_cost = self.dropout_layers[objective_layer].output
        self.cost = []

        self._cook_datastream(verbose = verbose)

        self._cook_optimizer(params = params,
                             optimizer = self.cooked_optimizer,
                             objective = self.dropout_cost,
                             verbose = verbose )

        if self.network_type == 'classifier':
            self._initialize_test (classifier = classifier_layer,
                                   verbose = verbose)
            self._initialize_predict ( classifier = classifier_layer,
                                   verbose = verbose)
            self._initialize_posterior (classifier = classifier_layer,
                                   verbose = verbose)
        else:
            self._initialize_test (value = objective_layer,
                                   verbose = verbose)
        self._initialize_train ( verbose = verbose )

        self._cook_resultor(resultor = self.cooked_resultor, verbose = verbose)
        self._cook_visualizer(verbose = verbose) # always cook visualizer last.
        self.visualize (epoch = 0, verbose = verbose)

        self.validation_accuracy = []
        self.best_validation_errors = numpy.inf
        self.best_training_errors = numpy.inf
        self.training_accuracy = []
        self.best_params = []
        # Let's bother only about learnable params. This avoids the problem when weights are
        # shared

        for param in params:
            self.best_params.append(theano.shared(param.get_value(borrow = self.borrow)))

    def print_status (self, epoch , verbose = 2):
        """
        This function prints the cost of the current epoch, learning rate and momentum of the
        network at the moment. This also calls the resultor to process results.

        Todo:
            This needs to to go to visualizer.

        Args:
            verbose: Just as always.
            epoch: Which epoch are we at ?
        """

        if self.cooked_datastream is None:
            raise Exception(" Cook first then run this.")

        if verbose >=2 :
            if len(self.cost) < self.batches2train * self.mini_batches_per_batch[0]:
                print(".. Cost                : " + str(self.cost[-1]))
            else:
                print(".. Cost                : " + str(numpy.mean(self.cost[-1 *
                                    self.batches2train * self.mini_batches_per_batch[0]:])))

        if verbose >= 3:
            print("... Learning Rate       : " + str(self.learning_rate.get_value(borrow=\
                                                                                 self.borrow)))
            print("... Momentum            : " + str(self.current_momentum(epoch)))

        lr = self.learning_rate.get_value(borrow =  self.borrow)
        mom = self.current_momentum(epoch)

        self.cooked_resultor.process_results(cost = cost,
                                           lr = lr,
                                           mom = mom,
                                           verbose = verbose)

    def _print_layer (self, id, prefix = " ", nest = True, last = True):
        """
        Internal funcrion used for recursion purposes.

        Args:
            id: ``id`` of the layer that is to be used as a root to print.
            prefix : string.. what to print first
            nest: To print more or not.
        """
        prefix_entry = self.layers[id].print_layer(prefix = prefix, nest=True, last = last)
        destinations = self.layers[id].destination
        count = len(destinations) - 1
        for id in destinations:
            if count <= 0:
                prefix = self._print_layer( id = id,
                                        prefix = prefix_entry,
                                        nest = nest,
                                        last = True)
            else:
                prefix = self._print_layer( id = id,
                                        prefix = prefix_entry,
                                        nest = nest,
                                        last = False)
                count = count - 1
        return prefix_entry

    def pretty_print (self, verbose = 2):
        """
        This method is used to pretty print the network's connections
        This is going to be deprecated with the use of visualizer module.
        """
        if verbose >=2:
            print(".. This method will be deprecated with the implementation of a visualizer," + \
                    "also this works only for tree-like networks. This will cause errors in " + \
                    "printing DAG-style networks.")
        input_layers = []
        # collect all begining of streams
        for id, layer in self.layers.iteritems():
            if layer.type == 'input' or layer.type == 'random':
                input_layers.append(id)

        for input_layer in input_layers:
            prefix = self._print_layer(id = input_layer, prefix = " ", nest = True)

    def validate(self, epoch = 0, training_accuracy = False, show_progress = False, verbose = 2):
        """
        Method is use to run validation. It will also load the validation dataset.

        Args:
            verbose: Just as always
            show_progress: Display progressbar ?
            training_accuracy: Do you want to print accuracy on the training set as well ?
        """
        best = False
        if  not (epoch % self.validate_after_epochs == 0) :
            return best

        validation_errors = 0
        training_errors = 0

        # Similar to the trianing loop
        if training_accuracy is True:
            total_mini_batches = self.batches2train * self.mini_batches_per_batch [0] \
                                 + self.batches2validate * self.mini_batches_per_batch [1]

        else:
            total_mini_batches =  self.batches2validate * self.mini_batches_per_batch[1]

        if show_progress is True:
            bar = progressbar.ProgressBar(maxval=total_mini_batches, \
                  widgets=[progressbar.AnimatedMarker(), \
                ' validation ', ' ', progressbar.Percentage(), ' ',progressbar.ETA(), ]).start()

        batch_counter = 0
        for batch in xrange (self.batches2validate):
            if verbose >= 3:
                print("... validating batch " + str(batch))
            self._cache_data ( batch = batch , type = 'valid', verbose = verbose )
            for minibatch in xrange(self.mini_batches_per_batch[1]):
                validation_errors = validation_errors + self.mini_batch_test (minibatch)
                if verbose >= 3:
                    print("... validation error after mini batch " + str(batch_counter) + \
                                                              " is " + str(validation_errors))
                batch_counter = batch_counter + 1
                if show_progress is True:
                    bar.update(batch_counter)


        batch_counter = 0
        if training_accuracy is True:
            if verbose >= 3:
                print("... training accuracy of batch " + str(batch))
            for batch in xrange (self.batches2train):
                self._cache_data(batch = batch, type = 'train', verbose = verbose )

                for minibatch in xrange(self.mini_batches_per_batch[0]):
                    training_errors = training_errors + self.mini_batch_test (minibatch)
                    if verbose >= 3:
                        print("... training error after mini batch " + str(batch_counter) + \
                                                                      " is " + str(training_errors))
                    batch_counter = batch_counter + 1
                    if show_progress is True:
                        bar.update(batch_counter + \
                                            self.batches2validate * self.mini_batches_per_batch[1])

        if show_progress is True:
            bar.finish()

        total_valid_samples = (self.batches2validate*self.mini_batches_per_batch[1]* \
                                                                            self.mini_batch_size)
        total_train_samples = (self.batches2train*self.mini_batches_per_batch[0]* \
                                                                              self.mini_batch_size)
        if self.network_type == 'classifier':
            validation_accuracy = (total_valid_samples -  \
                                                    validation_errors)*100. / total_valid_samples
            self.validation_accuracy = self.validation_accuracy + [validation_accuracy]
            if verbose >=2 :
                print(".. Validation accuracy : " +str(validation_accuracy))

            if training_accuracy is True:
                training_accuracy = (total_train_samples - \
                                                        training_errors)*100. / total_train_samples
                self.training_accuracy = self.training_accuracy + [training_accuracy]
                if verbose >=2 :
                    print(".. Training accuracy : " +str(training_accuracy))

                if training_errors < self.best_training_errors:
                    self.best_training_errors = training_errors
                    if verbose >= 2:
                        print(".. Best training accuracy")

            if validation_errors < self.best_validation_errors:
                self.best_validation_errors = validation_errors
                best = True
                if verbose >= 2:
                    print(".. Best validation accuracy")

        elif self.network_type == 'generator':
            validation_accuracy = validation_errors / total_valid_samples
            self.validation_accuracy = self.validation_accuracy + [validation_accuracy]

            if verbose >= 2:
                print(".. Mean Validation Error : " + str(validation_accuracy))

            if training_accuracy is True:
                training_accuracy = training_errors / total_train_samples
                self.training_accuracy = self.training_accuracy + [training_accuracy]

                if verbose >=2:
                    print(".. Mean Trianing Error : " + str(training_accuracy))

                if training_errors < self.best_training_errors:
                    self.best_training_errors = training_errors
                    if verbose >= 2:
                        print(".. Best training error")

            if validation_errors < self.best_validation_errors:
                self.best_validation_errors = validation_errors
                best = True
                if verbose >= 2:
                    print(".. Best validation error")

        return best



    def train(self, verbose = 2, **kwargs):
        """
        Training function of the network. Calling this will begin training.

        Args:
            epochs: ``(num_epochs for each learning rate... )`` to train Default is ``(20, 20)``
            validate_after_epochs: 1, after how many epochs do you want to validate ?
            show_progress: default is ``True``, will display a clean progressbar.
                             If ``verbose`` is ``3`` or more - False
            early_terminate: ``True`` will allow early termination.
            learning_rates: (annealing_rate, learning_rates ... ) length must be one more than
                         ``epochs`` Default is ``(0.05, 0.01, 0.001)``

        """
        start_time = time.clock()

        if verbose >= 1:
            print(". Training")

        if self.cooked_datastream is None:
            raise Exception ("Cannot train without cooking the network first")

        if not 'epochs' in kwargs.keys():
            epochs = (20, 20)
        else:
            epochs = kwargs["epochs"]

        if not 'validate_after_epochs' in kwargs.keys():
            self.validate_after_epochs = 1
        else:
            self.validate_after_epochs = kwargs["validate_after_epochs"]

        if not 'visualize_after_epochs' in kwargs.keys():
            self.visualize_after_epochs = self.validate_after_epochs
        else:
            self.visualize_after_epochs = kwargs['visualize_after_epochs']

        if not 'show_progress' in kwargs.keys():
            show_progress = True
        else:
            show_progress = kwargs["show_progress"]

        if progressbar_installed is False:
            show_progress = False

        if verbose == 3:
            show_progress = False

        if not 'training_accuracy' in kwargs.keys():
            training_accuracy = False
        else:
            training_accuracy = kwargs["training_accuracy"]

        if not 'early_terminate' in kwargs.keys():
            patience = 5
        else:
            if kwargs["early_terminate"] is True:
                patience = numpy.inf
            else:
                patience = 5
        # (initial_learning_rate, fine_tuning_learning_rate, annealing)
        if not 'learning_rates' in kwargs.keys():
            learning_rates = (0.05, 0.01, 0.001)
        else:
            learning_rates = kwargs["learning_rates"]

        # Just save some backup parameters
        nan_insurance = []
        for param in self.active_params:
            nan_insurance.append(theano.shared(param.get_value(borrow = self.borrow)))

        self.learning_rate.set_value(learning_rates[1])
        patience_increase = 2
        improvement_threshold = 0.995
        best_iteration = 0
        epoch_counter = 0
        early_termination = False
        iteration= 0
        era = 0
        if isinstance(epochs, int):
            total_epochs = epochs
            change_era = epochs + 1
        elif len(epochs) > 1:
            total_epochs = sum(epochs)
            change_era = epochs[era]
        else:
            total_epochs = epochs
            change_era = epochs + 1
        final_era = False

        # main loop
        while (epoch_counter < total_epochs) and (not early_termination):
            nan_flag = False
            # check if its time for a new era.
            if (epoch_counter == change_era):
            # if final_era, while loop would have terminated.
                era = era + 1
                if era == len(epochs) - 1:
                    final_era = True
                if verbose >= 3:
                    print("... Begin era " + str(era))
                change_era = epoch_counter + epochs[era]
                if self.learning_rate.get_value(borrow = self.borrow) < learning_rates[era+1]:
                    if verbose >= 2:
                        print(".. Learning rate was already lower than specified. Not changing it.")
                    new_lr = self.learning_rate.get_value(borrow = self.borrow)
                else:
                    new_lr = learning_rates[era+1]
                self._new_era(new_learning_rate = new_lr, verbose = verbose)

            # This printing below and the progressbar should move to visualizer ?
            if verbose >= 1:
                print("."),
                if  verbose >= 2:
                    print("\n")
                    print (".. Epoch: " + str(epoch_counter) + " Era: " +str(era))

            if show_progress is True:
                total_mini_batches =  self.batches2train * self.mini_batches_per_batch[0]
                bar = progressbar.ProgressBar(maxval=total_mini_batches, \
                        widgets=[progressbar.AnimatedMarker(), \
                ' training ', ' ', progressbar.Percentage(), ' ',progressbar.ETA(), ]).start()

            # Go through all the large batches
            total_mini_batches_done = 0
            for batch in xrange (self.batches2train):

                if nan_flag is True:
                    # If NAN, restart the epoch, forget the current epoch.
                    break
                # do multiple cached mini-batches in one loaded batch
                if self.cache is True:
                    self._cache_data ( batch = batch , type = 'train', verbose = verbose )
                else:
                    # If dataset is not cached but need to be loaded all at once, check if trianing.
                    if not self.current_data_type == 'train':
                        # If cache is False, then there is only one batch to load.
                        self._cache_data(batch = 0, type = 'train', verbose = verbose )

                # run through all mini-batches in new batch of data that was loaded.
                for minibatch in xrange(self.mini_batches_per_batch[0]):
                    # All important part of the training function. Batch Train.
                    cost = self.mini_batch_train (minibatch, epoch_counter)
                    if numpy.isnan(cost):
                        nan_flag = True
                        new_lr = self.learning_rate.get_value( borrow = self.borrow ) * 0.1
                        self._new_era(new_learning_rate = new_lr, verbose =verbose )
                        if verbose >= 2:
                            print(".. NAN! Slowing learning rate by 10 times and restarting epoch.")
                        break
                    self.cost = self.cost + [cost]
                    total_mini_batches_done = total_mini_batches_done + 1

                    if show_progress is False and verbose >= 3:
                        print(".. Mini batch: " + str(total_mini_batches_done))
                        self.print_status(  epoch = epoch_counter, verbose = verbose )

                    if show_progress is True:
                        bar.update(total_mini_batches_done)

            if show_progress is True:
                bar.finish()

            # post training items for one loop of batches.
            if nan_flag is False:
                best = self.validate(   epoch = epoch_counter,
                                        training_accuracy = training_accuracy,
                                        show_progress = show_progress,
                                        verbose = verbose )
                self.visualize ( epoch = epoch_counter , verbose = verbose )
                self.print_status ( epoch = epoch_counter, verbose=verbose )

                if best is True:
                    copy_params(source = self.active_params, destination= nan_insurance ,
                                                                            borrow = self.borrow)
                    copy_params(source = self.active_params, destination= self.best_params,
                                                                            borrow = self.borrow)
                        # self.resultor.save_network()
                # self.resultor.something() # this function is dummy now. But resultor should use
                # self.visualizer.soemthing() # Again visualizer shoudl do something.
                self.decay_learning_rate(learning_rates[0])

                if patience < epoch_counter:
                    early_termination = True
                    if final_era is False:
                        if verbose >= 3:
                            print("... Patience ran out lowering learning rate.")
                        new_lr = self.learning_rate.get_value( borrow = self.borrow ) * 0.1
                        self._new_era(new_learning_rate = new_lr, verbose =verbose )
                        early_termination = False
                    else:
                        if verbose >= 2:
                            print(".. Early stopping")
                        break
                epoch_counter = epoch_counter + 1

        end_time = time.clock()
        if verbose >=2 :
            print(".. Training complete.Took " +str((end_time - start_time)/60) + " minutes")

    def test(self, show_progress = True, verbose = 2):
        """
        This function is used for producing the testing accuracy.

        Args:
            verbose: As usual
        """
        if verbose >= 2:
            print(".. Testing")
        start_time = time.clock()
        wrong = 0
        predictions = []
        if self.network_type == 'classifier':
            posteriors = []
        labels = []
        total_mini_batches =  self.batches2test * self.mini_batches_per_batch[2]

        if show_progress is True:
            bar = progressbar.ProgressBar(maxval=total_mini_batches, \
                  widgets=[progressbar.AnimatedMarker(), \
                ' testing ', ' ', progressbar.Percentage(), ' ',progressbar.ETA(), ]).start()

        batch_counter = 0
        for batch in xrange(self.batches2test):
            if verbose >= 3:
                print("... training batch " + str(batch))
            self._cache_data ( batch = batch , type = 'test', verbose = verbose )
            for minibatch in xrange (self.mini_batches_per_batch[2]):
                wrong = wrong + self.mini_batch_test(minibatch) # why casted?
                predictions = predictions + self.mini_batch_predictions(minibatch).tolist()
                if self.network_type == 'classifier':
                    posteriors = posteriors + self.mini_batch_posterior(minibatch).tolist()
                if verbose >= 3:
                    print("... testing error after mini batch " + str(batch_counter) + \
                                                              " is " + str(wrong))
                batch_counter = batch_counter + 1
                if show_progress is True:
                    bar.update(batch_counter)

        if show_progress is True:
            bar.finish()

        total_samples = total_mini_batches * self.mini_batch_size
        if self.network_type == 'classifier':
            testing_accuracy = (total_samples - wrong)*100. / total_samples

            if verbose >= 2:
                print(".. Testing accuracy : " + str(testing_accuracy))
        elif self.network_type == 'generator':
            testing_accuracy = wrong / total_samples

            if verbose >= 2:
                print(".. Mean testing error : " + str(testing_accuracy))

if __name__ == '__main__':
    pass
