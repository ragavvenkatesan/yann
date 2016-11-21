"""
TODO:
    * Something is wrong with the d3viz visualizer for html printing. The path is weird.
    * Visualizer module needs to make use of mathplotlib and print online graphs of outputs of cost
      and possibly display first layer filters for CNNs
    * Datastream should include fuel interface and also needs interface for COCO, PASCAL and 
      IMAGENET. Also consider migrating to hd5 for larger datasets ? Should also be able to create
      datasets from images in python. Right now its a roundabout way of going via matlab.
    
"""
import os
import numpy
import cPickle
from collections import OrderedDict

import numpy
import theano
import theano.tensor as T
from theano.ifelse import ifelse

static_printer_import = True
dynamic_printer_import = True
try:
    from theano.printing import pydotprint as static_theano_print
except:
    static_printer_import = False
try:
    from theano.d3viz import d3viz as dynamic_theano_print # refer todo on top.
except:
    dynamic_printer_import = False

from utils.dataset import create_shared_memory_dataset
from utils.dataset import check_type

class module(object):
    """
    Prototype for what a layer should look like. Every layer should inherit from this.
    """
    def __init__(self, id, type, verbose = 2):
        self.id = id
        self.type = type
        # Every layer must have these four properties.
        if verbose >= 3:
            print "... Initializing a new module " + self.id + " of type " + self.type        

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
            self.shuffle_batch_ind = numpy.arange(self.mini_batch_size)
            numpy.random.shuffle(self.shuffle_batch_ind)
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
        filename = self.root + '/computational_graphs/static/' + function.name 
        # this try and except is bad coding, but this seems to be OS dependent and I don't want to 
        # bother with this.

        if static_printer_import is True:
            try:
                static_theano_print(fct = function, outfile = filename + '.' + format, 
                                                                print_output_file = False,
                                                                format = format,
                                                        var_with_name_simple = short_variable_names)
            except:
                if verbose >= 3:
                    print "... Something is wrong with the setup of installers for pydot"
        
        if dynamic_printer_import is True:
            try:
                dynamic_theano_print(fct = function, outfile = filename + '.html') 
                                                # this is not working for something is 
                                                # wrong with path. Refer todo on top of the code.
            except:
                if verbose >= 3:
                    print "... Something is wrong with the setup of installers for dv3viz"

class resultor(module):
    """
    Resultor of the network saves down resultor. The initilizer initializes the directories 
    for storing results.

    Args:
        verbose:  Similar to any 3-level verbose in the toolbox.
        resultor_init_args: ``resultor_params`` is a dictionary of the form 

            .. code-block:: none

                resultor_init_args    =    { 
                    "root"      : "<root directory to save stuff inside>",
                    "results"   : "<results_file_name>.txt",      
                    "errors"    : "<error_file_name>.txt",
                    "costs"     : "<cost_file_name>.txt",
                    "confusion" : "<confusion_file_name>.txt",
                    "network"   : "<network_save_file_name>.pkl"
                    "id"        : id of the resultor
                                }                                                          

            While the filenames are optional, ``root`` must be provided. If a particular file is 
            not provided, that value will not be saved. 

    Returns: 
        yann.modules.resultor: A resultor object
                                                                                                                                            
    """                    
    def __init__( self, resultor_init_args, verbose = 1):  
        if "id" in resultor_init_args.keys(): 
            id = resultor_init_args["id"]
        else:
            id = '-1'
        super(resultor,self).__init__(id = id, type = 'resultor')
            
        if verbose >= 3:
            print "... Creating resultor directories"
            
        for item, value in resultor_init_args.iteritems():            
            if item == "root":
                self.root                   = value                
            elif item == "results":
                self.results_file           = value                  
            elif item == "errors":
                self.error_file             = value
            elif item == "costs":
                self.cost_file              = value
            elif item == "confusion":
                self.confusion_file         = value
            elif item == "network":
                self.network_file           = value

        if not hasattr(self, 'root'): raise Exception('root variable has not been provided. \
                                            Without a root folder, no save can be performed')
        if not os.path.exists(self.root):
            if verbose >= 3:
                print "... Creating a root directory for save files"
            os.makedirs(self.root)
                    
        if verbose >= 3:
            print "... Resultor is initiliazed"
  
class optimizer(module):    
    """

    TODO:

        * AdaDelta

    WARNING:
        Adam is not fully tested.
        
    Optimizer is an important module of the toolbox. Optimizer creates the protocols required 
    for learning. ``yann``'s optimizer supports the following optimization techniques:

        * Stochastic Gradient Descent
        * AdaGrad [1]
        * RmsProp [2]
        * Adam [3]

    Optimizer also supports the following momentum techniques:

        * Polyak [4]
        * Nesterov [5]

    .. [#]   John Duchi, Elad Hazan, and Yoram Singer. 2011. Adaptive subgradient methods for 
             online learning and stochastic optimization. JMLR
    .. [#]   Yann N. Dauphin, Harm de Vries, Junyoung Chung, Yoshua Bengio,"RMSProp and 
             equilibrated adaptive learning rates for non-convex optimization", or 
             arXiv:1502.04390v1
    .. [#]   Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic optimization." 
             arXiv preprint arXiv:1412.6980 (2014).
    .. [#]   Polyak, Boris Teodorovich. "Some methods of speeding up the convergence of 
             iteration methods." USSR Computational Mathematics and Mathematical Physics 4.5 
             (1964): 1-17. Implementation was adapted from Sutskever, Ilya, et al. "On the 
             importance of initialization and momentum in deep learning." Proceedings of the
             30th international conference on machine learning (ICML-13). 2013.
    .. [#]   Nesterov, Yurii. "A method of solving a convex programming problem with 
             convergence rate O (1/k2)."   Soviet Mathematics Doklady. Vol. 27. No. 2. 1983.
             Adapted from `Sebastien Bubeck's`_ blog.
    .. _Sebastien Bubeck's:
                     https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/
    


    Args:
        verbose: Similar to any 3-level verbose in the toolbox.
        optimizer_init_args: ``optimizer_init_args`` is a dictionary like: 

            .. code-block:: none

                optimizer_params =  {    
                    "momentum_type"   : <option>  'false' <no momentum>, 'polyak', 'nesterov'. 
                                        Default value is 'polyak'
                    "momentum_params" : (<option [0,1]>, <option [0,1]>, <int>)
                                        (momentum coeffient at start,at end, 
                                        at what epoch to end momentum increase)
                                        Default is the tuple (0.5, 0.95,50)                                                           
                    "regularization" : (l1_coeff, l2_coeff). Default is (0.001, 0.001)                
                    "optimizer_type" : <option>, 'sgd', 'adagrad', 'rmsprop', 'adam'. 
                                       Default is 'rmsprop'
                    "objective_function": <option>,'nll'- log likelihood,
                                        'cce'-categorical cross entropy, 
                                            'bce'-binary cross entropy.    
                                         Default is 'nll'
                    "id"        : id of the optimizer
                            }  
                                
    Returns:
        yann.modules.optimizer: Optimizer object
    """            
    def __init__(self, optimizer_init_args, verbose = 1):
        """
        Refer to the class description for inputs
        """
        if "id" in optimizer_init_args.keys(): 
            id = optimizer_init_args["id"]
        else:
            id = '-1'
        super(optimizer,self).__init__(id = id, type = 'optimizer')
                            
        if "momentum_params" in optimizer_init_args.keys():                
            self.momentum_start     = optimizer_init_args [ "momentum_params" ][0]
            self.momentum_end       = optimizer_init_args [ "momentum_params" ][1]
            self.momentum_epoch_end = optimizer_init_args [ "momentum_params" ][2]
        else:
            self.momentum_start                  = 0.5
            self.momentum_end                    = 0.99
            self.momentum_epoch_end              = 50
            
        if "momentum_type" in optimizer_init_args.keys():
            self.momentum_type   = optimizer_init_args [ "momentum_type" ]
        else:
            self.momentum_type                   = 'polyak'   

        if "optimizer_type" in optimizer_init_args.keys():                
            self.optimizer_type = optimizer_init_args [ "optimizer_type" ]
        else:
            self.optimizer_type                  = 'rmsprop'

        if verbose >= 3:
            print "... Optimizer is initiliazed"

        if verbose>=3 :
            print "... Applying momentum"
            
        self.epoch = T.scalar('epoch')
        self.momentum = ifelse(self.epoch <= self.momentum_epoch_end,
                        self.momentum_start * (1.0 - self.epoch / self.momentum_epoch_end) +  
                        self.momentum_end * (self.epoch / self.momentum_epoch_end),                        
                        self.momentum_end)  

        if verbose>=3 :
            print "... Creating learning rate"    
        # just setup something for now. Trainer will reinitialize
        self.learning_rate = theano.shared(numpy.asarray(0.1,dtype=theano.config.floatX))

    def calculate_gradients(self, params, objective, verbose = 1):
        """
        This method initializes the gradients.

        Args:
            params: Supply learnable active parameters of a network.
            objective: supply a theano graph connecting the params to a loss
            verbose: Just as always

        Notes:
            Once this is setup, ``optimizer.gradients`` are available
        """
        if verbose >=3 :
            print "... Estimating gradients"

        self.gradients = []      
        for param in params: 
            try:
                gradient = T.grad( objective ,param)
                self.gradients.append ( gradient )
            except:
                raise Exception ("Cannot learn a layer that is disconnected with objective. " +
                        "Try cooking again by making the particular layer learnable as False")


    def create_updates(self, params, verbose = 1):
        """
        This basically creates all the updates and update functions which trainers can iterate 
        upon.

        Args:
            params: Supply learnable active parameters of a network.
            objective: supply a theano graph connecting the params to a loss        
            verbose: Just as always
        """

        # accumulate velocities for momentum
        if verbose >=3: 
            print "... creating internal parameters for all the optimizations"
        velocities = []
        for param in params:
            velocity = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
                                                                dtype=theano.config.floatX))
            velocities.append(velocity)

        # these are used for second order optimizers. 
        accumulator_1 =[]
        accumulator_2 = []
        for param in params:
            eps = numpy.zeros_like(param.get_value(borrow=True), dtype=theano.config.floatX)   
            accumulator_1.append(theano.shared(eps, borrow=True))
            accumulator_2.append(theano.shared(eps, borrow=True)) 

        # these are used for adam.
        timestep = theano.shared(numpy.asarray(0., dtype=theano.config.floatX))
        delta_t = timestep + 1
        b1=0.9                       # for ADAM
        b2=0.999                     # for ADAM
        a = T.sqrt (   1-  b2  **  delta_t )   /   (   1   -   b1  **  delta_t )     # for ADAM

        # to avoid division by zero
        fudge_factor = 1e-7
        if verbose>=3: 
            print "... Building backprop network."

        # This is copied straight from my old toolbox: Samosa. I hope this is working correctly.
        # There might be a better way to have written these... different methods for different 
        # optimizers perhaps ?
        if verbose >=3 :
            print "... Applying " + self.optimizer_type
            print "... Applying " + self.momentum_type
        self.updates = OrderedDict()
        for velocity, gradient, acc_1 , acc_2, param in zip(velocities, self.gradients, 
                                                        accumulator_1, accumulator_2, params):        
            if self.optimizer_type == 'adagrad':
    
                """ Adagrad implemented from paper:
                John Duchi, Elad Hazan, and Yoram Singer. 2011. Adaptive subgradient methods
                for online learning and stochastic optimization. JMLR
                """
                current_acc_1 = acc_1 + T.sqr(gradient) # Accumulates Gradient 
                self.updates[acc_1] = current_acc_1          # updates accumulation at timestamp

            elif self.optimizer_type == 'rmsprop':
                """ Tieleman, T. and Hinton, G. (2012):
                Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
                Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)"""
                rms_rho = 0.9    
                current_acc_1 = rms_rho * acc_1 + (1 - rms_rho) * T.sqr(gradient) 
                self.updates[acc_1] = current_acc_1

            elif self.optimizer_type == 'sgd':
                current_acc_1 = 1.

            elif self.optimizer_type == 'adam':
                """ Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic optimization." 
                     arXiv preprint arXiv:1412.6980 (2014)."""  
                if not self.momentum_type == '_adam': 
                    if verbose >= 3 and not self.momentum_type == 'false':
                        print "... ADAM doesn't need explicit momentum. Momentum is removed." 
                    self.momentum_type = '_adam'
                                                        

                current_acc_2 = b1 * acc_2 + (1-b1) * gradient
                current_acc_1 = b2 * acc_1 + (1-b2) * T.sqr( gradient )                
                self.updates[acc_2] = current_acc_2
                self.updates[acc_1] = current_acc_1
                
            if self.momentum_type == '_adam':
                self.updates[velocity] = a * current_acc_2 / (T.sqrt(current_acc_1) +
                                                                                 fudge_factor)
                
            elif self.momentum_type == 'false':               # no momentum
                self.updates[velocity] = - (self.learning_rate / T.sqrt(current_acc_1 + 
                                                                     fudge_factor)) * gradient                                            
            elif self.momentum_type == 'polyak':       # if polyak momentum    
                """ Momentum implemented from paper:  
                Polyak, Boris Teodorovich. "Some methods of speeding up the convergence of 
                iteration methods."  USSR Computational Mathematics and Mathematical 
                Physics 4.5 (1964): 1-17.
    
                Adapted from Sutskever, Ilya, Hinton et al. "On the importance of initialization
                and momentum in deep learning.", Proceedings of the 30th international 
                conference on machine learning (ICML-13). 2013. equation (1) and equation (2)"""   
    
                self.updates[velocity] = self.momentum * velocity - (1.- self.momentum) * \
                                 ( self.learning_rate / T.sqrt(current_acc_1 + fudge_factor)) \
                                                                                    * gradient                             
    
            elif self.momentum_type == 'nesterov':             # Nestrov accelerated gradient 
                """Nesterov, Yurii. "A method of solving a convex programming problem with 
                convergence rate O (1/k2)." Soviet Mathematics Doklady. Vol. 27. No. 2. 1983.
                Adapted from 
                https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/ 
    
                Instead of using past params we use the current params as described in this link
                https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,"""
      
                self.updates[velocity] = self.momentum * velocity - (1.-self.momentum) * \
                                ( self.learning_rate / T.sqrt(current_acc_1 + fudge_factor)) \
                                                                                    * gradient                                 
                self.updates[param] = self.momentum * self.updates[velocity] 
    
            else:
                if verbose >= 3:
                    print "... Unrecognized mometum type, switching to no momentum."
                self.momentum_type = 'false'
                self.updates[velocity] = - (self.learning_rate / T.sqrt(current_acc_1 + 
                                                                    fudge_factor))  * gradient                 
            stepped_param = param + self.updates[velocity]
            if self.momentum_type == 'nesterov':
                stepped_param = stepped_param + self.updates[param]

            column_norm = True #This I don't fully understand if 
                                #its needed after BN is implemented.
                                # This is been around since my first ever 
                                # implementation of samosa, and I haven't tested it out.
            if param.get_value(borrow=True).ndim == 2 and column_norm is True:
                """ constrain the norms of the COLUMNs of the weight, according to
                https://github.com/BVLC/caffe/issues/109 """
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(15))
                scale = desired_norms / (fudge_factor + col_norms)
                self.updates[param] = stepped_param * scale
            else:            
                self.updates[param] = stepped_param
                
        if self.optimizer_type == 'adam':
           self.updates[timestep] = delta_t       
                                                                               

class datastream(module):
    """
    This module initializes the dataset to the network class and provides all dataset related 
    functionalities. It also provides for dynamically loading and caching dataset batches.
    :mod: ``add_layer`` will use this to initialize. 

    Args:
        dataset_init_args: Is a dictionary of the form:
        borrow: Theano's borrow. Default value is ``True``.

            .. code-block:: python

                dataset_init_args = {
                            "dataset":  <location>
                            "svm"    :  False or True 
                                 ``svm`` if ``True``, a one-hot label set will also be setup.
                            "n_classes": <int>
                                ``n_classes`` if ``svm`` is ``True``, we need to know how 
                                 many ``n_classes`` are present.
                            "id": id of the datastream
                    }

        verbose: Similar to verbose throughout the toolbox.

    Returns: 
        dataset: A dataset module object that has the details of loader and other things.

    TODO:
        * Datastream should work with Fuel perhaps ?
        * Support HDf5 perhaps

    """    
            
    # this loads up the data_params from a folder and sets up the initial databatch.         
    def __init__ ( self, 
                   dataset_init_args,   
                   borrow = True,                                   
                   verbose = 1):

        if "id" in dataset_init_args.keys(): 
            id = dataset_init_args["id"]
        else:
            id = '-1'
        super(datastream,self).__init__(id = id, type = 'datastream')
        
        dataset = dataset_init_args ["dataset"]
        self.dataset = dataset
        self.borrow = borrow

        if verbose >= 3:
            print "... Initializing datastream with " + dataset

        f = open(dataset + '/data_params.pkl', 'rb')
        data_params = cPickle.load(f)
        f.close()
        
        self.dataset_location    = data_params [ "location"]
        self.mini_batch_size     = data_params [ "mini_batch_size" ]    
        self.mini_batches_per_batch  = data_params [ "cache_batches" ]
        self.batches2train       = data_params [ "batches2train" ]
        self.batches2test        = data_params [ "batches2test" ]
        self.batches2validate    = data_params [ "batches2validate" ]         
        self.height              = data_params [ "height" ]
        self.width               = data_params [ "width" ]
        self.channels            = data_params [ "channels" ]        
        self.cache               = data_params [ "cache" ]

        self.current_type = 'train'       
        if 'svm' in dataset_init_args.keys():
            self.svm = dataset_init_args["svm"]
        else:
            self.svm = False

        if self.svm is True:
            if "n_classes" in dataset_init_args.keys():
                self.n_classes = dataset_init_args ["n_classes"]            
            else:
                self.n_classes = False

        self.initialize_dataset(verbose = verbose)
        self.batch = 0# initialize the batch to zero. Changing this will produce a new stream. 
         
        self.cached_zeros_x = numpy.zeros((1,),dtype = theano.config.floatX)
        self.cached_zeros_y = numpy.zeros((1,),dtype = theano.config.floatX)

        if verbose >= 3:
            print "... Datastream is initiliazed"
        
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.one_hot_y = T.matrix('one_hot_y')

    def load_data (self, type = 'train', batch = 0, verbose = 2):
        """
        Will load the data from the file and will return the data. The important thing to note 
        is that all the datasets in :mod: ``yann`` all require a ``y`` or a variable to 
        predict. In case of auto-encoder for instance, the thing to predict is the image 
        itself. Setup dataset thusly.

        Args: 
            type: ``train``, ``test`` or ``valid``.
                  default is ``train``
            batch: Supply an integer
                   
            verbose: Simliar to verbose in toolbox.

        Returns:
            numpy.ndarray: ``data_x, data_y`` 
        """
        if verbose >= 3: 
            print "... loading " + type + " data batch " + str(batch) 

        f = open(self.dataset + '/' + type + '/batch_' + str(batch) +'.pkl', 'rb')
        data_x, data_y = cPickle.load(f)
        f.close()   

        if verbose >= 3:
            print "... data is loaded"  
        
        data_x = check_type (data_x, theano.config.floatX)
        data_y = check_type (data_y, theano.config.floatX)
        # Theano recommends storing on gpus only as floatX and casts them to ints during use.
        # I don't know why, but I am following their recommendations blindly.
        return data_x, data_y         

    def set_data (self, type = 'train', batch = 0, verbose = 2):
        """
        This can work only after network is cooked.

        Args:
            batch: which batch of data to load and set
            verbose: as usual
        """
        if verbose >=3 :
            print "... Setting batch " + str(batch) + " of data of type " + type 

        data_x, data_y = self.load_data (batch = batch, type = type, verbose = verbose )
        # Doing this just so that I can use set_value instead of set_sub_tensor.
        # Also, I see some elegance in zeroing out stuff.
        
        if data_x.shape[0] < self.data_cache_size:
            # This will probably used by non-cached datasets heavily. 
            data_size_needed = (self.data_cache_size - data_x.shape[0], self.height * 
                                                            self.width * self.channels)  
            if not self.cached_zeros_x.shape[0] == data_size_needed[0]:
                self.cached_zeros_x = numpy.zeros(data_size_needed,
                                                     dtype = data_x.dtype)
                if verbose >= 3:
                    print "... Cache miss in loading data "                  
            if not self.cached_zeros_y.shape[0] == data_size_needed[0]:
                self.cached_zeros_y =  numpy.zeros((data_size_needed[0],),
                                                        dtype = data_y.dtype)
                                                        
            data_x = numpy.concatenate((data_x, self.cached_zeros_x), axis=0)
            data_y = numpy.concatenate((data_y, self.cached_zeros_y), axis = 0)
    
        elif data_x.shape[0] > self.data_cache_size:
            # don't know if this case will ever be used.
            data_x = data_x[:self.data_cache_size,]
            data_y = data_y[:self.data_cache_size,]

        self.data_x.set_value (data_x, borrow = self.borrow )
        self.data_y_uncasted.set_value (data_y, borrow = self.borrow )                        
        if self.svm is True:
            data_one_hot_y = self.one_hot_labels( data_y, verbose = verbose )
            self.data_one_hot_y.set_value ( data_one_hot_y , borrow = self.borrow )

        self.current_type = type
        
    def one_hot_labels (self, y, verbose = 1):
        """
        Function takes in labels and returns a one-hot encoding. Used for max-margin loss.
        Args:
            y: Labels to be encoded.n_classes
            verbose: Typical as in the rest of the toolbox.

        Notes:
            ``self.n_classes``: Number of unique classes in the labels.

                       This could be found out using the following:
                       .. code-block: python
                           
                           import numpy
                           n_classes = len(numpy.unique(y))
                        
                      This might be potentially dangerous in case of cached dataset. Although 
                      this is the default if ``n_classes`` is not provided as input to this 
                      module, I discourage anyone from using this. 
        Returns:
            numpy ndarray: one-hot encoded label list.
        """

        if self.n_classes is False:
            if verbose >= 3:
                print "... Making a decision to create n_classes variable, not a good idea."
            self.n_classes = len(numpy.unique(y)) 

        # found this technique online somewhere, forgot where couldn't cite.
        y1 = -1 * numpy.ones((y.shape[0], self.n_classes))
        y1[numpy.arange(y.shape[0]), y] = 1	   
        y1 = check_type(y1, theano.config.floatX)         
        return y1
        

    def initialize_dataset( self, verbose = 1 ):
        """
        Load the initial training batch of data on to ``data_x`` and ``data_y`` variables
        and create shared memories. 
        
        TODO:
            I am assuming that training has the largest number of data. This is immaterial when
            caching but during set_data routine, I need to be careful. 
        Args:
            verbose: Toolbox style verbose.
        """
        if verbose >= 3:
            print ".. Initializing the dataset by loading 0th batch"

        # every dataset will have atleast one batch ..load that.  
        # Assumimg that train has more number of data than any other. 

        data_x, data_y = self.load_data(type = 'train', batch = 0, verbose = verbose)                                                   
        self.data_cache_size = data_x.shape[0]

        if self.svm is False:
            self.data_x, self.data_y_uncasted = create_shared_memory_dataset(
                                                           (data_x, data_y),
                                                            borrow = self.borrow,
                                                            verbose = verbose)
        else:
            data_y1 = self.one_hot_labels (data_y, verbose = verbose)
            self.data_x, self.data_y_uncasted, self.data_one_hot_y = create_shared_memory_dataset(
                                                  (data_x, data_y, data_y1),
                                                            borrow = self.borrow,
                                                            svm = True,
                                                            verbose = verbose)
        self.data_y = T.cast(self.data_y_uncasted, 'int32')

        if verbose >=3:
            print "... dataset is initialized"


if __name__ == '__main__':
    pass              