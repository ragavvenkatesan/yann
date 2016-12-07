static_printer_import = True
dynamic_printer_import = True

import os
from abstract import module

try:
    from theano.printing import pydotprint as static_theano_print
except:
    static_printer_import = False
try:
    from theano.d3viz import d3viz as dynamic_theano_print # refer todo on top.
except:
    dynamic_printer_import = False

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

if __name__ == '__main__':
    pass  