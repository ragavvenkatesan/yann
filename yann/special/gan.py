"""
Support for the implementation from 

Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, 
Aaron Courville, and Yoshua Bengio. "Generative adversarial nets." In Advances in Neural Information
Processing Systems, pp. 2672-2680. 2014.
"""
import time 
import numpy 
import theano.tensor as T
import theano
from collections import OrderedDict
import imp
try:
    imp.find_module('progressbar')
    progressbar_installed = True
except ImportError:
    progressbar_installed = False
if progressbar_installed is True:
    import progressbar
from yann.network import network
from yann.core.operators import copy_params

class gan (network):
    """
    This class is inherited from the network class and has its own methods modified in support of gan
    networks. 

    Args:
        Same as the network class
    """
    def __init__ (self, verbose = 2 ,**kwargs):
        super(gan,self).__init__(verbose = verbose, kwargs = kwargs)
    
    def initialize_train(self, verbose = 2):
        """
        Internal function that creates a train methods for the GAN network

        Args:
            verbose: as always
        """
        if verbose >=3:
            print "... creating the classifier training theano function "

        #D_c(x)
        index = T.lscalar('index')     
        if self.cooked_datastream.svm is False:   
            self.mini_batch_train_softmax = theano.function(
                    inputs = [index, self.cooked_softmax_optimizer.epoch],
                    outputs = self.dropout_softmax_cost,
                    name = 'train',
                    givens={
            self.x: self.data_x[index * self.mini_batch_size:(index + 1) * self.mini_batch_size],
            self.y: self.data_y[index * self.mini_batch_size:(index + 1) * self.mini_batch_size]},
                    updates = self.cooked_softmax_optimizer.updates, 
                    on_unused_input = 'ignore')
        else:                                                                        
            self.mini_batch_train_softmax = theano.function(
                    inputs = [index, self.cooked_softmax_optimizer.epoch],
                    outputs = self.dropout_softmax_sot,
                    name = 'train',                    
                    givens={
            self.x: self.data_x[ index * self.mini_batch_size:(index + 1) * self.mini_batch_size],
            self.one_hot_y: self.data_one_hot_y[index * self.mini_batch_size:(index + 1) * 
                                                                    self.mini_batch_size]},
                    updates = self.cooked_softmax_optimizer.updates, 
                    on_unused_input = 'ignore')
        
        #D(x)
        self.mini_batch_train_real = theano.function(
                inputs = [index, self.cooked_real_optimizer.epoch],
                outputs = self.dropout_real_cost,
                name = 'train',                    
                givens={
        self.x: self.data_x[ index * self.mini_batch_size:(index + 1) * self.mini_batch_size] },
                updates = self.cooked_real_optimizer.updates, 
                on_unused_input = 'ignore')         
        
        #D(G(z))
        self.mini_batch_train_fake = theano.function(
                inputs = [index, self.cooked_fake_optimizer.epoch],
                outputs = self.dropout_fake_cost,
                name = 'train',                    
                givens={
        self.x: self.data_x[ index * self.mini_batch_size:(index + 1) * self.mini_batch_size]},
                updates = self.cooked_fake_optimizer.updates, 
                on_unused_input = 'ignore')                                      

        #Update for G(z) weights
        self.mini_batch_train_gen = theano.function(
                inputs = [index, self.cooked_gen_optimizer.epoch],
                outputs = self.dropout_gen_cost,
                name = 'train',                    
                givens={
        self.x: self.data_x[ index * self.mini_batch_size:(index + 1) * self.mini_batch_size] },
                updates = self.cooked_gen_optimizer.updates, 
                on_unused_input = 'ignore') 

    def cook_softmax_optimizer ( self, optimizer_params, verbose = 2):
        """
        This method cooks the softmax optimizer.

        Args: 
            verbose: as always 
        """    
        optimizer_params["id"] = 'softmax_optimizer'
        if verbose >=3 :
            print "... Building the softmax classifier backprop network."
        self.add_module( type = 'optimizer', params = optimizer_params, verbose =verbose )
        self.cooked_softmax_optimizer = self.optimizer["softmax_optimizer"]

        self._cook_optimizer(params = self.classifier_active_params,
                             objective = self.dropout_softmax_cost,
                             optimizer = self.cooked_softmax_optimizer,
                             verbose = verbose )                                 
        self.softmax_learning_rate = self.learning_rate
        self.softmax_decay_learning_rate = self.decay_learning_rate       
        self.softmax_current_momentum = self.current_momentum

    def cook_real_optimizer ( self, optimizer_params, verbose = 2):
        """
        This method cooks the real optimizer.

        Args: 
            verbose: as always 
        """    
        optimizer_params["id"] = 'real_optimizer'
        if verbose >=3 :
            print "... Building the real classifier backprop network."
        self.add_module( type = 'optimizer', params = optimizer_params, verbose =verbose )
        self.cooked_real_optimizer = self.optimizer["real_optimizer"]
        
        self._cook_optimizer(params = self.discriminator_active_params,
                             objective = self.dropout_real_cost,
                             optimizer = self.cooked_real_optimizer,
                             verbose = verbose )                                 
        self.real_learning_rate = self.learning_rate
        self.real_decay_learning_rate = self.decay_learning_rate       
        self.real_current_momentum = self.current_momentum

    def cook_fake_optimizer ( self, optimizer_params, verbose = 2):
        """
        This method cooks the fake optimizer.

        Args: 
            verbose: as always 
        """    
        optimizer_params["id"] = 'fake_optimizer'
        if verbose >=3 :
            print "... Building the fake classifier backprop network."
        self.add_module( type = 'optimizer', params = optimizer_params, verbose =verbose )
        self.cooked_fake_optimizer = self.optimizer["fake_optimizer"]
            
        self._cook_optimizer(params = self.discriminator_active_params,
                             objective = self.dropout_fake_cost,
                             optimizer = self.cooked_fake_optimizer,
                             verbose = verbose )                                 
        self.fake_learning_rate = self.learning_rate
        self.fake_decay_learning_rate = self.decay_learning_rate       
        self.fake_current_momentum = self.current_momentum 

    def cook_gen_optimizer ( self, optimizer_params, verbose = 2):
        """
        This method cooks the generator optimizer.
        
        Args: 
            verbose: as always 
        """    
        optimizer_params["id"] = 'gen_optimizer'
        if verbose >=3 :
            print "... Building the generator classifier backprop network."
        self.add_module( type = 'optimizer', params = optimizer_params, verbose =verbose )
        self.cooked_gen_optimizer = self.optimizer["gen_optimizer"]
            
        self._cook_optimizer(params = self.generator_active_params,
                             objective = self.dropout_gen_cost, # ascend on the same obj
                             optimizer = self.cooked_gen_optimizer,
                             verbose = verbose )                                 
        self.gen_learning_rate = self.learning_rate
        self.gen_decay_learning_rate = self.decay_learning_rate       
        self.gen_current_momentum = self.current_momentum 

    def cook(   self,   
                objective_layers, 
                discriminator_layers,
                classifier_layers,
                generator_layers,
                softmax_layer,
                optimizer_params = None, 
                verbose = 2, 
                **kwargs ):
        """
        This function builds the backprop network, and makes the trainer, tester and validator
        theano functions. The trainer builds the trainers for a particular objective layer and 
        optimizer.  

        Args:
            optimizer_params: Supply optimizer_params.
            datastream: Supply which datastream to use.
                            Default is the last datastream created.
            visualizer: Supply a visualizer to cook with.

            objective_layers: Supply a tuple of layer ids of layers that have the objective 
                                functions (classification, real, fake)

            classifier: supply the classifier layer of the discriminator.
            discriminator: supply the fake/real layer of the data stream.  
            generator: supply the last generator layer.       
            generator_layers: list or tuple of all generator layers
            discriminator_layers: list or tuple of all discriminator layers   
            classifier_layers: list or tuple of all classifier layers                 
            verbose: Similar to the rest of the toolbox.


        """
        if verbose >= 2:
            print ".. Cooking the network"
        if verbose >= 3:
            print "... Building the network's objectives, gradients and backprop network"            

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
            
        if not 'datastream' in kwargs.keys():
            datastream = None
        else:
            datastream = kwargs['datastream'] 

        if not 'params' in kwargs.keys():
            params = None
        else:
            params = params

        self.network_type = 'gan'

        if datastream is None:
            if self.last_datastream_created is None:
                raise Exception("Cannot build trainer without having an datastream initialized")
            
            if verbose >= 3:
                print "... datastream not provided, assuming " + self.last_datastream_created
            datastream = self.last_datastream_created    
        else:
            if not datastream in self.datastream.keys():
                raise Exception ("Datastream " + datastream + " not found.")
        self.cooked_datastream = self.datastream[datastream]

        self.generator_active_params = []
        self.discriminator_active_params = []
        self.classifier_active_params = []

        for lyr in generator_layers:
            self.generator_active_params = self.generator_active_params + \
                                           self.dropout_layers[lyr].params
                                           
        for lyr in discriminator_layers:
            self.discriminator_active_params = self.discriminator_active_params + \
                                               self.dropout_layers[lyr].params
        for lyr in classifier_layers:
            self.classifier_active_params = self.classifier_active_params + \
                                            self.dropout_layers[lyr].params

        if optimizer_params is None:
            optimizer_params =  {        
                            "momentum_type"       : 'polyak',             
                            "momentum_params"     : (0.9, 0.95, 30),      
                            "regularization"      : (0.0001, 0.0001),       
                            "optimizer_type"      : 'rmsprop',                
                                }
            
        if datastream is None:
            if self.last_datastream_created is None:
                raise Exception("Cannot build trainer without having an datastream initialized")
            
            if verbose >= 3:
                print "... datastream not provided, assuming " + self.last_datastream_created
            datastream = self.last_datastream_created    
        else:
            if not datastream in self.datastream.keys():
                raise Exception ("Datastream " + datastream + " not found.")

        self.cooked_datastream = self.datastream[datastream]         
        self._cook_datastream(verbose = verbose)

        self.data_y = self.cooked_datastream.data_y
        if self.cooked_datastream.svm is True:
            self.data_one_hot_y = self.cooked_datastream.data_one_hot_y
        self.y = self.cooked_datastream.y
        self.one_hot_y = self.cooked_datastream.one_hot_y             
        self.current_data_type = self.cooked_datastream.current_type

        self.softmax_cost = self.layers[objective_layers[0]].output
        self.dropout_softmax_cost = self.dropout_layers[objective_layers[0]].output
        self.real_cost = self.layers[objective_layers[1]].output
        self.dropout_real_cost = self.dropout_layers[objective_layers[1]].output
        self.fake_cost = self.layers[objective_layers[2]].output
        self.dropout_fake_cost = self.dropout_layers[objective_layers[2]].output
        self.gen_cost = -self.fake_cost 
        self.dropout_gen_cost = -self.dropout_fake_cost 

        self._cook_datastream(verbose = verbose)
        self.cook_softmax_optimizer(optimizer_params = optimizer_params,
                                    verbose = verbose)
        self.cook_real_optimizer(   optimizer_params = optimizer_params,
                                    verbose = verbose)                                       
        self.cook_fake_optimizer(   optimizer_params = optimizer_params,
                                    verbose = verbose)                                                                     
        self.cook_gen_optimizer(   optimizer_params = optimizer_params,
                                    verbose = verbose)                                    
        self._initialize_test (classifier = softmax_layer,
                               verbose = verbose)
        self._initialize_predict ( classifier = softmax_layer,
                                 verbose = verbose)
        self._initialize_posterior (classifier = softmax_layer,
                                   verbose = verbose)

        self.initialize_train ( verbose = verbose )   
        self.validation_accuracy = []
        self.best_validation_errors = numpy.inf
        self.best_training_errors = numpy.inf
        self.training_accuracy = []
        self.best_params = []
        
        # Let's bother only about learnable params. This avoids the problem when weights are 
        # shared
        self.validation_accuracy = []
        self.best_validation_errors = numpy.inf
        self.best_training_errors = numpy.inf
        self.training_accuracy = []
        self.best_params = []
        # Let's bother only about learnable params. This avoids the problem when weights are 
        # shared
        self.active_params = self.classifier_active_params + self.discriminator_active_params + \
                                    self.generator_active_params 
        if params is None:
            params = self.active_params
        for param in params:
            self.best_params.append(theano.shared(param.get_value(borrow = self.borrow)))

        self.gen_cost = []  
        self.real_cost = []
        self.fake_cost = []
        self.softmax_cost = []
        self.cooked_visualizer = self.visualizer[visualizer]
        self._cook_visualizer(verbose = verbose) # always cook visualizer last.
        self.visualize (epoch = 0, verbose = verbose)
        # Cook Resultor.

    
    def _new_era ( self, new_learning_rate = 0.01, verbose = 2):
        """
        This re-initializes the learning rate to the learning rate variable. This also reinitializes
        the parameters of the network to best_params.

        Args:
            new_learning_rate: rate at which you want fine tuning to begin.
            verbose: Just as the rest of the toolbox. 
        """    
        if verbose >= 3:
            print "... setting up new era"
        self.softmax_learning_rate.set_value(numpy.asarray(new_learning_rate,
                                                        dtype = theano.config.floatX))
        self.real_learning_rate.set_value(numpy.asarray(new_learning_rate,
                                                        dtype = theano.config.floatX))
        self.fake_learning_rate.set_value(numpy.asarray(new_learning_rate,
                                                        dtype = theano.config.floatX))  
        self.gen_learning_rate.set_value(numpy.asarray(new_learning_rate,
                                                        dtype = theano.config.floatX))                                                                                                                                                                        
        # copying and removing only active_params. Is that a porblem ?
        copy_params ( source = self.best_params, destination = self.active_params , 
                                                                            borrow = self.borrow)
    def print_status (self, epoch , verbose = 2):
        """
        This function prints the costs of the current epoch, learning rate and momentum of the 
        network at the moment. 
        
        Todo:
            This needs to to go to visualizer.

        Args:
            verbose: Just as always. 
            epoch: Which epoch are we at ?
        """

        if self.cooked_datastream is None:
            raise Exception(" Cook first then run this.")

        if verbose >=2 :
            if len(self.gen_cost) < self.batches2train * self.mini_batches_per_batch[0]:
                print ".. Generator Cost                : " + str(self.gen_cost[-1])
            else:
                print ".. Generator Cost                : " + str(numpy.mean(self.gen_cost[-1 * 
                                    self.batches2train * self.mini_batches_per_batch[0]:]))  

        if len(self.real_cost) < self.batches2train * self.mini_batches_per_batch[0]:
            print ".. Discriminator Real Images Cost    : " + str(self.real_cost[-1])
        else:
            print ".. Discriminator Real Images Cost    : " + str(numpy.mean(self.real_cost[-1 * 
                                self.batches2train * self.mini_batches_per_batch[0]:]))     


        if len(self.fake_cost) < self.batches2train * self.mini_batches_per_batch[0]:
            print ".. Discriminator Fake Images Cost    : " + str(self.fake_cost[-1])
        else:
            print ".. Discriminator Fake Images Cost    : " + str(numpy.mean(self.fake_cost[-1 * 
                                self.batches2train * self.mini_batches_per_batch[0]:]))   
        
        if len(self.softmax_cost) < self.batches2train * self.mini_batches_per_batch[0]:
            print ".. Discriminator Softmax Cost        : " + str(self.softmax_cost[-1])
        else:
            print ".. Discriminator Softmax Cost        : " + str(numpy.mean(self.softmax_cost[-1 * 
                                self.batches2train * self.mini_batches_per_batch[0]:])) 

        if verbose >= 3:
            print "... Learning Rate       : " + str(self.learning_rate.get_value(borrow=\
                                                                                 self.borrow))
            print "... Momentum            : " + str(self.current_momentum(epoch))  

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
            print "... creating the activities of all layers "

        if self.cooked_datastream is None:
           raise Exception ("This needs to be run only after network is cooked")

        index = T.lscalar('index')             
        self.layer_activities_created = True
        for id, _layer in self.layers.iteritems():
            if verbose >=3 :
                print "... collecting the activities of layer " + id
            activity = _layer.output  
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

    def validate (self, epoch = 0, training_accuracy = False, show_progress = False, verbose = 2):
        """
        Method is use to run validation. It will also load the validation dataset.

        Args:
            verbose: Just as always
            show_progress: Display progressbar ?
            training_accuracy: Do you want to print accuracy on the training set as well ?
        """        
        self.network_type = 'classifier'        
        best = super(gan,self).validate(epoch = epoch,
                                 training_accuracy = training_accuracy,
                                 show_progress = show_progress,
                                 verbose = verbose)
        return best
    def train ( self, verbose, **kwargs):        
        """
        Training function of the network. Calling this will begin training.

        Args:
            epochs: ``(num_epochs for each learning rate... )`` to train Default is ``(20, 20)``
            validate_after_epochs: 1, after how many epochs do you want to validate ?
            show_progress: default is ``True``, will display a clean progressbar.
                                If ``verbose`` is ``3`` or more - False 
            early_terminate: ``True`` will allow early termination.
            k : how many discriminator updates for every generator update.
            learning_rates: (annealing_rate, learning_rates ... ) length must be one more than 
                            ``epochs`` Default is ``(0.05, 0.01, 0.001)``
            
        """
        start_time = time.clock()  

        if verbose >= 1:
            print ". Training"

        if self.cooked_datastream is None:
            raise Exception ("Cannot train without cooking the network first")

        if not 'epochs' in kwargs.keys():
            epochs = (20, 20)
        else:
            epochs = kwargs["epochs"]

        if not 'k' in kwargs.keys():
            k = 1
        else:
            k = kwargs["k"]

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

        self.softmax_learning_rate.set_value(learning_rates[1])        
        self.real_learning_rate.set_value(learning_rates[1]) 
        self.fake_learning_rate.set_value(learning_rates[1])     
        self.gen_learning_rate.set_value(learning_rates[1])   
                
        patience_increase = 2  
        improvement_threshold = 0.995       
        best_iteration = 0
        epoch_counter = 0
        early_termination = False
        iteration= 0        
        era = 0
        total_epochs = sum(epochs) 
        change_era = epochs[era] 
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
                    print "... Begin era " + str(era)
                change_era = epoch_counter + epochs[era]                     
                if self.learning_rate.get_value(borrow = self.borrow) < learning_rates[era+1]:
                    if verbose >= 2:
                        print ".. Learning rate was already lower than specified. Not changing it."
                    new_lr = self.gen_learning_rate.get_value(borrow = self.borrow)
                else:
                    new_lr = learning_rates[era+1]
                self._new_era(new_learning_rate = new_lr, verbose = verbose)
            
            # This printing below and the progressbar should move to visualizer ?
            if verbose >= 1:
                print ".",
                if  verbose >= 2:
                    print "\n"
                    print ".. Epoch: " + str(epoch_counter) + " Era: " +str(era)
                    
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
                    # This is the classifier for discriminator. I will run this later.

                    fake_cost = self.mini_batch_train_fake (minibatch, epoch_counter)                    
                    real_cost = self.mini_batch_train_real (minibatch, epoch_counter)                    
                    softmax_cost = self.mini_batch_train_softmax (minibatch, epoch_counter)
                                            
                    if minibatch % k == 0:                  
                        gen_cost = self.mini_batch_train_gen (minibatch, epoch_counter)
                    

                    if numpy.isnan(gen_cost) or \
                        numpy.isnan(softmax_cost) or \
                        numpy.isnan(fake_cost) or \
                        numpy.isnan(real_cost):                  
                        nan_flag = True
                        new_lr = self.gen_learning_rate.get_value( borrow = self.borrow ) * 0.1 
                        self._new_era(new_learning_rate = new_lr, verbose =verbose )
                        if verbose >= 2:
                            print ".. NAN! Slowing learning rate by 10 times and restarting epoch."                                      
                        break                 
                    self.fake_cost = self.fake_cost + [fake_cost]
                    self.real_cost = self.real_cost + [real_cost]
                    self.softmax_cost = self.softmax_cost + [softmax_cost]                    
                    self.gen_cost = self.gen_cost + [gen_cost]
                    
                    total_mini_batches_done = total_mini_batches_done + 1                     
                    
                    if show_progress is False and verbose >= 3:
                        print ".. Mini batch: " + str(total_mini_batches_done)
                        self.print_status(  epoch = epoch_counter, verbose = verbose ) 

                    if show_progress is True:
                        bar.update(total_mini_batches_done)                         

            if show_progress is True:
                bar.finish()               
            
            
            # post training items for one loop of batches.    
            if nan_flag is False:    
                if verbose >= 2:
                    self.print_status ( epoch = epoch_counter, verbose = verbose )    
                
                best = self.validate(   epoch = epoch_counter,
                                        training_accuracy = training_accuracy,
                                        show_progress = show_progress,
                                        verbose = verbose )
                self.visualize ( epoch = epoch_counter , verbose = verbose)
                
                if best is True:
                    copy_params(source = self.params, destination= nan_insurance , 
                                                                            borrow = self.borrow)
                    copy_params(source = self.params, destination= self.best_params, 
                                                                            borrow = self.borrow)                        

                self.decay_learning_rate(learning_rates[0])  

                if patience < epoch_counter:
                    early_termination = True
                    if final_era is False:
                        if verbose >= 3:
                            print "... Patience ran out lowering learning rate."
                        new_lr = self.learning_rate.get_value( borrow = self.borrow ) * 0.1 
                        self._new_era(new_learning_rate = new_lr, verbose =verbose )              
                        early_termination = False
                    else:
                        if verbose >= 2:
                            print ".. Early stopping"
                        break   
                epoch_counter = epoch_counter + 1
         
        end_time = time.clock()
        if verbose >=2 :
            print ".. Training complete.Took " +str((end_time - start_time)/60) + " minutes"                