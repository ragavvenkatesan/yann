import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from collections import OrderedDict
from abstract import module

class optimizer(module):
    """
    Optimizer is an important module of the toolbox. Optimizer creates the protocols required
    for learning. ``yann``'s optimizer supports the following optimization techniques:

        * Stochastic Gradient Descent
        * AdaGrad [1]
        * RmsProp [2]
        * Adam [3]
        * Adadelta [4]

    Optimizer also supports the following momentum techniques:

        * Polyak [5]
        * Nesterov [6]

    .. [#]   John Duchi, Elad Hazan, and Yoram Singer. 2011. Adaptive subgradient methods for
             online learning and stochastic optimization. JMLR
    .. [#]   Yann N. Dauphin, Harm de Vries, Junyoung Chung, Yoshua Bengio,"RMSProp and
             equilibrated adaptive learning rates for non-convex optimization", or
             arXiv:1502.04390v1
    .. [#]   Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic optimization."
             arXiv preprint arXiv:1412.6980 (2014).
    .. [#]   Zeiler, Matthew D. "ADADELTA: an adaptive learning rate method." arXiv preprint 
             arXiv:1212.5701 (2012).
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
                                        Default value is 'false'
                    "momentum_params" : (<option in range [0,1]>, <option in range [0,1]>, <int>)
                                        (momentum coeffient at start,at end,
                                        at what epoch to end momentum increase)
                                        Default is the tuple (0.5, 0.95,50)
                    "optimizer_type" : <option>, 'sgd', 'adagrad', 'rmsprop', 'adam'.
                                       Default is 'sgd'
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
            self.momentum_type                   = 'false'
        
        if self.momentum_type == 'false':
            self.momentum = None

        if "optimizer_type" in optimizer_init_args.keys():
            self.optimizer_type = optimizer_init_args [ "optimizer_type" ]
        else:
            self.optimizer_type                  = 'sgd'

        if verbose >= 3:
            print "... Optimizer is initiliazed"

        self.epoch = T.scalar('epoch')

        if verbose>=3 :
            print "... Creating learning rate"
        # just setup something for now. Trainer will reinitialize
        self.learning_rate = theano.shared(np.asarray(0.1,dtype=theano.config.floatX))

    def _init_momentum (self, verbose = 2):
        """
        Intitializes momentum 

        Args:
            verbose: as always
        """
        if verbose>=3 :
            print "... Setting up momentum"

        self.momentum = ifelse(self.epoch <= self.momentum_epoch_end,
                        self.momentum_start * (1.0 - self.epoch / self.momentum_epoch_end) +
                        self.momentum_end * (self.epoch / self.momentum_epoch_end),
                        self.momentum_end)

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
        self.params = params
        for param in self.params:  
            if verbose >=3 :           
                print "... Estimating gradient of parameter ", 
                print param 
            try:
                gradient = T.grad( objective ,param)
                self.gradients.append ( gradient )
            except:
                print param
                raise Exception ("Cannot learn a layer that is disconnected with objective. " +
                        "Try cooking again by making the particular layer learnable as False")

    def _polyak (self, verbose = 1):  
        """
        Apply polyak momentum

        Args: 
            verbose: as usual 

        Notes:
                Momentum implemented from paper:
                Polyak, Boris Teodorovich. "Some methods of speeding up the convergence of
                iteration methods."  USSR Computational Mathematics and Mathematical
                Physics 4.5 (1964): 1-17.

                Adapted from Sutskever, Ilya, Hinton et al. "On the importance of initialization
                and momentum in deep learning.", Proceedings of the 30th international
                conference on machine learning (ICML-13). 2013. equation (1) and equation (2)        
        """
        self._init_momentum () 
        if verbose>=3 :
            print "... Applying polyak momentum"
        for param in self.params:
            if verbose >=3 :           
                print "... Momentum of parameter " , 
                print param                         
            velocity = theano.shared(np.zeros(param.shape.eval(), dtype=theano.config.floatX))
            step = self.momentum * velocity + self.updates[param]
            self.updates[velocity] = step - param
            self.updates[param] = step

    def _nesterov (self, verbose = 1):
        """
        Apply Nesterov momentum

        Args: 
            verbose: as usual 

        Notes:
                Nesterov, Yurii. "A method of solving a convex programming problem with
                convergence rate O (1/k2)." Soviet Mathematics Doklady. Vol. 27. No. 2. 1983.
                Adapted from
                https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/

                Instead of using past params we use the current params as described in this link
                https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,
        """
        self._init_momentum () 
        if verbose>=3 :
            print "... Applying nesterov momentum"

        for param in self.params:
            if verbose >=3 :           
                print "... Momentum of parameter " , 
                print param                         
            velocity = theano.shared(np.zeros(param.shape.eval(), dtype=theano.config.floatX))
            step = self.momentum * velocity + self.updates[param] - param
            self.updates[velocity] = step
            self.updates[param] = self.momentum * step + self.updates[param]

    def _adam (self, rho=0.9, beta=0.999, verbose = 1):
        """
        Apply ADAM updates.

        Args: 
            rho: Suppy as was mentioned in the paper. Sometimes refered to also as beta1 (0.9)
            beta: Supply as was mentioend in paper, sometimes refered to as beta2 (0.999)
            verbose: is as usual

        Notes: 
                Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic optimization."
                 arXiv preprint arXiv:1412.6980 (2014).            
        """      
        if verbose>=3 :
            print "... Applying ADAM"

        one = T.constant(1)          
        t_accumulate = theano.shared(np.asarray(0., dtype = theano.config.floatX))
        t_current = t_accumulate + 1
        a_current = self.learning_rate * T.sqrt( one - beta ** t_current ) / (one-rho ** t_current)

        for param, gradient in zip(self.params, self.gradients):
            if verbose >=3 :           
                print "... Momentum of parameter " , 
                print param             
            m_accumulate = theano.shared(np.zeros(param.shape.eval(), dtype=theano.config.floatX))
            v_accumulate = theano.shared(np.zeros(param.shape.eval(), dtype=theano.config.floatX))
            m_current = rho * m_accumulate + (one-rho) * gradient
            v_current = beta * v_accumulate + (one-beta) * gradient ** 2
            step = a_current * m_current / (T.sqrt(v_current) + 1e-8)
            self.updates[m_accumulate] = m_current
            self.updates[v_accumulate] = v_current
            self.updates[param] = param - step

        self.updates[t_accumulate] = t_current
     
    def _sgd (self, verbose = 1):
        """
        Apply sgd updates.

        Args: 
            verbose: is as usual
        """
        if verbose>=3 :
            print "... Applying SGD"        
        for param, gradient in zip (self.params, self.gradients):
            if verbose >=3 :           
                print "... Backprop of parameter " , 
                print param             
            self.updates[param] = param  - self.learning_rate  * gradient

    def _rmsprop (self, rho=0.9, verbose = 1):
        """
        Apply rmsprop updates.

        Args: 
            rho: suppy as was mentioned in the slides.
            verbose: is as usual

        Notes: 
            Applied as per:
                [1] Tieleman, T. and Hinton, G. (2012):
                Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
                Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
        """     
        if verbose>=3 :
            print "... Applying RMSPROP"        
        accumulator =[]
        one = T.constant(1)
        for param in self.params:
            if verbose >=3 :           
                print "... Accumulating gradinent of parameter " , 
                print param 
            eps = np.zeros_like(param.get_value(borrow=True), dtype=theano.config.floatX)
            accumulator.append(theano.shared(eps, borrow=True))

        for param, gradient, accumulate in zip(self.params, self.gradients, accumulator):
            if verbose >=3 :           
                print "... Backprop of parameter " , 
                print param             
            current_accumulate = rho * accumulate + (one - rho) * T.sqr(gradient)
            self.updates[accumulate] = current_accumulate
            self.updates[param] = param - (self.learning_rate * gradient / T. sqrt ( \
                                                        current_accumulate + 1e-8 ))
                        
    def _adagrad (self, verbose = 1):
        """
        Apply adagrad updates.

        Args: 
            verbose: is as usual

        Notes: 
            Applied as per:
                John Duchi, Elad Hazan, and Yoram Singer. 2011. Adaptive subgradient methods
                for online learning and stochastic optimization. JMLR
        """     
        if verbose>=3 :
            print "... Applying AdaGrad"        
        accumulator =[]
        one = T.constant(1)
        for param in self.params:
            if verbose >=3 :           
                print "... Accumulating gradinent of parameter " , 
                print param 
            eps = np.zeros_like(param.get_value(borrow=True), dtype=theano.config.floatX)
            accumulator.append(theano.shared(eps, borrow=True))

        for param, gradient, accumulate in zip(self.params, self.gradients, accumulator):
            if verbose >=3 :           
                print "... Backprop of parameter " , 
                print param             
            current_accumulate = accumulate +  T.sqr(gradient)
            self.updates[accumulate] = current_accumulate
            self.updates[param] = param - (self.learning_rate * gradient / T. sqrt ( \
                                                        current_accumulate + 1e-8 ))

        
    def _adadelta (self, rho = 0.95, verbose = 1 ):
        """
        Apply adadelta updates.

        Args:
            rho: As described in paper.
            verbose: As usual
        
        Notes:
            Zeiler, Matthew D. "ADADELTA: an adaptive learning rate method." arXiv preprint 
            arXiv:1212.5701 (2012).
        """
        one = T.constant(1)
        if verbose>=3 :
            print "... Applying AdaDelta"  
        for param, gradient in zip(self.params, self.gradients):
            accumulate = theano.shared(np.zeros(param.shape.eval(), dtype=theano.config.floatX))
            delta = theano.shared(np.zeros(param.shape.eval(), dtype=theano.config.floatX))
            current_accumulate = rho * accumulate + (one - rho) * gradient ** 2
            self.updates[accumulate] = current_accumulate
            step = (gradient * T.sqrt(delta + 1e-8) / T.sqrt(current_accumulate + 1e-8))
            self.updates[param] = param - self.learning_rate * step
            current_delta = rho * delta + (one - rho) * step ** 2
            self.updates[delta] = current_delta

    def _constrain (self, verbose = 1):
        """
        constrain the norms of the COLUMNs of the weight, according to
        https://github.com/BVLC/caffe/issues/109      

        Args:
            verbose: As usual    
        """
        if verbose >= 3:
            print ("... Applying constraints")
        for param in self.params:
            if param.get_value(borrow=True).ndim == 2: 
                col_norms = T.sqrt(T.sum(T.sqr(self.updates[param]), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(15))
                scale = desired_norms / (1e-8 + col_norms)
                self.updates[param] = self.updates[param] * scale
      
    def create_updates(self, verbose = 1):
        """
        This basically creates all the updates and update functions which trainers can iterate
        upon.

        Args:
            verbose: Just as always
        """
        if verbose>=3 :
            print "... Creating backprop"        
        self.updates = OrderedDict()


        if verbose >=3 :
            print "... Applying " + self.optimizer_type
            print "... Applying " + self.momentum_type

        if self.optimizer_type == 'adagrad':
            self._adagrad(verbose = verbose)
        elif self.optimizer_type == 'rmsprop':
            self._rmsprop(verbose = verbose)
        elif self.optimizer_type == 'sgd':
            self._sgd(verbose = verbose)
        elif self.optimizer_type == 'adam':
            self._adam(verbose = verbose)
        elif self.optimizer_type == 'adadelta':
            self._adadelta(verbose = verbose)
        else:
            self._sgd(verbose = verbose)

        if self.momentum_type == 'polyak':       
            self._polyak(verbose = verbose)
        if self.momentum_type == 'nesterov':      
            self._nesterov(verbose = verbose)
        
        self._constrain(verbose = verbose)
if __name__ == '__main__':#pragma: no cover
    pass
