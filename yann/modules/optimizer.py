import numpy
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from collections import OrderedDict
from abstract import module

class optimizer(module):
    """

    Todo:

        * AdaDelta

    Warning:
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

        if "optimizer_type" in optimizer_init_args.keys():
            self.optimizer_type = optimizer_init_args [ "optimizer_type" ]
        else:
            self.optimizer_type                  = 'sgd'

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
            if verbose >=3 :           
                print ".. Estimating gradient of parameter ", 
                print param 
            try:
                gradient = T.grad( objective ,param)
                self.gradients.append ( gradient )
            except:
                print param
                raise Exception ("Cannot learn a layer that is disconnected with objective. " +
                        "Try cooking again by making the particular layer learnable as False")


    def create_updates(self, params, verbose = 1):
        """
        This basically creates all the updates and update functions which trainers can iterate
        upon.

        Todo:
            Need to modularize this method more. I need to split these into many methods one for
            each type of optimizer. if-then breaks are fine for now.

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
            if verbose >=3 :           
                print ".. Estimating velocity  of parameter ",
                print param 
            velocity = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
                                                                dtype=theano.config.floatX))
            velocities.append(velocity)

        # these are used for second order optimizers.
        accumulator_1 =[]
        accumulator_2 = []
        for param in params:
            if verbose >=3 :           
                print ".. Accumulating gradinent of parameter " , 
                print param 
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
            if verbose >=3 :           
                print ".. Backprop of parameter ", 
                print param 

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


if __name__ == '__main__':
    pass
