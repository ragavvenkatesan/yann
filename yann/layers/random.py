from abstract import layer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano
import numpy

class random_layer (layer):
    """
    This is a random generation layer.

    Args:
        num_neurons: List of the shapes of all inputs.
        distribution: ``'binomial'``, ``'uniform'``, ``'normal'`` ``'gaussian'``
        limits: tuple for uniform
        mu: mean for gaussian
        sigma: variance for gaussian
        p: if ``type`` is ``'binomial'`` supply a ``p`` variable. Default is 0.5
        id: Supply a layer id
        num_neurons: Supply the output shape of the layer desired.
        verbose: As always

    """
    def __init__ (  self,
                    num_neurons,
                    id = -1,
                    distribution = 'binomial',
                    verbose = 2,
                    options = None):

        if verbose >=3:
            print("... Creating a " + distribution + "random layer of " + \
                  "output_shape " +  str(num_neurons))

        super(random_layer,self).__init__(id = id, type = 'random', verbose = verbose)
        rng = numpy.random
        srng = RandomStreams(rng.randint(1,2147462468), use_cuda=None)

        if isinstance(num_neurons, int):
            num_neurons = (num_neurons,)

        if distribution == 'binomial':
            if not 'p' in options.keys():
                if verbose >= 3:
                    print("... Needs input p, by default assuming 0.5")
                p = 0.5
            else:
                p = options["p"]

            self.output = srng.binomial(n=1, p=p, size=num_neurons, dtype = theano.config.floatX )


        elif distribution == 'uniform':
            if not 'limits' in options.keys():
                if verbose >= 3:
                    print("... Needs limits, assuming default (0,1)")
                limits = (0,1)
            else:
                limits = options['limits']

            self.output = srng.uniform(size = num_neurons, low =limits[0], high =limits[1],
                                                dtype = theano.config.floatX )

        elif distribution == 'gaussian' or distribution == 'normal':
            if not 'mu' in options.keys():
                if verbose >= 3:
                    print("... Needs mu, assuming default 0")
                mu = 0
            else:
                mu = options['mu']
            if not 'sigma' in options.keys():
                if verbose >= 3:
                    print("... Needs sigma, assuming default 1")
                sigma = 1
            else:
                sigma = options['sigma']

            self.output = srng.normal(size = num_neurons, avg = mu, std = sigma,
                                                dtype = theano.config.floatX)

        self.output_shape = num_neurons
        self.num_neurons = num_neurons

        self.inference = self.output
        if verbose >=3:
            print("... Random layer is created with output shape " + str(self.output_shape))


if __name__ == '__main__':#pragma: no cover
    pass
