import unittest
import numpy
import theano
from yann.modules.optimizer import optimizer as opt

try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock,patch


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.verbose = 3
        self.channels = 1
        self.borrow = True
        self.input_layer_name = "input"
        self.input_dropout_layer_name = "dinput"
        self.input_tensor_layer_name = "tinput"
        self.input_dropout_tensor_layer_name = "dtinput"
        self.dropout_rate = 1
        self.rng = None
        self.mean_subtract = False
        self.input_shape = (1,1,10,10)
        self.input_ndarray = numpy.random.rand(1,1,10,10)
        self.output_dropout_ndarray= numpy.zeros((1,1,10,10))
        self.input_tensor = theano.shared(self.input_ndarray)
        self.optimizer_id = "optid"
        self.momentum_params = (0.9, 0.95, 30)
        self.regularization = (0.000, 0.000)
        self.optimizer_type = 'sgd'
        self.sample = 1
        self.learning_val = 0.5
        self.momentum_type = 'false'
        self.scalar_value = 1
        self.optimizer_params = {
            "id" : self.optimizer_id,
            "momentum_type": self.momentum_type,
            "momentum_params": self.momentum_params,
            "regularization": self.regularization,
            "optimizer_type": self.optimizer_type,
        }
        self.input_params = [1]
        self.input_objective = 1
        self.gradiant_exception_msg = "Cannot learn a layer that is disconnected with objective. " +"Try cooking again by making the particular layer learnable as False"
        self.input_params_ndarray = numpy.zeros((1,1,10,10))

    def f(self, X):
        return ([0.1, 0.2, 0.3] * X**2).sum()

    @patch('theano.shared')
    @patch('theano.tensor.scalar')
    def test1_optimizer(self,mock_scalar,mock_shared):
        mock_shared.return_value = self.learning_val
        mock_scalar.return_value = self.scalar_value
        self.opt = opt(
                optimizer_init_args = self.optimizer_params,
                verbose = self.verbose)
        self.assertEqual(self.opt.id,self.optimizer_id)
        self.assertEqual(self.opt.momentum_start,self.momentum_params[0])
        self.assertEqual(self.opt.momentum_end, self.momentum_params[1])
        self.assertEqual(self.opt.momentum_epoch_end, self.momentum_params[2])
        self.assertEqual(self.opt.momentum_type,self.momentum_type)
        self.assertEqual(self.opt.epoch,self.scalar_value)
        self.assertEqual(self.opt.learning_rate,self.learning_val)

    @patch('yann.modules.optimizer.ifelse')
    @patch('theano.shared')
    @patch('theano.tensor.scalar')
    def test2_optimizer_init_momentum(self,mock_scalar,mock_shared, mock_ifelse):
        mock_ifelse.return_value = self.sample
        mock_shared.return_value = self.learning_val
        mock_scalar.return_value = self.scalar_value
        self.opt = opt(
                optimizer_init_args = self.optimizer_params,
                verbose = self.verbose)
        self.opt._init_momentum()
        self.assertEqual(self.opt.momentum,self.scalar_value)


    @patch('yann.modules.optimizer.T.grad')
    @patch('theano.shared')
    @patch('theano.tensor.scalar')
    def test3_optimizer_calculate_gradients(self,mock_scalar,mock_shared, mock_grad):
        mock_grad.return_value = 1
        mock_shared.return_value = self.learning_val
        mock_scalar.return_value = self.scalar_value
        self.opt = opt(
                optimizer_init_args = self.optimizer_params,
                verbose = self.verbose)
        self.opt.calculate_gradients(params= self.input_params, objective=self.input_objective, verbose = self.verbose)
        self.assertTrue(numpy.allclose(self.opt.gradients,self.input_params))


    @patch('theano.shared')
    @patch('theano.tensor.scalar')
    def test4_optimizer_calculate_gradients_exception(self,mock_scalar,mock_shared):
        try:
            mock_shared.return_value = self.learning_val
            mock_scalar.return_value = self.scalar_value
            self.opt = opt(
                    optimizer_init_args = self.optimizer_params,
                    verbose = self.verbose)
            self.opt.calculate_gradients(params= self.input_params, objective=False, verbose = self.verbose)
        except Exception, c:
            self.assertEqual(c.message, self.gradiant_exception_msg)


    @patch('yann.modules.optimizer.ifelse')
    @patch('theano.shared')
    @patch('theano.tensor.scalar')
    def test5_optimizer_polyak(self,mock_scalar, mock_shared,mock_ifelse):#mock_shared
        A = theano.shared(numpy.asarray([1, 1, 1], dtype=theano.config.floatX))
        B = theano.shared(numpy.asarray([1, 1, 1], dtype=theano.config.floatX))
        mock_ifelse.return_value = 0.9
        mock_shared.return_value = self.learning_val
        mock_scalar.return_value = self.scalar_value
        self.opt = opt(
                optimizer_init_args = self.optimizer_params,
                verbose = self.verbose)
        self.opt._init_momentum()
        self.opt.params = [A, B]
        self.opt.updates = self.f(A) + self.f(B)
        self.opt._polyak()
        self.assertTrue(numpy.allclose(self.opt.updates[0],A))



    @patch('yann.modules.optimizer.ifelse')
    @patch('theano.shared')
    @patch('theano.tensor.scalar')
    def test5_optimizer_nesterov(self,mock_scalar, mock_shared,mock_ifelse):#mock_shared
        A = theano.shared(numpy.asarray([1, 1, 1], dtype=theano.config.floatX))
        B = theano.shared(numpy.asarray([1, 1, 1], dtype=theano.config.floatX))
        mock_ifelse.return_value = 0.9
        mock_shared.return_value = self.learning_val
        mock_scalar.return_value = self.scalar_value
        self.opt = opt(
                optimizer_init_args = self.optimizer_params,
                verbose = self.verbose)
        self.opt._init_momentum()
        self.opt.params = [A, B]
        self.opt.updates = self.f(A) + self.f(B)
        self.opt._nesterov()
        self.assertTrue(numpy.allclose(self.opt.updates[0],A))

    @patch('theano.shared')
    def test6_optimizer_adam(self,mock_shared):
        A = theano.shared(numpy.asarray([1, 1, 1], dtype=theano.config.floatX))
        B = theano.shared(numpy.asarray([1, 1, 1], dtype=theano.config.floatX))
        mock_shared.return_value = self.learning_val
        self.opt = opt(
                optimizer_init_args = self.optimizer_params,
                verbose = self.verbose)
        self.opt.params = [A, B]
        self.opt.updates = self.f(A) + self.f(B)
        self.opt.gradients = [0,0]
        self.opt.learning_rate = 0
        self.opt._adam(rho=0, beta=0,verbose=self.verbose)
        self.assertTrue(numpy.allclose(self.opt.updates[0],A))

    @patch('theano.shared')
    def test7_optimizer_sdg(self,mock_shared):
        A = theano.shared(numpy.asarray([1, 1, 1], dtype=theano.config.floatX))
        B = theano.shared(numpy.asarray([1, 1, 1], dtype=theano.config.floatX))
        mock_shared.return_value = self.learning_val
        self.opt = opt(
                optimizer_init_args = self.optimizer_params,
                verbose = self.verbose)
        self.opt.params = [A, B]
        self.opt.updates = self.f(A) + self.f(B)
        self.opt.gradients = [0,0]
        self.opt.learning_rate = 0
        self.opt._sgd(verbose=self.verbose)
        self.assertTrue(numpy.allclose(self.opt.updates[0],A))

    @patch('theano.shared')
    def test8_optimizer_rmsprop(self,mock_shared):
        A = theano.shared(numpy.asarray([1, 1, 1], dtype=theano.config.floatX))
        B = theano.shared(numpy.asarray([1, 1, 1], dtype=theano.config.floatX))
        mock_shared.return_value = self.learning_val
        self.opt = opt(
                optimizer_init_args = self.optimizer_params,
                verbose = self.verbose)
        self.opt.params = [A, B]
        self.opt.updates = self.f(A) + self.f(B)
        self.opt.gradients = [0,0]
        self.opt.learning_rate = 0
        self.opt._rmsprop(rho = 0,verbose=self.verbose)
        self.assertTrue(numpy.allclose(self.opt.updates[0],A))

    @patch('theano.shared')
    def test9_optimizer_adagrad(self,mock_shared):
        A = theano.shared(numpy.asarray([1, 1, 1], dtype=theano.config.floatX))
        B = theano.shared(numpy.asarray([1, 1, 1], dtype=theano.config.floatX))
        mock_shared.return_value = self.learning_val
        self.opt = opt(
                optimizer_init_args = self.optimizer_params,
                verbose = self.verbose)
        self.opt.params = [A, B]
        self.opt.updates = self.f(A) + self.f(B)
        self.opt.gradients = [0,0]
        self.opt.learning_rate = 0
        self.opt._adagrad(verbose=self.verbose)
        self.assertTrue(numpy.allclose(self.opt.updates[0],A))

    @patch('theano.shared')
    def test10_optimizer_adadelta(self,mock_shared):
        A = theano.shared(numpy.asarray([1, 1, 1], dtype=theano.config.floatX))
        B = theano.shared(numpy.asarray([1, 1, 1], dtype=theano.config.floatX))
        mock_shared.return_value = self.learning_val
        self.opt = opt(
                optimizer_init_args = self.optimizer_params,
                verbose = self.verbose)
        self.opt.params = [A, B]
        self.opt.updates = self.f(A) + self.f(B)
        self.opt.gradients = [0,0]
        self.opt.learning_rate = 0
        self.opt._adadelta(rho =0, verbose=self.verbose)
        self.assertTrue(numpy.allclose(self.opt.updates[0],A))

    @patch('theano.shared')
    def test11_optimizer_constrain(self,mock_shared):
        A = theano.shared(numpy.asarray([1, 1, 1], dtype=theano.config.floatX))
        B = theano.shared(numpy.asarray([1, 1, 1], dtype=theano.config.floatX))
        mock_shared.return_value = self.learning_val
        self.opt = opt(
                optimizer_init_args = self.optimizer_params,
                verbose = self.verbose)
        self.opt.params = [A, B]
        self.opt.updates = self.f(A) + self.f(B)
        self.opt.gradients = [0,0]
        self.opt.learning_rate = 0
        self.opt._constrain(verbose=self.verbose)
        self.assertTrue(numpy.allclose(self.opt.updates[0],A))


    @patch('yann.modules.optimizer.optimizer._nesterov')
    @patch('yann.modules.optimizer.optimizer._adadelta')
    @patch('yann.modules.optimizer.optimizer._adam')
    @patch('yann.modules.optimizer.optimizer._sgd')
    @patch('yann.modules.optimizer.optimizer._rmsprop')
    @patch('yann.modules.optimizer.optimizer._constrain')
    @patch('yann.modules.optimizer.optimizer._adagrad')
    @patch('yann.modules.optimizer.optimizer._polyak')
    @patch('theano.shared')
    def test12_optimizer_create_updates(self,mock_shared,mock_polyak,mock_adagrad,mock_constrain,mock_rmsprop,mock_sgd,mock_adam,mock_adadelta,mock_nesterov):
        try:
            mock_polyak.return_value = ""
            mock_adagrad.return_value = ""
            mock_constrain.return_value = ""
            mock_rmsprop.return_value = ""
            mock_sgd.return_value = ""
            mock_adam.return_value = ""
            mock_adadelta.return_value = ""
            mock_nesterov.return_value = ""
            mock_shared.return_value = self.learning_val
            self.opt = opt(
                    optimizer_init_args = self.optimizer_params,
                    verbose = self.verbose)
            self.opt.optimizer_type ="adagrad"
            self.opt.momentum_type ="polyak"
            self.opt.create_updates(verbose=self.verbose)
            self.opt.optimizer_type ="rmsprop"
            self.opt.momentum_type ="nesterov"
            self.opt.create_updates(verbose=self.verbose)
            self.opt.optimizer_type ="sgd"
            self.opt.momentum_type ="polyak"
            self.opt.create_updates(verbose=self.verbose)
            self.opt.optimizer_type ="adam"
            self.opt.momentum_type ="polyak"
            self.opt.create_updates(verbose=self.verbose)
            self.opt.optimizer_type ="adadelta"
            self.opt.momentum_type ="polyak"
            self.opt.create_updates(verbose=self.verbose)
            self.assertEqual(True, True)
        except Exception,c:
            self.assertEqual(True,False)
