# Convolutional Neural Networks:

This is a CNN implementaion in theano with many features such as dropouts, adagrad, momentum, svm layer and access to many datasets through skdata. If you're a beginner in neural networks and deep learning as I was five months ago, you would find yourself overwhelmed with literature and materials. Afterspending a long time, based on your needs, you will end up deciding to use among many software, Theano (http://deeplearning.net/software/theano/) . You'll follow the CNN tutorials (http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb or http://www.iro.umontreal.ca/~lisa/pointeurs/tutorial_hpcs2011_fixed.pdf or http://deeplearning.net/software/theano/tutorial/index.html#tutorial ).

Once you are done with the tutorials (and that takes about a couple of months to get everything going), while even though you may feel a little confident, you'll find that to stand up to the state-of-the-art, you need to implement SVM layers, All kinds of activation functions, dropout, classical momentum, accelerated gradients, adaptive gradient descent, AdaGrad and all these before you can start your research. Theano is a beautiful environment for it provides you gradients automatically. Its a symbolic language. IT IS NOT A DEEP LEARNING TOOLBOX. I have used it recently to code a k-svd dictionary system also. It is hard to get around, particularly, if like me you are also new to python and to symbolic programming in general.  

The code that is here in this repository has the following features, among many others, already written:

1. CNNs with easy architecture management. 
    Create any architecture you want with just changes of small parameters in the boiler plate. 
2. Dropouts
    Srivastava, Nitish, et al. "Dropout: A simple way to prevent neural networks
    from overfitting." The Journal of Machine Learning Research 15.1 (2014): 1929-1958.
3. adaGrad 
    John Duchi, Elad Hazan, and Yoram Singer. 2011. Adaptive subgradient methods
    for online learning and stochastic optimization. JMLR
4. Polyak Momentum 
    Polyak, Boris Teodorovich. "Some methods of speeding up the convergence of iteration methods." 
    USSR Computational Mathematics and Mathematical Physics 4.5 (1964): 1-17.
    Adapted from Sutskever, Ilya, et al. "On the importance of initialization and momentum in deep learning." 
    Proceedings of the 30th international conference on machine learning (ICML-13). 2013.
5. Data handling capabilities.
    I have also provided a wrapper to skdata's dataset interface (under construction, as of now, I have cifar10, mnist,             caltech101) with boilerplate samples for all of them. 
    
and many others .... 

Requirements:

Running this code essentially requires:
1. python 2.x
2. theano 0.6 + 
3. numpy 
4. scipy
5. skdata
6. cPickle
7. opencv (cv2) 
8. gzip

Most of these could be installed by installing anaconda of continuum analytics (http://docs.continuum.io/anaconda/install.html) The code is reasonably well documented. It is not that difficult to find out from the boilerplate what is happening. If you need to understand really well what is happening before you jump into the code, use verbose = True flag in the boiler plate. 

Thanks for using the code.
Ragav Venkatesan
