# The story behind the repository

This is a CNN implementaion in [theano](http://deeplearning.net/software/theano/) with many modern day upgrades features such as dropouts[1], adagrad[2], momentum[3], max-margin layer and access to many datasets through [skdata](https://jaberg.github.io/skdata/). If you're a beginner in neural networks and deep learning as I was five months ago, you would find yourself overwhelmed with literature and materials. After spending a long time, based on your needs, you will end up deciding to use among many software, [theano](http://deeplearning.net/software/theano/) . [Theano](http://deeplearning.net/software/theano/) is a beautiful environment for it provides you gradients automatically. Its a symbolic language. IT IS NOT A DEEP LEARNING TOOLBOX. It is hard to get around, particularly, if like me you are also new to python and to symbolic programming in general. 

You'll probably begin by following the [Caffe CNN tutorials](http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb) or [Lisa lab's tutorial](http://www.iro.umontreal.ca/~lisa/pointeurs/tutorial_hpcs2011_fixed.pdf) or the [theano tutorial](http://deeplearning.net/software/theano/tutorial/index.html#tutorial) itself.

Once you are done with the tutorials (and that takes about a couple of months to get everything going), while even though you may feel a little confident, you'll find that to stand up to the state-of-the-art, you need to implement SVM layers, All kinds of activation functions, dropout[1], classical momentum[3], accelerated gradients[4], adaptive gradient descent, AdaGrad[2] and all these before you can start your research.  

*** 

## What is in this repository? 

The code that is here in this repository has the following features, among many others:
* CNNs with easy architecture management. You can create any architecture you want with just changes of small parameters in the boiler plate. 
* Dropouts[1]
* adaGrad[2]
* Polyak Momentum[3]
* Data handling capabilities.
* Also provided a wrapper to [skdata's dataset](https://jaberg.github.io/skdata/) interface (under construction, as of now, I have cifar10, mnist, extended mnists , caltech101). 
* [new] Data visualization capabilities: Visualize the activities of select images (random) from the trainset on each layer after select number of epochs of training. Also view filters after select number of epochs. If the input images are color, the first layer saves down color features. 
   
More features will be added as and when I am implementing them. You can check the `to_do.txt` in the repository for expected updates.  I will add more detailed description of the implementation as and when I have time to so. But don't expect it soon.

*** 
## System requirements

Running this code essentially requires:

    1. `python 2.x`

    2. `theano 0.6 +`    

    3. `numpy` 
 
    4. `scipy`

    5. `skdata`

    6. `cPickle`

    7. `opencv (cv2)` 

    8. `gzip`

Most of these could be installed by installing [anaconda of continuum analytics](http://docs.continuum.io/anaconda/install.html) The code is reasonably well documented. It is not that difficult to find out from the boilerplate what is happening. If you need to understand really well what is happening before you jump into the code, use `verbose = True` flag in the boiler plate. The code has been tested on exhaustively in both MacOSX and Linux Ubuntu 14.x and 15.x by virtue of constant use. 

*** 

## Who is this code most useful for ?

I wrote this code essentially for my labmates, those who are interested in starting deep learning to make a fast transition into theano. I reckon that this will be useful for someone who is starting out to be a grad student getting to start research into deep learning (like me) or to someone who wants to do a try out on some kaggle challenge. Parts of the code are directly lifted from [theano tutorials](http://deeplearning.net/software/theano/tutorial/) or from [Misha Denil's repository](https://github.com/mdenil). 

This might not be really useful for advanced deep learning researchers. It's quite basic. If you are a serious researcher and you either find a bug or you want to just make suggestions please feel free, I will be grateful.  

***

# References
[1]   Srivastava, Nitish, et al. "Dropout: A simple way to prevent neural networks from overfitting." The Journal of Machine Learning Research 15.1 (2014): 1929-1958.

[2]   John Duchi, Elad Hazan, and Yoram Singer. 2011. Adaptive subgradient methods for online learning and stochastic optimization. JMLR

[3]   Polyak, Boris Teodorovich. "Some methods of speeding up the convergence of iteration methods." USSR Computational Mathematics and Mathematical Physics 4.5 (1964): 1-17. Implementation was adapted from Sutskever, Ilya, et al. "On the importance of initialization and momentum in deep learning." Proceedings of the 30th international conference on machine learning (ICML-13). 2013.

[4]   Nesterov, Yurii. "A method of solving a convex programming problem with convergence rate O (1/k2)."   Soviet Mathematics Doklady. Vol. 27. No. 2. 1983. Adapted from [Sebastien Bubeck's](https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/) blog.
*** 

Thanks for using the code, hope you had fun.
Ragav Venkatesan
