.. _setup:

==================
Installation Guide
==================

Yann is built on top of `Theano`_. `Theano`_ and all its pre-requisites are mandatory.
Once theano and its pre-requisites are setup you may setup and run this toolbox.
Theano setup is documented in the `theano toolbox documentation`_. Yann is built with theanoo 0.8 
but should be forward compatible unless theano makes a drastic release. 

.. _Theano: http://deeplearning.net/software/theano/ 
.. _theano toolbox documentation: http://deeplearning.net/software/theano/install.html

Quick fire Installation
***********************

Now before going through the full-fledged installation procedure, you can run through the entire
installation in one command that will install the basics required to run the toolbox. To install
the toolbox quickly do the following:

.. code-block:: bash

   pip install git+git://github.com/ragavvenkatesan/yann.git

If it showed any errors, install ``numpy`` first. ``skdata`` has some issue that requires ``numpy``
installed first. If you use anaconda, just install the numpy and scipy using ``conda install`` 
instead of ``pip install``. This will setup the toolbox for all intentions and purposes.

For a full-fledged installation procedure, don't do the above but run through the following set of 
instructions. If you want to install all other supporting features like datasets, visualizers and 
others, do the following: 

.. code-block:: bash

  pip install -r requirements_full.txt
  pip install git+git://github.com/ragavvenkatesan/yann.git

Full installation
*****************

Dependencies
============


Python + pip / conda
--------------------

  Yann needs Python 2.7. 
  Please install it for your OS.. Some modules that are required
  don't come with default python. But don't worry python comes with a package installer
  called pip. You can use pip to install additional packages.  
  
  For a headache free installation, the 
  `anaconda <https://www.continuum.io/downloads>`_ distribution of python is 
  very strongly recommended because it comes with a lot of goodies pre-packaged.  

C compiler
----------

  You need a C compiler, not because yann needs C, but theano and probably numpy
  requires C compilers. Make sure that your OS has one. Apple osX or macOS users, if you are using 
  Cuda and cuDNN, prefer using command line tools 7.x+. 8 doesn't work with cuDNN at the moment of 
  writing this documentation. You can download older versions of xcode and command line tools 
  `here <https://developer.apple.com/download/more/>`_.

numpy/scipy 
-----------

  Numpy 1.6 and Scipy 0.11 are needed for yann. Make sure these work well with a blas system. Prefer 
  `Intel MKL <https://software.intel.com/en-us/intel-mkl>`_ for blas, which is also availabe from 
  anaconda. MKL is free for students and researchers and is available for a small price for others.

  If you use pip use 

  .. code-block:: bash

     pip install numpy
     pip install scipy
  
  to install these. If you use anaconda, use

  .. code-block:: bash

    conda install mkl
    conda install numpy
    conda install scipy

  to set these up. If not, yann installer will ``pip install numpy scipy`` anyway as part of its 
  requirements.

Theano
------

Once all the pre-requisites are setup, install `theano`_ version 0.8 or higher.

.. _theano: http://deeplearning.net/software/theano/ 

The following ``.theanorc`` configuration can be used as a sample normally, 
but you may choose other options. As an example one can use the following:

.. code-block:: bash

  [global]
  floatX=float32
  device=cuda0
  optimizer_including=cudnn
  mode = FAST_RUN

  [nvcc]
  nvcc.fastmath=True
  allow_gc=False

  [cuda]
  root=/usr/local/cuda/

  [blas]
  ldflags = -lmkl

  [lib]
  cnmem = 0.5

If you use the `libgpuarray <http://deeplearning.net/software/libgpuarray/installation.html>`_ 
backend instead of the CUDA backend, use ``device=cuda0`` or whichever device you want to run on.
If you are using CUDA backed use ``device=gpu0``. Refer theano documentation for more on this.

Optional Dependencies
=====================

These are some optional dependencies that yann doesn't use directly but are used by yann's 
dependencies like theano. I highly recommend these before installing theano.

Cuda 
----

  This is an optional dependency. If you need the capability of a Nvidia GPU, you will need a 
  suitable `CUDA toolkit and drivers <https://developer.nvidia.com/cuda-toolkit>`_. If you do not  
  have this dependency installed, you won't be able to run the code on Nvidia GPUs.Some compoenents
  of the code depend on `cuDNN <https://developer.nvidia.com/cudnn>`_ for speeding things up, so 
  `cuDNN <https://developer.nvidia.com/cudnn>`_ is highly recommended although optional.
  Nvidia has the awesome cuDNN library that is free as long as you
  register as a `developer <https://developer.nvidia.com/cudnn>`_. 
  If you didn't install CUDA, you can still run the toolbox, but it will be much slower running on a
  CPU.


Libgpuarray
-----------

  `libgpuarray <http://deeplearning.net/software/libgpuarray/installation.html>`_  
  is now fully supported, cuda backend is strongly recommended for macOS, but for the Pascal 
  architecture of GPUs, ``libgpuarray`` seems to be performing much better. This is also an 
  optional but highly recommended tool 


Additional Dependencies
==============================

Yann also needs the following as additional dependencies that opens up additional features. 

Networkx
--------

  For those who are networking geeks, a neural network is a directed acyclic graph. So Yann 
  internally has the ability for every network to create a ``networkx`` style graph and do things 
  with it if you need. `Networkx <https://networkx.github.io/>`_ is a tremendously popular 
  tool for network realted tasks and we are still exploring and testing its capabilities. This might 
  only ever be used for visualization of network purposes, but some researcher somewhere might 
  use this once in the future networks get sophisticated, we never know. This is an optional 
  dependency, not having this dependency doesn't affect the toolbox, except for the purposes it is 
  needed for. 
  
  You can install ``networkx`` as follows:

  .. code-block:: bash
   
    pip install networkx


skdata
------

Used as a port for datasets. This is Needed if you are using some common benchmark datasets. 
Although this is an additional dependency, skdata is the core of the datasets module and most 
datasets in this toolbox are ported through skdata unless you have matlab. Work is on-going in
integrating with fuel and other ports. 

Install by using the following command:

.. code-block:: bash

  pip install skdata

progressbar
-----------
  
  Yann uses `progressbar <https://pypi.python.org/pypi/progressbar>`_ for aesthetic printing. You 
  can install it easily by using 

  .. code-block:: bash

    pip install progressbar
    
  If you don't have progressbar, yann will simply ignore it and print progress on terminal.

Dependencies for visualization
------------------------------

  Theano needs pydot and graphviz for visualization. We use theano's visualization for printing
  theano functions as shown 
  `here <https://github.com/ragavvenkatesan/yann/blob/master/docs/source/pantry/samples/train.pdf>`_.
   
  These visualizations are highly useful during debugging. If you want the capability of producing 
  these for your networks, install the dependencises using the following commands:

  .. code-block:: bash

    apt-get install graphviz
    pip install graphviz
    pip install pydot pydot-ng

  Not needed now, but might need in future. 
  Yann will switch from openCV to matplotlib or browser matplotlib for visualization. Install it by 

  .. code-block:: bash

    pip insall matplotlib
  
cPickle, gzip and hdf5py 
------------------------

  Most often the case is that `cPickle` and `gzip` these come with the python installation, 
  if not please install them.  Yann uses these for saving down models and such.

  For datasets, at the moment, yann uses cpickle. In the future, yann will migrate to hdf5 for 
  datasets. Install hdf5py by running either,

  .. code-block:: bash

      conda install h5py

  or, 

  .. code-block:: bash

      pip install h5py


Yann Toolbox Setup
====================
 
Finally to install the toolbox run, 

.. code-block:: bash

    pip install git+git://github.com/ragavvenkatesan/yann.git

If you have already setup the toolbox and want to just update to the bleeding-edge use,

.. code-block:: bash

    pip install --upgrade git+git://github.com/ragavvenkatesan/yann.git

If you want to build by yourself you may clone from git and then run using setuptools. Ensure that 
you have setuptools installed first. 

.. code-block:: bash

  pip install git setuptools

Once you are done, you clone the repository from git.

.. code-block:: bash

  git clone http://github.com/ragavvenkatesan/yann

Once cloned, enter the directory and run installer.

.. code-block:: bash

  cd yann
  python setup.py install

You can run a bunch of tests ( working on it ) by running the following code:

.. code-block:: bash

  python setup.py test

