MXNet 2.0 API reference
=======================

.. image:: _static/zombie.jpg
   :alt: MXNet lives
   :width: 680
   :align: center

This is the Python API reference for the `smolix/mxnet
<https://github.com/smolix/mxnet>`_ fork — a maintained port of Apache MXNet 2.0
to CUDA 13 / Blackwell GPUs and native Apple Silicon CPU.

It is generated directly from the docstrings in the installed ``mxnet`` package
(version |release|). There is no hand-written narrative or tutorial content; for
install, build, fork changes, and known issues see the repository's
``README.md``, ``FIXED.md``, and ``OPEN_ISSUES.md``.

.. note::

   ``mxnet.np`` / ``mxnet.npx`` are runtime aliases for the modules documented
   below as :mod:`mxnet.numpy` and :mod:`mxnet.numpy_extension`.

Packages and modules
---------------------

.. autosummary::
   :toctree: generated
   :recursive:

   mxnet.numpy
   mxnet.numpy_extension
   mxnet.ndarray
   mxnet.symbol
   mxnet.gluon
   mxnet.optimizer
   mxnet.lr_scheduler
   mxnet.io
   mxnet.image
   mxnet.autograd
   mxnet.kvstore
   mxnet.device
   mxnet.profiler
   mxnet.runtime
   mxnet.contrib

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
