================
Trame MNIST
================

Example application using **trame** for exploring MNIST dataset in the context of AI training and XAI thanks to **XAITK**.

* Free software: BSD License
* `XAITK Saliency with MNIST <https://github.com/XAITK/xaitk-saliency/blob/master/examples/MNIST_scikit_saliency.ipynb>`_
* `XAI Discovery Platform | MNIST Sample Data <http://obereed.net:3838/mnist/>`_

Installing
----------

For the Python layer it is recommended to use conda to properly install the various ML packages.

macOS conda setup
^^^^^^^^^^^^^^^^^

.. code-block:: console

    brew install miniforge
    conda init zsh

venv creation for AI
^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    # Needed in order to get py3.9 with lzma
    # PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install 3.9.9

    conda create --name trame-mnist python=3.9
    conda activate trame-mnist
    conda install "pytorch==1.9.1" "torchvision==0.10.1" -c pytorch
    conda install scipy "scikit-learn==0.24.2" "scikit-image==0.18.3" -c conda-forge
    pip install -e .



Run the application

.. code-block:: console

    conda activate trame-mnist
    trame-mnist
