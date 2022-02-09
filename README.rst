================
Trame AI LeNet-5
================

Visualization exploration for AI/XAI


* Free software: BSD License


Installing
----------
Build and install the Vue components

.. code-block:: console

    cd vue-components
    npm i
    npm run build
    cd -

For the Python layer it is recommended to use conda to properly install the various ML packages.

macOS conda setup
^^^^^^^^^^^^^^^^^

.. code-block:: console

    brew install miniforge
    conda init zsh

venv creation for AI
^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    # PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install 3.9.9
    conda create --name lenet5 python=3.9
    conda activate lenet5
    conda install "pytorch==1.9.1" "torchvision==0.10.1" -c pytorch
    # conda install scipy "scikit-learn==0.24.2" "scikit-image==0.18.3" -c conda-forge # For XAITK
    pip install -e .



Run the application

.. code-block:: console

    conda activate lenet5
    trame_ai_lenet_5
