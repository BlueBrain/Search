Entrypoints
===========

This section describes how to use the entrypoints for common operations.


Compute sentence embeddings
---------------------------

We will compute sentence embeddings:
* with the model BioBERT NLI+STS CORD-19 v1,
* for CORD-19 version 47,
* using 4 GPUs,
* on DGX-1.

The same How-To could be applied to other models, other CORD-19 versions,
other number of used GPUs, and other platforms than DGX-1.

1 - Login to DGX-1:

.. code-block:: bash

    ssh dgx1.bbp.epfl.ch

2 - Launch a Docker container with CUDA support and access to 4 GPUs:

.. code-block:: bash

    docker run -it -v /raid:/raid --rm --user root --gpus '"device=0,1,2,3"' \
      --name embeddings_computation bbs_base

3 - Upgrade :code:`pip`:

.. code-block:: bash

    python -m pip install --upgrade pip

4 - Install Blue Brain Search:

.. code-block:: bash

    pip install git+https://github.com/BlueBrain/BlueBrainSearch.git

5 - Define the path to the HDF5 file containing the embeddings:

.. code-block:: bash

    export EMBEDDINGS=/raid/sync/proj115/bbs_data/cord19_v47/embeddings/embeddings.h5

6 - If embeddings where computed for other models, backup the existing embeddings:

.. code-block:: bash

    cp  $EMBEDDINGS ${EMBEDDINGS}.backup

7 - Launch the parallel computation of the embeddings:

.. code-block:: bash

    compute_embeddings SentTransformer $EMBEDDINGS \
      --checkpoint /raid/sync/proj115/bbs_data/trained_models/biobert_nli_sts_cord19_v1 \
      --db-url dgx1.bbp.epfl.ch:8853/cord19_v47 \
      --gpus 0,1,2,3 \
      --h5_dataset_name 'BioBERT NLI+STS CORD-19 v1' \
      --n-processes 4 \
      --temp-dir .


Create the MySQL database
-------------------------

We will create the MySQL database:
* for CORD-19 version 65,
* on DGX-1.

The same How-To could be applied to other CORD-19 versions and other platforms
than DGX-1.

1 - Login to DGX-1:

.. code-block:: bash

    ssh dgx1.bbp.epfl.ch

2 - Launch a Docker container:

.. code-block:: bash

    docker run -it -v /raid:/raid --rm --user root --name database_creation bbs_base

3 - Upgrade :code:`pip`:

.. code-block:: bash

    python -m pip install --upgrade pip

4 - Install Blue Brain Search:

.. code-block:: bash

    pip install git+https://github.com/BlueBrain/BlueBrainSearch.git

5 - Launch the creation of the database:

.. code-block:: bash

    create_database --data-path /raid/sync/proj115/bbs_data/cord19_v65
