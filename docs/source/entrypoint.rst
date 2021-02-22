.. Blue Brain Search is a text mining toolbox focused on scientific use cases.
   Copyright (C) 2020  Blue Brain Project, EPFL.
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public License
   along with this program. If not, see <https://www.gnu.org/licenses/>.

Entry points
============

This section describes how to use the entry points for common operations.


Compute sentence embeddings
---------------------------
We will compute sentence embeddings:

- with the model BioBERT NLI+STS CORD-19 v1
- for CORD-19 version 47
- using 4 GPUs

The same instructions can be applied to other models, other CORD-19 versions, and
other GPU configurations. To run on a CPU please consistently remove the ``--gpus``
parameter everywhere.

Launch a Docker container with CUDA support and access to 4 GPUs:

.. code-block:: bash

    docker run \
      -it \
      --rm \
      --volume <local_path>:<container_path>
      --user 'root' \
      --gpus '"device=0,1,2,3"' \
      --name 'embedding_computation' \
      bbs_base

Note that we use the ``--volume`` parameter to mount all local paths that should be accessible
from the container, for example the output directory for the embedding file, or the path to
the embedding model checkpoint.

All following commands are executed in this interactive container.

Upgrade ``pip``:

.. code-block:: bash

    python -m pip install --upgrade pip

Install Blue Brain Search:

.. code-block:: bash

    pip install git+https://github.com/BlueBrain/Search.git

Define the path to the output HDF5 file with the embeddings:

.. code-block:: bash

    export EMBEDDINGS=<some_path>/embeddings.h5

It is possible to write different embedding datasets to the same h5-file. If the file specified
in ``EMBEDDINGS`` already exists and we're adding a new embedding dataset, then one might consider
creating a backup copy:

.. code-block:: bash

    cp  "$EMBEDDINGS" "${EMBEDDINGS}.backup"

Launch the parallel computation of the embeddings:

.. code-block:: bash

    compute_embeddings SentTransformer "$EMBEDDINGS" \
      --checkpoint '<biobert_model_checkpoint_path>' \
      --db-url <mysql_host>:<mysql_port>/<mysql_database> \
      --gpus '0,1,2,3' \
      --h5-dataset-name 'BioBERT NLI+STS CORD-19 v1' \
      --n-processes 4 \
      --temp-dir .

Create the MySQL database
-------------------------
Launch an interactive Docker container:

.. code-block:: bash

    docker run \
      -it \
      --rm \
      --volume <local_path>:<container_path> \
      --user 'root' \
      --name 'database_creation' \
      bbs_base

Note that we use the ``--volume`` parameter to mount all local paths that should be accessible
from the container, for example the directory with the CORD data (see below).

All following commands are executed in this interactive container.

Upgrade ``pip``:

.. code-block:: bash

    python -m pip install --upgrade pip

Install Blue Brain Search:

.. code-block:: bash

    pip install git+https://github.com/BlueBrain/Search.git

Launch the creation of the database:

.. code-block:: bash

    create_database --data-path <data_path>

The parameter ``data_path`` should point to the directory with the original CORD-19 data,
which can be obtained from
`Kaggle <https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge>`_.
