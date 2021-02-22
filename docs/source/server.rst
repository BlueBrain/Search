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

Servers
=======
This section describes how to launch and use various servers.

Embedding Server
----------------
The REST API Server runs in a docker container, so in order to use it a docker
image needs to be build, and a container needs to be spawned.

Build
~~~~~
To build the docker image open a terminal in the root directory of the project
and run the following command

.. code-block:: bash

    docker build -f docker/embedding.Dockerfile -t bbs_embedding .

This will create a docker image with the tag ``bbs_embedding:latest``.

Configure
~~~~~~~~~
Prior to starting the server one should check its configuration. We configure
our servers in a ``.env`` file. This file is not distributed with the package
and needs to be created from scratch in your working directory. We provide a
template configuration in ``.env.example`` that can be used as a starting point.
All variables relevant to the embedding server start with ``BBS_EMBEDDING_``.

Launch
~~~~~~
Next we need to start the embedding server by spawning a docker container
from the image that we built above:

.. code-block:: bash

      docker run \
        --detach \
        --rm \
        --publish <public_port>:8080 \
        --volume <real_path>:<container_path> \
        --env-file .env \
        --name bbs_embedding \
        bbs_embedding

Note the ``--volume`` parameter. It should be used to mount the paths to all model
checkpoints and logging directories that were specified in the configuration.
This parameter can be repeated multiple times to mount multiple paths. The flag
``--rm`` will ensure that the container is removed after it is stopped. The flag
``--publish`` specifies a port under which the server will run.

The server will take some time to initialize and to download pre-trained
models, so give it some time before trying to send requests.

Some embedding models are known to have had memory leaks. If memory consumption
starts becoming an issue one can try adding the following two parameters to
the ``docker run`` command:

- ``--memory 100g``: limit the memory of the container to 100GB, once this threshold is
  reached the container will be stopped.
- ``--restart unless-stopped``: make sure that the container is restarted automatically
  (unless stopped manually by using ``docker stop bbs_embedding``)

Use
~~~
Let's assume that ``bbs_url = http://<bbs_embedding_host>:<bbs_embedding_port>``.

To see a short welcome page with a few usage hints open the ``bbs_url`` in your browser.

To get a summary of the API interface send a ``POST`` request to ``bbs_url/help``. It
will respond with a JSON file containing instructions on how to use the
embedding API.
