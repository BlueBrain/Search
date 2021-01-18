.. BBSearch is a text mining toolbox focused on scientific use cases.
   Copyright (C) 2020  Blue Brain Project, EPFL.
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   You should have received a copy of the GNU Lesser General Public License
   along with this program. If not, see <https://www.gnu.org/licenses/>.

Servers
=======
This section describes how to launch and use various servers.

Search server
-------------

Setup
~~~~~
The REST API Server runs in a docker container, so in order to use it a docker
image needs to be build, and a container needs to be spawned.

To build the docker image open a terminal in the root directory of the project
and run the following command

.. code-block:: bash

    docker build -f docker/Dockerfile-embedding_server -t embedding_server .


This will create a docker image with the tag :code:`embedding_server:latest`.

Next a docker container has to be spawned from the image just created. In order
to function properly, the docker container needs to have access to the
pre-trained models stored on disk. Currently the server only loads one model
from disk, namely the :code:`BioSentVec` model, and will therefore look for the file
named :code:`BioSentVec_PubMed_MIMICIII-bigram_d700.bin`. As the path of this model
is by default the one on :code:`/raid` of DGX, this code can only run there. Assuming
this is the case, all of the above can be done by spawning the container
with the following command:

.. code-block:: bash

      docker run \
        --detach \
        --rm \
        --publish <public_port>:8080 \
        --volume /raid:/raid \
        --name bbs_embedding \
        embedding_server

The server will take some time to initialize and to download pre-trained
models, so give it a bit of time before trying to send requests.

The flag :code:`--rm` will ensure that the container is removed after it is stopped. The
flag :code:`--publish` specifies a port under which the server will run.

Some embedding models are known to have had memory leaks. If memory consumption
starts becoming an issue one can use the following command to partially alleviate
the issue:


.. code-block:: bash

      docker run \
        --detach \
        --memory 100g \
        --restart unless-stopped \
        --publish <public_port>:8080 \
        --volume /raid:/raid \
        --name bbs_embedding \
        embedding_server

The flag :code:`--memory 100g` will limit the memory of the container to 100 gigabytes.
Once this threshold is reached the container will be stopped. The flag
:code:`--restart unless-stopped` will then make sure that the container is restarted
automatically (unless stopped manually by using :code:`docker stop bbs_embedding`).

Usage
~~~~~
Let's assume that :code:`bbs_url = http://<bbs_embedding_host>:<bbs_embedding_port>`.

To see a short welcome page with a few usage hints open the :code:`bbs_url` in your browser.

To get a summary of the API interface send a :code:`POST` request to :code:`bbs_url/help`. It
will respond with a JSON file containing instructions on how to access the
embedding API.
