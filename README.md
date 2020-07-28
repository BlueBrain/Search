# BBP's Effort for the COVID-19 Challenge

[![Build Status](https://travis-ci.com/BlueBrain/BlueBrainSearch.svg?token=DiSGfujs1Bszyq2UxayG&branch=master)](https://travis-ci.com/BlueBrain/BlueBrainSearch)

## Installation
Before installation, please make sure you have a recent `pip` installed (`>=19.1`)

```bash
pip install --upgrade pip
```

To install `bbsearch` run

```bash
pip install .
```

## The Docker Image
We provide a docker file, `docker/Dockerfile` that allows to build a docker
image with all dependencies of `BlueBrainSearch` pre-installed. Note that
`BlueBrainSearch` itself is not installed, which needs to be done manually
on each container that is spawned.

To build the docker image open a terminal in the root directory of the project
and run the following command

```bash
$ docker build -f docker/Dockerfile -t bbs .
```

To spawn an interactive container session run
```bash
$ docker run -it --rm bbs
```

## Sentence Embedding REST API
### Setting Up the Server
The REST API Server runs in a docker container, so in order to use it a docker
image needs to be build, and a container needs to be spawned.

To build the docker image open a terminal in the root directory of the project
and run the following command

```shell script
$ docker build -f docker/Dockerfile-embedding_server -t embedding_server .
```

This will create a docker image with the tag `embedding_server:latest`.

Next a docker container has to be spawned from the image just created. In order
to function properly, the docker container needs to have access to the 
pre-trained models stored on disk. Currently the server only loads one model
from disk, namely the `BioSentVec` model, and will therefore look for the file
named `BioSentVec_PubMed_MIMICIII-bigram_d700.bin`. As the path of this model
is by default the one on `/raid` of DGX, this code can only run there. Assuming
this is the case, all of the above can be done by spawning the container
with the following command:

```shell script
$ docker run \
    --detach \
    --rm \
    --publish <public_port>:8080 \
    --volume /raid:/raid \
    --name bbs_embedding \
    embedding_server
```

The server will take some time to initialize and to download pre-trained
models, so give it a bit of time before trying to send requests.

The flag `--rm` will ensure that the container is removed after it is stopped. The
flag `--publish` specifies a port under which the server will run.

Some embedding models are known to have had memory leaks. If memory consumption
starts becoming an issue one can use the following command to partially alleviate
the issue:

```shell script
$ docker run \
    --detach \
    --memory 100g \
    --restart unless-stopped \
    --publish <public_port>:8080 \
    --volume /raid:/raid \
    --name bbs_embedding \
    embedding_server
```

The flag `--memory 100g` will limit the memory of the container to 100 gigabytes.
Once this threshold is reached the container will be stopped. The flag
`--restart unless-stopped` will then make sure that the container is restarted
automatically (unless stopped manually by using `docker stop bbs_embedding`).

### Using the Embedding Server REST API
Let's assume that `bbs_url = http://<bbs_embedding_host>:<bbs_embedding_port>`.

To see a short welcome page with a few usage hints open the `bbs_url` in your browser.

To get a summary of the API interface send a `POST` request to `bbs_url/help`. It
will respond with a JSON file containing instructions on how to access the
embedding API.
