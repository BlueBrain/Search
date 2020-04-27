# BBP's Effort for the COVID-19 Challenge

## Data and Assets
The notebooks in this repository assume the existence of the following
folders in the root folder:
- `data`
- `assets`

This folders are not part of the repository and have to be created locally.

The folder `assests` contains assets that do not depend on the CORD-19 dataset
and do not have to be versioned. These are for example
- synonym lists
- pre-trained models (e.g. BioSentVec)

The data folder contains the CORD-19 dataset and all files generated/derived from it.
Since this dataset gets updated on a regular basis, different versions of it need to be
kept separated. Therefore the `data` folder contains sub-folders corresponding to
different versions and named by the date on which the CORD-19 dataset was downloaded.
The same subfolder should also contain all files derived from that dataset.

## The Docker Image
We provide a docker file, `docker/Dockerfile` that allows to build a docker
image with all package dependencies pre-installed.

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

```bash
$ docker build -f docker/Dockerfile-embedding_server -t embedding_server .
```

This will create a docker image with the tag `embedding_server:latest`.

Next a docker container has to be spawned form the image just created. In order
to function properly, the docker container needs to have access to the 
pre-trained models stored on disk. Currently the server only loads one model
from disk, namely the `BioSentVec` model, and will therefore look for the file
named `BioSentVec_PubMed_MIMICIII-bigram_d700.bin`. To have access to this file
a volume needs to be mounted into the container that contains the model file
and an environmental variable `ASSETS_PATH` that points to
the directory with the model file needs to be set.

Assuming that the model file is located in the folder `/raid/assets/`, all
of the above can be done by spawning the container with the following command:

```bash
$ docker run \
    --rm \
    --publish-all \
    --volume /raid/assets:/assets \
    --env ASSETS_PATH="/assets" \
    --name embedding_server \
    embedding_server
```

The server will take some time to initialize and to download pre-trained
models, so give it a bit of time before trying to send requests.

The flag `--rm` will ensure that the container is removed after it is stopped. The
flag `--publish-all` opens up a port for sending requests to the server. Docker selects
a random port number that is available, and to find out which port number was assigned
run 

```bash
$ docker port embedding_server
```

The output should be of the following form:

```
8080/tcp -> 0.0.0.0:32774
```

which means the the public port number is `32744`. Test if the server
started successfully by opening `localhost:32744` in your browser. You
might want to adjust the port number and the server URL if your docker
container is running on a remote host.

### Using the REST API
To request a sentence embedding, send a `GET` request to
`/v1/embed/<output_type>`. The output type can be either `json` or
`csv`. The request should be a JSON file of the following form:

```json
{
    "model": "<embedding model name>",
    "text": "<text>"
}
```

where `"text"` is the sentence you want to embed, and `"model"` can be one
of the following:
- `"USE"` (Universal Sentence Embedding)
- `"SBERT"` (Sentence BERT)
- `"SBIOBERT"` (Sentence BioBERT)
- `"BSV"` (BioSentVec)

The server will respond with either a JSON file or a CSV file, according to
the `output_type` that was requested. The JSON file will contain a key
named `"embedding"`, which holds the sentence embedding. The CSV file
will contain one row and a number of columns, each of which correspond to
an entry in the embedding vector.

See the notebook `notebooks/demo_embedding_api.ipynb` for sample code
demonstrating sending requests to the server.