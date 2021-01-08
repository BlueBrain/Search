# BBP's Effort for the COVID-19 Challenge

[![GitHub](https://img.shields.io/github/license/BlueBrain/BlueBrainSearch)](https://github.com/BlueBrain/BlueBrainSearch/blob/master/LICENSE)
[![Build Status](https://travis-ci.com/BlueBrain/BlueBrainSearch.svg?token=DiSGfujs1Bszyq2UxayG&branch=master)](https://travis-ci.com/BlueBrain/BlueBrainSearch)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/BlueBrain/BlueBrainsearch)](https://github.com/BlueBrain/BlueBrainSearch/releases)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)


<!--- TODO: add code coverage badge --->
<!--- TODO: add binder / colab --->
<!--- TODO: add pypi version(s) --->


Blue Brain Search is a Python package to find sentences semantically similar
to a query in documents and to extract structured information from the
returned and ranked documents.

At the moment, the documents which are used are scientific publications about
COVID-19 from the [CORD-19](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge).


## Graphical Interface

The graphical interface is composed of [widgets](https://github.com/jupyter-widgets/ipywidgets)
to be used in [Jupyter notebooks](https://github.com/jupyterlab/jupyterlab).

For the graphical interface to work, the steps of the [Getting Started](getting-started)
should have been completed successfully.

### Find documents based on sentence semantic similarity

To find sentences semantically similar to the query *'Glucose is a risk factor
for COVID-19'* in the documents, you could just click on the blue button named
`Search Literature!`. You could also enter the query of your choice by editing
the text in the top field named `Query`. 

The returned results are ranked by decreasing semantic similarity. This means
that the first results have a similar meaning to the query. Thanks to the
state-of-the-art approach based on deep learning used by Blue Brain Search,
this is true even if the query and the sentences from the documents don't
share the same words (e.g. they are synonyms, they have a similar meaning, ...).

![Search Widget](screenshots/search_widget.png)

### Extract structured information from documents

The extraction could be done either on documents found by the search above or
on the text content of a document pasted in the widget.

#### Found documents

To extract structured information from the found documents, you could just
click on the blue button named `Mine Selected Articles!`.

At the moment, the returned results are named entities. For each named entity,
the structured information is: the mention (e.g. 'COVID-19'), the type
(e.g. 'DISEASE'), and its location up to the character in the document.

![Mining Widget (articles)](screenshots/mining_widget_articles.png)

#### Pasted document content

This is also possible to extract structured information from the pasted
content of a document. To switch to this mode, you could just click on the
tab named `Mine Text`. Then, you could launch the extracting by just clicking 
on the blue button named `Mine This Text!`. You could also enter the content
of your choice by editing the text field. 

![Mining Widget (text)](screenshots/mining_widget_text.png)


## Getting Started

There are 8 steps which need to be done in order:

1. Retrieve the documents.
2. Initialize the database server.
3. Install Blue Brain Search.
4. Create the database.
5. Compute the sentence embeddings.
6. Create the mining cache.
7. Initialize the search and mining servers.
8. Open the Jupyter notebook.

At the moment, note that these instructions suppose you have access to
Blue Brain resources.

### Retrieve the documents

This will download the CORD-19 version corresponding to version 73 on Kaggle.

```bash
export VERSION=2021-01-03
export CORD19=cord-19_$VERSION
```

```bash
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/${CORD19}.tar.gz
tar xf ./$CORD19
```

### Initialize the database server

```bash
export PORT=8853
export PASSWORD=1a2b3c4d
```

```bash
mkdir ./mysql_data
docker build -f ./docker/mysql.Dockerfile -t bbs_mysql .
docker run -d -v ./mysql_data:/var/lib/mysql -p $PORT:3306 -e MYSQL_ROOT_PASSWORD=$PASSWORD \
  --name bbs_mysql bbs_mysql
```

```bash
export HOST=dgx1.bbp.epfl.ch
export DATABASE=$HOST:$PORT/$CORD19
```

### Install Blue Brain Search

This will build a Docker image where Blue Brain Search is installed. Besides,
this will launch using this image an interactive session in a Docker container.
The next steps of this *Getting Started* will need to be run in this session.

FIXME needs `--env-file .env`?

```bash
git clone https://github.com/BlueBrain/BlueBrainSearch
cd BlueBrainSearch
docker build -f ./docker/base.Dockerfile -t bbs_base .
docker run -it -v /raid:/raid --rm --user root --gpus all \
  --name bbs_base bbs_base
pip install ./BlueBrainSearch
```

### Create the database

```bash
create_database \
  --data-path ./$CORD19 \
  --database-url $DATABASE
```

### Compute the sentence embeddings

```bash
compute_embeddings SentTransformer embeddings.h5 \
  --checkpoint ./biobert_nli_sts_cord19_v1 \
  --db-url $DATABASE \
  --gpus 0,1,2,3,4,5 \
  --h5-dataset-name 'BioBERT NLI+STS CORD-19 v1' \
  --n-processes 6
```

### Create the mining cache

```bash
create_mining_cache \
  --database-url $DATABASE
```

### Initialize the search and mining servers

Exit the interactive session of the `bbs_base` container with `CTRL+C`.

FIXME create .env from example
FIXME needs `--env-file .env`?

```bash
docker-compose up
```

### Open the Jupyter notebook

```bash
jupyter lab ./notebooks/BBS_BBG_poc.ipynb
```


## FIXME (below)

## Installation (virtual environment)
We currently support the following Python versions.
Make sure you are using one of them.
 - Python 3.6
 - Python 3.7
 - Python 3.8

Before installation, please make sure you have a recent `pip` installed (`>=19.1`)

```bash
pip install --upgrade pip
```

To install `bbsearch` run

```bash
pip install .
```

## Installation (Docker)
We provide a docker file, `docker/Dockerfile` that allows to build a docker
image with all dependencies of `BlueBrainSearch` pre-installed. Note that
`BlueBrainSearch` itself is not installed, which needs to be done manually
on each container that is spawned.

To build the docker image open a terminal in the root directory of the project
and run the following command.

```bash
$ docker build -f docker/Dockerfile -t bbs .
```

Then, to spawn an interactive container session run
```bash
$ docker run -it --rm bbs
```

## Documentation
We provide additional information on the package in the documentation. It can be
generated by sphinx. Make sure to install the `bbsearch` package with `dev`
extras to get the necessary dependencies.

```bash
pip install -e .[dev]
```


To generate the documentation run

```bash
cd docs
make clean && make html
```

You can then open it in a browser by navigating to `docs/_build/html/index.html`.

## Testing
We use `tox` to run all our tests. Running `tox` in the terminal will execute
the following environments:
- `lint`: code style and documentation checks
- `docs`: test doc build
- `check-packaging`: test packaging
- `py36`: run unit tests (using pytest) with python3.6
- `py37`: run unit tests (using pytest) with python3.7
- `py38`: run unit tests (using pytest) with python3.8

Each of these environments can be run separately using the following syntax:
```shell script
$ tox -e lint
```
This will only run the `lint` environment.

We provide several convenience tox environments that are not run automatically
and have to be triggered by hand:
- `format`
- `benchmarks`

The `format` environment will reformat all source code using `isort` and `black`.

The `benchmark` environment will run pre-defined pytest benchmarks. Currently
these benchmarks only test various servers and therefore need to know the server
URL. These can be passed to `tox` via the following environment variables:
```shell script
export EMBEDDING_SERVER=http://<url>:<port>
export MINING_SERVER=http://<url>:<port>
export MYSQL_SERVER=<url>:<port>
export SEARCH_SERVER=http://<url>:<port>
```
If a server URL is not defined, then the corresponding tests will be skipped.

It is also possible to provide additional positional arguments to pytest using
the following syntax:
```shell script
$ tox -e benchmarks -- <positional arguments>
```
for example:
```shell script
$ tox -e benchmarks -- \
  --benchmark-histogram=my_histograms/benchmarks \
  --benchmark-max-time=1.5 \
  --benchmark-min-rounds=1
```
See `pytest --help` for additional options.
