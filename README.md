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
this is true even if the query and the sentences from the documents do not
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

At the moment, please note that these instructions suppose you have access to
Blue Brain resources.

Please also note that in this *Getting Started* the ports, the Docker
image names, and the Docker container names are modified to safely test
the instructions on a machine where BBS Docker images would have already
been built, BBS Docker containers would already run, and BBS servers would
already run. If this is not the case, the prefix `test_` could be removed
from the Docker image and container names and the `sed`commands could be
omitted. For the ports, the default values start with `88` and not `89`.

### Prerequisites

```bash
export DIRECTORY=$(pwd)
git clone https://github.com/BlueBrain/BlueBrainSearch
```

### Retrieve the documents

This will download and decompress the CORD-19 version corresponding to the
version 73 on Kaggle. Note that the data are around 7 GB.

```bash
export VERSION=2021-01-03
export ARCHIVE=cord-19_${VERSION}.tar.gz
```

```bash
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/$ARCHIVE
tar xf $ARCHIVE
tar xf $VERSION/document_parses.tar.gz -C $VERSION
```

CORD-19 contains more than 400,000 publications. The next sections could run
for several hours, even days, depending on your computing power. For testing
purposes, you might want to consider a subset of CORD-19. The following code
select around 1,400 articles about *glucose* and *risk factors*:

```bash
cd $VERSION
mv metadata.csv metadata.csv.original
python
```

```python
import pandas as pd
metadata = pd.read_csv('metadata.csv.original')
sample = metadata[
    metadata.title.str.contains('glucose', na=False)
    | metadata.title.str.contains('risk factor', na=False)
  ]
print('The subset contains', sample.shape[0], 'articles.')
sample.to_csv('metadata.csv', index=False)
exit()
```

```bash
cd ..
```

### Initialize the database server

```bash
export PORT=8953
export PASSWORD=1234
export URL=$(hostname):$PORT/cord19
```

This will build a Docker image where MySQL is installed. Besides, this will
launch using this image a MySQL server running in a Docker container.

FIXME `docker build` fails because of `Connection refused`: proxy configuration issue?

```bash
mkdir mysql_data
cd BlueBrainSearch
docker build -f docker/mysql.Dockerfile -t test_bbs_mysql .
docker run \
  --network=test_bbs_network -p $PORT:3306 \
  --volume $DIRECTORY/mysql_data:/var/lib/mysql \
  --env MYSQL_ROOT_PASSWORD=$PASSWORD \
  --detach \
  --name test_bbs_mysql test_bbs_mysql
```

You will be asked to enter the MySQL root password defined above (`PASSWORD`).

```bash
docker exec --interactive --tty test_bbs_mysql bash
mysql -u root -p
> CREATE DATABASE cord19;
> CREATE USER 'guest'@'%' IDENTIFIED WITH mysql_native_password BY 'guest';
> GRANT SELECT ON cord19.* TO 'guest'@'%';
> exit;
exit
```

### Install Blue Brain Search

This will build a Docker image where Blue Brain Search is installed. Besides,
this will launch using this image an interactive session in a Docker container.
The next steps of this *Getting Started* will need to be run in this session.

FIXME ? needs `--env-file .env`
FIXME ? .env used for BBS_HTTP_PROXY BBS_http_proxy BBS_HTTPS_PROXY BBS_https_proxy

```bash
docker build \
  --build-arg BBS_HTTP_PROXY --build-arg BBS_http_proxy \
  --build-arg BBS_HTTPS_PROXY --build-arg BBS_https_proxy \
  -f docker/base.Dockerfile -t test_bbs_base .
docker run \
  --network=test_bbs_network \
  --volume /raid:/raid \
  --env VERSION --env URL --env DIRECTORY \
  --gpus all \
  --interactive --tty --rm --user root --workdir $DIRECTORY \
  --name test_bbs_base test_bbs_base
pip install --editable ./BlueBrainSearch
```

NB: At the moment, `--editable` is needed for DVC.load_ee_models_library()`.

### Create the database

You will be asked to enter the MySQL root password defined above (`PASSWORD`).

```bash
create_database \
  --data-path $VERSION \
  --database-url $URL
```

### Compute the sentence embeddings

```bash
compute_embeddings SentTransformer embeddings.h5 \
  --checkpoint biobert_nli_sts_cord19_v1 \
  --db-url $URL \
  --gpus 0,1 \
  --h5-dataset-name 'BioBERT NLI+STS CORD-19 v1' \
  --n-processes 2
```

### Create the mining cache

```bash
cd BlueBrainSearch/data_and_models/pipelines/ner
dvc pull ee_models_library.csv.dvc
for i in 1 2 3 4 5;
do dvc pull ../../models/ner_er/model$i;
done;
cd $DIRECTORY
```

You will be asked to enter the MySQL root password defined above (`PASSWORD`).

```bash
create_mining_cache \
  --database-url $URL \
  --verbose
```

NB: At the moment, `--verbose` is needed to show the INFO logs.

### Initialize the search and mining servers

Exit the interactive session of the `test_bbs_base` Docker container.

```bash
exit
```

FIXME create .env from example
FIXME should include BBS_SSH_USERNAME

*Common elements*

```bash
sed -i 's/ bbs_/ test_bbs_/g' docker/search.Dockerfile
sed -i 's/ bbs_/ test_bbs_/g' docker/mining.Dockerfile
```

*Search server*

```bash
docker build -f docker/search.Dockerfile -t test_bbs_search .
docker run \
  --network=test_bbs_network -p 8950:8080 \
  --volume /raid:/raid \
  --env-file .env \
  --detach \
  --name test_bbs_search test_bbs_search
```

*Mining server*

```bash
docker build -f docker/mining.Dockerfile -t test_bbs_mining .
docker run \
  --network=test_bbs_network -p 8952:8080 \
  --volume /raid:/raid \
  --env-file .env \
  --detach \
  --name test_bbs_mining test_bbs_mining
```

### Open the Jupyter notebook

FIXME ? cd $DIRECTORY (now in $DIRECTORY/BlueBrainSearch) vs docker run test_bbs_base

```bash
jupyter lab notebooks/BBS_BBG_poc.ipynb
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
