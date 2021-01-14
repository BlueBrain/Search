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
7. Initialize the search, mining, and notebooks servers
8. Open the example notebook.

Before proceeding, four things need to be noted.

First, at the moment, these instructions suppose the machine is inside Blue
Brain's network.

Second, the setup of Blue Brain Search requires the launch of 4 servers
(database, search, mining, notebooks). The instructions are supposed to be
executed on a powerful remote machine and the notebooks are supposed to be
accessed from a personal local machine through the network.

Third, the ports, the Docker image names, and the Docker container names are
modified (see below) to safely test the instructions on a machine where the
Docker images would have already been built, the Docker containers would
already run, and the servers would already run.

Fourth, if you are in a production setting, the database password and the
notebooks server token should be changed, the prefix `test_` should be removed
from the Docker image and container names, the `sed` commands should be
omitted, and the second digit of the ports should be replaced by `8`.

### Prerequisites

The instructions are written for GNU/Linux machines. However, any machine
with the equivalent of `git`, `wget`, `tar`, `mkdir` and `sed` (optional)
could be used.

The software named `Docker is also needed. To install `Docker`, please refer to
the [official Docker documentation](https://docs.docker.com/engine/install/).

An optional part is using the programming language `Python` and its package
manager `pip`. To install `Python` and `pip` please refer to the
[official Python documentation](https://wiki.python.org/moin/BeginnersGuide/Download).

Otherwise, let's define the environment variables commons to the instructions.

```bash
export DIRECTORY=$(pwd)

export DATABASE_PORT=8953
export SEARCH_PORT=8950
export MINING_PORT=8952
export NOTEBOOKS_PORT=8954

export DATABASE_PASSWORD=1234
export NOTEBOOKS_TOKEN=1a2b3c4d

export http_proxy=http://bbpproxy.epfl.ch:80/
export https_proxy=http://bbpproxy.epfl.ch:80/
```

Then, please clone the Blue Brain Search repository.

```bash
git clone https://github.com/BlueBrain/BlueBrainSearch
```

### Retrieve the documents

This will download and decompress the CORD-19 version corresponding to the
version 73 on Kaggle. Note that the data are around 7 GB.

```bash
export CORD19_VERSION=2021-01-03
export CORD19_ARCHIVE=cord-19_${CORD19_VERSION}.tar.gz
```

```bash
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/$CORD19_ARCHIVE
tar xf $CORD19_ARCHIVE
tar xf $CORD19_VERSION/document_parses.tar.gz -C $CORD19_VERSION
```

CORD-19 contains more than 400,000 publications. The next sections could run
for several hours, even days, depending on the power of the machine.

For testing purposes, you might want to consider a subset of the CORD-19. The
following code select around 1,400 articles about *glucose* and *risk factors*:

```bash
cd $CORD19_VERSION
mv metadata.csv metadata.csv.original
pip install pandas
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
export DATABASE_STORE=mysql_data
export DATABASE_NAME=cord19
export DATABASE_URL=$HOSTNAME:$DATABASE_PORT/$DATABASE_NAME
mkdir $DATABASE_STORE
cd BlueBrainSearch
```

This will build a Docker image where MySQL is installed. Besides, this will
launch using this image a MySQL server running in a Docker container.

```bash
docker build \
  --build-arg http_proxy --build-arg https_proxy  \
  -f docker/mysql.Dockerfile -t test_bbs_mysql .
```

NB:`HTTP_PROXY` and `HTTPS_PROXY`, in upper case, are not working here.

```bash
docker run \
  --publish $DATABASE_PORT:3306 \
  --volume $DIRECTORY/$DATABASE_STORE:/var/lib/mysql \
  --env MYSQL_ROOT_PASSWORD=$DATABASE_PASSWORD \
  --detach \
  --name test_bbs_mysql test_bbs_mysql
```

NB: The paths need to be absolute for `--volume`.

You will be asked to enter the MySQL root password defined above
(`DATABASE_PASSWORD`).

```bash
docker exec --interactive --tty test_bbs_mysql bash
mysql -u root -p
```

Please replace `<database name>` by the value of `DATABASE_NAME`.

```sql
CREATE DATABASE <database name>;
CREATE USER 'guest'@'%' IDENTIFIED WITH mysql_native_password BY 'guest';
GRANT SELECT ON <database name>.* TO 'guest'@'%';
exit;
```

Please exit the interactive session on the `test_bbs_mysql` container.

```bash
exit
```

### Install Blue Brain Search

This will build a Docker image where Blue Brain Search is installed. Besides,
this will launch using this image an interactive session in a Docker container.
The immediate next sections will need to be run in this session.

```bash
docker build \
  --build-arg BBS_HTTP_PROXY=$HTTP_PROXY --build-arg BBS_http_proxy=$HTTP_PROXY \
  --build-arg BBS_HTTPS_PROXY=$HTTPS_PROXY --build-arg BBS_https_proxy=$HTTPS_PROXY \
  -f docker/base.Dockerfile -t test_bbs_base .
```

NB: At the moment, `HTTP_PROXY`, `HTTPS_PROXY`, `http_proxy`, and `https_proxy`
are not working here.

```bash
docker run \
  --volume /raid:/raid \
  --env CORD19_VERSION --env DATABASE_URL --env DIRECTORY \
  --gpus all \
  --interactive --tty --rm --user root --workdir $DIRECTORY \
  --name test_bbs_base test_bbs_base
pip install --editable ./BlueBrainSearch
```

NB: At the moment, `--editable` is needed for `DVC.load_ee_models_library()`
and `--user root` is needed for `pip`.

### Create the database

You will be asked to enter the MySQL root password defined above
(`DATABASE_PASSWORD`).

If you are using the CORD-19 subset of around 1,400 articles, this would take
around 3 minutes.

```bash
create_database \
  --data-path $CORD19_VERSION \
  --database-url $DATABASE_URL
```

### Compute the sentence embeddings

FIXME retrieve model checkpoint or use known model

If you are using the CORD-19 subset of around 1,400 articles, this would take
around 1 minute (on 2 Tesla V100 16 GB).

```bash
compute_embeddings SentTransformer embeddings.h5 \
  --checkpoint biobert_nli_sts_cord19_v1 \
  --db-url $DATABASE_URL \
  --gpus 0,1 \
  --h5-dataset-name 'BioBERT NLI+STS CORD-19 v1' \
  --n-processes 2
```

### Create the mining cache

FIXME use utils.sh

```bash
cd BlueBrainSearch/data_and_models/pipelines/ner
dvc pull ee_models_library.csv.dvc
for i in 1 2 3 4 5;
do dvc pull ../../models/ner_er/model$i;
done;
cd $DIRECTORY
```

You will be asked to enter the MySQL root password defined above
(`DATABASE_PASSWORD`).

If you are using the CORD-19 subset of around 1,400 articles, this would take
around 4 minutes.

```bash
create_mining_cache \
  --database-url $DATABASE_URL \
  --verbose
```

NB: At the moment, `--verbose` is needed to show the `INFO` logs.

### Initialize the search, mining, and notebooks servers

Please exit the interactive session of the `test_bbs_base` container.

```bash
exit
```

#### Search server

```bash
sed -i 's/bbsearch.entrypoints/bbsearch.entrypoint/g' docker/search.Dockerfile
sed -i 's/ bbs_/ test_bbs_/g' docker/search.Dockerfile
docker build \
  -f docker/search.Dockerfile -t test_bbs_search .
```

NB: At the moment, `bbsearch.entrypoints` needs to be renamed into `bbsearch.entrypoint`.

```bash
export BBS_SEARCH_MYSQL_URL=$DATABASE_URL
export BBS_SEARCH_MYSQL_USER=guest  # FIXME ? parametrize
export BBS_SEARCH_MYSQL_PASSWORD=guest  # FIXME ? parametrize

export BBS_SEARCH_MODELS='BioBERT NLI+STS CORD-19 v1'  # FIXME ? reuse
export BBS_SEARCH_MODELS_PATH=$DIRECTORY
export BBS_SEARCH_EMBEDDINGS_PATH=$DIRECTORY/embeddings.h5  # FIXME ? reuse
```

```bash
docker run \
  --publish $SEARCH_PORT:8080 \
  --volume /raid:/raid \
  --env BBS_SEARCH_MYSQL_URL --env BBS_SEARCH_MYSQL_USER --env BBS_SEARCH_MYSQL_PASSWORD \
  --env BBS_SEARCH_MODELS --env BBS_SEARCH_MODELS_PATH --env BBS_SEARCH_EMBEDDINGS_PATH \
  --detach \
  --name test_bbs_search test_bbs_search
```

#### Mining server

```bash
sed -i 's/bbsearch.entrypoints/bbsearch.entrypoint/g' docker/mining.sh
sed -i 's/ bbs_/ test_bbs_/g' docker/mining.Dockerfile
docker build \
  -f docker/mining.Dockerfile -t test_bbs_mining .
```

NB: At the moment, `bbsearch.entrypoints` needs to be renamed into `bbsearch.entrypoint`.

```bash
export BBS_SSH_USERNAME=$(id --user --name)

export BBS_MINING_DB_TYPE=mysql
export BBS_MINING_MYSQL_URL=$DATABASE_URL
export BBS_MINING_MYSQL_USER=guest  # FIXME ? parametrize
export BBS_MINING_MYSQL_PASSWORD=guest  # FIXME ? parametrize
```

```bash
docker run \
  --publish $MINING_PORT:8080 \
  --volume /raid:/raid \
  --env BBS_SSH_USERNAME \
  --env BBS_MINING_DB_TYPE --env BBS_MINING_MYSQL_URL --env BBS_MINING_MYSQL_USER --env BBS_MINING_MYSQL_PASSWORD \
  --detach \
  --name test_bbs_mining test_bbs_mining
```

#### Notebooks server

```bash
sed -i 's/cord19_v47/'$DATABASE_NAME'/g' notebooks/BBS_BBG_poc.ipynb
export MYSQL_DB_URI=$HOSTNAME:$DATABASE_PORT
export SEARCH_ENGINE_URL=http://$HOSTNAME:$SEARCH_PORT
export TEXT_MINING_URL=http://$HOSTNAME:$MINING_PORT
```

```bash
docker run \
  --publish $NOTEBOOKS_PORT:8888 \
  --volume /raid:/raid \
  --env NOTEBOOKS_TOKEN \
  --env MYSQL_DB_URI --env SEARCH_ENGINE_URL --env TEXT_MINING_URL \
  --interactive --tty --rm --user root --workdir $DIRECTORY \
  --name test_bbs_notebooks test_bbs_base
pip install ./BlueBrainSearch
jupyter lab BlueBrainSearch/notebooks --NotebookApp.token=$NOTEBOOKS_TOKEN
```

To detach from the Docker container, please hit `CTRL+P` and then `CTRL+Q`.

NB: At the moment, `--user root` is needed for the widgets to write in
`BlueBrainSearch/notebooks/untracked/.widgets_checkpoints`.

### Open the example notebook

```bash
cd $DIRECTORY
echo http://$HOSTNAME:$NOTEBOOKS_PORT/lab/tree/BBS_BBG_poc.ipynb
```

To open the example notebook, please open the link returned above in a browser,
then please enter the token above (`NOTEBOOKS_TOKEN`), and finally please click
on `Log in`.

FIXME mining cache table name is inconsistent => crash

*Voilà!* You could now use the graphical interface.




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
