#!/bin/bash

set -e
set -x


if [ "$BUILD_DOCS" = true ]; then
  git checkout master
  git pull
fi

if [[ -d .tox ]]; then
  echo "Removing .tox folder to prevent caching of enviroments"
  rm -rf .tox
fi

if [[ -d venv ]]; then
  echo "Removing venv folder to prevent caching of enviroments"
  rm -rf venv
fi

# Load modules
module load archive/2019-01
module load python/3.6.5


# Install tox in a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install tox

# Run tox environments
pip install --upgrade black isort
isort --version
isort --honor-noqa --diff --profile black setup.py src tests
tox -e lint -vv
tox -e docs
tox -e py36 -- --color=yes

if [ "$BUILD_DOCS" = true ]; then
  .tox/docs/bin/python -c "import bbsearch; print('Installed version BBSearch: ', bbsearch.__version__)"
  cd docs || exit  # docs is already built by tox
  pip install -i https://bbpteam.epfl.ch/repository/devpi/simple docs-internal-upload
  docs-internal-upload --docs-path _build/html --metadata-path _build/html/metadata.md
fi