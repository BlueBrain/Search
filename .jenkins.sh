#!/bin/bash

# BBSearch is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

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
module load archive/2020-11
module load python/3.7.4


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
tox -e py37 -- --color=yes

if [ "$BUILD_DOCS" = true ]; then
  .tox/docs/bin/python -c "import bbsearch; print('Installed version BBSearch: ', bbsearch.__version__)"
  cd docs || exit  # docs is already built by tox
  pip install -i https://bbpteam.epfl.ch/repository/devpi/simple docs-internal-upload
  docs-internal-upload --docs-path _build/html --metadata-path _build/html/metadata.md
fi