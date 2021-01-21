# BBSearch is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# This image has the CUDA toolkit pre-installed and so the GPUs
# work out of the box. Just include the "--gpus all" flag in
# docker run.
#
# Note that this is a big development image. If you don't need
# the development packages consider changing the image tag to
# "10.2-base" or "10.2-runtime". See the information on docker
# hub for more details: https://hub.docker.com/r/nvidia/cuda
#
# If the GPU support is not necessary, then another image,
# for example "python:3.6" can be used.
FROM nvidia/cuda:10.2-devel

# ARGs are only visible at build time and can be provided in
# the docker-compose.yml file in the "args:" section or with the
# --build-arg parameter of docker build
ARG BBS_http_proxy
ARG BBS_https_proxy
ARG BBS_HTTP_PROXY
ARG BBS_HTTPS_PROXY

# ENVs are visible both at image build time and container run time.
# We want the http proxys to be visible in both cases and therefore
# set them equal to the values of the ARGs.
ENV http_proxy=$BBS_http_proxy
ENV https_proxy=$BBS_https_proxy
ENV HTTP_PROXY=$BBS_HTTP_PROXY
ENV HTTPS_PROXY=$BBS_HTTPS_PROXY

# Debian's default LANG=C breaks python3.
# See commends in the official python docker file:
# https://github.com/docker-library/python/blob/master/3.6/buster/Dockerfile
ENV LANG=C.UTF-8

# Install system packages
#
# The environment variable $DEBIAN_FRONTENT is necessary to
# prevent apt-get from prompting for the timezone and keyboard
# layout configuration.
#
# The first RUN command (that installs python3.6) is necessary because
# the base image (nvidia/cuda) does not have python pre-installed. This
# command can be omitted on images that already have python, for example
# "python:3.6"
RUN apt-get update && apt-get upgrade -y
RUN \
DEBIAN_FRONTEND="noninteractive" \
TZ="Europe/Zurich" \
apt-get install -y \
    dpkg-dev gcc libbluetooth-dev libbz2-dev libc6-dev libexpat1-dev \
    libffi-dev libgdbm-dev liblzma-dev libncursesw5-dev libreadline-dev \
    libsqlite3-dev libssl-dev make tk-dev wget xz-utils zlib1g-dev \
    python3.6-dev python3-setuptools python3-venv python3-pip
RUN \
apt-get install -y \
    gcc g++ build-essential \
    curl git htop less man vim
RUN \
curl -sL https://deb.nodesource.com/setup_10.x -o nodesource_setup.sh &&\
bash nodesource_setup.sh &&\
rm nodesource_setup.sh &&\
apt-get install -y nodejs
RUN \
DEBIAN_FRONTEND="noninteractive" \
apt-get install -y \
    libfontconfig1 wkhtmltopdf \
    libmysqlclient-dev default-libmysqlclient-dev

# Create soft links to python binaries and upgrade pip
RUN \
ln -s $(which python3) /usr/local/bin/python &&\
ln -s $(which pip3) /usr/local/bin/pip &&\
pip install --upgrade pip

# Install Jupyter & IPython
RUN \
pip install ipython jupyter jupyterlab ipywidgets &&\
jupyter nbextension enable --py widgetsnbextension &&\
jupyter labextension install @jupyter-widgets/jupyterlab-manager &&\
jupyter labextension install @jupyterlab/toc
EXPOSE 8888

# Install BBS requirements
COPY requirements.txt /tmp
RUN \
pip install Cython numpy &&\
pip install --no-cache-dir -r /tmp/requirements.txt &&\
rm /tmp/requirements.txt &&\
jupyter-lab build --name="BBS | Base"

# Download the scispaCy models
RUN \
pip install \
  https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_craft_md-0.2.5.tar.gz \
  https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_jnlpba_md-0.2.5.tar.gz \
  https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_bc5cdr_md-0.2.5.tar.gz \
  https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_bionlp13cg_md-0.2.5.tar.gz \
  https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_lg-0.2.5.tar.gz

# Configure Jupyter (for the root)
# Set --allow-root --no-browser --ip=0.0.0.0
RUN \
jupyter-lab --generate-config &&\
sed -i"" \
  -e "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '0.0.0.0'/g" \
  -e "s/#c.NotebookApp.open_browser = True/c.NotebookApp.open_browser = False/g" \
  -e "s/#c.NotebookApp.allow_root = False/c.NotebookApp.allow_root = True/g" \
  /root/.jupyter/jupyter_notebook_config.py

# Download the NLTK data (for the root)
RUN python -m nltk.downloader punkt stopwords

# Add and select a non-root user (bbsuser)
RUN groupadd -g 999 docker
RUN useradd --create-home --uid 1000 --gid docker bbsuser
USER bbsuser

# Configure Jupyter (for the bbsuser) 
# Set --no-browser --ip=0.0.0.0
RUN \
jupyter-lab --generate-config &&\
sed -i"" \
  -e "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '0.0.0.0'/g" \
  -e "s/#c.NotebookApp.open_browser = True/c.NotebookApp.open_browser = False/g" \
  /home/bbsuser/.jupyter/jupyter_notebook_config.py

# Download the NLTK data (for the bbsuser)
RUN python -m nltk.downloader punkt stopwords

WORKDIR /home/bbsuser
ENTRYPOINT ["bash"]

