# Blue Brain Search is a text mining toolbox focused on scientific use cases.
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
# for example "python:3.7" can be used.
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
# https://github.com/docker-library/python/blob/master/3.7/buster/Dockerfile
ENV LANG=C.UTF-8

# Install system packages
#
# The environment variable $DEBIAN_FRONTENT is necessary to
# prevent apt-get from prompting for the timezone and keyboard
# layout configuration.
RUN apt-get update && apt-get upgrade -y && apt-get update
RUN \
DEBIAN_FRONTEND="noninteractive" \
TZ="Europe/Zurich" \
apt-get install -y \
    dpkg-dev gcc libbluetooth-dev libbz2-dev libc6-dev libexpat1-dev \
    libffi-dev libgdbm-dev liblzma-dev libncursesw5-dev libreadline-dev \
    libsqlite3-dev libssl-dev make tk-dev wget xz-utils zlib1g-dev
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

# Install Python 3.7 & pip 3.7
#
# The base image ("nvidia/cuda") does not have Python pre-installed. The
# following command can be omitted on images that already have Python, for
# example "python:3.7".
#
# The package "python3.7-pip" doesn't exist. The package "python3-pip" needs
# to be installed instead. After its installation:
#   - "pip" isn't an existing command,
#   - "pip3" refers to pip for Python 3.6,
#   - "pip3.7" isn't an existing command,
#   - "python3.7 -m pip" works.
#
# The command "update-alternatives" makes the command "python" refers to
# "python3.7". Otherwise, "python" refers to "python2".
RUN \
apt-get install -y python3.7 python3.7-dev python3.7-venv python3-pip && \
python3.7 -m pip install --upgrade pip setuptools wheel && \
update-alternatives --install /usr/local/bin/python py37 /usr/bin/python3.7 0

# Install Jupyter & IPython
RUN \
pip install ipython "jupyterlab<3.0.0" ipywidgets &&\
jupyter nbextension enable --py widgetsnbextension &&\
jupyter labextension install --no-build @jupyter-widgets/jupyterlab-manager &&\
jupyter labextension install --no-build @jupyterlab/toc
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

# Install the spaCy models.
RUN \
pip install \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz

# Add custom users specified in $BBS_USERS="user1/id1,user2/id2,etc"
ARG BBS_USERS
COPY ./docker/utils.sh /tmp
RUN \
. /tmp/utils.sh && \
groupadd -g 999 docker && \
create_users "$BBS_USERS" "docker" && \
add_aliases "/root" && \
improve_prompt "/root" "03" "36" && \
config_jupyter "root" "/root" && \
download_nltk "root"

# Add and select a non-root user (bbsuser)
RUN . /tmp/utils.sh && create_users "bbsuser/1000" "docker"
USER bbsuser
ENTRYPOINT ["env"]
CMD ["bash", "-l"]
