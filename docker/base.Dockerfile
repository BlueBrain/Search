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
FROM nvidia/cuda:10.2-runtime

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
#
# There are two dev-packages that are installed:
# - libmysqlclient-dev
# - python3.7-dev
# This is intentional because otherwise "pip install SQLAlchemy[mysql]" breaks.
RUN \
apt-get update &&\
apt-get upgrade -y --no-install-recommends &&\
apt-get update &&\
TZ="Europe/Zurich" \
DEBIAN_FRONTEND="noninteractive" \
apt-get install -y --no-install-recommends \
    libbluetooth3 libbz2-1.0 libc6 libexpat1 \
    libffi6 libgdbm5 liblzma5 libncursesw5 libreadline7 \
    libsqlite3-0 libssl1.1 tk xz-utils zlib1g \
    gcc g++ build-essential make \
    curl git htop less man vim wget \
    libfontconfig1 libmysqlclient-dev
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
# The command "apt install python3-pip" does the following:
# - Install the pip module into the python version agnostic directory /usr/lib/python3/dist-packages
# - Install /usr/bin/python3.6
# - Link /usr/bin/python3 to /usr/bin/python3.6
# - Install the script /usr/bin/pip3 that has the /usr/bin/python3 shebang
#   and so load the pip module from python 3.6's site-packages
#
# How to make pip refer to the python 3.7 site-packages?
# Run "python3.7 -m pip install pip". This will
# - Use the pip module from the python version agnostic directory /usr/lib/python3/dist-packages
#   to install a pip module into the version specific directory /usr/local/lib/python3.7/dist-packages
#   (You can verify using "python3.7 -m site" that the version specific one has precedence)
# - Install the scripts
#   - /usr/local/bin/pip
#   - /usr/local/bin/pip3
#   - /usr/local/bin/pip3.7
#   which are all copies of each other and have the correct python 3.7 shebang
#
# The command "update-alternatives" makes the command "python" refer to
# "python3.7". Otherwise, "python" refers to "python2".
# 
RUN \
DEBIAN_FRONTEND="noninteractive" \
apt-get install -y --no-install-recommends \
python3.7-dev python3.7-venv python3-pip && \
python3.7 -m pip install --upgrade pip setuptools wheel && \
update-alternatives --install /usr/local/bin/python python /usr/bin/python3.7 0

# Install BBS requirements
COPY requirements.txt /tmp
COPY requirements-data_and_models.txt /tmp
RUN \
pip install --no-cache-dir -r /tmp/requirements.txt -r /tmp/requirements-data_and_models.txt &&\
rm /tmp/requirements.txt /tmp/requirements-data_and_models.txt

# Add and configure users
SHELL ["/bin/bash", "-c"]
ARG BBS_USERS
COPY docker/utils.sh /tmp
RUN \
. /tmp/utils.sh && \
groupadd -g 999 docker && \
create_users "${BBS_USERS},guest/1000" "docker" && \
configure_user

# Entry point
EXPOSE 8888
RUN mkdir /workdir && chmod a+rwX /workdir
WORKDIR /workdir
USER guest
ENTRYPOINT ["env"]
CMD ["bash", "-l"]
