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

FROM continuumio/miniconda3:4.9.2

ENV HTTP_PROXY='http://bbpproxy.epfl.ch:80/'
ENV HTTPS_PROXY='http://bbpproxy.epfl.ch:80/'
ENV http_proxy='http://bbpproxy.epfl.ch:80/'
ENV https_proxy='http://bbpproxy.epfl.ch:80/'

# Update conda, install additional system packages
RUN true \
	&& conda update conda \
	&& apt-get update \
	&& apt-get install -y gcc g++ build-essential vim libfontconfig1 wkhtmltopdf
RUN conda install -c carta mysqlclient

# Install Prodigy -- assume that we have Prodigy wheel there!
RUN mkdir -p /bbs/tmp
COPY prodigy-1.10.7-cp36.cp37.cp38-cp36m.cp37m.cp38-linux_x86_64.whl /bbs/tmp
RUN pip install /bbs/tmp/prodigy-1.10.7-cp36.cp37.cp38-cp36m.cp37m.cp38-linux_x86_64.whl

# Instal BlueBrainSearach -- revision can be a branch, sha, or tag
ARG BBS_REVISION=v0.1.1
ADD . /src
WORKDIR /src
RUN git checkout $BBS_REVISION
# remove ruamel-yaml: https://github.com/pypa/pip/issues/5247#issuecomment-381550610
RUN rm -rf /opt/conda/lib/python3.8/site-packages/ruamel*
RUN pip install -r requirements.txt
RUN pip install $PWD


EXPOSE 8888

RUN groupadd -g 999 docker
RUN useradd --create-home --uid 1000 --gid docker bbsuser

WORKDIR /bbs
RUN rm -rf /bbs/tmp
ENTRYPOINT ["/bin/bash"]
