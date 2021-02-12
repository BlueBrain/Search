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

FROM ubuntu:latest

LABEL maintainer="Stanislav Schmidt <stanislav.schmidt@epfl.ch>"
LABEL version="1.0"
LABEL description="CoreNLP Server"

# ENV HTTP_PROXY='http://bbpproxy.epfl.ch:80/'
# ENV HTTPS_PROXY='http://bbpproxy.epfl.ch:80/'
# ENV http_proxy='http://bbpproxy.epfl.ch:80/'
# ENV https_proxy='http://bbpproxy.epfl.ch:80/'

# Install git, gcc, and g++
RUN apt-get update && apt-get install -y \
	default-jre \
	unzip \
	wget

# Download and install CoreNLP 4.0.0 (2020-04-19)
# See https://stanfordnlp.github.io/CoreNLP/history.html
# COPY corenlp_download.zip .
RUN true \
	&& export CORENLP_VERSION=4.0.0 \
	&& URL=http://nlp.stanford.edu/software/stanford-corenlp-${CORENLP_VERSION}.zip \
	&& wget -q --show-progress --progress=bar:force -O corenlp_download.zip $URL 2>&1 \
	&& unzip -q -j corenlp_download.zip -d /corenlp \
	&& rm corenlp_download.zip


# Add a user
RUN useradd corenlpuser
WORKDIR /corenlp
USER corenlpuser

# Expose a port
EXPOSE 9000

ENTRYPOINT exec java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -annotators "tokenize,ssplit,pos,depparse"

