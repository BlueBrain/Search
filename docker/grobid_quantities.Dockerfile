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
LABEL description="GROBID Quantities Server"

# ENV HTTP_PROXY='http://bbpproxy.epfl.ch:80/'
# ENV HTTPS_PROXY='http://bbpproxy.epfl.ch:80/'
# ENV http_proxy='http://bbpproxy.epfl.ch:80/'
# ENV https_proxy='http://bbpproxy.epfl.ch:80/'


# Install java, git, unzip and wget
RUN apt-get update && apt-get install -y \
	default-jre \
	git \
	unzip \
	wget

# Add a user
RUN useradd --create-home grobiduser
WORKDIR /home/grobiduser
USER grobiduser

# Download and install GROBID
RUN true \
	&& git clone --depth=1 https://github.com/kermitt2/grobid.git grobid \
	&& cd grobid \
#	&& echo "systemProp.https.proxyHost=bbpproxy.epfl.ch" >> gradle.properties \
	&& ./gradlew clean install

# Download and install GROBID Quantities
RUN true \
	&& git clone --depth=1 https://github.com/kermitt2/grobid-quantities.git grobid/grobid-quantities \
	&& cd grobid/grobid-quantities/ \
#	&& echo "\nsystemProp.https.proxyHost=bbpproxy.epfl.ch" >> gradle.properties \
	&& ./gradlew copyModels \
	&& ./gradlew clean install

# Expose a port and set working directory
EXPOSE 8060
WORKDIR /home/grobiduser/grobid/grobid-quantities

ENTRYPOINT exec java -jar $(find build/libs -name "grobid-*onejar.jar") server resources/config/config.yml
# ENTRYPOINT exec java -jar build/libs/grobid-quantities-0.6.1-SNAPSHOT-onejar.jar server resources/config/config.yml

