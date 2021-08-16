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

FROM bbs-base

USER root

# Install the app
ADD . /src
WORKDIR /src
RUN pip install .
RUN pip install gunicorn

# Set image version
LABEL maintainer="BBP-EPFL Machine Learning team <bbp-ou-machinelearning@groupes.epfl.ch>"
LABEL description="REST API Server for Blue Brain Search"

# Add a user
RUN useradd --create-home serveruser
WORKDIR /home/serveruser
USER serveruser

# Run the entry point
# Note the "timeout" parameter. That's to let the server initialisation finish before
# gunicorn decides that the worker is not responsive and restarts it again.
# Might think about a better solution in the future... (initialize in a threading?)
EXPOSE 8080
ENTRYPOINT [\
"gunicorn", \
"--bind", "0.0.0.0:8080", \
"--workers", "1", \
"--timeout", "7200", \
"bluesearch.entrypoint.search_server:get_search_app()"]
