#!/bin/bash

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

source /src/docker/utils.sh
# If $BBS_SSH_USERNAME is empty then this is a no-op
dvc_configure_ssh_remote_authentication "$BBS_SSH_USERNAME"
# Not usable in README as it works only when inside the `bbs_` containers.
# If $DATA_DIR is empty then this will fail
dvc_pull_models "$BBS_DATA_AND_MODELS_DIR"

# Launch mining server
gunicorn --bind 0.0.0.0:8080 --workers 1 --timeout 7200 'bluesearch.entrypoint:get_mining_app()'
