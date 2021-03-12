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
# Commented to allow the use of a DVC remote from a different type than SSH.
# ssh_check
# Not usable in README as it works only when inside the `bbs_` containers.
# dvc_pull_models

# Launch mining server
gunicorn --bind 0.0.0.0:8080 --workers 1 --timeout 7200 'bluesearch.entrypoint:get_mining_app()'
