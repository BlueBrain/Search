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
dvc_configure_ssh_remote_authentication "$BBS_SSH_USERNAME"
dvc_pull_models "$DATA_DIR"

# Launch mining cache creation, using arguments only if defined
create_mining_cache \
  $([ -n "$BBS_MINING_CACHE_TARGET_TABLE" ] && echo "--target-table-name $BBS_MINING_CACHE_TARGET_TABLE") \
  $([ -n "$BBS_MINING_CACHE_PROCESSORS_PER_MODEL" ] && echo "--n-processes-per-model $BBS_MINING_CACHE_PROCESSORS_PER_MODEL") \
  $([ -n "$BBS_MINING_CACHE_LOG_FILE" ] && echo "--log-file $BBS_MINING_CACHE_LOG_FILE") \
  $([ -n "$BBS_MINING_CACHE_LOG_LEVEL" ] && echo "--log-level $BBS_MINING_CACHE_LOG_LEVEL")
