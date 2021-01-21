#!/bin/bash

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

# Ensure that the correct username is used for ssh
ssh_setup() {
  if [[ -z "$BBS_SSH_USERNAME" ]]; then
    echo "Env var BBS_SSH_USERNAME unset!" 1>&2
    exit 1
  fi
  mkdir ~/.ssh
  printf "Host *\n    User %s" "$BBS_SSH_USERNAME"> ~/.ssh/config
}

# Pull models with DVC
dvc_pull_models() {
  dvc remote modify gpfs_ssh ask_password true
  pushd /src/data_and_models/pipelines/ner/ || exit
  dvc pull $(< dvc.yaml grep -oE '\badd_er_[0-9]+\b' | xargs)
  popd || exit
}
