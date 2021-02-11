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

# Ensure that username provided
ssh_check() {
  if [[ -z "$BBS_SSH_USERNAME" ]]; then
    echo "Env var BBS_SSH_USERNAME unset!" 1>&2
    exit 1
  fi
}

# Pull models with DVC
dvc_pull_models() {
  dvc remote modify gpfs_ssh ask_password true
  dvc remote modify gpfs_ssh user $BBS_SSH_USERNAME
  pushd /src/data_and_models/pipelines/ner/ || exit
  egrep -o '\badd_er_[0-9]+\b' dvc.yaml | xargs dvc pull
  popd || exit
}


add_aliases() {
  # Write some useful aliases to "$HOME_DIR/.bash_aliases$
  local HOME_DIR="$1"

  echo "
  alias ll='ls -lah'\n
  " >> "${HOME_DIR}/.bash_aliases"
}

improve_prompt() {
  # Change the prompt appearance and add current git branch
  local HOME_DIR="$1"
  local USER_MODE="$2"
  local USER_COLOR="$3"
  if [ -z "$USER_MODE" ]; then USER_MODE="01"; fi
  if [ -z "$USER_COLOR" ]; then USER_COLOR="33"; fi

  local USER_STR="\[\e[${USER_MODE};${USER_COLOR}m\]\u\[\e[00m\]"
  local WORKDIR_STR="\[\e[01;34m\]\w\[\e[00m\]"
  local GIT_STR="\[\e[0;35m\]\$(parse_git_branch)\[\e[00m\]"

  echo "
function parse_git_branch {
  local ref
  ref=\$(command git symbolic-ref HEAD 2> /dev/null) || return 0
  echo \"‹\${ref#refs/heads/}› \"
}

PS1='${USER_STR} :: ${WORKDIR_STR} ${GIT_STR}$ '
" >> "${HOME_DIR}/.bashrc"
}

config_jupyter() {
# Configure Jupyter
# Set --no-browser --ip=0.0.0.0
  local user_name="$1"
  local home_dir="$2"

  if [ "$user_name" = "root" ]
  then
    jupyter-lab --generate-config
  else
    su "$user_name" -c 'jupyter-lab --generate-config'
  fi 
  sed -i"" \
    -e "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '0.0.0.0'/g" \
    -e "s/#c.NotebookApp.open_browser = True/c.NotebookApp.open_browser = False/g" \
    "$home_dir/.jupyter/jupyter_notebook_config.py"
}

download_nltk() {
  local user_name="$1"
  if [ "$user_name" = "root" ]
  then
    python -m nltk.downloader punkt stopwords
  else
    su "$user_name" -c 'python -m nltk.downloader punkt stopwords'
  fi
}

create_users() {
  local USERS="$1"
  local GROUP="$2"

  for x in $(echo "$USERS" | tr "," "\n")
  do
    if [ -z "$x" ]
    then
      continue
    fi
    user_name="${x%/*}"
    user_id="${x#*/}"
    user_home="/home/${user_name}"
    useradd --create-home --uid "$user_id" --gid "$GROUP" --home-dir "$user_home" "$user_name"

    # If this directory doesn't exist it won't be included in the $PATH
    # and python entrypoints for user-installed packages won't work
    su "$user_name" -c "mkdir -p $user_home/.local/bin"

    # pre-download the nltk data
    download_nltk "$user_name"

    # miscellaneous tweaks and settings
    add_aliases "$user_home"
    improve_prompt "$user_home"
    config_jupyter "$user_name" "$user_home"

    echo "Added user ${user_name} with ID ${user_id}"
  done
}
