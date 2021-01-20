#!/bin/bash

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
