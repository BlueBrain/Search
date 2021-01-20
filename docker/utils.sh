#!/bin/bash

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
  dvc pull ee_models_library.csv.dvc
  dvc pull $(< dvc.yaml grep -oE '\badd_er_[0-9]+\b' | xargs)
  popd || exit
}
