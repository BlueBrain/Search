#!/bin/bash

# Ensure that the correct username is used for ssh
if [[ -z "$BBS_SSH_USERNAME" ]]; then
  echo "Env var BBS_SSH_USERNAME unset!" 1>&2
  exit 1
fi
mkdir ~/.ssh
printf "Host *\n    User %s" "$BBS_SSH_USERNAME"> ~/.ssh/config

# Pull models with DVC
dvc remote modify gpfs_ssh ask_password true
cd /src/data_and_models/pipelines/ner/ || exit
dvc pull ee_models_library.csv.dvc
dvc pull "$(< dvc.yaml grep -oE '\badd_er_[0-9]+\b' | xargs)"

# Launch mining server
gunicorn --bind 0.0.0.0:8080 --workers 1 --timeout 7200 'bbsearch.entrypoints:get_mining_app()'
