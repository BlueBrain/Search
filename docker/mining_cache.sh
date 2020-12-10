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

# Launch mining cache creation, using arguments only if defined
create_mining_cache \
  $([ -n "$BBS_MINING_CACHE_TARGET_TABLE" ] && echo "--target-table-name $BBS_MINING_CACHE_TARGET_TABLE") \
  $([ -n "$BBS_MINING_CACHE_PROCESSORS_PER_MODEL" ] && echo "--n_processes_per_model $BBS_MINING_CACHE_PROCESSORS_PER_MODEL") \
  $([ -n "$BBS_MINING_CACHE_LOG_FILE" ] && echo "--log-file $BBS_MINING_CACHE_LOG_FILE") \
  $([ -n "$BBS_MINING_VERBOSE" ] && echo "--verbose $BBS_MINING_VERBOSE")
