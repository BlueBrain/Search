#!/bin/bash

# DVC pulling logic
cd /src/data_and_models/pipelines/ner/ || exit

dvc remote default gpfs_local

dvc pull ee_models_library.csv.dvc 
dvc pull add_er_1 add_er_2 add_er_3 add_er_4 add_er_5
# dvc pull $(cat dvc.yaml | egrep '^ *add_er_[0-9]+' | sed -e 's/://g' |  xargs)

# Launch mining cache creation, using arguments only if defined
create_mining_cache \
  $([ -n "$BBS_MINING_CACHE_TARGET_TABLE" ] && echo "--target-table-name $BBS_MINING_CACHE_TARGET_TABLE") \
  $([ -n "$BBS_MINING_CACHE_PROCESSORS_PER_MODEL" ] && echo "--n_processes_per_model $BBS_MINING_CACHE_PROCESSORS_PER_MODEL") \
  $([ -n "$BBS_MINING_CACHE_LOG_FILE" ] && echo "--log-file $BBS_MINING_CACHE_LOG_FILE") \
  $([ -n "$BBS_MINING_VERBOSE" ] && echo "--verbose $BBS_MINING_VERBOSE")
