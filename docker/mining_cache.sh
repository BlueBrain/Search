#!/bin/bash

# shellcheck source=.
source "$($0)"/utils.sh
ssh_setup
dvc_pull_models

# Launch mining cache creation, using arguments only if defined
create_mining_cache \
  $([ -n "$BBS_MINING_CACHE_TARGET_TABLE" ] && echo "--target-table-name $BBS_MINING_CACHE_TARGET_TABLE") \
  $([ -n "$BBS_MINING_CACHE_PROCESSORS_PER_MODEL" ] && echo "--n-processes-per-model $BBS_MINING_CACHE_PROCESSORS_PER_MODEL") \
  $([ -n "$BBS_MINING_CACHE_LOG_FILE" ] && echo "--log-file $BBS_MINING_CACHE_LOG_FILE") \
  $([ -n "$BBS_MINING_VERBOSE" ] && echo "$BBS_MINING_VERBOSE")
