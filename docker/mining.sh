#!/bin/bash

# shellcheck source=.
source "$($0)"/utils.sh
ssh_setup
dvc_pull_models

# Launch mining server
gunicorn --bind 0.0.0.0:8080 --workers 1 --timeout 7200 'bbsearch.entrypoints:get_mining_app()'
