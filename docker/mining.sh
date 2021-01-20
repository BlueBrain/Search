#!/bin/bash

source /src/docker/utils.sh
ssh_check
dvc_pull_models

# Launch mining server
gunicorn --bind 0.0.0.0:8080 --workers 1 --timeout 7200 'bbsearch.entrypoints:get_mining_app()'
