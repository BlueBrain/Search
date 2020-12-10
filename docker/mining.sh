#!/bin/bash

# DVC pulling logic
cd /src/data_and_models/pipelines/ner/ || exit

mkdir ~/.ssh
printf "Host *\n    User $BBS_SSH_USERNAME" > ~/.ssh/config

dvc remote modify gpfs_ssh ask_password true

dvc pull ee_models_library.csv.dvc 
dvc pull add_er_1 add_er_2 add_er_3 add_er_4 add_er_5
# dvc pull $(cat dvc.yaml | egrep '^ *add_er_[0-9]+' | sed -e 's/://g' |  xargs)


# Launch mining server
gunicorn --bind 0.0.0.0:8080 --workers 1 --timeout 7200 'bbsearch.entrypoints:get_mining_app()'
