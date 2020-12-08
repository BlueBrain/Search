# docker/mining.sh
# DVC pulling logic
cd /src/data_and_models/pipelines/ner/

dvc remote default gpfs_local

dvc pull ee_models_library.csv.dvc 
dvc pull add_er_1 add_er_2 add_er_3 add_er_4 add_er_5
# dvc pull $(cat dvc.yaml | egrep '^ *add_er_[0-9]+' | sed -e 's/://g' |  xargs)


gunicorn --bind 0.0.0.0:8080 --workers 1 --timeout 7200 'bbsearch.entrypoints:get_mining_app()'
