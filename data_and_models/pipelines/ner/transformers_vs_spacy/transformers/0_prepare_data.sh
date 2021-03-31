#!/usr/bin/env bash

export LANG=C.UTF-8


train_data="../../../../annotations/ner/annotations15_EmmanuelleLogette_2020-09-22_raw9_Pathway.jsonl"
test_data_1="../../../../annotations/ner/annotations10_EmmanuelleLogette_2020-08-28_raw1_raw5_10EntityTypes.jsonl"
test_data_2="../../../../annotations/ner/annotations12_EmmanuelleLogette_2020-08-28_raw7_10EntityTypes.jsonl"

python3 francesco_script.py --annotation-files "$train_data" -o train_data.txt --keep-punctuation --entity-type "PATHWAY"
python3 francesco_script.py --annotation-files "$test_data_1,$test_data_2" -o test_data.txt --keep-punctuation --entity-type "PATHWAY"

python3 create_pickle.py train_data.txt train_data.pkl
python3 create_pickle.py test_data.txt test_data.pkl

rm train_data.txt
rm test_data.txt

