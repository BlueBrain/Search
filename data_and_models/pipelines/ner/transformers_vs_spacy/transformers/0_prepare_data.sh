#!/usr/bin/env bash

# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

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

