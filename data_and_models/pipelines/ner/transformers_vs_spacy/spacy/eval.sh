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

DVC_BASE="../../../.."
test_data_1="$DVC_BASE/annotations/ner/annotations10_EmmanuelleLogette_2020-08-28_raw1_raw5_10EntityTypes.jsonl"
test_data_2="$DVC_BASE/annotations/ner/annotations12_EmmanuelleLogette_2020-08-28_raw7_10EntityTypes.jsonl"


python eval.py \
  --annotation_files "$test_data_1,$test_data_2" \
  --model "$DVC_BASE/models/ner/model5" \
  --output_file "pathway_metrics.json" \
  --etype "PATHWAY"
