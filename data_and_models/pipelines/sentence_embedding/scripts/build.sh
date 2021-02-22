#!/bin/bash

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

# Tested on Python 3.6.12 with:
#   - sentence-transformers v0.3.9
#   - transformers v3.4.0
#   - tokenizers v0.9.2
#   - torch v1.7.0

export MODEL=biobert_nli_sts_cord19_v1
export TEMP=biobert_cord19_v1
export BASE=clagator/biobert_v1.1_pubmed_nli_sts
export TRAIN=sentences-filtered_11-527-877.txt
export DEV=biosses_sentences.txt

echo -e "\n\nTrain...\n\n"
time python train.py \
  --model_name_or_path $BASE \
  --train_data_file $TRAIN \
  --eval_data_file $DEV \
  --line_by_line \
  --mlm \
  --output_dir $TEMP \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --logging_dir ${TEMP}__logs \
  --logging_steps 5000 \
  --save_steps 5000 \
  --save_total_limit 20 \
  --dataloader_num_workers 16

echo -e "\n\nFine-tune...\n\n"
time python fine_tune.py $TEMP $MODEL

echo -e "\n\nDone!\n\n"
