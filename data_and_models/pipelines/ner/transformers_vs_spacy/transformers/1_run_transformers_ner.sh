#!/usr/bin/env bash

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


exp_name="evaluate_transformers"

#  --model_name_or_path bert-base-uncased \
#  --lr_scheduler_type "constant" \
DS="train_data.pkl"
DS_VAL="test_data.pkl"
python3 1_run_transformers_ner.py \
  --model_name_or_path "dmis-lab/biobert-large-cased-v1.1" \
  --output_dir "checkpoints/$exp_name" \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy "steps" \
  --eval_steps 10 \
  --train_file "$DS" \
  --validation_file "$DS_VAL" \
  --test_file "$DS_VAL" \
  --num_train_epochs 50 \
  --learning_rate "1e-4" \
  --logging_strategy "epoch" \
  --logging_dir "logs/$exp_name" \
  $@
  # --dataset_name conll2003 \
