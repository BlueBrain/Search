#!/bin/bash

# Tested on Python 3.6.12 with:
#   - sentence-transformers v0.3.9
#   - transformers v3.4.0
#   - tokenizers v0.9.2
#   - torch v1.7.0

export MODEL=biobert_nli_sts_cord19_v1
export TEMP=biobert_cord19_v1
export BASE=clagator/biobert_v1.1_pubmed_nli_sts
export DATA="$1"
export TRAIN=$DATA/sentences-filtered_11-527-877.txt
export DEV=$DATA/biosses_sentences.txt

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
