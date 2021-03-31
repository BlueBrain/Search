#!/usr/bin/env bash


DVC_BASE="../../../.."
test_data_1="$DVC_BASE/annotations/ner/annotations10_EmmanuelleLogette_2020-08-28_raw1_raw5_10EntityTypes.jsonl"
test_data_2="$DVC_BASE/annotations/ner/annotations12_EmmanuelleLogette_2020-08-28_raw7_10EntityTypes.jsonl"


python eval.py \
  --annotation_files "$test_data_1,$test_data_2" \
  --model "$DVC_BASE/models/ner/model5" \
  --output_file "pathway_metrics.json" \
  --etype "PATHWAY"
