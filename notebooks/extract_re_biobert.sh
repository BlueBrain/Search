export TASK_NAME=chemprot
export RE_DIR=/raid/covid_data/data/ChemProt_Corpus/chemprot_test_gs/
export OUTPUT_DIR=$PWD/results_re_chemprot/
export BIOBERT_DIR=/raid/covid_data/assets/biobert_v1.1_pubmed/

export BIOBERT_RE_EXEC=/raid/users/casalegn/biobert/run_re.py

python $BIOBERT_RE_EXEC --task_name=$TASK_NAME \
				 --do_train=false \
				 --do_eval=true \
				 --do_predict=true \
				 --vocab_file=$BIOBERT_DIR/vocab.txt \
				 --bert_config_file=$BIOBERT_DIR/bert_config.json \
				 --init_checkpoint=$BIOBERT_DIR/model.ckpt-1000000 \
				 --max_seq_length=128 \
				 --train_batch_size=32 \
				 --learning_rate=2e-5 \
				 --num_train_epochs=3.0 \
				 --do_lower_case=false \
				 --data_dir=$RE_DIR \
				 --output_dir=$OUTPUT_DIR
    
#--output_path=$OUTPUT_DIR/test_results.tsv --answer_path=$RE_DIR/test.tsv
