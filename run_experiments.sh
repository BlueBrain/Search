set -e
set -x 

# HYPERPARAMETERS
BATCH_SIZE=32
EVAL_FREQUENCY=10
GPU_ID=0  # -1 means a CPU
N_EPOCHS=3
PATIENCE=0
SEED=2
WITH_TRANSFORMER="no"
[[ $GPU_ID = -1 ]] && DEVICE="CPU" || DEVICE="GPU"


CONFIG_PATH=config_${WITH_TRANSFORMER}_transformers.cfg
#CONFIG_PATH=config_transformers.cfg
OUTPUT_PATH=repro_results_${DEVICE}_${WITH_TRANSFORMER}-transformer/
TRAIN_PATH=data_and_models/annotations/ner/annotations15_EmmanuelleLogette_2020-09-22_raw9_Pathway.train.spacy 
VAL_PATH=data_and_models/annotations/ner/annotations15_EmmanuelleLogette_2020-09-22_raw9_Pathway.dev.spacy


# Prepare experiments
experiments=(1 2)

# Print spacy version
python -c "import spacy;print(spacy.__version__)"

for i in ${experiments[@]}
do
	EXPERIMENT_PATH=${OUTPUT_PATH}exp_${i}
	if [ -d $EXPERIMENT_PATH ]
		then rm -rf $EXPERIMENT_PATH;
	fi

	echo $i
	python -m spacy train \
		$CONFIG_PATH \
		--output $EXPERIMENT_PATH \
		--gpu-id $GPU_ID \
		--nlp.batch_size $BATCH_SIZE \
		--paths.train $TRAIN_PATH \
		--paths.dev $VAL_PATH \
		--system.seed $SEED \
		--training.eval_frequency $EVAL_FREQUENCY \
		--training.max_epochs $N_EPOCHS \
		--training.patience 0 \

	echo $EXPERIMENT_PATH
done

diff -qr ${OUTPUT_PATH}exp_1 ${OUTPUT_PATH}exp_2
