set -e
set -x 

# HYPERPARAMETERS
BATCH_SIZE=32
EVAL_FREQUENCY=10
N_EPOCHS=6
SEED=2

TRAIN_PATH=data_and_models/annotations/ner/annotations15_EmmanuelleLogette_2020-09-22_raw9_Pathway.train.spacy
VAL_PATH=data_and_models/annotations/ner/annotations15_EmmanuelleLogette_2020-09-22_raw9_Pathway.dev.spacy

# Print spacy version
python -m pip show spacy | tail -n +1 | head -3


for WITH_TRANSFORMER in "yes" "no"
do
  for GPU_ID in -1 0
  do
    [[ $GPU_ID = -1 ]] && DEVICE="CPU" || DEVICE="GPU"
    CONFIG_PATH=config_${WITH_TRANSFORMER}_transformers.cfg
    OUTPUT_PATH=repro_results_${DEVICE}_${WITH_TRANSFORMER}-transformer/

    for EXPERIMENT_ID in 1 2
    do
      EXPERIMENT_PATH=${OUTPUT_PATH}exp_${EXPERIMENT_ID}
      if [ -d $EXPERIMENT_PATH ]
        then rm -rf $EXPERIMENT_PATH;
      fi

      echo ${EXPERIMENT_ID}
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
  done
done
