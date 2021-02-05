<!---
BBSearch is a text mining toolbox focused on scientific use cases.

Copyright (C) 2020  Blue Brain Project, EPFL.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
-->

This README describes how to use the script `convert_chemprot_fmt.py`. Its goal
is to convert the raw ChemProt dataset to a format that can be directly
used with [BioBERT](https://github.com/dmis-lab/biobert-pytorch).

# Getting ChemProt
1. Download https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/
2. Unzip the `ChemProt_Corpus.zip` file and then unzip all the inner `.zip` files.
3. One should then have the following files and folders:

```bash
├── chemprot_development
│   ├── Readme.pdf
│   ├── chemprot_development_abstracts.tsv
│   ├── chemprot_development_entities.tsv
│   ├── chemprot_development_gold_standard.tsv
│   └── chemprot_development_relations.tsv
├── chemprot_sample
│   ├── Readme.pdf
│   ├── chemprot_sample_abstracts.tsv
│   ├── chemprot_sample_entities.tsv
│   ├── chemprot_sample_gold_standard.tsv
│   ├── chemprot_sample_predictions.tsv
│   ├── chemprot_sample_predictions_eval.txt
│   ├── chemprot_sample_relations.tsv
│   └── guidelines
├── chemprot_test_gs
│   ├── chemprot_test_abstracts_gs.tsv
│   ├── chemprot_test_entities_gs.tsv
│   ├── chemprot_test_gold_standard.tsv
│   ├── chemprot_test_relations_gs.tsv
│   └── readme_test_gs.pdf
└── chemprot_training
    ├── Readme.pdf
    ├── chemprot_training_abstracts.tsv
    ├── chemprot_training_entities.tsv
    ├── chemprot_training_gold_standard.tsv
    └── chemprot_training_relations.tsv
```


# Using the conversion script
The script converts one "folder" (see above) at a time. One needs to provide it
as the first argument. Additionally, there are multiple additional options
that one can use.


```bash
usage: convert_chemprot_fmt.py [-h] [--binary-classification] --annotation-style {scibert,biobert} [--discard-non-eval] [--keep-undefined-relations] input_dir output_dir

positional arguments:
  input_dir
  output_dir

optional arguments:
  -h, --help            show this help message and exit
  --binary-classification, -b
  --annotation-style {scibert,biobert}
  --discard-non-eval, -d
  --keep-undefined-relations, -k
```

Note that in what follows we are going to use the flag `-b, --binary-classification`
since BioBert only supports binary classification. When it comes to the
`--annotation-style` one can choose either `scibert` or `biobert`.

Let us now generate the training and the test set for all relations.

```bash
python convert_chemprot_fmt.py -b --annotation-style biobert ChemProt_Corpus/chemprot_training train
python convert_chemprot_fmt.py -b --annotation-style biobert ChemProt_Corpus/chemprot_test_gs test
```
The folders `train` and `test` should look like this

```bash
├── test
│   ├── test_gs_ACTIVATOR.tsv
│   ├── test_gs_AGONIST-ACTIVATOR.tsv
│   ├── test_gs_AGONIST-INHIBITOR.tsv
│   ├── test_gs_AGONIST.tsv
│   ├── test_gs_ANTAGONIST.tsv
│   ├── test_gs_COFACTOR.tsv
│   ├── test_gs_DIRECT-REGULATOR.tsv
│   ├── test_gs_DOWNREGULATOR.tsv
│   ├── test_gs_INDIRECT-DOWNREGULATOR.tsv
│   ├── test_gs_INDIRECT-REGULATOR.tsv
│   ├── test_gs_INDIRECT-UPREGULATOR.tsv
│   ├── test_gs_INHIBITOR.tsv
│   ├── test_gs_MODULATOR-ACTIVATOR.tsv
│   ├── test_gs_MODULATOR-INHIBITOR.tsv
│   ├── test_gs_MODULATOR.tsv
│   ├── test_gs_NOT.tsv
│   ├── test_gs_PART-OF.tsv
│   ├── test_gs_PRODUCT-OF.tsv
│   ├── test_gs_REGULATOR.tsv
│   ├── test_gs_SUBSTRATE.tsv
│   └── test_gs_UPREGULATOR.tsv
└── train
    ├── training_ACTIVATOR.tsv
    ├── training_AGONIST-ACTIVATOR.tsv
    ├── training_AGONIST-INHIBITOR.tsv
    ├── training_AGONIST.tsv
    ├── training_ANTAGONIST.tsv
    ├── training_COFACTOR.tsv
    ├── training_DIRECT-REGULATOR.tsv
    ├── training_DOWNREGULATOR.tsv
    ├── training_INDIRECT-DOWNREGULATOR.tsv
    ├── training_INDIRECT-REGULATOR.tsv
    ├── training_INDIRECT-UPREGULATOR.tsv
    ├── training_INHIBITOR.tsv
    ├── training_MODULATOR-ACTIVATOR.tsv
    ├── training_MODULATOR-INHIBITOR.tsv
    ├── training_MODULATOR.tsv
    ├── training_NOT.tsv
    ├── training_PART-OF.tsv
    ├── training_PRODUCT-OF.tsv
    ├── training_REGULATOR.tsv
    ├── training_SUBSTRATE.tsv
    ├── training_SUBSTRATE_PRODUCT-OF.tsv
    └── training_UPREGULATOR.tsv
```
We now pick the relation that we are interested in (e.g. ACTIVATOR) and
only take the corresponding training and test `.tsv` for that relation.

```bash
test/test_gs_ACTIVATOR.tsv
train/training_ACTIVATOR.tsv
```

# Training a binary classifier with BioBERT
Take the above `tsv` files, rename them and put them in a folder.

```bash
mkdir bert_input
cp test/test_gs_ACTIVATOR.tsv bert_input/train.tsv
cp train/training_ACTIVATOR.tsv bert_input/test.tsv
```

Clone the [BioBERT](https://github.com/dmis-lab/biobert-pytorch) and
navigate to `biobert-pytorch/relation-extraction/`. Finally run the following
script (it requires `transformers==3.0.0` !!!).

```
export SAVE_DIR=./output
export DATA_DIR=bert_input   # Provide the correct path

export MAX_LENGTH=128
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=1000
export SEED=1

python run_re.py \
    --task_name SST-2 \
    --config_name bert-base-cased \
    --data_dir ${DATA_DIR} \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
    --max_seq_length ${MAX_LENGTH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --save_steps ${SAVE_STEPS} \
    --seed ${SEED} \
    --do_train \
    --do_predict \
    --learning_rate 5e-5 \
    --output_dir ${SAVE_DIR} \
    --overwrite_output_dir
```
The trained model will be located at `./output`.

For additional details see the BioBERT repository.
