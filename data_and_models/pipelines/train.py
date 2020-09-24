import argparse
from pathlib import Path

from prodigy.recipes.commands import db_in
from prodigy.recipes import train

parser = argparse.ArgumentParser()
parser.add_argument("--annotation_files",
                    default="data_and_models/annotations/ner/annotations5_EmmanuelleLogette_2020"
                            "-06-30_raw2_Disease.jsonl",
                    type=str,
                    help="The annotation file used to train the model")
parser.add_argument("--evaluation_split",
                    default=0.1,
                    type=float,
                    help="Evaluation split for the training")
parser.add_argument("--spacy_model",
                    default='en_ner_bc5cdr_md',
                    type=str,
                    help="Spacy model to train on (if training from scratch blank:en)")
parser.add_argument("--output_dir",
                    default='data_and_models/models/ner/model_DISEASE/',
                    type=str,
                    help="Spacy model to train on (if training from scratch blank:en)")
args = parser.parse_args()


def main():

    print('STARTING...')
    datasets = []
    print('FILLING OF THE DATABASE....')
    for annotation in args.annotation_files.split(';'):
        annotation_file = Path(annotation)
        db_in(set_id=annotation_file.stem, in_file=annotation_file)
        datasets += [annotation_file.stemtt

    print(datasets)
    print('TRAINING...')
    train.train('ner', datasets=datasets, spacy_model=args.spacy_model, output=args.output_dir)


if __name__ == "__main__":
    main()
