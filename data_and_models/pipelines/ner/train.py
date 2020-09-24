import argparse
from pathlib import Path

from prodigy.recipes.commands import db_in, drop
from prodigy.components.db import connect
from prodigy.recipes import train

parser = argparse.ArgumentParser()
parser.add_argument("--annotation_files",
                    type=str,
                    help="The annotation file used to train the model")
parser.add_argument("--evaluation_split",
                    default=0.1,
                    type=float,
                    help="Evaluation split for the training")
parser.add_argument("--spacy_model",
                    type=str,
                    help="Spacy model to train on (if training from scratch blank:en)")
parser.add_argument("--output_dir",
                    type=str,
                    help="Spacy model to train on (if training from scratch blank:en)")
args = parser.parse_args()


def main():

    print('READ params.yaml...')

    print('STARTING...')
    datasets = []
    print('CONNECTION TO DATABASE...')
    db = connect()
    existing_datasets = db.datasets
    print('FILLING OF THE DATABASE....')
    for annotation in args.annotation_files.split(';'):
        annotation_file = Path(annotation)
        if annotation_file.stem in existing_datasets:
            drop(set_id=annotation_file.stem)
        db_in(set_id=annotation_file.stem, in_file=annotation_file)
        datasets += [annotation_file.stem]

    print(datasets)
    print('TRAINING...')
    train.train('ner', datasets=datasets, spacy_model=args.spacy_model, output=args.output_dir)


if __name__ == "__main__":
    main()
