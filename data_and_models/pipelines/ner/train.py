import argparse
from pathlib import Path

import yaml

from prodigy.recipes.commands import db_in
from prodigy.components.db import connect
from prodigy.recipes import train

parser = argparse.ArgumentParser()
parser.add_argument(
    "--annotation_files",
    type=str,
    help="Input annotation file(s) used to train the model. If more than one, "
         "they should be comma-separated.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="Output directory where the trained spacy model will be saved.",
)
parser.add_argument(
    "--model_name",
    required=True,
    type=str,
    help="Name of the new model.",
)
args = parser.parse_args()


def main():

    print("Read params.yaml...")
    params = yaml.safe_load(open("params.yaml"))

    datasets = []
    print("Connect to db...")
    db = connect()
    existing_datasets = db.datasets
    print("Fill db with annotations...")
    for annotation in args.annotation_files.split(","):
        annotation_file = Path(annotation)
        if annotation_file.stem in existing_datasets:
            db.drop_dataset(annotation_file.stem)
            print(
                f"Found and removed pre-existing dataset {annotation_file.stem} from prodigy.db"
            )
        db_in(set_id=annotation_file.stem, in_file=annotation_file)
        datasets += [annotation_file.stem]

    print(datasets)
    print("Training starting...")

    prodigy_train_kwargs = params["train"][args.model_name]["prodigy_train_kwargs"]
    train.train(
        "ner",
        datasets=datasets,
        output=args.output_dir,
        **prodigy_train_kwargs,
    )
    print("Training completed!")


if __name__ == "__main__":
    main()
