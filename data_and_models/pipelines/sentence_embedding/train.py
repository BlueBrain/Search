import argparse
import os
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--sentences_file",
    type=str,
    help="Input file used to train the model, should contain .",
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="Output directory where the trained model will be saved.",
)
parser.add_argument(
    "--model",
    choices=["tf_idf", "count"],
    required=True,
    type=str,
    help="Name of the model to train.",
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="Seed for initializing random number generators.",
)
args = parser.parse_args()


def main():
    np.random.seed(args.seed)

    print("Reading params.yaml...")
    params = yaml.safe_load(open("params.yaml"))["train"][args.model]

    print("Reading training set...")
    with open(args.sentences_file, "r") as f:
        corpus = f.readlines()

    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.model == "tf_idf":
        model = TfidfVectorizer(**params["init_kwargs"])
        print("Training model...")
        model.fit(corpus)
        # hack: https://github.com/scikit-learn/scikit-learn/issues/18669
        model.vocabulary_ = OrderedDict(
            sorted(model.vocabulary_.items(), key=lambda kv: kv[1])
        )
        model._stop_words_id = 0
        print("Saving model to disk...")
        with open(out_dir / "model.pkl", "wb") as f:
            joblib.dump(model, f)
    elif args.model == "count":
        model = CountVectorizer(**params["init_kwargs"])
        print("Training model...")
        model.fit(corpus)
        # hack: https://github.com/scikit-learn/scikit-learn/issues/18669
        model.vocabulary_ = OrderedDict(
            sorted(model.vocabulary_.items(), key=lambda kv: kv[1])
        )
        model._stop_words_id = 0
        print("Saving model to disk...")
        with open(out_dir / "model.pkl", "wb") as f:
            joblib.dump(model, f)
    else:
        raise ValueError(f"Training not available for model {args.model}!")

    print("Training completed!")


if __name__ == "__main__":
    main()
