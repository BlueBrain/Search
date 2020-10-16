import argparse
import os
import pickle
from pathlib import Path

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
    "--model", required=True, type=str, help="Name of the model to train.",
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
        print("Saving model to disk...")
        pickle.dump(model, out_dir / "model.pkl")
    elif args.model == "count":
        model = CountVectorizer(**params["init_kwargs"])
        print("Training model...")
        model.fit(corpus)
        print("Saving model to disk...")
        pickle.dump(model, out_dir / "model.pkl")
    else:
        raise ValueError(f"Training not available for model {args.model}!")

    print("Training completed!")


if __name__ == "__main__":
    main()
