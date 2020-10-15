"""Evaluation script for NER models."""
import importlib
import json
import pathlib
from argparse import ArgumentParser
from collections import OrderedDict

import pandas as pd
import spacy
import yaml
from sklearn.metrics.pairwise import cosine_similarity

parser = ArgumentParser()
parser.add_argument(
    "--annotation_files",
    required=True,
    type=str,
    help="The CSV file(s) with the test set, i.e. containing sentences pairs "
         "with the annotated ground-truth simliarity. If more than one, should "
         "be comma-separated.",
)
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="Name of the model to evaluate.",
)
parser.add_argument(
    "--output_file",
    required=True,
    type=str,
    help="Output json file where metrics results should be written.",
)
args = parser.parse_args()


def main():
    print("Read params.yaml...")
    params = yaml.safe_load(open("params.yaml"))["eval"][args.model]
    module = importlib.import_module('bbsearch.embedding_models')
    class_ = getattr(module, params['class'])
    model = class_(**params['init_kwargs'])

    print(model.embed("This is a sentence."))


if __name__ == "__main__":
    main()
