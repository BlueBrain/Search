"""Evaluation script for NER models."""

# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import pathlib
import json
from argparse import ArgumentParser
from collections import OrderedDict
from pprint import pprint

import pandas as pd
import spacy
import yaml

from bluesearch.mining.eval import (
    annotations2df,
    spacy2df,
    remove_punctuation,
    ner_report,
)

parser = ArgumentParser()
parser.add_argument(
    "--annotation_files",
    required=True,
    type=str,
    help="The JSONL file(s) with the test set, i.e. containing "
    "sentences with ground truth NER annotations. If more than "
    "one, should be comma-separated.",
)
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="SpaCy model to evaluate.",
)
parser.add_argument(
    "--etype", required=True, type=str, help="Name of the entity type.",
)
parser.add_argument(
    "--output_file",
    required=True,
    type=str,
    help="Output json file where metrics results should be written.",
)
args = parser.parse_args()


def main():
    # Load and preprocess the annotations
    print("Loading data and model")
    df = annotations2df(args.annotation_files.split(","))
    ner_model = spacy.load(args.model)

    print("Computing predictions")
    df_pred = []
    for source, df_ in df.groupby("source"):
        df_ = df_.sort_values(by="id", ignore_index=True)
        df_sentence = spacy2df(
            spacy_model=ner_model, ground_truth_tokenization=df_["text"].to_list()
        )
        df_sentence["id"] = df_["id"].values
        df_sentence["source"] = source
        df_pred.append(df_sentence)

    print("Formatting predctions")
    df_pred = pd.concat(df_pred, ignore_index=True).rename(columns={"class": "class_pred"})
    df = df.merge(df_pred, on=["source", "id", "text"], how="inner")
    #df = remove_punctuation(df)
    iob_true = df["class"]
    iob_pred = df["class_pred"]
    
    print("Saving predictions")
    df.to_pickle("df_test_pred.pkl")

    print("Computing and saving metrics")
    output_file = pathlib.Path(args.output_file)
    metrics_dict = ner_report(
        iob_true,
        iob_pred,
        mode="token",
        return_dict=True,
    )
    metrics_dict = dict(metrics_dict[args.etype])
    with output_file.open("w") as f:
        json.dump(metrics_dict, f)
        f.write("\n")
    pprint(metrics_dict)


if __name__ == "__main__":
    main()
