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

from argparse import ArgumentParser
from collections import OrderedDict
import pathlib
import json

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
    print("Read params.yaml...")
    params = yaml.safe_load(open("params.yaml"))["eval"][args.etype]

    # Load and preprocess the annotations
    df = annotations2df(args.annotation_files.split(","))
    ner_model = spacy.load(args.model)

    df_pred = []
    for source, df_ in df.groupby("source"):
        df_ = df_.sort_values(by="id", ignore_index=True)
        df_sentence = spacy2df(
            spacy_model=ner_model, ground_truth_tokenization=df_["text"].to_list()
        )
        df_sentence["id"] = df_["id"].values
        df_sentence["source"] = source
        df_pred.append(df_sentence)

    df_pred = pd.concat(df_pred, ignore_index=True).rename(columns={"class": "class_pred"})

    df = df.merge(df_pred, on=["source", "id", "text"], how="inner")

    df = remove_punctuation(df)

    iob_true = df["class"]
    iob_pred = df["class_pred"]

    output_file = pathlib.Path(args.output_file)
    with output_file.open("w") as f:
        all_metrics_dict = OrderedDict()
        for mode in ["entity", "token"]:
            metrics_dict = ner_report(
                iob_true,
                iob_pred,
                mode=mode,
                return_dict=True,
            )[args.etype.upper()]
            metrics_dict = OrderedDict(
                [(f"{mode}_{k}", v) for k, v in metrics_dict.items()]
            )
            all_metrics_dict.update(metrics_dict)
        json.dump(all_metrics_dict, f)


if __name__ == "__main__":
    main()
