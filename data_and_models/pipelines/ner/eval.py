"""Evaluation script for NER models."""
from argparse import ArgumentParser
import pathlib
import json

import pandas as pd
import spacy
import yaml

from bbsearch.mining.eval import (
    annotations2df,
    spacy2df,
    remove_punctuation,
    ner_report,
)

parser = ArgumentParser()
parser.add_argument(
    "--annotations_file",
    required=True,
    type=str,
    help="The JSONL file with the test set, i.e. containing "
         "sentences with ground truth NER annotations.",
)
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="SpaCy model to evaluate. Can either be a SciSpacy model"
         '(e.g. "en_ner_jnlpba_md"_ or the path to a custom'
         "trained model.",
)
parser.add_argument(
    "--stage_name",
    required=True,
    type=str,
    help="Name of the stage in the DVC pipeline.",
)
args = parser.parse_args()


def main():
    params = yaml.safe_load(open("params.yaml"))[args.stage_name]
    etypes = list(params["etypes"].keys())

    # Load and preprocess the annotations
    df = annotations2df(args.annotations_file)
    ner_model = spacy.load(args.model)

    df_pred = []
    for source, df_ in df.groupby("source"):
        df_ = df_.sort_values(by="id", inplace=False, ignore_index=True)
        df_sentence = spacy2df(
            spacy_model=ner_model, ground_truth_tokenization=df_["text"].to_list()
        )
        df_sentence["id"] = df_["id"].values
        df_sentence["source"] = source
        df_pred.append(df_sentence)

    df_pred = pd.concat(df_pred, ignore_index=True)
    df_pred.rename(columns={"class": "class_pred"}, inplace=True)

    df = df.merge(df_pred, on=["source", "id", "text"], how="inner")

    df = remove_punctuation(df)

    iob_true = df["class"]
    iob_pred = df["class_pred"]

    # Evaluation
    final_dict = {
        etype: {
            mode: ner_report(
                iob_true,
                iob_pred,
                mode=mode,
                return_dict=True,
                etypes_map=params["etypes"],
            )[etype]
            for mode in ["entity", "token"]
        }
        for etype in etypes
    }

    output_file = (
            pathlib.Path.cwd().parent.parent
            / "metrics"
            / "ner"
            / f"{args.stage_name}_metrics.json"
    )

    with output_file.open("w") as f:
        f.write(json.dumps(final_dict))


if __name__ == "__main__":
    main()
