"""Evaluation script for NER models."""
from argparse import ArgumentParser
from collections import OrderedDict
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
    "--annotation_files1",
    required=True,
    type=str,
    help="The JSONL file(s) with the test set, i.e. containing "
    "sentences NER annotations of the first annotator, the one we will consider as ground truth. "
    "If more than one, should be comma-separated.",
)
parser.add_argument(
    "--annotation_files2",
    required=True,
    type=str,
    help="The JSONL file(s) with the test set, i.e. containing "
    "sentences NER annotations of the second annotator, the one we will consider as human "
    "annotator. If more than one, should be comma-separated.",
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
    df_a1 = annotations2df(args.annotation_files1.split(","))
    df_a2 = annotations2df(args.annotation_files2.split(","))
    # Merge the common sentences between annotators
    df = df_a2.merge(df_a1,
                     on=['source', 'id', 'text', 'start_char', 'end_char'],
                     suffixes=('_annotator_1', '_annotator_2'),
                     how='inner')

    # Load NER model
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

    iob_true = df["class_annotator_1"]
    iob_pred = df["class_pred"]
    iob_interrater = df["class_annotator_2"]

    labels = ["eval", "interrater"]
    annots = [iob_pred, iob_interrater]
    etypes_maps = [{args.etype: params["etype_name"]}, None]

    output_file = pathlib.Path(args.output_file)
    with output_file.open("w") as f:
        all_metrics_dict = OrderedDict()
        for mode in ["entity", "token"]:
            for label, annot, etypes_map in zip(labels, annots, etypes_maps):
                metrics_dict = ner_report(
                    iob_true,
                    annot,
                    mode=mode,
                    return_dict=True,
                    etypes_map=etypes_map,
                )[args.etype]
                metrics_dict = OrderedDict(
                    [(f"{mode}_{k}_{label}", v) for k, v in metrics_dict.items()]
                )
                all_metrics_dict.update(metrics_dict)
        json.dump(all_metrics_dict, f)


if __name__ == "__main__":
    main()
