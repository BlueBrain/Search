"""Evaluation script for NER models."""
from argparse import ArgumentParser
from collections import OrderedDict
import pathlib
import json

import pandas as pd
import spacy
import yaml

from bbsearch.mining import remap_entity_type
from bbsearch.mining.eval import (
    annotations2df,
    spacy2df,
    remove_punctuation,
    ner_report,
)
from bbsearch.utils import JSONL

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
parser.add_argument(
    "--patterns_file",
    type=str,
    help="Path to the patterns file.",
)
args = parser.parse_args()


def main():
    print("Read params.yaml...")
    params = yaml.safe_load(open("params.yaml"))["eval"][args.etype]

    # Load and preprocess the annotations
    df = annotations2df(args.annotation_files.split(","))
    ner_model = spacy.load(args.model)

    if args.patterns_file is not None:
        print("Loading patterns")
        path_patterns = pathlib.Path(args.patterns_file)
        er = spacy.pipeline.EntityRuler(ner_model, validate=True, overwrite_ents=True)
        patterns = JSONL.load_jsonl(path_patterns)
        modified_patterns = remap_entity_type(patterns, {args.etype: params["etype_name"]})
        er.add_patterns(modified_patterns)
        ner_model.add_pipe(er, after="ner")

    df_pred = []
    for source, df_ in df.groupby("source"):
        df_ = df_.sort_values(by="id", inplace=False, ignore_index=True)
        df_sentence = spacy2df(
            spacy_model=ner_model,
            ground_truth_tokenization=df_["text"].to_list(),
            excluded_entity_type="NaE"
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

    output_file = pathlib.Path(args.output_file)
    with output_file.open("w") as f:
        all_metrics_dict = OrderedDict()
        for mode in ["entity", "token"]:
            metrics_dict = ner_report(
                iob_true,
                iob_pred,
                mode=mode,
                return_dict=True,
                etypes_map={args.etype: params["etype_name"]},
            )[args.etype]
            metrics_dict = OrderedDict(
                [(f"{mode}_{k}", v) for k, v in metrics_dict.items()]
            )
            all_metrics_dict.update(metrics_dict)
        json.dump(all_metrics_dict, f)


if __name__ == "__main__":
    main()
