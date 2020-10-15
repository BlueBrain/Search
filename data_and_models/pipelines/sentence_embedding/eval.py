"""Evaluation script for NER models."""
import importlib
import json
import pathlib
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau, pearsonr, spearmanr
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
    "--model", required=True, type=str, help="Name of the model to evaluate.",
)
parser.add_argument(
    "--output_file",
    required=True,
    type=str,
    help="Output json file where metrics results should be written.",
)
args = parser.parse_args()


def main():
    print("Reading params.yaml...")
    params = yaml.safe_load(open("params.yaml"))["eval"][args.model]
    module = importlib.import_module("bbsearch.embedding_models")
    class_ = getattr(module, params["class"])
    model = class_(**params["init_kwargs"])

    print("Reading test set...")
    df_sents = pd.read_csv(args.annotation_files)

    print("Computing test scores...")
    y_true = df_sents["score"]
    embeddings_1 = model.embed_many(model.preprocess_many(df_sents["sentence_1"]))
    embeddings_2 = model.embed_many(model.preprocess_many(df_sents["sentence_2"]))
    y_pred = np.array(
        [
            cosine_similarity(e1[np.newaxis], e2[np.newaxis]).item()
            for e1, e2 in zip(embeddings_1, embeddings_2)
        ]
    )

    metrics_dict = OrderedDict()
    metrics_dict["kendall_tau"] = kendalltau(y_true, y_pred).correlation
    metrics_dict["pearson_r"] = pearsonr(y_true, y_pred)[0]
    metrics_dict["spearman_rho"] = spearmanr(y_true, y_pred).correlation

    print("Writing to file...")
    output_file = pathlib.Path(args.output_file)
    with output_file.open("w") as f:
        json.dump(metrics_dict, f)

    print("Done.")


if __name__ == "__main__":
    main()
