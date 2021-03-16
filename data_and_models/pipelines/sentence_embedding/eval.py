"""Evaluation of sentence embedding models."""

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

import importlib
import json
import pathlib
from argparse import ArgumentParser
from collections import OrderedDict

import matplotlib.pyplot as plt
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
    "--model",
    choices=["biobert_nli_sts_cord19_v1", "bsv", "biobert_nli_sts", "tf_idf", "count", "use", "sbert", "sbiobert"],
    required=True,
    type=str,
    help="Name of the model to evaluate.",
)
parser.add_argument(
    "--output_dir",
    required=True,
    type=str,
    help="Output dir where metrics and outputs  will be written.",
)

args = parser.parse_args()


def main():
    print("Reading params.yaml...")
    params = yaml.safe_load(open("params.yaml"))["eval"][args.model]
    module = importlib.import_module("bluesearch.embedding_models")
    class_ = getattr(module, params["class"])
    model = class_(**params["init_kwargs"])

    print("Reading test set...")
    df_sents = pd.read_csv(args.annotation_files)

    print("Computing test scores...")
    print(" - test 1: correlation metrics")
    y_true = df_sents["score"]
    embeddings_1 = model.embed_many(model.preprocess_many(df_sents["sentence_1"].tolist()))
    embeddings_2 = model.embed_many(model.preprocess_many(df_sents["sentence_2"].tolist()))
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

    print(" - test 2: semantic similarities matrix")
    # Matrix inspired by Cer D., et al. "Universal sentence encoder." arXiv
    # preprint arXiv:1803.11175 (2018).
    words_of_interest = [
        "COVID-19",
        "SARS-CoV-2",
        "coronavirus",
        "sugar",
        "glucose",
        "obesity",
        "glycosylation",
        "ketosis",
        "potato",
        "carrot",
    ]
    embeddings_w = model.embed_many(model.preprocess_many(words_of_interest))
    mm = cosine_similarity(embeddings_w, embeddings_w)

    print("Writing results to file...")
    out_file = pathlib.Path(args.output_dir) / args.model
    # Write metrics
    output_json = out_file.with_suffix(".json")
    with output_json.open("w") as f:
        json.dump(metrics_dict, f)

    # Write matrix
    output_csv = out_file.with_suffix(".csv")
    output_png = out_file.with_suffix(".png")
    pd.DataFrame(mm, columns=words_of_interest, index=words_of_interest).to_csv(
        output_csv
    )
    # Plot matrix
    f, ax = plt.subplots(figsize=(5,) * 2)
    ax.imshow(mm)
    nw = len(embeddings_w)
    for i in range(nw):
        for j in range(nw):
            c = "white" if mm[i, j] < 0.5 else "black"
            ax.text(i, j, f"{mm[i, j]:.1f}", ha="center", va="center", c=c)
    ax.set_xticks(np.arange(nw))
    ax.set_yticks(np.arange(nw))
    ax.set_xticklabels(words_of_interest, rotation=45)
    ax.set_yticklabels(words_of_interest)
    ax.xaxis.tick_top()
    f.tight_layout()
    f.savefig(output_png, transparent=True, dpi=200)

    print("Done.")


if __name__ == "__main__":
    main()
