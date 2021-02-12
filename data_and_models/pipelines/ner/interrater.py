"""Computation of inter-rater agreement for NER models."""

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

from bluesearch.mining.eval import (
    annotations2df,
    remove_punctuation,
    ner_report,
)

parser = ArgumentParser()
parser.add_argument(
    "--annotations1",
    required=True,
    type=str,
    help="The JSONL file(s) with the test set, i.e. containing "
    "sentences with NER annotations considered as GROUND TRUTH. If more than "
    "one, should be comma-separated.",
)
parser.add_argument(
    "--annotations2",
    required=True,
    type=str,
    help="The JSONL file(s) with the test set, i.e. containing "
    "sentences with NER annotations considered as human annotations. If more than "
    "one, should be comma-separated.",
)
parser.add_argument(
    "--output_dir",
    required=True,
    type=str,
    help="Output json directory where metrics results should be written.",
)
args = parser.parse_args()


def main():
    # Load and preprocess the annotations
    df_a1 = annotations2df(args.annotations1.split(","))
    df_a2 = annotations2df(args.annotations2.split(","))
    # Merge the common sentences between annotators
    df = df_a2.merge(
        df_a1,
        on=["source", "id", "text", "start_char", "end_char"],
        suffixes=("_annotator_2", "_annotator_1"),
        how="inner",
    )

    df = remove_punctuation(df)

    iob_true = df["class_annotator_1"]
    iob_pred = df["class_annotator_2"]

    metrics_dict = {}
    for mode in ["entity", "token"]:
        metrics_dict[mode] = ner_report(
            iob_true,
            iob_pred,
            mode=mode,
            return_dict=True,
        )
    for etype in metrics_dict["entity"]:
        all_metrics_dict = OrderedDict()
        for mode in ["entity", "token"]:
            all_metrics_dict.update(
                [(f"{mode}_{k}", v) for k, v in metrics_dict[mode][etype].items()]
            )
        filename = etype.lower() + ".json"
        output_file = pathlib.Path(args.output_dir) / filename
        with output_file.open("w") as f:
            json.dump(all_metrics_dict, f)


if __name__ == "__main__":
    main()
