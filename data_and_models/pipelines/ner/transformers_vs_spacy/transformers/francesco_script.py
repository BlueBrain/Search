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

import argparse
import json
import pathlib
import random
import string
import sys

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument(
    "--annotation-files",
    required=True,
    type=str,
    help="JSONL annotation file(s) used to train the model. If more than one, "
    "they should be comma-separated.",
)
parser.add_argument(
    "--entity-type",
    type=str,
    help="Name of the entity type of interest."
)
parser.add_argument(
    "--output-file",
    "-o",
    required=True,
    help="Output file where annotations are to be written.",
)
parser.add_argument(
    "--seed",
    default=42,
    help="Output file where annotations are to be written.",
)
parser.add_argument(
    "--keep-punctuation",
    action="store_true",
    help="Whether to set all punctuation tokens to 'O'.",
)
args = parser.parse_args()


def punctuation_to_o(df: pd.DataFrame) -> pd.DataFrame:
    """Sets all punctuation characters to out-of-entity 'O'."""
    is_punctuation = df["text"].isin(list(string.punctuation))
    df = df.copy()
    print(df)

    for idx in df.index[is_punctuation & df["class"].str.startswith("B-")]:
        i = idx
        while df.loc[i, "text"] in list(string.punctuation):
            i += 1
        df.loc[i, "class"] = "B" + df.loc[i, "class"][1:]

    df.loc[is_punctuation, "class"] = "O"
    return df


def annotations2df(annots_files, not_entity_symbol="O"):
    """Convert prodigy annotations in JSONL format into a pd.DataFrame.

    Parameters
    ----------
    annots_files : str, list of str, path or list of path
        Name of the annotation file(s) to load.
    not_entity_symbol : str
        A symbol to use for tokens that are not an entity.

    Returns
    -------
    final_table : pd.DataFrame
        Each row represents one token, the columns are 'source', 'sentence_id', 'class',
        'start_char', end_char', 'id', 'text'.
    """
    final_table_rows = []

    if isinstance(annots_files, list):
        final_tables = [annotations2df(ann, not_entity_symbol) for ann in annots_files]
        final_table = pd.concat(final_tables, ignore_index=True)
        return final_table
    elif not (isinstance(annots_files, str) or isinstance(annots_files, Path)):
        raise TypeError(
            "Argument 'annots_files' should be a string or an " "iterable of strings!"
        )

    with open(annots_files) as f:
        for row in f:
            content = json.loads(row)

            if content["answer"] != "accept":
                continue

            # annotations for the sentence: list of dict (or empty list)
            spans = content.get("spans", [])

            classes = {}
            for ent in spans:
                for ix, token_ix in enumerate(
                    range(ent["token_start"], ent["token_end"] + 1)
                ):
                    ent_label = ent["label"].upper()

                    classes[token_ix] = "{}-{}".format(
                        "B" if ix == 0 else "I", ent_label
                    )

            for token in content["tokens"]:
                final_table_rows.append(
                    {
                        "source": content["meta"]["source"],
                        "class": classes.get(token["id"], not_entity_symbol),
                        "start_char": token["start"],
                        "end_char": token["end"],
                        "id": token["id"],
                        "text": token["text"],
                    }
                )

    final_table = pd.DataFrame(final_table_rows)

    return final_table


def filter_entity_type(target_entity_type):
    def inner(entity_type):
        if entity_type.endswith(target_entity_type):
            return entity_type
        else:
            return "O"
    return inner


def main():
    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # read all sentences
    annotated_sentences = []
    for fn in args.annotation_files.split(","):
        df = annotations2df(fn)
        for sentence_id, df_slice in df.groupby("source"):
            if not args.keep_punctuation:
                df_slice = punctuation_to_o(df_slice)
            if args.entity_type is not None:
                df_slice["class"] = df_slice["class"].apply(filter_entity_type(args.entity_type))
            annotated_sentences.append(df_slice[["text", "class"]])

    with pathlib.Path(args.output_file).open("a") as f:
        for i, df_sentence in enumerate(annotated_sentences):
            if i > 0:
                f.write("\n")
            df_sentence.to_csv(f, sep=" ", header=False, index=False, mode="a")


if __name__ == "__main__":
    sys.exit(main())
