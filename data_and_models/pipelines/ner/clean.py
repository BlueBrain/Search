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

"""
This script cleans annotations exported with Prodigy as .jsonl files.

Run 'analyze.py' before to identify relevant cleaning rules.

Usage: clean.py [OPTIONS] INPUT_PATH KEEP_LABEL [RENAME_INTO]

First, this script keeps only valid texts from the annotations (see next paragraph).
Second, this script normalizes entity labels (use upper case for labels in lower case).
Third, this script keeps only the given entity label and renames it if necessary.

Valid texts are texts which have been accepted, have spans (i.e. entities), and have
a unique 'input_hash'.

The output is one file. The file is put in the same directory as the input file. It is
named following the pattern 'annotations_<entity_label>.jsonl'.
"""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import srsly
import typer


def is_valid(example: Dict[str, Any], duplicated_hashes: set) -> bool:
    accepted = example["answer"] == "accept"
    spans = example["spans"]
    with_spans = isinstance(spans, list) and spans
    unique_hash = example["_input_hash"] not in duplicated_hashes
    return all((accepted, with_spans, unique_hash))


def main(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    keep_label: str = typer.Argument(...),
    rename_into: Optional[str] = typer.Argument(None),
):
    print("/!\\ Run 'analyze.py' before to identify relevant cleaning rules.")

    keep_label = keep_label.upper()
    rename_into = None if rename_into is None else rename_into.upper()

    print("Read annotations...")
    corpus = list(srsly.read_jsonl(input_path))
    print(f"...read {len(corpus)} texts")

    print("Identify duplicated 'input_hash'...")
    seen_hashes = set()
    duplicated_hashes = set()
    for x in corpus:
        hash = x["_input_hash"]
        if hash in seen_hashes:
            duplicated_hashes.add(hash)
        else:
            seen_hashes.add(hash)
    hashes_count = len(duplicated_hashes)
    print(f"...identified {hashes_count} hash(es)")
    if hashes_count > 0:
        debug = " ".join(map(str, sorted(duplicated_hashes)))
        print(f">>> {debug}")

    print("Keep only valid texts...")
    valid_texts = [x for x in corpus if is_valid(x, duplicated_hashes)]
    print(f"...kept {len(valid_texts)} texts")

    print("Normalize labels...")
    total_spans = 0
    normalized_spans = 0
    for text in valid_texts:
        for span in text["spans"]:
            label = span["label"]
            total_spans += 1
            if label.islower():
                span["label"] = label.upper()
                normalized_spans += 1
    print(f"...normalized {normalized_spans} label spans (on {total_spans})")

    print(f"Keep only label {keep_label}...")
    if rename_into is not None:
        print(f"Rename label {keep_label} into {rename_into}...")
    filtered_texts = []
    spans_count = 0
    for text in valid_texts:
        filtered_spans = []
        for span in text["spans"]:
            if span["label"] == keep_label:
                if rename_into is not None:
                    span["label"] = rename_into
                filtered_spans.append(span)
        if filtered_spans:
            text["spans"] = filtered_spans
            spans_count += len(filtered_spans)
            filtered_texts.append(text)
    print(f"...kept {len(filtered_texts)} texts and {spans_count} spans")

    print("Write cleaned annotations...")
    target = keep_label if rename_into is None else rename_into
    output_path = input_path.parent / f"annotations_{target.lower()}.jsonl"
    srsly.write_jsonl(output_path, filtered_texts)
    print(f"...wrote {output_path}")


if __name__ == "__main__":
    typer.run(main)
