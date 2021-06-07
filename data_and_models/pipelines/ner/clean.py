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

First, this script keeps only valid texts from the annotations (see next paragraph).
Second, this script normalizes entity labels (use upper case for labels in lower case).
Third, this script keeps only the given entity label and renames it if necessary.

Valid texts are texts which have been accepted, have or not spans (i.e. entities), and
have a unique 'input_hash'.

The output is one file. The file is put in the same directory as the input file. It is
named following the pattern 'annotations_<entity_label>.jsonl'.
"""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Set

import srsly

parser = ArgumentParser()
parser.add_argument(
    "input_path",
    type=Path,
    help="path of the annotations exported with Prodigy (.jsonl file)",
)
parser.add_argument(
    "kept_label",
    type=str,
    help="keep only this entity label in the annotations (converted to upper case)",
)
parser.add_argument(
    "--renamed-label",
    type=str,
    help="rename the kept entity label with this new name (converted to upper case)",
)
args = parser.parse_args()


def is_valid(example: Dict[str, Any], duplicated_hashes: Set[str]) -> bool:
    accepted = example["answer"] == "accept"
    unique_hash = example["_input_hash"] not in duplicated_hashes
    return all((accepted, unique_hash))


def main():
    kept_label = args.kept_label.upper()
    renamed_label = None if args.renamed_label is None else args.renamed_label.upper()

    print("Read annotations...")
    corpus = list(srsly.read_jsonl(args.input_path))
    print(f"...read {len(corpus)} texts")

    print("Identify duplicated 'input_hash'...")
    counter = Counter(x["_input_hash"] for x in corpus)
    duplicated_hashes = [
        input_hash for input_hash, count in counter.items() if count > 1
    ]
    hashes_count = len(duplicated_hashes)
    print(f"...identified {hashes_count} hash(es)")
    if hashes_count > 0:
        debug = " ".join(map(str, sorted(duplicated_hashes)))
        print(f">>> {debug}")

    print("Keep only valid texts...")
    valid_texts = [x for x in corpus if is_valid(x, duplicated_hashes)]
    print(f"...kept {len(valid_texts)} texts")
    print(f"...that's {len(valid_texts) / len(corpus):.0%} of the total")

    print("Normalize labels...")
    total_spans = 0
    normalized_spans = 0
    for text in valid_texts:
        for span in text.get("spans", []):
            label = span["label"]
            total_spans += 1
            # Testing islower() is not sufficient (i.e. "ENTITy".islower() == False).
            if not label.isupper():
                span["label"] = label.upper()
                normalized_spans += 1
    print(f"...normalized {normalized_spans} label spans (on {total_spans})")
    print(f"...that's {normalized_spans/total_spans:.0%} of the total")

    renaming = (
        "" if renamed_label is None else f" (renamed into {renamed_label})"
    )
    print(f"Keep only label {kept_label}{renaming}...")
    output_texts = []
    kept_spans = 0
    for text in valid_texts:
        filtered_spans = []
        for span in text.get("spans", []):
            if span["label"] == kept_label:
                if renamed_label is not None:
                    span["label"] = renamed_label
                filtered_spans.append(span)
        text["spans"] = filtered_spans
        kept_spans += len(text["spans"])
        output_texts.append(text)
    print(f"...kept {kept_spans} label spans (on {total_spans})")
    print(f"...that's {kept_spans / total_spans:.0%} of the total")

    print("Write cleaned annotations...")
    output_label = kept_label if renamed_label is None else renamed_label
    output_path = args.input_path.parent / f"annotations_{output_label.lower()}.jsonl"
    srsly.write_jsonl(output_path, output_texts)
    print(f"...wrote {output_path}")


if __name__ == "__main__":
    main()
