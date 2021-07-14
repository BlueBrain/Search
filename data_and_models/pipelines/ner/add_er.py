"""Append an entity ruler to a spacy pipeline."""

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
import pathlib

import spacy

from bluesearch.mining import global2model_patterns
from bluesearch.utils import JSONL

parser = ArgumentParser()
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="SpaCy model without an entity ruler.",
)
parser.add_argument(
    "--output_file",
    required=True,
    type=str,
    help="File to which we save the enhanced spacy pipeline.",
)
parser.add_argument(
    "--patterns_file",
    required=True,
    type=str,
    help="Path to the patterns file used for rule-based entity recognition.",
)
args = parser.parse_args()


def main():
    # Load and preprocess the annotations
    ner_model = spacy.load(args.model)

    print("Loading patterns")
    path_patterns = pathlib.Path(args.patterns_file)
    patterns = JSONL.load_jsonl(path_patterns)
    _, _, entity_type = args.model.rpartition("-")
    modified_patterns = global2model_patterns(patterns, entity_type.upper())
    er_config = {"validate": True, "overwrite_ents": True}
    er = ner_model.add_pipe("entity_ruler", after="ner", config=er_config)
    er.add_patterns(modified_patterns)

    print("Saving model with an entity ruler")
    ner_model.to_disk(args.output_file)


if __name__ == "__main__":
    main()
