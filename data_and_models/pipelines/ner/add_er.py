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
from unittest.mock import PropertyMock, patch

import spacy
import yaml

from bluesearch.mining import remap_entity_type
from bluesearch.utils import JSONL

parser = ArgumentParser()
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="SpaCy model without an entity ruler. Can either be a SciSpacy model"
         '(e.g. "en_ner_jnlpba_md"_ or the path to a custom'
         "trained model.",
)
parser.add_argument(
    "--etypes", required=True, type=str, help="Comma separated list of entity types.",
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
    print("Read params.yaml...")
    params = yaml.safe_load(open("params.yaml"))["eval"]
    external_etypes = [x.strip() for x in args.etypes.split(",")]
    etype_mapping = {external: params[external]["etype_name"] for external in external_etypes}
    # Load and preprocess the annotations
    ner_model = spacy.load(args.model)

    print("Loading patterns")
    path_patterns = pathlib.Path(args.patterns_file)
    er = spacy.pipeline.EntityRuler(ner_model, validate=True, overwrite_ents=True)
    patterns = JSONL.load_jsonl(path_patterns)
    modified_patterns = remap_entity_type(patterns, etype_mapping)
    er.add_patterns(modified_patterns)
    ner_model.add_pipe(er, after="ner")

    sorted_labels = tuple(sorted(er.labels))

    # See https://github.com/explosion/spaCy/issues/7352 for more info
    with patch("spacy.pipeline.EntityRuler.labels", new_callable=PropertyMock) as mock:
        mock.return_value = sorted_labels

        print("Saving model with an entity ruler")
        ner_model.to_disk(args.output_file)


if __name__ == "__main__":
    main()
