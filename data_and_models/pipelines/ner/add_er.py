"""Append an entity ruler to a spacy pipeline."""

from argparse import ArgumentParser
import pathlib

import spacy
import yaml

from bbsearch.mining import remap_entity_type
from bbsearch.utils import JSONL

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

    print("Saving model with an entity ruler")
    ner_model.to_disk(args.output_file)


if __name__ == "__main__":
    main()
