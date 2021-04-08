#!/usr/bin/env python
import pathlib
import sys
from collections import Counter, defaultdict

import pandas as pd
import spacy
import yaml
from spacy.tokens import DocBin

from bluesearch.mining.eval import annotations2df


DATA_DIR = pathlib.Path("..").resolve()
NLP = spacy.load("en_core_web_sm")


def read_yaml(file_name):
    with open(file_name) as fp:
        yaml_dict = yaml.safe_load(fp)

    return yaml_dict


def count_entity_types_jsonl(file_name):
    annotations_df = annotations2df(DATA_DIR / file_name)
    labels = annotations_df["class"]
    return Counter(labels)


def count_entity_types_spacy(file_name):
    docbin = DocBin().from_disk(DATA_DIR / file_name)

    all_entity_types = []
    for doc in docbin.get_docs(NLP.vocab):
        for token in doc:
            if token.ent_iob_ == "O":
                entity_type = "O"
            else:
                entity_type = token.ent_iob_ + "-" + token.ent_type_
            all_entity_types.append(entity_type)

    return Counter(all_entity_types)


def count_entity_types(file_name):
    if file_name.endswith(".spacy"):
        counts = count_entity_types_spacy(file_name)
    elif file_name.endswith(".jsonl"):
        counts = count_entity_types_jsonl(file_name)
    else:
        raise ValueError("Unknown file type")

    return counts


def filter_entity_type(counts, entity_type):
    new_counts = defaultdict(int)
    for name, count in counts.items():
        if name.endswith(entity_type):
            new_counts[name] += count
        else:
            new_counts["O"] += count

    return new_counts


def do_evaluation(entity_type, split_files, entity_type_map):
    # Do the counting
    counts = {}
    for split_name, files in split_files.items():
        if split_name not in counts:
            counts[split_name] = Counter()
        for file_name in files:
            counts[split_name].update(count_entity_types(file_name))

    # Map entity types
    counts["train"] = {entity_type_map[name]: value for name, value in counts["train"].items()}
    counts["dev"] = {entity_type_map[name]: value for name, value in counts["dev"].items()}

    # Filter entity types
    counts = {name: filter_entity_type(c, entity_type) for name, c in counts.items()}

    # Package into a data frame
    df_counts = pd.DataFrame(counts)
    df_counts = df_counts.fillna(0)
    for column in df_counts.columns:
        # Columns containing NaNs are converted to float...
        df_counts[column] = df_counts[column].astype(int)
        # Compute percentages
        df_counts[f"{column} %"] = df_counts[column] / df_counts[column].sum() * 100

    # Print results
    print(df_counts)


def main():
    # Read configs
    config = read_yaml("config.yaml")
    entity_type_map = read_yaml("entity_type_map.yaml")
    new_entity_type_map = {"O": "O"}
    for our_name, spacy_name in entity_type_map.items():
        new_entity_type_map["B-" + spacy_name] = "B-" + our_name
        new_entity_type_map["I-" + spacy_name] = "I-" + our_name

    # Evaluate
    for entity_type, split_files in config.items():
        print("=" * 80)
        print("Entity Type:", entity_type)
        print("=" * 80)
        do_evaluation(entity_type, split_files, new_entity_type_map)
        print()


if __name__ == "__main__":
    sys.exit(main())
