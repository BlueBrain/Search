"""Parse chemprot into tsv files.

This script is parsing chemprot dataset into tsv files that are compatible with
biobert train scripts. Those files are containing sentences and
corresponding labels columns. To use this script:
- Download chemprot dataset on
https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/
- Unzip ChemProt_Corpus.zip file and then unzip chemprot_training.zip, ... files
- Launch command line:
python chemprot.py chemprot_test_gs/ output_dir/ --annotation-style {"biobert", "scibert"}
"""
import argparse
import logging
import pathlib
import re
import sys

import pandas as pd


logger = logging.getLogger("chemprot")


def find_sentences_naive(text):
    """Parse some text into sentences with naive approach.

    Parameters
    ----------
    text : str
        Text to parse into sentences.

    Returns
    -------
    sentence_boundaries : list of Tuple
        List of sentences boundaries (start_char, end_char+1).
    """
    sentence_ends = [match.span()[0] + 1 for match in re.finditer(r"\.( |\n)", text)]
    sentence_boundaries = []
    start = 0
    for end in sentence_ends:
        sentence_boundaries.append((start, end))
        start = end
        while start < len(text) and text[start] == " ":
            start += 1

    # TODO: think about fixing bad cases.
    # corrected_boundaries = []
    # for i, (b1, b2) in enumerate(zip(sentence_boundaries, sentence_boundaries[1:])):
    #     (s1, e1), (s2, e2) = b1, b2
    #     sent1 = text[s1:e1]
    #     sent2 = text[s2:e2]
    #     if sent1.endswith("i.e.") or sent1.endswith("e.g."):
    #         corrected_boundaries.append((s1, e2))
    #     else:
    #         corrected_boundaries.append((s1, e1))

    return sentence_boundaries


def get_name(split_type, file_type):
    """Get name of the chemprot file given split and file types.

    Parameters
    ----------
    split_type : str
        Split type in {"training", "test_gs", "development", "sample"}.
    file_type : str
        File type in {"abstracts", "entities", "gold_standard", "relations_gs"}

    Returns
    -------
    str
        Entire file name.
    """
    if split_type == "test_gs":
        return f"chemprot_test_{file_type}_gs.tsv"
    else:
        return f"chemprot_{split_type}_{file_type}.tsv"


def read_files(input_dir):
    """Read chemprot files of a given directory.

    Parameters
    ----------
    input_dir : pathlib.Path
        Directory where chemprot dataset is located.

    Returns
    -------
    df_abstracts : pd.DataFrame
        Dataframe containing all information contained in abstracts file.
    df_entities : pd.DataFrame
        Dataframe containing all information contained in entities file.
    df_relations : pd.DataFrame
        Dataframe containing all information contained in relations file.
    """
    _, _, split_type = input_dir.name.partition("_")

    file_abstracts = input_dir / get_name(split_type, "abstracts")
    file_entities = input_dir / get_name(split_type, "entities")
    file_relations = input_dir / get_name(split_type, "relations")

    df_abstracts = pd.read_csv(
        file_abstracts,
        sep="\t",
        header=None,
        names=["abstract_id", "title", "text"],
    )
    df_entities = pd.read_csv(
        file_entities,
        sep="\t",
        header=None,
        names=[
            "abstract_id",
            "entity_id",
            "entity_type",
            "start_char",
            "end_char",
            "entity",
        ],
    )
    # Replace GENE-Y and GENE-N by GENE
    df_entities.loc[df_entities["entity_type"].isin(["GENE-Y", "GENE-N"]), "entity_type"] = "GENE"

    df_relations = pd.read_csv(
        file_relations,
        sep="\t",
        header=None,
        names=[
            "abstract_id",
            "group",
            "evaluate",
            "relation",
            "arg_1",
            "arg_2",
        ],
        converters={
            "group": lambda s: int(s[4:]),
            "evaluate": lambda s: s.strip() == "Y",
            "arg_1": lambda s: s[5:],
            "arg_2": lambda s: s[5:],
        },
    )
    return df_abstracts, df_entities, df_relations


def main(argv=None):
    """Parse chemprot dataset and write tsv files.

    Parameters
    ----------
    argv : sequence of str
        Command lines parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--binary-classification", "-b", action="store_true")
    parser.add_argument("--annotation-style", choices=["scibert", "biobert"], required=True)
    parser.add_argument("--discard-non-eval", "-d", action="store_true")
    parser.add_argument("--keep-undefined-relations", "-k", action="store_true")
    args = parser.parse_args(argv)
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)

    df_abstracts, df_entities, df_relations = read_files(input_dir)

    # Process --discard-non-eval
    if args.discard_non_eval:
        df_relations = df_relations[df_relations["evaluate"]]

    # Collect all abstracts
    logger.info("Collecting all abstracts")
    abstracts = {}
    for abstract_id, title, text in df_abstracts.itertuples(index=False):
        full_text = title + " " + text
        sentence_boundaries = find_sentences_naive(full_text)
        abstracts[abstract_id] = (full_text, sentence_boundaries)

    # Collect all entities
    logger.info("Collecting all entities")
    entities = {}
    for abstract_id, entity_id, entity_type, start_char, end_char, entity in df_entities.itertuples(index=False):
        if abstract_id not in entities:
            entities[abstract_id] = {}
        entities[abstract_id][entity_id] = (entity_type, start_char, end_char, entity)

    # Building the sentence dataframe
    logger.info("Building the sentence dataframe")
    bad_sentence_count = 0
    output_rows = []
    column_names = ("sentence", "start_1", "end_1", "type_1", "start_2", "end_2", "type_2", "relation", "group")
    for abstract_id, group, evaluate, relation, arg_1, arg_2 in df_relations.itertuples(index=False):
        text, sentence_boundaries = abstracts[abstract_id]
        type_1, start_1, end_1, entity_1 = entities[abstract_id][arg_1]
        type_2, start_2, end_2, entity_2 = entities[abstract_id][arg_2]

        # Check that entity 1 and entity 2 are disjoint
        # equivalent check:
        # if not (end_1 < start_2 or end_2 < start_1)
        if start_2 < end_1 and start_1 < end_2:
            logger.warning(f"Overlapping entities: {start_1}:{end_1} and {start_2}:{end_2}")
            continue

        for start, end in sentence_boundaries:
            if start <= start_1 < end:
                # consistency check
                if not (start < end_1 <= end) or not (start <= start_2 < end) or not (start < end_2 <= end):
                    # raise ValueError("Consistency check failed")
                    logger.warning(f"Bad sentence: {text[start:end]}")
                    bad_sentence_count += 1
                    continue

                sentence = text[start:end]
                row = (
                    sentence,
                    start_1 - start,
                    end_1 - start,
                    type_1,
                    start_2 - start,
                    end_2 - start,
                    type_2,
                    relation,
                    group
                )
                output_rows.append(row)
                break

    df_sentences = pd.DataFrame(output_rows, columns=column_names)

    # Process --keep-undefined-relations
    if not args.keep_undefined_relations:
        df_sentences = df_sentences[df_sentences["group"] != 0]

    # Annotate entity types
    def annotate_scibert(row):
        text = row["sentence"]
        start_1 = row["start_1"]
        end_1 = row["end_1"]
        start_2 = row["start_2"]
        end_2 = row["end_2"]

        if start_1 > start_2:
            start_1, start_2 = start_2, start_1
            end_1, end_2 = end_2, end_1

        part_1 = text[:start_1]
        entity_1 = text[start_1:end_1]
        part_2 = text[end_1:start_2]
        entity_2 = text[start_2:end_2]
        part_3 = text[end_2:]

        text = part_1 + "<< " + entity_1 + " >>" + part_2 + "[[ " + entity_2 + " ]]" + part_3

        return text

    def annotate_biobert(row):
        text = row["sentence"]
        start_1 = row["start_1"]
        end_1 = row["end_1"]
        type_1 = row["type_1"]
        start_2 = row["start_2"]
        end_2 = row["end_2"]
        type_2 = row["type_2"]

        if start_1 > start_2:
            start_1, start_2 = start_2, start_1
            end_1, end_2 = end_2, end_1
            type_1, type_2 = type_2, type_1

        part_1 = text[:start_1]
        part_2 = text[end_1:start_2]
        part_3 = text[end_2:]

        text = part_1 + "@" + type_1 + "$" + part_2 + "@" + type_2 + "$" + part_3

        return text

    if args.annotation_style == "scibert":
        df_sentences["sentence"] = df_sentences.apply(annotate_scibert, axis=1)
    elif args.annotation_style == "biobert":
        df_sentences["sentence"] = df_sentences.apply(annotate_biobert, axis=1)
    else:
        raise ValueError("unreachable")

    logger.info(" Output ".center(80, "="))
    logger.info(str(df_sentences.head()))
    logger.info(f"Number of bad sentences: {bad_sentence_count} of {len(df_sentences)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, split_type = input_dir.name.partition("_")

    # Writing the output
    if args.binary_classification:
        all_relations = df_sentences["relation"].unique()
        for relation in all_relations:
            df_binary = df_sentences[["sentence", "relation"]].copy()
            df_binary["relation"] = df_binary["relation"].apply(lambda r: 1 if r == relation else 0)
            df_binary.to_csv(
                output_dir / f"{split_type}_{relation}.tsv",
                sep="\t",
                index=False,
            )
    else:
        df_sentences[["sentence", "relation"]].to_csv(
            output_dir / f"{split_type}.tsv",
            sep="\t",
            index=False,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
