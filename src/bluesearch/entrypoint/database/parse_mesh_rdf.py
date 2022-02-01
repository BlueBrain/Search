#  Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
#  Copyright (C) 2022 Blue Brain Project, EPFL.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
"""CLI sub-command for parsing MeSH RDF files."""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import logging
import pathlib
import re
import typing

logger = logging.getLogger(__name__)


def init_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initialise the argument parser for the parse-mesh-rdf subcommand.

    Parameters
    ----------
    parser
        The argument parser to initialise.

    Returns
    -------
    argparse.ArgumentParser
        The initialised argument parser. The same object as the `parser`
        argument.
    """
    parser.description = "Parse a MeSH RDF file in N-Triples format."
    parser.add_argument(
        "mesh_nt_gz_file",
        type=pathlib.Path,
        help="""
        Path to a "mesh*.nt.gz" file downloaded from
        https://nlmpubs.nlm.nih.gov/projects/mesh/rdf/
        """,
    )
    parser.add_argument(
        "output_json_file",
        type=pathlib.Path,
        help="""
        The output file for parsing results. The JSON file will contain a
        flat dictionary with MeSH tree names as keys and corresponding topic
        labels as values.
        """,
    )
    return parser


def run(*, mesh_nt_gz_file: pathlib.Path, output_json_file: pathlib.Path) -> int:
    """Parse a MeSH RDF file to extract the topic tree structure.

    See the description of the `init_parser` command for more information on
    the command and its parameters.
    """
    if not mesh_nt_gz_file.exists():
        logger.error(f"The file {mesh_nt_gz_file} does not exist.")
        return 1
    if not mesh_nt_gz_file.is_file():
        logger.error(f"The path {mesh_nt_gz_file} must be a file.")
        return 1

    logger.info(f"Parsing the MeSH file {mesh_nt_gz_file.resolve().as_uri()}")
    with gzip.open(mesh_nt_gz_file, "rt") as fh:
        tree_number_to_label = parse_tree_numbers(fh)

    logger.info(f"Saving results to {output_json_file.resolve().as_uri()}")
    with open(output_json_file, "w") as fh:
        json.dump(tree_number_to_label, fh)

    logger.info("Done")
    return 0


def parse_tree_numbers(nt_stream: typing.TextIO) -> dict[str, str]:
    """Parse the MeSH topic tree from a stream of MeSH RDF N-tuples.

    Parameters
    ----------
    nt_stream
        A text stream of MeSH RDF N-tuples. This is intended to work with
        the content of the MeSH files downloaded from the following website:
        https://nlmpubs.nlm.nih.gov/projects/mesh/rdf

    Returns
    -------
    dict[str, str]
        A dictionary representing the parsed MeSH topic tree. The keys are
        the tree numbers that uniquely identify a topic. The values are the
        corresponding topic labels. Note that the topic labels are not
        unique. For example, the two tree numbers `F04.096.628.255.500` and
        `H01.158.610.030` have both the same label "Cognitive Neuroscience".
    """
    id_to_label = {}
    id_to_tree_numbers = collections.defaultdict(list)

    # Regexes we need for parsing
    # Each line must be a triple subject, predicate, object.
    p_line = re.compile(r"(<.*>) (<.*>) (.*) \.")
    # We're only interested in subjects that represent descriptors. It appears
    # their ID is of the form "Dxxx..." where "xxx" are digits.
    p_desc = re.compile(r"<http://id\.nlm\.nih\.gov/mesh/\d{4}/(D\d{3,})>")
    # The topic label is in quotes and is followed by a language suffix. We'll
    # only keep labels that are in English
    p_en_label = re.compile(r"\"(.*)\"@en")
    # The "\d{4}" part is going to be the year, e.g. "2022", the actual tree
    # number is some combination of characters that we leave open.
    p_tree_number = re.compile(r"<http://id\.nlm\.nih\.gov/mesh/\d{4}/(.*)>")

    # The two predicates we'll be looking for
    pred_label = "<http://www.w3.org/2000/01/rdf-schema#label>"
    pred_tree_number = "<http://id.nlm.nih.gov/mesh/vocab#treeNumber>"

    for i, line in enumerate(nt_stream):
        if i % 1_000_000 == 0:
            logger.info(f"Parsed {i:,d} lines")

        # Parse the triple
        subj, pred, obj = p_line.fullmatch(line.strip()).groups()

        # Extract the descriptor ID
        m_desc = p_desc.fullmatch(subj)
        if not m_desc:
            # Subject is not a descriptor
            continue
        id_ = m_desc.group(1)

        # Parse the descriptor label or tree number
        if pred == pred_label:
            m_label = p_en_label.fullmatch(obj)
            if not m_label:
                continue  # not an English label
            label = m_label.group(1)
            if id_ in id_to_label:
                raise RuntimeError(
                    f"Multiple labels for ID={id_}: {id_to_label[id_]}, {label}"
                )
            id_to_label[id_] = label
        elif pred == pred_tree_number:
            m_tree_number = p_tree_number.fullmatch(obj)
            if not m_tree_number:
                raise RuntimeError(f"Cannot parse tree number: {obj}")
            id_to_tree_numbers[id_].append(m_tree_number.group(1))

    # Given "id => label" and "id => tree numbers" find "tree number => label"
    logger.info("Labeling tree numbers")
    tree_number_to_label = {}
    for id_, label in id_to_label.items():
        for tree_number in id_to_tree_numbers[id_]:
            if tree_number in tree_number_to_label:
                raise RuntimeError(f"Duplicate tree number: {tree_number}")
            tree_number_to_label[tree_number] = label

    return tree_number_to_label
