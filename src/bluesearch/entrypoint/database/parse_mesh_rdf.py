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
import gzip
import json
import logging
import pathlib

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
    from bluesearch.database import mesh

    if not mesh_nt_gz_file.exists():
        logger.error(f"The file {mesh_nt_gz_file} does not exist.")
        return 1
    if not mesh_nt_gz_file.is_file():
        logger.error(f"The path {mesh_nt_gz_file} must be a file.")
        return 1

    logger.info(f"Parsing the MeSH file {mesh_nt_gz_file.resolve().as_uri()}")
    with gzip.open(mesh_nt_gz_file, "rt") as fh:
        tree_number_to_label = mesh.parse_tree_numbers(fh)

    logger.info(f"Saving results to {output_json_file.resolve().as_uri()}")
    with open(output_json_file, "w") as fh:
        json.dump(tree_number_to_label, fh)

    logger.info("Done")
    return 0
