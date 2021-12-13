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
"""Extract topic of articles."""
import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def init_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initialise the argument parser for the add subcommand.

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
    parser.description = "Extract topic of articles."

    parser.add_argument(
        "input_source",
        type=str,
        choices=(
            "arxiv",
            "biorxiv",
            "medrxiv",
            "pmc",
            "pubmed",
        ),
        help="""
        Format of the input.
        If extracting topic of several articles, all articles must have the same format.
        """,
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="""
        Path to a file or directory. If a directory, topic will be extracted for
        all articles inside the directory.
        """,
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="""
        Path to the file where the results will be saved.
        If it does not exist yet, the file is created.
        """,
    )
    parser.add_argument(
        "-R",
        "--recursive",
        action="store_true",
        help="""
        Find articles recursively.
        """,
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="""
        If output_file exists and overwrite is true, the output file is overwrote.
        """,
    )
    return parser


def run(
        *,
        input_source: str,
        input_path: Path,
        output_file: Path,
        recursive: bool,
        overwrite: bool,
) -> int:
    """Extract topic of articles.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """

    return 0
