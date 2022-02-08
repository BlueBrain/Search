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
"""Run the overall pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Iterator

from defusedxml import ElementTree

from bluesearch.database.article import ArticleSource

logger = logging.getLogger(__name__)

def convert_to_datetime(s: str) -> datetime:
    """Try to convert a string to a datetime.

    Parameters
    ----------
    s
        String to be check as a valid date.

    Returns
    -------
    datetime
        The date specified in the input string.

    Raises
    ------
    ArgumentTypeError
        When the specified string has not a valid date format.
    """
    try:
        return datetime.strptime(s, "%Y-%m")
    except ValueError:
        msg = f"{s} is not a valid date"
        raise argparse.ArgumentTypeError(msg)


def init_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initialise the argument parser for the run subcommand.

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
    parser.description = "Run the overall pipeline."

    parser.add_argument(
        "source",
        type=str,
        choices=[member.value for member in ArticleSource],
        help="Source of the articles.",
    )
    parser.add_argument(
        "from_month",
        type=convert_to_datetime,
        help="The starting month (included) for the download in format YYYY-MM. "
        "All papers from the given month until today will be downloaded.",
    )
    parser.add_argument(
        "filter_config",
        type=Path,
        help="""
        Path to a .JSONL file that defines all the rules for filtering.
        """,
    )
    return parser


def run(
    *,
    source: str,
    from_month: datetime,
    filter_config: Path,
) -> int:
    """Run overall pipeline.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    logger.info("Starting the overall pipeline")

    return 0
