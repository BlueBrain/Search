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
"""Download articles from different sources."""
import argparse
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def valid_date(s: str) -> datetime:
    """Check the input is a valid date.

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
        msg = "not a valid date: {0!r}".format(s)
        raise argparse.ArgumentTypeError(msg)


def init_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initialise the argument parser for the download subcommand.

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
    parser.description = "Download articles."

    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Directory to save the downloaded articles.",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        required=True,
        choices=("arxiv", "biorxiv", "medrxiv", "pmc", "pubmed"),
        help="Source of the download.",
    )
    parser.add_argument(
        "--from-month",
        "-f",
        type=valid_date,
        required=True,
        help="The start date for the download in format YYYY-MM",
    )
    return parser


def run(
    output_dir: Path,
    source: str,
    from_month: datetime,
) -> int:
    """Download articles of a source from a specific date.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    return 0
