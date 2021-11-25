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
        "source",
        type=str,
        choices=("arxiv", "biorxiv", "medrxiv", "pmc", "pubmed"),
        help="Source of the download.",
    )
    parser.add_argument(
        "from_month",
        type=convert_to_datetime,
        help="The starting month (included) for the download in format YYYY-MM. "
        "All papers from the given month until today will be downloaded.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save the downloaded articles.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="""
        Display requests for the download.
        """,
    )
    return parser


def run(source: str, from_month: datetime, output_dir: Path, dry_run: bool) -> int:
    """Download articles of a source from a specific date.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    from bluesearch.database.download import download_pmc_articles, get_pmc_urls

    if source == "pmc":
        url_dict = {}
        for component in {"author_manuscript", "oa_comm", "oa_noncomm"}:
            url_dict[component] = get_pmc_urls(component, from_month)

        if dry_run:
            for component, url_list in url_dict.items():
                print(f"URL requests from {component}:")
                print(*url_list, sep="\n")
            return 0

        logger.info("Start downloading PMC papers.")
        for component, url_list in url_dict.items():
            component_dir = output_dir / component
            logger.info(
                f"Start downloading {component} in {component_dir.resolve()}"
            )
            component_dir.mkdir(exist_ok=True, parents=True)
            download_pmc_articles(url_list, component_dir)
        return 0

    return 0
