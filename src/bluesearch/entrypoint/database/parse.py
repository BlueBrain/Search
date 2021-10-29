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
"""Parsing articles."""
import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Iterable, Iterator

from defusedxml import ElementTree

from bluesearch.database.article import (
    Article,
    ArticleParser,
    CORD19ArticleParser,
    PMCXMLParser,
    PubMedXMLParser,
)

logger = logging.getLogger(__name__)


def init_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initialise the argument parser for the parse subcommand.

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
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.description = "Parse one or several articles."

    parser.add_argument(
        "input_type",
        type=str,
        choices=("cord19-json", "pmc-xml", "pubmed-xml", "pubmed-xml-set"),
        help="""
        Format of the input. If parsing several articles, all articles
        must have same format.
        """,
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="""
        Path to a file or directory. If a directory, all articles
        inside the directory will be parsed.
        """,
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="""
        Path to a directory where parsed article(s) will be saved.
        If it does not exist yet, a directory with this path is created.
        """,
    )
    return parser


def iter_parsers(input_type: str, input_path: Path) -> Iterator[ArticleParser]:
    """Return an iterator of initialized parsers for the given input."""
    if input_type == "cord19-json":
        with input_path.open() as f:
            data = json.load(f)
            yield CORD19ArticleParser(data)

    elif input_type == "pmc-xml":
        yield PMCXMLParser(input_path)

    elif input_type == "pubmed-xml":
        yield PubMedXMLParser(input_path)

    elif input_type == "pubmed-xml-set":
        articles = ElementTree.parse(str(input_path))
        for article in articles.iter("PubmedArticle"):
            yield PubMedXMLParser(article)

    else:
        raise ValueError(f"Unsupported input type '{input_type}'!")


def run(
    *,
    input_type: str,
    input_path: Path,
    output_dir: Path,
) -> int:
    """Parse one or several articles.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    inputs: Iterable[Path]
    if input_path.is_file():
        inputs = [input_path]
    elif input_path.is_dir():
        inputs = sorted(input_path.glob("*"))
    else:
        raise ValueError(
            "Argument 'input_path' should be a path to an existing file or directory!"
        )

    output_dir.mkdir(exist_ok=True)

    for input_path in inputs:
        logger.info(f"Parsing {input_path.name}")

        try:
            parsers = iter_parsers(input_type, input_path)

            for parser in parsers:
                article = Article.parse(parser)
                output_file = output_dir / f"{article.uid}.json"

                if output_file.exists():
                    raise FileExistsError(f"Output '{output_file}' already exists!")
                else:
                    serialized = article.to_json()
                    output_file.write_text(serialized, "utf-8")

        except Exception as e:
            warnings.warn(
                f'Failed parsing file "{input_path}":\n {e}', category=RuntimeWarning
            )

    logger.info("Parsing done")

    return 0
