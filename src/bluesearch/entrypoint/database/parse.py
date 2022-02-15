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
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Iterator

from defusedxml import ElementTree

from bluesearch.database.article import (
    Article,
    ArticleParser,
    CORD19ArticleParser,
    JATSXMLParser,
    PubMedXMLParser,
    TEIXMLParser,
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
    parser.description = "Parse one or several articles."

    parser.add_argument(
        "input_type",
        type=str,
        choices=(
            "cord19-json",
            "jats-xml",
            "jats-meca",
            "pubmed-xml",
            "pubmed-xml-set",
            "tei-xml",
            "tei-xml-arxiv",
        ),
        help="""
        Format of the input.
        If parsing several articles, all articles must have the same format.
        'jats-xml' could be used for articles from PubMed Central.
        'jats-meca' could be used for articles from bioRxiv and medRxiv.
        'tei-xml-arxiv' should be used for TEI XML generated from arXiv PDF articles.
        """,
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="""
        Path to a file or directory. If a directory, all articles
        inside the directory will be parsed. If not provided the
        user is supposed to pipe a space separated list of article paths to
        the standard input.
        """,
        nargs="?",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="""
        Path to a directory where parsed article(s) will be saved.
        If it does not exist yet, a directory with this path is created.
        """,
    )
    parser.add_argument(
        "-m",
        "--match-filename",
        type=str,
        help="""
        Parse only files with a name matching the given regular expression.
        Ignored when 'input_path' is a path to a file.
        """,
    )
    parser.add_argument(
        "-R",
        "--recursive",
        action="store_true",
        help="""
        Parse files recursively.
        """,
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="""
        Display files to parse without parsing them.
        Especially useful when using '--match-filename' and / or '--recursive'.
        """,
    )
    return parser


def iter_parsers(input_type: str, input_path: Path) -> Iterator[ArticleParser]:
    """Return an iterator of initialized parsers for the given input."""
    if input_type == "cord19-json":
        with input_path.open() as f:
            data = json.load(f)
            yield CORD19ArticleParser(data)

    elif input_type == "jats-xml":
        yield JATSXMLParser.from_xml(input_path)

    elif input_type == "jats-meca":
        yield JATSXMLParser.from_zip(input_path)

    elif input_type == "pubmed-xml":
        yield PubMedXMLParser(input_path)

    elif input_type == "pubmed-xml-set":
        with gzip.open(input_path) as mygzip:
            articles_str = mygzip.read()
        articles = ElementTree.fromstring(articles_str)
        for article in articles.iter("PubmedArticle"):
            yield PubMedXMLParser(article)

    elif input_type.startswith("tei-xml"):
        if input_type == "tei-xml-arxiv":
            is_arxiv = True
        else:
            is_arxiv = False
        yield TEIXMLParser(input_path, is_arxiv=is_arxiv)

    else:
        raise ValueError(f"Unsupported input type '{input_type}'!")


def run(
    *,
    input_type: str,
    input_path: Path | None,
    output_dir: Path,
    match_filename: str | None,
    recursive: bool,
    dry_run: bool,
) -> int:
    """Parse one or several articles.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    from bluesearch.utils import find_files

    if input_path is None:
        if sys.stdin.isatty():
            # Real terminal session
            logger.error("No input files provided")
            return 1
        else:
            # Piped session
            input_lines = sys.stdin.read().split()
            inputs = []

            for line in input_lines:
                path = Path(line)
                if path.exists():
                    inputs.append(path)

    else:
        try:
            inputs = find_files(input_path, recursive, match_filename)
        except ValueError:
            logger.error(
                "Argument 'input_path' should be a path "
                "to an existing file or directory!"
            )
            return 1

    if dry_run:
        # Inputs are already sorted.
        print(*inputs, sep="\n")
        return 0

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
