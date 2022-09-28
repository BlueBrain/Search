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
"""Adding articles to the database."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from bluesearch.database.article import Article

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
    parser.description = "Add entries to a database. \
        ENV variables are used to configure the database connection.\
        ES_URL and ES_PASS are required."

    parser.add_argument(
        "parsed_path",
        type=Path,
        help="Path to a parsed file or to a directory of parsed files.",
    )
    return parser


def bulk_articles(
    inputs: Iterable[Path], progress: Optional[tqdm.std.tqdm] = None
) -> Iterable[dict[str, Any]]:
    """Yield an article mapping as a document to upload to Elasticsearch.

    Parameters
    ----------
    inputs
        Paths to the parsed files.
    progress
        Progress bar to update.

    Yields
    ------
    dict[str, Any]
        A document to upload to Elasticsearch.
    """
    for inp in inputs:
        serialized = inp.read_text("utf-8")
        article = Article.from_json(serialized)
        doc = {
            "_index": "articles",
            "_id": article.uid,
            "_source": {
                "article_id": article.uid,
                "authors": article.authors,
                "title": article.title,
                "abstract": article.abstract,
                "pubmed_id": article.pubmed_id,
                "pmc_id": article.pmc_id,
                "doi": article.doi,
            },
        }
        if progress:
            progress.update(1)
        yield doc


def bulk_paragraphs(
    inputs: Iterable[Path], progress: Optional[tqdm.std.tqdm] = None
) -> Iterable[dict[str, Any]]:
    """Yield a paragraph mapping as a document to upload to Elasticsearch.

    Parameters
    ----------
    paragraph_mappings
        The mappings of the paragraphs to upload.
    progress
        Progress bar to update.

    Yields
    ------
    dict[str, Any]
        A document to upload to Elasticsearch.
    """
    for inp in inputs:
        serialized = inp.read_text("utf-8")
        article = Article.from_json(serialized)
        # add abstract to paragraphs in order to be able to search for abstracts
        for i, abstract in enumerate(article.abstract):
            doc = {
                "_index": "paragraphs",
                "_source": {
                    "article_id": article.uid,
                    "section": "abstract",
                    "text": abstract,
                    "paragraph_id": i,
                },
            }
            yield doc
        # add body paragraphs
        for ppos, (section, text) in enumerate(article.section_paragraphs):
            doc = {
                "_index": "paragraphs",
                "_source": {
                    "article_id": article.uid,
                    "section": section,
                    "text": text,
                    "paragraph_id": ppos,
                },
            }
            if progress:
                progress.update(1)
            yield doc


def run(
    client: Elasticsearch,
    parsed_path: Path,
) -> int:
    """Add an entry to the database.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    inputs: Iterable[Path]
    if parsed_path.is_file():
        inputs = [parsed_path]
    elif parsed_path.is_dir():
        inputs = sorted(parsed_path.glob("*.json"))
    else:
        raise ValueError(
            "Argument 'parsed_path' should be a path to an existing file or directory!"
        )

    if len(inputs) == 0:
        raise RuntimeWarning(f"No articles found at '{parsed_path}'!")

    logger.info("Uploading articles to the database...")
    progress = tqdm.tqdm(desc="Uploading articles", total=len(inputs), unit="articles")
    resp = bulk(client, bulk_articles(inputs, progress))
    logger.info(f"Uploaded {resp[0]} articles.")

    if resp[0] == 0:
        raise RuntimeWarning(f"No articles were loaded to ES from '{parsed_path}'!")

    logger.info("Uploading articles to the database...")
    progress = tqdm.tqdm(
        desc="Uploading paragraphs", total=len(inputs), unit="articles"
    )
    resp = bulk(client, bulk_paragraphs(inputs, progress))
    logger.info(f"Uploaded {resp[0]} paragraphs.")

    if resp[0] == 0:
        raise RuntimeWarning(f"No paragraphs were loaded to ES from '{parsed_path}'!")

    logger.info("Adding done")
    return 0
