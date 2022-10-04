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
from elasticsearch.helpers import bulk

from bluesearch.database.article import Article
from bluesearch.k8s.connect import connect

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
    parser.add_argument(
        "--articles-index-name",
        type=str,
        default="articles",
        help="Desired name of the index holding articles.",
    )
    parser.add_argument(
        "--paragraphs-index-name",
        type=str,
        default="paragraphs",
        help="Desired name of the index holding paragraphs.",
    )
    return parser


def bulk_articles(
    inputs: Iterable[Path],
    index: str,
    progress: Optional[tqdm.std.tqdm] = None,
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
            "_index": index,
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
    inputs: Iterable[Path],
    index: str,
    progress: Optional[tqdm.std.tqdm] = None,
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
        for ppos, (section, text) in enumerate(article.section_paragraphs):
            doc = {
                "_index": index,
                "_source": {
                    "article_id": article.uid,
                    "section_name": section,
                    "text": text,
                    "paragraph_id": ppos,
                },
            }
            if progress:
                progress.update(1)
            yield doc


def run(
    parsed_path: Path,
    articles_index_name: str,
    paragraphs_index_name: str,
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
        raise ValueError(f"No articles found at '{parsed_path}'!")

    # Creating a client
    client = connect()
    logger.info("Uploading articles to the {articles_index_name} index...")
    progress = tqdm.tqdm(desc="Uploading articles", total=len(inputs), unit="articles")
    resp = bulk(client, bulk_articles(inputs, articles_index_name, progress))
    logger.info(f"Uploaded {resp[0]} articles.")

    if resp[0] == 0:
        raise ValueError(f"No articles were loaded to ES from '{parsed_path}'!")

    logger.info("Uploading articles to the {paragraphs_index_name} index...")
    progress = tqdm.tqdm(
        desc="Uploading paragraphs", total=len(inputs), unit="articles"
    )
    resp = bulk(client, bulk_paragraphs(inputs, paragraphs_index_name, progress))
    logger.info(f"Uploaded {resp[0]} paragraphs.")

    if resp[0] == 0:
        raise ValueError(f"No paragraphs were loaded to ES from '{parsed_path}'!")

    logger.info("Adding done")
    return 0
