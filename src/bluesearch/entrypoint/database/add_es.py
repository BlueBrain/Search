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
from typing import Any, Iterable

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


def _upload_es(
    article_mappings: list[dict[str, Any]],
    paragraph_mappings: list[dict[str, Any]],
) -> None:
    """Upload the mappings to an Elasticsearch database.

    Parameters
    ----------
    article_mappings
        The mappings of the articles to upload.
    paragraph_mappings
        The mappings of the paragraphs to upload.

    Returns
    -------
    None
    """
    import tqdm
    import urllib3
    from bluesearch.k8s.connect import connect
    from elasticsearch.helpers import bulk

    urllib3.disable_warnings()

    client = connect()

    progress = tqdm.tqdm(
        desc="Uploading articles", total=len(article_mappings), unit="articles"
    )

    def bulk_articles(
        article_mappings: list[dict[str, Any]], progress: Any = None
    ) -> Iterable[dict[str, Any]]:
        for article in article_mappings:
            doc = {
                "_index": "articles",
                "_id": article["article_id"],
                "_source": {
                    "article_id": article["article_id"],
                    "authors": article["title"],
                    "title": article["authors"],
                    "abstract": article["abstract"],
                    "pubmed_id": article["pubmed_id"],
                    "pmc_id": article["pmc_id"],
                    "doi": article["doi"],
                },
            }
            if progress:
                progress.update(1)
            yield doc

    resp = bulk(client, bulk_articles(article_mappings, progress))
    logger.info(f"Uploaded {resp[0]} articles.")

    progress = tqdm.tqdm(
        desc="Uploading paragraphs", total=len(paragraph_mappings), unit="paragraphs"
    )

    def bulk_paragraphs(
        paragraph_mappings: list[dict[str, Any]], progress: Any = None
    ) -> Iterable[dict[str, Any]]:
        """Yield a paragraph mapping as a document to upload to Elasticsearch.

        Parameters
        ----------
        paragraph_mappings
            The mappings of the paragraphs to upload.

        Returns
        -------
        Iterable[dict[str, Any]]
            A generator of documents to upload to Elasticsearch.
        """
        for paragraph in paragraph_mappings:
            doc = {
                "_index": "paragraphs",
                "_source": {
                    "article_id": paragraph["article_id"],
                    "section_name": paragraph["section_name"],
                    "text": paragraph["text"],
                    "paragraph_id": paragraph["paragraph_pos_in_article"],
                },
            }
            if progress:
                progress.update(1)
            yield doc

    resp = bulk(client, bulk_paragraphs(paragraph_mappings, progress))
    logger.info(f"Uploaded {resp[0]} paragraphs.")


def run(
    *,
    parsed_path: Path,
) -> int:
    """Add an entry to the database.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    from typing import Iterable

    from bluesearch.database.article import Article

    inputs: Iterable[Path]
    if parsed_path.is_file():
        inputs = [parsed_path]
    elif parsed_path.is_dir():
        inputs = sorted(parsed_path.glob("*.json"))
    else:
        raise ValueError(
            "Argument 'parsed_path' should be a path to an existing file or directory!"
        )

    articles = []
    for inp in inputs:
        serialized = inp.read_text("utf-8")
        article = Article.from_json(serialized)
        articles.append(article)

    if not articles:
        raise RuntimeWarning(f"No article was loaded from '{parsed_path}'!")

    logger.info("Splitting text into paragraphs")
    article_mappings = []
    paragraph_mappings = []

    for article in articles:
        logger.info(f"Processing {article.uid}")

        article_mapping = {
            "article_id": article.uid,
            "title": article.title,
            "authors": ", ".join(article.authors),
            "abstract": "\n".join(article.abstract),
            "pubmed_id": article.pubmed_id,
            "pmc_id": article.pmc_id,
            "doi": article.doi,
        }
        article_mappings.append(article_mapping)

        for ppos, (section, text) in enumerate(article.section_paragraphs):
            paragraph_mapping = {
                "section_name": section,
                "text": text,
                "article_id": article.uid,
                "paragraph_pos_in_article": ppos,
            }
            paragraph_mappings.append(paragraph_mapping)

    if not paragraph_mappings:
        raise RuntimeWarning(f"No sentence was extracted from '{parsed_path}'!")

    # Persistence.
    _upload_es(article_mappings, paragraph_mappings)

    logger.info("Adding done")
    return 0
