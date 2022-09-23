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
    parser.description = "Add entries to a database."

    parser.add_argument(
        "db_url",
        type=str,
        help="""
        The location of the database depending on the database type.

        For MySQL and MariaDB the server URL should be provided, for SQLite the
        location of the database file. Generally, the scheme part of
        the URL should be omitted, e.g. for MySQL the URL should be
        of the form 'my_sql_server.ch:1234/my_database' and for SQLite
        of the form '/path/to/the/local/database.db'.
        """,
    )
    parser.add_argument(
        "parsed_path",
        type=Path,
        help="Path to a parsed file or to a directory of parsed files.",
    )
    parser.add_argument(
        "--db-type",
        default="sqlite",
        type=str,
        choices=("mariadb", "mysql", "postgres", "sqlite", "elasticsearch"),
        help="Type of the database.",
    )
    return parser


def _upload_sql(
    db_url: str,
    db_type: str,
    article_mappings: list[dict[str, Any]],
    paragraph_mappings: list[dict[str, Any]],
) -> None:
    """Upload the mappings to a SQL database.

    Parameters
    ----------
    db_url
        The location of the database.
    db_type
        Type of the database.
    article_mappings
        The mappings of the articles to upload.
    paragraph_mappings
        The mappings of the paragraphs to upload.

    Returns
    -------
    None
    """
    import sqlalchemy

    if db_type == "sqlite":
        engine = sqlalchemy.create_engine(f"sqlite:///{db_url}")
    elif db_type in {"mariadb", "mysql"}:
        engine = sqlalchemy.create_engine(f"mysql+pymysql://{db_url}")
    elif db_type == "postgres":
        engine = sqlalchemy.create_engine(f"postgresql+pg8000://{db_url}")
    else:
        # This branch never reached because of `choices` in `argparse`
        raise ValueError(f"Unrecognized database type {db_type}.")  # pragma: nocover

    article_keys = [
        "article_id",
        "title",
        "authors",
        "abstract",
        "pubmed_id",
        "pmc_id",
        "doi",
    ]
    article_fields = ", ".join(article_keys)
    article_binds = f":{', :'.join(article_keys)}"
    article_query = sqlalchemy.text(
        f"INSERT INTO articles({article_fields}) VALUES({article_binds})"
    )

    logger.info("Adding entries to the articles table")
    with engine.begin() as con:
        con.execute(article_query, *article_mappings)

    paragraphs_keys = [
        "section_name",
        "text",
        "article_id",
        "paragraph_pos_in_article",
    ]
    paragraphs_fields = ", ".join(paragraphs_keys)
    paragraphs_binds = f":{', :'.join(paragraphs_keys)}"
    paragraph_query = sqlalchemy.text(
        f"INSERT INTO sentences({paragraphs_fields}) VALUES({paragraphs_binds})"
    )

    logger.info("Adding entries to the sentences table")
    with engine.begin() as con:
        con.execute(paragraph_query, *paragraph_mappings)


def _upload_es(
    db_url: str,
    article_mappings: list[dict[str, Any]],
    paragraph_mappings: list[dict[str, Any]],
) -> None:
    """Upload the mappings to an Elasticsearch database.

    Parameters
    ----------
    db_url
        The location of the database.
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
    from decouple import config
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk

    urllib3.disable_warnings()

    client = Elasticsearch(
        db_url,
        basic_auth=("elastic", config("ES_PASS")),
        verify_certs=False,
    )

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
    print(resp)

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
    print(resp)


def run(
    *,
    db_url: str,
    parsed_path: Path,
    db_type: str,
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
    if db_type in ["mariadb", "mysql", "postgres", "sqlite"]:
        _upload_sql(db_url, db_type, article_mappings, paragraph_mappings)
    elif db_type == "elasticsearch":
        _upload_es(db_url, article_mappings, paragraph_mappings)
    else:
        raise RuntimeError("Database {db_type} not supported")

    logger.info("Adding done")
    return 0
