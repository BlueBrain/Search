import argparse
import json
import sys
from typing import List, Optional

import sqlalchemy

import bluesearch.database.article as article_module


def get_parser() -> argparse.ArgumentParser:
    """Create a parser."""
    parser = argparse.ArgumentParser(
        description="Add entries.",
    )
    parser.add_argument(
        "db_url",
        type=str,
        help="""
        The location of the database depending on the database type.

        For MySQL the server URL should be provided, for SQLite the
        location of the database file. Generally, the scheme part of
        the URL should be omitted, e.g. for MySQL the URL should be
        of the form 'my_sql_server.ch:1234/my_database' and for SQLite
        of the form '/path/to/the/local/database.db'.
        """,
    )
    parser.add_argument(
        "parser",
        type=str,
        help="""Parser class.""",
    )
    parser.add_argument(
        "path",
        type=str,
        help="""Path to the file/directory to be parsed.""",
    )
    parser.add_argument(
        "--db-type",
        default="sqlite",
        type=str,
        choices=("mysql", "sqlite"),
        help="Type of the database.",
    )
    return parser


def run(
    *,
    db_url: str,
    parser: str,
    path: str,
    db_type: str,
) -> None:
    """Add an entry to the database.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """

    if db_type == "sqlite":
        engine = sqlalchemy.create_engine(f"sqlite:///{db_url}")

    elif db_type == "mysql":
        raise NotImplementedError

    else:
        raise ValueError(f"Unrecognize database type {db_type}.")  # pragma: nocover

    valid_parsers = [x.__name__ for x in article_module.ArticleParser.__subclasses__()]

    if parser not in valid_parsers:
        raise ValueError(f"Unsupported parser {parser}. Valid parsers: {valid_parsers}")

    parser_cls = getattr(article_module, parser)

    # We should unify this somehow to make sure all parsers have the same constructor
    if parser == "CORD19ArticleParser":
        with open(path, "r") as f:
            parser_inst = parser_cls(json.load(f))
    else:
        parser_inst = parser_cls(path)  # not covered since we do not have other parsers

    article = article_module.Article.parse(parser_inst)

    with engine.connect() as con:
        query = sqlalchemy.text("INSERT INTO articles(title) VALUES(:title)")
        con.execute(query, {"title": article.title})
