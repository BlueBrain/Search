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
import argparse
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    """Create a parser."""
    parser = argparse.ArgumentParser(
        description="Add entries to the database.",
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
        "parsed_path",
        type=Path,
        help="Path to a parsed file or to a directory of parsed files.",
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
    parsed_path: Path,
    db_type: str,
) -> None:
    """Add an entry to the database.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    import pickle  # nosec
    from typing import Iterable

    import sqlalchemy

    from bluesearch.database.identifiers import generate_uid
    from bluesearch.utils import load_spacy_model

    if db_type == "sqlite":
        engine = sqlalchemy.create_engine(f"sqlite:///{db_url}")

    elif db_type == "mysql":
        engine = sqlalchemy.create_engine(f"mysql+pymysql://{db_url}")

    else:
        # This branch never reached because of `choices` in `argparse`
        raise ValueError(f"Unrecognized database type {db_type}.")  # pragma: nocover

    inputs: Iterable[Path]
    if parsed_path.is_file():
        inputs = [parsed_path]
    elif parsed_path.is_dir():
        inputs = sorted(parsed_path.glob("*.pkl"))
    else:
        raise ValueError(
            "Argument 'parsed_path' should be a path to an existing file or directory!"
        )

    articles = []
    for inp in inputs:
        with inp.open("rb") as f:
            article = pickle.load(f)  # nosec
            articles.append(article)

    nlp = load_spacy_model("en_core_sci_lg", disable=["ner"])

    article_mappings = []
    sentence_mappings = []

    for article in articles:

        # TODO At the moment, no identifiers are extracted. This is a patch meanwhile.
        article_id = generate_uid((article.title,))
        article_mapping = {
            "article_id": article_id,
            "title": article.title,
            "authors": ", ".join(article.authors),
            "abstract": "\n".join(article.abstract),
        }
        article_mappings.append(article_mapping)

        swapped = (
            (text, (section, ppos))
            for ppos, (section, text) in enumerate(article.section_paragraphs)
        )
        for text, (section, ppos) in swapped:
            doc = nlp(text)
            for spos, sent in enumerate(doc.sents):
                sentence_mapping = {
                    "section_name": section,
                    # Using 'sent.text' gives an issue on Python 3.7 and Ubuntu 20.04.
                    "text": str(sent),
                    "article_id": article_id,
                    "paragraph_pos_in_article": ppos,
                    "sentence_pos_in_paragraph": spos,
                }
                sentence_mappings.append(sentence_mapping)

    # Persistence.

    article_keys = ["article_id", "title", "authors", "abstract"]
    article_fields = ", ".join(article_keys)
    article_binds = f":{', :'.join(article_keys)}"
    article_query = sqlalchemy.text(
        f"INSERT INTO articles({article_fields}) VALUES({article_binds})"
    )

    with engine.begin() as con:
        con.execute(article_query, *article_mappings)

    sentences_keys = [
        "section_name",
        "text",
        "article_id",
        "paragraph_pos_in_article",
        "sentence_pos_in_paragraph",
    ]
    sentences_fields = ", ".join(sentences_keys)
    sentences_binds = f":{', :'.join(sentences_keys)}"
    sentence_query = sqlalchemy.text(
        f"INSERT INTO sentences({sentences_fields}) VALUES({sentences_binds})"
    )

    with engine.begin() as con:
        con.execute(sentence_query, *sentence_mappings)
