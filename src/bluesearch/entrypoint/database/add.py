"""Adding an article to the database."""
import argparse
import pickle  # nosec

import sqlalchemy


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
        "path",
        type=str,
        help="""Path to the parsed file.""",
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
        # This branch never reached because of `choices` in `argparse`
        raise ValueError(f"Unrecognized database type {db_type}.")  # pragma: nocover

    with open(path, "rb") as f:
        article = pickle.load(f)  # nosec

    # Article table.

    articles_mapping = {
        "title": article.title,
        "authors": ", ".join(article.authors),
        "abstract": "\n".join(article.abstract),
    }
    articles_keys = articles_mapping.keys()
    articles_fields = ", ".join(articles_keys)
    articles_binds = f":{', :'.join(articles_keys)}"

    with engine.connect() as con:
        query = sqlalchemy.text(
            f"INSERT INTO articles({articles_fields}) VALUES({articles_binds})"
        )
        con.execute(query, articles_mapping)

    article_id = 1  # FIXME

    # Sentence table.

    mappings = []
    for position, (section, text) in enumerate(article.section_paragraphs):
        sentences_mapping = {
            "section_name": section,
            "text": text,
            "article_id": article_id,
            "paragraph_pos_in_article": position,
            "sentence_pos_in_paragraph": 1,  # FIXME
        }
        mappings.append(sentences_mapping)

    sentences_keys = sentences_mapping.keys()
    sentences_fields = ", ".join(sentences_keys)
    sentences_binds = f":{', :'.join(sentences_keys)}"

    with engine.connect() as con:
        for mapping in mappings:
            query = sqlalchemy.text(
                f"INSERT INTO sentences({sentences_fields}) VALUES({sentences_binds})"
            )
            con.execute(query, mapping)
