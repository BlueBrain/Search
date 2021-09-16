"""Initialization of the database."""
import argparse

import sqlalchemy


def get_parser() -> argparse.ArgumentParser:
    """Create a parser."""
    parser = argparse.ArgumentParser(
        description="Initialize.",
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
    db_type: str,
) -> None:
    """Initialize database.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    if db_type == "sqlite":
        engine = sqlalchemy.create_engine(f"sqlite:///{db_url}")

    elif db_type == "mysql":
        engine = sqlalchemy.create_engine(f"mysql+pymysql://{db_url}")

    else:
        # This branch never reached because of `choices` in `argparse`
        raise ValueError(f"Unrecognized database type {db_type}")  # pragma: nocover

    metadata = sqlalchemy.MetaData()

    # Creation of the schema of the tables
    sqlalchemy.Table(
        "articles",
        metadata,
        sqlalchemy.Column(
            "article_id", sqlalchemy.Integer(), primary_key=True, autoincrement=True
        ),
        sqlalchemy.Column("title", sqlalchemy.Text()),
    )

    # Construction
    with engine.begin() as connection:
        metadata.create_all(connection)