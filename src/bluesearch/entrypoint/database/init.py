"""Initialization of the database."""
import argparse
import logging

logger = logging.getLogger(__name__)


def init_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initialise the argument parser for the init subcommand.

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
    parser.description = "Initialize a database."

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
        "--db-type",
        default="sqlite",
        type=str,
        choices=("mariadb", "mysql", "postgres", "sqlite"),
        help="Type of the database.",
    )
    return parser


def run(
    *,
    db_url: str,
    db_type: str,
) -> int:
    """Initialize database.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    logger.info("Importing dependencies")
    import sqlalchemy

    from bluesearch.entrypoint.database.schemas import schema_articles, schema_sentences

    if db_type == "sqlite":
        engine = sqlalchemy.create_engine(f"sqlite:///{db_url}")

    elif db_type in {"mariadb", "mysql"}:
        engine = sqlalchemy.create_engine(f"mysql+pymysql://{db_url}")

    elif db_type == "postgres":
        engine = sqlalchemy.create_engine(f"postgresql+pg8000://{db_url}")

    else:
        # This branch never reached because of `choices` in `argparse`
        raise ValueError(f"Unrecognized database type {db_type}")  # pragma: nocover

    metadata = sqlalchemy.MetaData()

    # Creation of the schema of the tables
    schema_articles(metadata)
    schema_sentences(metadata)

    # Construction
    with engine.begin() as connection:
        metadata.create_all(connection)

    logger.info("Initialization done")

    return 0
