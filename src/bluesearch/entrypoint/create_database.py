"""EntryPoint for the creation of the database."""

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

import argparse
import getpass
import logging
import pathlib
import sys

import sqlalchemy

from ._helper import CombinedHelpFormatter, configure_logging, parse_args_or_environment


def run_create_database(argv=None):
    """Run the CLI entry point.

    Parameters
    ----------
    argv : list_like of str
        The command line arguments.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=CombinedHelpFormatter,
    )
    parser.add_argument(
        "--log-file",
        "-l",
        type=str,
        metavar="<filepath>",
        default=None,
        help="In addition to stderr, log messages to a file.",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="""
        The logging level. Possible values:
        - 50 for CRITICAL
        - 40 for ERROR
        - 30 for WARNING
        - 20 for INFO
        - 10 for DEBUG
        - 0 for NOTSET
        """,
    )
    parser.add_argument(
        "--cord-data-path",
        type=str,
        help="""
        The location of the CORD-19 database. It should contain the file
        "metadata.csv" in its root, as well as JSON files with the document
        parses that are referenced in the metadata file.
        """,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--db-type",
        default="mysql",
        type=str,
        choices=("mysql", "sqlite"),
        help="Type of the database.",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        help="""
        The location of the database depending on the database type.

        For MySQL the server URL should be provided, for SQLite the
        location of the database file. Generally, the scheme part of
        the URL should be omitted, e.g. for MySQL the URL should be
        of the form 'my_sql_server.ch:1234/my_database' and for SQLite
        of the form '/path/to/the/local/database.db'.

        If missing, then the environment variable DB_URL will
        be read.
        """,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--only-mark-bad-sentences",
        default=False,
        action="store_true",
        help="""
        If set, then the database creation will be skipped and only the
        routine for marking bad sentences will be run.
        """,
    )
    env_variable_names = {
        "db_url": "DB_URL",
        "cord_data_path": "CORD_DATA_PATH",
    }
    args = parse_args_or_environment(parser, env_variable_names, argv=argv)

    # Configure logging
    configure_logging(args.log_file, args.log_level)
    logger = logging.getLogger(pathlib.Path(__file__).stem)

    logger.info(" Configuration ".center(80, "-"))
    for k, v in vars(args).items():
        logger.info(f"{k:<32}: {v}")
    logger.info("-" * 80)

    # Import libraries
    logger.info("Loading libraries")

    from ..database import CORD19DatabaseCreation, mark_bad_sentences

    # Initialise SQL database engine
    logger.info("Initialising the SQL database engine")
    if args.db_type == "sqlite":
        database_path = pathlib.Path(args.db_url)
        if not database_path.exists():
            database_path.parent.mkdir(exist_ok=True, parents=True)
            database_path.touch()
        database_url = f"sqlite:///{database_path}"
    elif args.db_type == "mysql":
        # We assume the database already exists
        password = getpass.getpass("MySQL root password: ")
        database_url = f"mysql+pymysql://root:{password}@{args.db_url}"
    else:  # pragma: no cover
        # This is unreachable because of choices=("mysql", "sqlite") in argparse
        raise ValueError(f'"{args.db_type}" is not a supported db_type.')

    # Create the database engine
    logger.info("Creating the database engine")
    # The NullPool prevents the Engine from using any connection more than once
    # This is important for multiprocessing
    engine = sqlalchemy.create_engine(database_url)

    # Launch database creation
    if not args.only_mark_bad_sentences:
        logger.info("Starting the database creation")
        db = CORD19DatabaseCreation(data_path=args.cord_data_path, engine=engine)
        db.construct()

    # Mark bad sentences
    logger.info("Marking bad sentences")
    mark_bad_sentences(engine, "sentences")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_create_database())
