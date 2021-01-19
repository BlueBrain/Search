"""EntryPoint for the creation of the database."""

# BBSearch is a text mining toolbox focused on scientific use cases.
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

from ._helper import configure_logging


def run_create_database(argv=None):
    """Run the CLI entry point.

    Parameters
    ----------
    argv : list_like of str
        The command line arguments.
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        default="/raid/projects/bbs/logs/",
        type=str,
        help="The directory path where to save the logs.",
    )
    parser.add_argument(
        "--log-name",
        default="database_creation.log",
        type=str,
        help="The name of the log file.",
    )
    parser.add_argument(
        "--data-path",
        default="/raid/sync/proj115/bbs_data/cord19_v65",
        type=str,
        help="The directory path where the metadata.csv and JSON files are located, "
        "files needed to create the database",
    )
    parser.add_argument(
        "--db-type",
        default="mysql",
        type=str,
        choices=("mysql", "sqlite"),
        help="Type of the database.",
    )
    parser.add_argument(
        "--database-url",
        default="dgx1.bbp.epfl.ch:8853/cord19_v47",
        type=str,
        help=(
            "The location of the database depending on the database type. "
            "For MySQL the server URL should be provided, for SQLite the "
            "location of the database file. Generally, the scheme part of "
            "the URL should be omitted, e.g. for MySQL the URL should be "
            "of the form 'my_sql_server.ch:1234/my_database' and for SQLite "
            "of the form '/path/to/the/local/database.db'."
        ),
    )
    parser.add_argument(
        "--only-mark-bad-sentences",
        default=False,
        action="store_true",
        help=(
            "If set, then the database creation will be skipped and only the "
            "routine for marking bad sentences will be run"
        ),
    )
    args = parser.parse_args(argv)
    print(" Configuration ".center(80, "-"))
    print(f"log-dir                 : {args.log_dir}")
    print(f"log-name                : {args.log_name}")
    print(f"data-path               : {args.data_path}")
    print(f"db-type                 : {args.db_type}")
    print(f"only-mark-bad-sentences : {args.only_mark_bad_sentences}")
    print("-" * 80)
    input("Press any key to continue... ")

    # Configure logging
    log_file = pathlib.Path(args.log_dir) / args.log_name
    configure_logging(log_file, logging.INFO)
    logger = logging.getLogger(pathlib.Path(__file__).stem)

    # Import libraries
    logger.info("Loading libraries")

    from ..database import CORD19DatabaseCreation, mark_bad_sentences

    # Initialise SQL database engine
    logger.info("Initialising the SQL database engine")
    if args.db_type == "sqlite":
        database_path = pathlib.Path(args.database_url)
        if not database_path.exists():
            database_path.parent.mkdir(exist_ok=True, parents=True)
            database_path.touch()
        database_url = f"sqlite:///{database_path}"
    elif args.db_type == "mysql":
        # We assume the database already exists
        password = getpass.getpass("MySQL root password: ")
        database_url = f"mysql+pymysql://root:{password}@{args.database_url}"
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
        db = CORD19DatabaseCreation(data_path=args.data_path, engine=engine)
        db.construct()

    # Mark bad sentences
    logger.info("Marking bad sentences")
    mark_bad_sentences(engine, "sentences")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_create_database())
