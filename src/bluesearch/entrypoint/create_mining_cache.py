"""EntryPoint for mining a database and saving of extracted items in a cache."""

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
from sqlalchemy.pool import NullPool

from ..utils import get_available_spacy_models
from ._helper import CombinedHelpFormatter, configure_logging, parse_args_or_environment


def run_create_mining_cache(argv=None):
    """Mine all texts in database and save results in a cache.

    Parameters
    ----------
    argv : list_like of str
        The command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Mine the CORD-19 database and cache the results.",
        formatter_class=CombinedHelpFormatter,
    )
    parser.add_argument(
        "--data-and-models-dir",
        type=str,
        help="""
        The local path to the "data_and_models" directory. It will
        be used to load the available spacy models from
        <data-and-models-dir>/models/ner_er/

        If missing, then the environment variable BBS_DATA_AND_MODELS_DIR
        will be read.
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
        "--target-table-name",
        default="mining_cache_temporary",
        type=str,
        help="The name of the target mining cache table",
    )
    parser.add_argument(
        "--n-processes-per-model",
        default=1,
        type=int,
        help="""
        Each mining model is run in parallel with respect to the others.
        In addition to that, n-processes-per-model are used to run in
        parallel a single mining model.
        """,
    )
    parser.add_argument(
        "--restrict-to-etypes",
        type=str,
        default=None,
        help="""
        Comma-separated list of entity types to detect
        to populate the cache. By default, all models in
        data_and_models/models/ner_er/ are run.
        """,
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

    # Parse CLI arguments
    env_variable_names = {
        "db_url": "DB_URL",
        "data_and_models_dir": "BBS_DATA_AND_MODELS_DIR",
    }
    args = parse_args_or_environment(parser, env_variable_names, argv=argv)

    # Configure logging
    configure_logging(args.log_file, args.log_level)

    logger = logging.getLogger("Mining cache entrypoint")

    logger.info(" Configuration ".center(80, "-"))
    for k, v in vars(args).items():
        logger.info(f"{k:<32}: {v}")
    logger.info("-" * 80)

    # Loading libraries
    logger.info("Loading libraries")
    from ..database import CreateMiningCache

    # Database type
    logger.info("Parsing the database type")
    if args.db_type == "sqlite":
        database_path = pathlib.Path(args.db_url)
        if not database_path.exists():
            raise FileNotFoundError(f"No database found at {database_path}.")
        database_url = f"sqlite:///{database_path}"
    elif args.db_type == "mysql":
        password = getpass.getpass("MySQL root password: ")
        database_url = f"mysql+pymysql://root:{password}@{args.db_url}"
    else:  # pragma: no cover
        # Will never get here because `parser.parse_args()` will fail first.
        # This is because we have choices=("mysql", "sqlite") in the
        # argparse parameters
        raise ValueError("Invalid database type specified under --db-type")

    # Create the database engine
    logger.info("Creating the database engine")
    # The NullPool prevents the Engine from using any connection more than once
    # This is important for multiprocessing
    database_engine = sqlalchemy.create_engine(database_url, poolclass=NullPool)

    # Load the models library
    logger.info("Loading the available spacy models")
    ee_models_paths = get_available_spacy_models(args.data_and_models_dir)

    # Restrict to given models
    if args.restrict_to_etypes is not None:
        logger.info("Restricting to a subset of entity types")
        etype_selection = args.restrict_to_etypes.split(",")
        etype_selection = set(map(lambda s: s.strip().upper(), etype_selection))
        for etype in etype_selection:
            if etype not in ee_models_paths:
                logger.warning(
                    f"Can't restrict to etype {etype} because it was not "
                    f"found in data_and_models folder. This entry will be ignored."
                )

        ee_models_paths = {
            etype: path
            for etype, path in ee_models_paths.items()
            if etype in etype_selection
        }

    # Create the cache creation class and run the cache creation
    logger.info("Creating the cache miner")
    cache_creator = CreateMiningCache(
        database_engine=database_engine,
        ee_models_paths=ee_models_paths,
        target_table_name=args.target_table_name,
        workers_per_model=args.n_processes_per_model,
    )

    logger.info("Launching the mining")
    cache_creator.construct()

    logger.info("All done, bye")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_create_mining_cache())
