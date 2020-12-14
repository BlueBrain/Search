"""EntryPoint for mining a database and saving of extracted items in a cache."""
import argparse
import getpass
import logging
import pathlib
import sys

from ..utils import DVC
from ._helper import configure_logging


def run_create_mining_cache(argv=None):  # pragma: no cover
    """Mine all texts in database and save results in a cache.

    Parameters
    ----------
    argv : list_like of str
        The command line arguments.
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        description="Mine the CORD-19 database and cache the results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db-type",
        default="mysql",
        type=str,
        choices=("mysql", "sqlite"),
        help="Type of the database.",
    )
    parser.add_argument(
        "--database-uri",
        default="dgx1.bbp.epfl.ch:8853/cord19_v47",
        type=str,
        help="The URI to the MySQL database.",
    )
    parser.add_argument(
        "--target-table-name",
        default="mining_cache_temporary",
        type=str,
        help="The name of the target mining cache table",
    )
    parser.add_argument(
        "--n_processes_per_model",
        default=1,
        type=int,
        help="Each mining model is run in parallel with respect to the others. In "
        "addition to that, n_processes_per_model are used to run in parallel"
        "a single mining model.",
    )
    parser.add_argument(
        "--restrict-to-models",
        type=str,
        default=None,
        help="Comma-separated list of models (as called in ee_models_library_file)"
        "to be run to populate the cache. By default, all models in "
        "ee_models_library_file are run.",
    )
    parser.add_argument(
        "--log-file",
        "-l",
        type=str,
        metavar="<filename>",
        default=None,
        help="The file for the logs. If not provided the stdout will be used.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="The logging level, -v correspond to INFO, -vv to DEBUG",
    )
    args = parser.parse_args(argv)

    import sqlalchemy
    from sqlalchemy.pool import NullPool

    from ..database import CreateMiningCache

    # Configure logging
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    else:
        level = logging.WARNING
    configure_logging(args.log_file, level)

    logger = logging.getLogger("Mining cache entrypoint")
    logger.info("Welcome to the mining cache creation")
    logger.info("Parameters:")
    logger.info(f"db_type                : {args.db_type}")
    logger.info(f"database_uri           : {args.database_uri}")
    logger.info(f"target_table_name      : {args.target_table_name}")
    logger.info(f"n_processes_per_model  : {args.n_processes_per_model}")
    logger.info(f"restrict_to_models     : {args.restrict_to_models}")
    logger.info(f"log_file               : {args.log_file}")
    logger.info(f"verbose                : {args.verbose}")

    # Database type
    logger.info("Parsing the database type")
    if args.db_type == "sqlite":
        database_path = "/raid/sync/proj115/bbs_data/cord19_v47/databases/cord19.db"
        if not pathlib.Path(database_path).exists():
            raise FileNotFoundError(f"No database found at {database_path}.")
        database_url = f"sqlite:///{database_path}"
    elif args.db_type == "mysql":
        password = getpass.getpass("MySQL root password: ")
        database_url = f"mysql+pymysql://root:{password}@{args.database_uri}"
    else:
        raise ValueError("This is not a valid db_type.")

    # Load the models library
    logger.info("Loading the models library")
    ee_models_library = DVC.load_ee_models_library()

    # Restrict to given models
    if args.restrict_to_models is not None:
        logger.info("Restricting to a subset of models")
        model_selection = args.restrict_to_models.split(",")
        model_selection = set(map(lambda s: s.strip(), model_selection))
        for model_path in model_selection:
            if model_path not in ee_models_library["model"].values:
                logger.warning(
                    f"Can't restrict to model {model_path} because it is not "
                    f"listed in the models library file {args.ee_models_library_file}. "
                    "This entry will be ignored."
                )
        keep_rows = ee_models_library["model"].isin(model_selection)
        ee_models_library = ee_models_library[keep_rows]

    # Create the database engine
    logger.info("Creating the database engine")
    # The NullPool prevents the Engine from using any connection more than once
    # This is important for multiprocessing
    database_engine = sqlalchemy.create_engine(database_url, poolclass=NullPool)

    # Create the cache creation class and run the cache creation
    logger.info("Creating the cache miner")
    cache_creator = CreateMiningCache(
        database_engine=database_engine,
        ee_models_library=ee_models_library,
        target_table_name=args.target_table_name,
        workers_per_model=args.n_processes_per_model,
    )

    logger.info("Launching the mining")
    cache_creator.construct()

    logger.info("All done, bye")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_create_mining_cache())
