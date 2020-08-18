"""EntryPoint for mining a database and saving of extracted items in a cache."""
import argparse
import getpass
import logging
import pathlib

from ._helper import configure_logging


def run_create_mining_cache(argv=None):
    """Mine all texts in database and save results in a cache.

    Parameters
    ----------
    argv : list_like of str
        The command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_type",
        default="mysql",
        type=str,
        help="Type of the database. Possible values: (sqlite, mysql)",
    )
    parser.add_argument(
        "--database_uri",
        default="dgx1.bbp.epfl.ch:8853/cord19_v35",
        type=str,
        help="The URI to the MySQL database.",
    )
    parser.add_argument(
        "--ee_models_library_file",
        default="/raid/sync/proj115/bbs_data/models_libraries/ee_models_library.csv",
        type=str,
        help="The csv file with info on which model to use to mine which entity type.",
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
        "--restrict_to_models",
        type=str,
        default=None,
        help="Comma-separated list of models (as called in ee_models_library_file)"
        "to be run to populate the cache. By default, all models in "
        "ee_models_library_file are run.",
    )
    parser.add_argument(
        "--log_file",
        "-l",
        type=str,
        metavar="<filename>",
        default="mining_cache_log.txt",
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

    import pandas as pd

    from ..database import CreateMiningCache

    # Configure logging
    log_file = pathlib.Path(args.log_file).resolve()
    log_dir = str(log_file.parent)
    log_file_name = log_file.name
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    else:
        level = logging.WARNING
    configure_logging(log_dir, log_file_name, level)
    logger = logging.getLogger(__name__)

    # Database type
    if args.db_type == "sqlite":
        database_path = "/raid/sync/proj115/bbs_data/cord19_v35/databases/cord19.db"
        if not pathlib.Path(database_path).exists():
            pathlib.Path(database_path).touch()
        database_url = f"sqlite:///{database_path}"
    elif args.db_type == "mysql":
        password = getpass.getpass()
        database_url = f"mysql+pymysql://root:{password}@{args.database_uri}"
    else:
        raise ValueError("This is not a valid db_type.")

    # Load the models library
    ee_models_library = pd.read_csv(args.ee_models_library_file)

    # Restrict to given models
    if args.restrict_to_models is None:
        restrict_to_models = ee_models_library.model.unique().tolist()
    else:
        restrict_to_models = [
            model_path.strip() for model_path in args.restrict_to_models.split(",")
        ]
        for model_path in restrict_to_models:
            if model_path not in ee_models_library["model"]:
                logger.warning(
                    f"Can't restrict to model {model_path} because it is not "
                    f"listed in the models library file {args.ee_models_library_file}. "
                    "This entry will be ignored."
                )

    # Create the cache creation class and run the cache creation
    cache_creator = CreateMiningCache(
        database_url=database_url,
        ee_models_library=ee_models_library,
        restrict_to_models=restrict_to_models,
        workers_per_model=args.n_processes_per_model,
    )
    cache_creator.construct()


if __name__ == "__main__":
    exit(run_create_mining_cache())
