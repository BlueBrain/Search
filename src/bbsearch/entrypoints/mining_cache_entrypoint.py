"""EntryPoint for mining a database and saving of extracted items in a cache."""
import argparse
import logging

from ._helper import configure_logging

parser = argparse.ArgumentParser()
parser.add_argument(
    "--db_type",
    default="mysql",
    type=str,
    help="Type of the database. Possible values: (sqlite, " "mysql)",
)
parser.add_argument(
    "--database_uri",
    default="dgx1.bbp.epfl.ch:8853/cord19_v35",
    type=str,
    help="The URI to the MySQL database.",
)
parser.add_argument(
    "--ee_models_library_file",
    default='/raid/sync/proj115/bbs_data/models_libraries/ee_models_library.csv',
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
args = parser.parse_args()


def main():
    """Mine all texts in database and save results in a cache."""
    import getpass
    import pathlib

    import pandas as pd

    from bbsearch.mining import CreateMiningCache

    print("Parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print()

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

    if args.db_type == "sqlite":
        database_path = "/raid/sync/proj115/bbs_data/cord19_v35/databases/cord19.db"
        if not pathlib.Path(database_path).exists():
            pathlib.Path(database_path).touch()
        database_url = f"sqlite:///{database_path}"
    elif args.db_type == "mysql":
        password = getpass.getpass("Password:")
        database_url = f"mysql+pymysql://root:{password}" f"@{args.database_uri}"
    else:
        raise ValueError("This is not an handled db_type.")

    ee_models_library = pd.read_csv(args.ee_models_library_file)

    if args.restrict_to_models is None:
        restrict_to_models = ee_models_library.model.unique().tolist()
    else:
        restrict_to_models = args.restrict_to_models.split(",")

    cache_creator = CreateMiningCache(
        database_url=database_url,
        ee_models_library=ee_models_library,
        restrict_to_models=restrict_to_models,
        workers_per_model=args.n_processes_per_model
    )
    cache_creator.construct()


if __name__ == "__main__":
    main()
