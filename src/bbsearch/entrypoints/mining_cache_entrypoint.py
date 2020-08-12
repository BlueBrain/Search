"""EntryPoint for mining a database and saving of extracted items in a cache."""
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    default='/raid/sync/proj115/bbs_data/models_libraries/ee_models_library.csv"',
    type=str,
    help="The csv file with info on which model to use to mine which entity type.",
)
parser.add_argument(
    "--n_processes",
    default="4",
    type=int,
    help="Max n of processes to run the different requested mining models in parallel.",
)
parser.add_argument(
    "--always_mine",
    dest="always_mine",
    action="store_true",
    help="Force running all mining models, even if extracted entities were "
    "already found in the cache for some models.",
)
args = parser.parse_args()


def main():
    """Mine all texts in database and save results in a cache."""
    import getpass
    import pathlib

    import pandas as pd
    import sqlalchemy

    from bbsearch.database import MiningCacheCreation

    if args.db_type == "sqlite":
        database_path = "/raid/sync/proj115/bbs_data/cord19_v35/databases/cord19.db"
        if not pathlib.Path(database_path).exists():
            pathlib.Path(database_path).touch()
        engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    elif args.db_type == "mysql":
        password = getpass.getpass("Password:")
        engine = sqlalchemy.create_engine(
            f"mysql+pymysql://guest:{password}" f"@{args.database_uri}"
        )
    else:
        raise ValueError("This is not an handled db_type.")

    import pdb

    pdb.set_trace()
    ee_models_library = pd.read_csv(args.ee_models_library_file)
    db = MiningCacheCreation(engine=engine)
    db.construct(
        ee_models_library=ee_models_library,
        n_processes=args.n_processes,
        always_mine=args.always_mine,
    )


if __name__ == "__main__":
    main()
