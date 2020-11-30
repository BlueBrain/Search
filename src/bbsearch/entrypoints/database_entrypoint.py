"""EntryPoint for the creation of the database."""
import argparse
import getpass
import logging
import pathlib
import sys

from ._helper import configure_logging


def main(argv=None):
    """Run database construction."""
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
    args = parser.parse_args(argv)
    print(" Configuration ".center(80, "-"))
    print(f"log-dir   : {args.log_dir}")
    print(f"log-name  : {args.log_name}")
    print(f"data-path : {args.data_path}")
    print(f"db-type   : {args.db_type}")
    print("-" * 80)
    input("Press any key to continue... ")

    # Configure logging
    log_file = pathlib.Path(args.log_dir) / args.log_name
    configure_logging(log_file, logging.INFO)
    logger = logging.getLogger(pathlib.Path(__file__).stem)

    # Import libraries
    logger.info("Loading libraries")
    import sqlalchemy

    from ..database import CORD19DatabaseCreation

    # Initialise SQL database engine
    logger.info("Initialising the SQL database engine")
    if args.db_type == "sqlite":
        database_path = pathlib.Path(
            "/raid/sync/proj115/bbs_data/cord19_v65/databases/cord19.db"
        )
        if not database_path.exists():
            database_path.parent.mkdir(exist_ok=True, parents=True)
            database_path.touch()
        engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    elif args.db_type == "mysql":
        # We assume the database `cord19_v65` already exists
        mysql_uri = input("MySQL URL: ")
        password = getpass.getpass("MySQL root password: ")
        engine = sqlalchemy.create_engine(
            f"mysql+pymysql://root:{password}@{mysql_uri}/cord19_v65"
        )
    else:
        raise ValueError(f'"{args.db_type}" is not a supported db_type.')

    # Launch database creation
    logger.info("Starting the database creation")
    db = CORD19DatabaseCreation(data_path=args.data_path, engine=engine)
    db.construct()


if __name__ == "__main__":
    sys.exit(main())
