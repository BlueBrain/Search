"""EntryPoint for the creation of the database."""
import argparse
import logging
import pathlib
import sys

from ._helper import configure_logging


def run_create_database(argv=None):
    """Run the CLI entry point.

    Parameters
    ----------
    argv : list_like of str
        The command line arguments.
    """
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
        default="/raid/sync/proj115/bbs_data/cord19_v47",
        type=str,
        help=(
            "The directory path where the metadata.csv and json files are "
            "located. Files needed to create the database."
        ),
    )
    parser.add_argument(
        "--db-type",
        default="sqlite",
        type=str,
        help="Type of database. Possible values: (sqlite, mysql)",
    )
    args = parser.parse_args(argv)

    """Run database construction."""
    # Configure logging
    log_file = pathlib.Path(args.log_dir) / args.log_name
    configure_logging(log_file, logging.INFO)

    import getpass
    from pathlib import Path

    import sqlalchemy

    from ..database import CORD19DatabaseCreation

    if args.db_type == "sqlite":
        database_path = "/raid/sync/proj115/bbs_data/cord19_v47/databases/cord19.db"
        if not Path(database_path).exists():
            Path(database_path).touch()
        engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    elif args.db_type == "mysql":
        # We assume the database `cord19_v35` already exists
        mysql_uri = input("MySQL URI:")
        password = getpass.getpass("Password:")
        engine = sqlalchemy.create_engine(
            f"mysql+pymysql://root:{password}" f"@{mysql_uri}/cord19_v47"
        )
    else:
        raise ValueError(f'"{args.db_type}" is not supported as a db_type.')

    db = CORD19DatabaseCreation(data_path=Path(args.data_path), engine=engine)
    db.construct()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_create_database())
