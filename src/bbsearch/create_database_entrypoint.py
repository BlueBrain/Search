"""EntryPoint for the creation of the database."""
import argparse
import logging
from pathlib import Path

from bbsearch.database import DatabaseCreation

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",
                    default="/raid/covid_data/data/v7/CORD-19-research-challenge/",
                    type=str,
                    help="The directory path where the metadata.csv and json files are located, "
                         "files needed to create the database")
parser.add_argument("--saving_directory",
                    default='/raid/covid_data/data/v7/',
                    type=str,
                    help="The directory path where the database is saved")
parser.add_argument("--version",
                    default='v1',
                    type=str,
                    help="The version of the database")
args = parser.parse_args()


def main():
    """Create Database."""
    db = DatabaseCreation(data_path=Path(args.data_path),
                          version=args.version,
                          saving_directory=Path(args.saving_directory))
    db.construct()


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    main()
