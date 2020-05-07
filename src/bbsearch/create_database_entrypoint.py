import argparse
import logging
from pathlib import Path

from bbsearch.sql import DatabaseCreation

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",
                    default="/raid/covid_data/data/v7/",
                    type=str,
                    help="The path to the dataset needed to create the database")
parser.add_argument("--saving_directory",
                    default='/raid/covid_data/data/v7/',
                    type=str,
                    help="The path to the directory where the database is saved")
parser.add_argument("--version",
                    default='v1',
                    type=str,
                    help="The version of the database")
parser.add_argument("--cord_path",
                    default='/raid/covid_data/data/v7/CORD-19-research-challenge/',
                    type=str,
                    help="The directory where the metadata.csv and json files are located, "
                         "files needed to create the database")
args = parser.parse_args()


def main():
    db = DatabaseCreation(data_path=Path(args.data_path),
                          version=args.version,
                          saving_directory=Path(args.saving_directory),
                          cord_path=Path(args.cord_path))
    db.construct()


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    main()
