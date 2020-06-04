"""EntryPoint for the creation of the database."""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",
                    default="/raid/covid_data/data/v7/CORD-19-research-challenge/",
                    type=str,
                    help="The directory path where the metadata.csv and json files are located, "
                         "files needed to create the database")
parser.add_argument("--out_dir",
                    default='/raid/bbs_data/cord19_v7/databases/',
                    type=str,
                    help="The directory path where the database is saved")
parser.add_argument("--version",
                    default='v1',
                    type=str,
                    help="The version of the database")
args = parser.parse_args()


def main():
    from pathlib import Path
    from .database import CORD19DatabaseCreation

    """Create Database."""
    db = CORD19DatabaseCreation(
        data_path=Path(args.data_path),
        version=args.version,
        saving_directory=Path(args.out_dir))
    db.construct()


if __name__ == "__main__":
    main()
