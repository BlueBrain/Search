"""EntryPoint for the creation of the database."""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",
                    default="/raid/covid_data/data/v7/CORD-19-research-challenge/",
                    type=str,
                    help="The directory path where the metadata.csv and json files are located, "
                         "files needed to create the database")
parser.add_argument("--db_type",
                    default="sqlite",
                    type=str,
                    help="Type of database. Possible values: (sqlite, mysql)")
args = parser.parse_args()


def main():
    """Run database construction."""
    from pathlib import Path
    import getpass
    import sqlalchemy
    from ..database import CORD19DatabaseCreation

    if args.db_type == 'sqlite':
        engine = sqlalchemy.create_engine('sqlite:///raid/bbs_data/cord19_v35/databases/')
    elif args.db_type == 'mysql':
        password = getpass.getpass('Password:' )
        engine = sqlalchemy.create_engine(f'mysql+pymysql://root:{password}'
                                          f'@dgx1.bbp.epfl.ch:8853/cord19_v35')
    else:
        raise ValueError('This is not an handled db_type.')

    db = CORD19DatabaseCreation(
        data_path=Path(args.data_path),
        engine=engine)
    db.construct()


if __name__ == "__main__":
    main()
