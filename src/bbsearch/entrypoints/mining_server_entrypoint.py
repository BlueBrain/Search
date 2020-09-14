"""The entrypoint script for the mining server."""
import argparse
import logging
import os
import pathlib

from ._helper import configure_logging

parser = argparse.ArgumentParser()
parser.add_argument("--host",
                    default="0.0.0.0",
                    type=str,
                    help="The server host IP")
parser.add_argument("--port",
                    default=8080,
                    type=int,
                    help="The server port")
parser.add_argument("--ee_models_lib",
                    default="/raid/sync/proj115/bbs_data/models_libraries/ee_models_library.csv",
                    type=str,
                    help="The csv file with info on which model to use to mine which entity type.")
parser.add_argument("--db_type",
                    default="mysql",
                    type=str,
                    help="Type of the database. Possible values: (sqlite, "
                         "mysql)")
args = parser.parse_args()


def main():
    """Execute the entry point."""
    # Configure logging
    log_dir = os.getenv("LOG_DIR", "/")
    log_name = os.getenv("LOG_NAME", "bbs_mining.log")
    log_file = pathlib.Path(log_dir) / log_name
    configure_logging(log_file, logging.INFO)

    # Start server
    import sqlalchemy
    from flask import Flask

    from ..server.mining_server import MiningServer

    app = Flask("BBS Mining Server")

    if args.db_type == 'sqlite':
        database_path = '/raid/sync/proj115/bbs_data/cord19_v47/databases/cord19.db'
        if not pathlib.Path(database_path).exists():
            pathlib.Path(database_path).touch()
        engine = sqlalchemy.create_engine(f'sqlite:///{database_path}')
    elif args.db_type == 'mysql':
        mysql_uri = input('MySQL URI:')
        engine = sqlalchemy.create_engine(f'mysql+mysqldb://guest:guest'
                                          f'@{mysql_uri}/cord19_v47?charset=utf8mb4')
    else:
        raise ValueError('This is not an handled db_type.')

    MiningServer(app=app,
                 models_libs={'ee': args.ee_models_lib},
                 connection=engine
                 )
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=False,
    )


if __name__ == "__main__":
    main()
