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
parser.add_argument("--database_uri",
                    default="dgx1.bbp.epfl.ch:8853/cord19_v47",
                    type=str,
                    help="The URI to the MySQL database.")
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

    engine = sqlalchemy.create_engine(f'mysql+mysqldb://guest:guest'
                                      f'@{args.database_uri}?charset=utf8mb4')

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
