"""The entrypoint script for the mining server."""
import argparse
import os

from ._helper import configure_logging


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--host",
                    default="0.0.0.0",
                    type=str,
                    help="The server host IP")
parser.add_argument("--port",
                    default=8080,
                    type=int,
                    help="The server port")
parser.add_argument("--ee_models_lib",
                    default="/raid/bbs_data/models_libraries/ee_models_library.csv",
                    type=str,
                    help="The csv file with info on which model to use to mine which entity type.")
parser.add_argument("--database_path",
                    default="/raid/bbs_data/cord19_v7/databases/cord19.db",
                    type=str,
                    help="The path to the database. ")
parser.add_argument("--version",
                    default=None,
                    help="Version.")
args = parser.parse_args()


def main():
    """Execute the entry point."""
    # Configure logging
    log_dir = os.getenv("LOG_DIR", "/")
    log_name = os.getenv("LOG_NAME", "bbs_mining.log")
    configure_logging(log_dir, log_name)

    # Start server
    from flask import Flask
    import sqlalchemy
    from ..server.mining_server import MiningServer

    app = Flask("BBS Mining Server")
    engine = sqlalchemy.create_engine(f"sqlite:///{args.database_path}")

    MiningServer(app=app,
                 models_libs={'ee': args.ee_models_lib},
                 connection=engine,
                 version=args.version
                 )
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=False,
    )


if __name__ == "__main__":
    main()
