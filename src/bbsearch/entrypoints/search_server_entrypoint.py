"""The entrypoint script for the search server."""
import argparse
import logging
import os

from ._helper import configure_logging

from bbsearch.utils import H5


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
parser.add_argument("--models_path",
                    default="/raid/sync/proj115/bbs_data/trained_models",
                    type=str,
                    help="The folder with pretrained models")
parser.add_argument("--embeddings_path",
                    default="/raid/sync/proj115/bbs_data/cord19_v35/embeddings/embeddings.h5",
                    type=str,
                    help="The path to an h5 file with the precomputed embeddings")
parser.add_argument("--database_uri",
                    default="dgx1.bbp.epfl.ch:8853/cord19_v35",
                    type=str,
                    help="The URI to the MySQL database.")
parser.add_argument("--debug",
                    action="store_true",
                    default=False,
                    help="Enable debug logging messages")
args = parser.parse_args()


def main():
    """Execute the entry point."""
    # Configure logging
    log_dir = os.getenv("LOG_DIR", "/")
    log_name = os.getenv("LOG_NAME", "bbs_search.log")
    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    configure_logging(log_dir, log_name, log_level)

    # Start server
    import pathlib
    from flask import Flask
    import sqlalchemy
    from ..server.search_server import SearchServer

    app = Flask("BBSearch Server")
    models_path = pathlib.Path(args.models_path)
    embeddings_path = pathlib.Path(args.embeddings_path)

    indices = H5.find_populated_rows(embeddings_path, 'BSV')

    engine = sqlalchemy.create_engine(f"mysql+pymysql://guest:guest@{args.database_uri}")

    SearchServer(app, models_path, embeddings_path, indices, engine)
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=False,
    )


if __name__ == "__main__":
    main()
