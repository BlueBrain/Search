"""The entrypoint script for the search server."""
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
parser.add_argument("--models_path",
                    default="/raid/bbs_data/trained_models",
                    type=str,
                    help="The folder with pretrained models")
parser.add_argument("--embeddings_path",
                    default="/raid/bbs_data/cord19_v7/embeddings",
                    type=str,
                    help="The folder with the precomputed embeddings")
parser.add_argument("--database_path",
                    default="/raid/bbs_data/cord19_v7/databases/cord19.db",
                    type=str,
                    help="The path to the SQL database.")
args = parser.parse_args()


def main():
    """Execute the entry point."""
    # Configure logging
    log_dir = os.getenv("LOG_DIR", "/")
    log_name = os.getenv("LOG_NAME", "bbs_search.log")
    configure_logging(log_dir, log_name)

    # Start server
    import pathlib
    from flask import Flask
    import sqlalchemy
    from ..server.search_server import SearchServer

    app = Flask("BBSearch Server")
    models_path = pathlib.Path(args.models_path)
    embeddings_path = pathlib.Path(args.embeddings_path)
    engine = sqlalchemy.create_engine(f"sqlite:///{args.database_path}")

    SearchServer(app, models_path, embeddings_path, engine)
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=False,
    )


if __name__ == "__main__":
    main()
