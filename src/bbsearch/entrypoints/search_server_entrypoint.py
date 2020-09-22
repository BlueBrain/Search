"""The entrypoint script for the search server."""
import argparse
import logging
import os
import pathlib

import numpy as np

from bbsearch.utils import H5

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
parser.add_argument("--models_path",
                    default="/raid/sync/proj115/bbs_data/trained_models",
                    type=str,
                    help="The folder with pretrained models")
parser.add_argument("--embeddings_path",
                    default="/raid/sync/proj115/bbs_data/cord19_v47/embeddings/embeddings.h5",
                    type=str,
                    help="The path to an h5 file with the precomputed embeddings")
parser.add_argument("--database_uri",
                    default="dgx1.bbp.epfl.ch:8853/cord19_v47",
                    type=str,
                    help="The URI to the MySQL database.")
parser.add_argument("--debug",
                    action="store_true",
                    default=False,
                    help="Enable debug logging messages")
parser.add_argument("--models",
                    default="USE,SBERT,SBioBERT,BSV,Sent2Vec",
                    type=str,
                    help="Models to load in the search server.")
args = parser.parse_args()


def main():
    """Execute the entry point."""
    # Configure logging
    log_dir = os.getenv("LOG_DIR", ".")
    log_name = os.getenv("LOG_NAME", "bbs_search.log")
    log_file = pathlib.Path(log_dir) / log_name
    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    configure_logging(log_file, log_level)

    # Start server
    import sqlalchemy
    from flask import Flask

    from ..server.search_server import SearchServer

    app = Flask("BBSearch Server")
    models_path = pathlib.Path(args.models_path)
    embeddings_path = pathlib.Path(args.embeddings_path)

    n_sentences, dim_embedding = H5.get_shape(embeddings_path, 'BSV')
    # 0th row is for padding and is filled with NaNs
    # here we're assuming that all embeddings (up to the 0th row)
    # are correctly populated
    indices = np.arange(1, n_sentences)

    engine = sqlalchemy.create_engine(f"mysql+mysqldb://guest:guest@{args.database_uri}")

    models = [model.strip() for model in args.models.split(",")]

    SearchServer(app, models_path, embeddings_path, indices, engine, models)

    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=False,
    )


if __name__ == "__main__":
    main()
