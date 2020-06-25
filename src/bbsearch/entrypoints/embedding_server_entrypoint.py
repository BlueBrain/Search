"""Entrypoint for launching an embedding server."""
import argparse
import logging
import os
import pathlib
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--host",
                    default="0.0.0.0",
                    type=str,
                    help="The server host IP")
parser.add_argument("--port",
                    default=8080,
                    type=int,
                    help="The server port")
parser.add_argument("--bsv_checkpoints",
                    default='/raid/covid_data/assets/BioSentVec_PubMed_MIMICIII-bigram_d700.bin',
                    type=str,
                    help="Path to file containing the checkpoints for the BSV model.")
args = parser.parse_args()


def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    """Exception handler for logging.

    For more information about the parameters see
    https://docs.python.org/3/library/sys.html#sys.exc_info

    Parameters
    ----------
    exc_type
        Type of the exception.
    exc_value
        Exception instance.
    exc_traceback
        Traceback option.

    Note
    ----
    Credit: https://stackoverflow.com/a/16993115/2804645
    """
    logging.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback))


def main():
    """Parse arguments and run Flask application."""

    # Configure logging
    log_dir = os.getenv("LOG_DIR", "/")
    log_name = os.getenv("LOG_NAME", "bbs_embedding.log")
    log_path = pathlib.Path(log_dir) / log_name
    log_path = log_path.resolve()

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s :: %(levelname)-8s :: %(name)s | %(message)s')
    sys.excepthook = handle_uncaught_exception

    # Start server
    from flask import Flask
    from ..server.embedding_server import EmbeddingServer
    from ..embedding_models import USE, SBERT, SBioBERT, BSV

    embedding_models = {
        'USE': USE(),
        'SBERT': SBERT(),
        'BSV': BSV(checkpoint_model_path=pathlib.Path(args.bsv_checkpoints)),
        'SBioBERT': SBioBERT()}

    # Create Server app
    app = Flask("BBSearch Embedding Server")
    EmbeddingServer(app, embedding_models)
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=False,
    )


if __name__ == "__main__":
    main()
