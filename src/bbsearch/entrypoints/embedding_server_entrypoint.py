"""Entrypoint for launching an embedding server."""
import argparse
import os

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
parser.add_argument("--bsv_checkpoints",
                    default='/raid/sync/proj115/bbs_data/trained_models/BioSentVec_PubMed_MIMICIII-bigram_d700.bin',
                    type=str,
                    help="Path to file containing the checkpoints for the BSV model.")
args = parser.parse_args()


def main():
    """Parse arguments and run Flask application."""
    # Configure logging
    log_dir = os.getenv("LOG_DIR", "/")
    log_name = os.getenv("LOG_NAME", "bbs_embedding.log")
    configure_logging(log_dir, log_name)

    # Start server
    import pathlib

    from flask import Flask

    from ..embedding_models import BSV, SBERT, USE, SBioBERT
    from ..server.embedding_server import EmbeddingServer

    embedding_models = {
        'USE': USE(),
        'SBERT': SBERT(),
        'BSV': BSV(checkpoint_model_path=pathlib.Path(args.bsv_checkpoints)),
        'SBioBERT': SBioBERT()}

    # Create Server app
    app = Flask("BBSearch Embedding Server")
    EmbeddingServer(app=app,
                    embedding_models=embedding_models
                    )
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=False,
    )


if __name__ == "__main__":
    main()
