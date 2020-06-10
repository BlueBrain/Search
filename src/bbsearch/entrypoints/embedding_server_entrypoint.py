"""Entrypoint for launching an embedding server."""
import argparse

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


def main():
    """Parse arguments and run Flask application."""
    import pathlib
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
        debug=True,
    )


if __name__ == "__main__":
    main()
