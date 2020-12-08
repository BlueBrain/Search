"""Entrypoint for launching an embedding server."""
import logging
import pathlib
import sys

from ._helper import configure_logging, get_var, run_server


def get_embedding_app():
    """Construct the embedding flask app."""
    from ..embedding_models import BSV, USE, SBioBERT, Sent2VecModel, SentTransformer
    from ..server.embedding_server import EmbeddingServer

    # Read configuration
    log_file = get_var("BBS_EMBEDDING_LOG_FILE", check_not_set=False)
    log_level = get_var("BBS_EMBEDDING_LOG_LEVEL", logging.INFO, var_type=int)

    bsv_checkpoint = get_var("BBS_EMBEDDING_BSV_CHECKPOINT_PATH")
    sent2vec_checkpoint = get_var("BBS_EMBEDDING_SENT2VEC_CHECKPOINT_PATH")

    # Configure logging
    configure_logging(log_file, log_level)
    logger = logging.getLogger(__name__)

    # Load embedding models
    logger.info("Loading embedding models")
    embedding_models = {
        "USE": USE(),
        "SBERT": SentTransformer("bert-base-nli-mean-tokens"),
        "BIOBERT NLI+STS": SentTransformer("clagator/biobert_v1.1_pubmed_nli_sts"),
        "Sent2Vec": Sent2VecModel(pathlib.Path(sent2vec_checkpoint)),
        "BSV": BSV(pathlib.Path(bsv_checkpoint)),
        "SBioBERT": SBioBERT(),
    }

    # Create Server app
    logger.info("Creating the server app")
    embedding_app = EmbeddingServer(embedding_models)

    return embedding_app


def run_embedding_server():
    """Run the embedding server."""
    run_server(get_embedding_app, "embedding")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_embedding_server)
