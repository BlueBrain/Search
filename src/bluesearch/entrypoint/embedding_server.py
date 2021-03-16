"""Entrypoint for launching an embedding server."""

# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import logging
import sys

from ..embedding_models import get_embedding_model
from ._helper import configure_logging, get_var, run_server


def get_embedding_app():
    """Construct the embedding flask app."""
    from ..server.embedding_server import EmbeddingServer

    # Read configuration
    log_file = get_var("BBS_EMBEDDING_LOG_FILE", check_not_set=False)
    log_level = get_var("BBS_EMBEDDING_LOG_LEVEL", logging.INFO, var_type=int)

    # Configure logging
    configure_logging(log_file, log_level)
    logger = logging.getLogger(__name__)

    logger.info(" Configuration ".center(80, "-"))
    logger.info(f"log-file            : {log_file}")
    logger.info(f"log-level           : {log_level}")
    logger.info("-" * 80)

    # Load embedding models
    logger.info("Loading embedding models")
    supported_models = ["SBERT", "SBioBERT", "BioBERT NLI+STS"]
    embedding_models = {
        model_name: get_embedding_model(model_name) for model_name in supported_models
    }

    # Create Server app
    logger.info("Creating the server app")
    embedding_app = EmbeddingServer(embedding_models)

    return embedding_app


def run_embedding_server():
    """Run the embedding server."""
    run_server(get_embedding_app, "embedding")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_embedding_server())
