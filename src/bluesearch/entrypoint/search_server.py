"""The entrypoint script for the search server."""

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
import pathlib
import sys

import sqlalchemy

from ._helper import configure_logging, get_var, run_server


def get_search_app():
    """Construct the search flask app."""
    from ..server.search_server import SearchServer
    from ..utils import H5

    # Read configuration
    log_file = get_var("BBS_SEARCH_LOG_FILE", check_not_set=False)
    log_level = get_var("BBS_SEARCH_LOG_LEVEL", logging.INFO, var_type=int)

    models_path = get_var("BBS_SEARCH_MODELS_PATH")
    embeddings_path = get_var("BBS_SEARCH_EMBEDDINGS_PATH")
    which_models = get_var("BBS_SEARCH_MODELS")

    mysql_url = get_var("BBS_SEARCH_DB_URL")
    mysql_user = get_var("BBS_SEARCH_MYSQL_USER")
    mysql_password = get_var("BBS_SEARCH_MYSQL_PASSWORD")

    # Configure logging
    configure_logging(log_file, log_level)
    logger = logging.getLogger(__name__)

    logger.info(" Configuration ".center(80, "-"))
    logger.info(f"log-file          : {log_file}")
    logger.info(f"log-level         : {log_level}")
    logger.info(f"models-path       : {models_path}")
    logger.info(f"embeddings-path   : {embeddings_path}")
    logger.info(f"which-models      : {which_models}")
    logger.info(f"mysql_url         : {mysql_url}")
    logger.info(f"mysql_user        : {mysql_user}")
    logger.info(f"mysql_password    : {mysql_password}")
    logger.info("-" * 80)

    # Initialize flask app
    logger.info("Creating the Flask app")
    models_path = pathlib.Path(models_path)
    embeddings_path = pathlib.Path(embeddings_path)
    engine_url = f"mysql://{mysql_user}:{mysql_password}@{mysql_url}"
    engine = sqlalchemy.create_engine(engine_url, pool_recycle=14400)
    models_list = [model.strip() for model in which_models.split(",")]
    indices = H5.find_populated_rows(embeddings_path, models_list[0])

    server_app = SearchServer(
        models_path, embeddings_path, indices, engine, models_list
    )
    return server_app


def run_search_server():
    """Run the search server."""
    run_server(get_search_app, "search")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_search_server())
